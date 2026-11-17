# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.triadic_gdn import chunk_triadic_gdn, fuse_key_factors

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


class TriadicGatedDeltaNet(nn.Module):
    """
    Triadic Gated DeltaNet layer (https://arxiv.org/abs/2609.36529), with an n-way state.

    Each head keeps a state of shape `[E_1, ..., E_m, K, V]` for `key_factors=(E_1, ..., E_m)`, written with the
    joint key `k_1 ⊗ ... ⊗ k_m ⊗ k`. `key_factors=(E,)` is Triadic GDN with `E` slices.
    Each factor has its own short conv, softplus + L2 norm and gate `-exp(A_a) * softplus(W_a x + b_a)`,
    so the second key, query and gate projections have `E_1 + ... + E_m` outputs per head.
    The cached recurrent state is stored flattened as `[N, H, E_1 * ... * E_m, K, V]`.
    The chunk kernel keeps the `E_1 * ... * E_m` slices of a state block in registers and is tested with up to 16 slices.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        key_factors (int | tuple[int, ...], Optional):
            The sizes `E_1, ..., E_m` of the second key factors. An int `E` means `(E,)`. Default: `(4,)`.
        mode (str, Optional):
            Which kernel to use. Currently only `chunk` is supported. Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution. Default: `False`.
        norm_eps (float, Optional):
            The epsilon value for the output normalization layer. Default: 1e-5.
        output_gate_activation (str, Optional):
            The activation of the output gate, one of `swish`, `silu` and `sigmoid`. Default: `sigmoid`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_heads: int = 16,
        expand_v: float = 1.0,
        key_factors: int | tuple[int, ...] = (4,),
        mode: str = 'chunk',
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        output_gate_activation: str = 'sigmoid',
        layer_idx: int | None = None,
        **kwargs,
    ) -> TriadicGatedDeltaNet:
        super().__init__()
        if mode != 'chunk':
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        key_factors = (key_factors,) if isinstance(key_factors, int) else tuple(int(e) for e in key_factors)
        if len(key_factors) == 0 or min(key_factors) < 1:
            raise ValueError(f"key_factors must be a non-empty tuple of positive sizes, got {key_factors}.")
        if output_gate_activation not in ('swish', 'silu', 'sigmoid'):
            raise ValueError(f"output_gate_activation must be 'swish', 'silu', or 'sigmoid', got {output_gate_activation!r}.")

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim
        self.key_factors = key_factors
        self.num_slices = math.prod(key_factors)
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.layer_idx = layer_idx

        factor_dim = num_heads * sum(key_factors)
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.k2_proj = nn.Linear(hidden_size, factor_dim, bias=False)
        self.q2_proj = nn.Linear(hidden_size, factor_dim, bias=False)
        self.a2_proj = nn.Linear(hidden_size, factor_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        A = torch.empty(factor_dim, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(factor_dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=conv_bias, activation='silu')
            self.k2_conv1d = ShortConvolution(factor_dim, conv_size, bias=conv_bias, activation=None)
            self.q2_conv1d = ShortConvolution(factor_dim, conv_size, bias=conv_bias, activation=None)

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps, activation=output_gate_activation)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _split_factors(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = rearrange(x, '... (h e) -> ... h e', h=self.num_heads)
        return list(torch.split(x, self.key_factors, dim=-1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        last_state = get_layer_cache(module=self, past_key_values=past_key_values)

        cu_seqlens = kwargs.get('cu_seqlens')
        hidden_states, indices, cu_seqlens = unpad_hidden_states(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            q_len=q_len,
        )

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        k2, q2 = self.k2_proj(hidden_states), self.q2_proj(hidden_states)
        conv_state = None
        if self.use_short_conv:
            conv_state = last_state['conv_state'] if last_state is not None else (None,) * 5
            convs = (self.q_conv1d, self.k_conv1d, self.v_conv1d, self.k2_conv1d, self.q2_conv1d)
            outputs = [
                conv(x=x, cache=cache, output_final_state=use_cache, cu_seqlens=cu_seqlens)
                for conv, x, cache in zip(convs, (q, k, v, k2, q2), conv_state)
            ]
            (q, k, v, k2, q2), conv_state = zip(*outputs)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)

        # the chunk kernel takes q, k, v, k2 and q2 in one dtype, the gates and beta in fp32
        dtype = v.dtype
        q = F.normalize(rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim).float(), dim=-1).to(dtype)
        k = F.normalize(rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim).float(), dim=-1).to(dtype)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        k2s = [F.normalize(F.softplus(x.float()), dim=-1).to(dtype) for x in self._split_factors(k2)]
        q2s = [F.normalize(F.softplus(x.float()), dim=-1).to(dtype) for x in self._split_factors(q2)]
        g = -self.A_log.float().exp() * F.softplus(self.a2_proj(hidden_states).float() + self.dt_bias)
        k2, q2, g = fuse_key_factors(k2s, q2s, self._split_factors(g))
        beta = self.b_proj(hidden_states).float().sigmoid()

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        o, recurrent_state = chunk_triadic_gdn(
            q=q,
            k=k,
            v=v,
            k2=k2,
            q2=q2,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        update_layer_cache(
            module=self,
            past_key_values=past_key_values,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            offset=q_len,
        )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim))
        o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))
        o = repad_hidden_states(hidden_states=o, indices=indices, batch_size=batch_size, q_len=q_len)

        return o, None, past_key_values
