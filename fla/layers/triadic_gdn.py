# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.triadic_gdn import chunk_triadic_gdn
from fla.ops.triadic_gdn.factors import fuse_key_factors


def _inv_softplus_dt(n: int) -> torch.Tensor:
    dt = torch.exp(torch.rand(n) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
    return dt + torch.log(-torch.expm1(-dt))


class TriadicGatedDeltaNet(nn.Module):
    """
    Triadic Gated DeltaNet layer (https://arxiv.org/abs/2609.36529), with an n-way state.

    `key_factors=(E,)` is Triadic GDN with `E` slices, `key_factors=(E_1, ..., E_m)` keeps an n-way state
    `[E_1, ..., E_m, K, V]` per head written with the joint key `k_1 ⊗ ... ⊗ k_m ⊗ k`, and `key_factors=(1,)` is GDN.
    Each factor has its own short conv, softplus + L2 norm and gate `-exp(A_a) * softplus(W_a x + b_a)`.
    With `channel_decay=True`, a KDA-style gate per key channel is added (Triadic KDA).

    Training and full-sequence forward only (no cache).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_heads: int = 16,
        expand_v: float = 1.0,
        key_factors: tuple[int, ...] = (4,),
        channel_decay: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
        **kwargs,
    ) -> TriadicGatedDeltaNet:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim
        self.key_factors = tuple(int(e) for e in key_factors)
        self.sum_e = sum(self.key_factors)
        self.channel_decay = channel_decay
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.k2_proj = nn.Linear(hidden_size, num_heads * self.sum_e, bias=False)
        self.q2_proj = nn.Linear(hidden_size, num_heads * self.sum_e, bias=False)
        self.a2_proj = nn.Linear(hidden_size, num_heads * self.sum_e, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.A2_log = nn.Parameter(torch.empty(num_heads * self.sum_e).uniform_(0, 16).log())
        self.A2_log._no_weight_decay = True
        self.dt2_bias = nn.Parameter(_inv_softplus_dt(num_heads * self.sum_e))
        self.dt2_bias._no_weight_decay = True

        if channel_decay:
            self.f_proj = nn.Sequential(
                nn.Linear(hidden_size, self.head_v_dim, bias=False),
                nn.Linear(self.head_v_dim, self.key_dim, bias=False),
            )
            self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1, 16).log())
            self.A_log._no_weight_decay = True
            self.dt_bias = nn.Parameter(_inv_softplus_dt(self.key_dim))
            self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(hidden_size=self.value_dim, kernel_size=conv_size, activation='silu')
            self.k2_conv1d = ShortConvolution(hidden_size=num_heads * self.sum_e, kernel_size=conv_size, activation=None)
            self.q2_conv1d = ShortConvolution(hidden_size=num_heads * self.sum_e, kernel_size=conv_size, activation=None)

        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(hidden_size=self.head_v_dim, eps=norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _factors(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = rearrange(x, '... (h e) -> ... h e', h=self.num_heads)
        return list(torch.split(x, self.key_factors, dim=-1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        if use_cache or past_key_values is not None:
            raise NotImplementedError("TriadicGatedDeltaNet runs full sequences only.")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError("TriadicGatedDeltaNet does not take padding masks.")
        x = hidden_states
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        k2, q2 = self.k2_proj(x), self.q2_proj(x)
        if self.use_short_conv:
            q, _ = self.q_conv1d(q)
            k, _ = self.k_conv1d(k)
            v, _ = self.v_conv1d(v)
            k2, _ = self.k2_conv1d(k2)
            q2, _ = self.q2_conv1d(q2)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
        # the chunk kernels take q, k, v, k2 and q2 in one dtype, the gates and beta in fp32
        dtype = v.dtype
        q = F.normalize(rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim).float(), dim=-1).to(dtype)
        k = F.normalize(rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim).float(), dim=-1).to(dtype)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        k2s = [F.normalize(F.softplus(t.float()), dim=-1).to(dtype) for t in self._factors(k2)]
        q2s = [F.normalize(F.softplus(t.float()), dim=-1).to(dtype) for t in self._factors(q2)]
        g2 = -self.A2_log.float().exp() * F.softplus(self.a2_proj(x).float() + self.dt2_bias)
        k2, q2, g2 = fuse_key_factors(k2s, q2s, self._factors(g2))
        beta = self.b_proj(x).float().sigmoid()

        if self.channel_decay:
            from fla.ops.triadic_kda import chunk_triadic_kda
            gk = F.softplus(self.f_proj(x).float() + self.dt_bias)
            gk = -self.A_log.float().exp()[:, None] * rearrange(gk, '... (h d) -> ... h d', d=self.head_k_dim)
            o, _ = chunk_triadic_kda(q=q, k=k, v=v, k2=k2, q2=q2, gk=gk, g2=g2, beta=beta)
        else:
            o, _ = chunk_triadic_gdn(q=q, k=k, v=v, k2=k2, q2=q2, g=g2, beta=beta)

        o = self.o_norm(o, rearrange(self.g_proj(x), '... (h d) -> ... h d', d=self.head_v_dim))
        o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))
        return o, None, past_key_values
