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
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import logging

from fla.layers.triadic_gdn import TriadicGatedDeltaNet
from fla.models.generation_utils import FLAUnsupportedCacheGenerationMixin
from fla.models.modeling_utils import FLAForCausalLM, FLAModel, FLAPreTrainedModel, get_attention
from fla.models.triadic_gdn.configuration_triadic_gdn import TriadicGatedDeltaNetConfig
from fla.modules import GatedMLP as TriadicGatedDeltaNetMLP
from fla.modules import RMSNorm

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.cache_utils import Cache


logger = logging.get_logger(__name__)


class TriadicGatedDeltaNetBlock(GradientCheckpointingLayer):

    def __init__(self, config: TriadicGatedDeltaNetConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = get_attention(config=config, layer_idx=layer_idx)
        if self.attn is None:
            self.attn = TriadicGatedDeltaNet(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                expand_v=config.expand_v,
                key_factors=config.key_factors,
                mode=config.attn_mode,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                output_gate_activation=config.output_gate_activation,
                layer_idx=layer_idx,
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TriadicGatedDeltaNetMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        attnres_states: list[torch.Tensor] | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(x=hidden_states, residual=residual, prenorm=True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values, attnres_states


class TriadicGatedDeltaNetPreTrainedModel(FLAPreTrainedModel):

    config_class = TriadicGatedDeltaNetConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['TriadicGatedDeltaNetBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module, prenorm_residual_strategy: str | None = None, num_residuals_per_layer: int = 2):
        if isinstance(module, TriadicGatedDeltaNet) and next(module.parameters()).device.type != 'meta':
            with torch.no_grad():
                if not getattr(module.A_log, '_is_hf_initialized', False):
                    module.A_log.copy_(nn.init.uniform_(module.A_log, a=0, b=16).log())
                module.A_log._no_weight_decay = True
                if not getattr(module.dt_bias, '_is_hf_initialized', False):
                    dt = torch.exp(
                        nn.init.uniform_(module.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
                    ).clamp(min=1e-4)
                    # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                    module.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
                module.dt_bias._no_weight_decay = True
        super()._init_weights(module, prenorm_residual_strategy, num_residuals_per_layer)


class TriadicGatedDeltaNetModel(TriadicGatedDeltaNetPreTrainedModel, FLAModel):

    def __init__(self, config: TriadicGatedDeltaNetConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TriadicGatedDeltaNetBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)],
        )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.use_attnres = False

        self.gradient_checkpointing = False

        self.post_init()


class TriadicGatedDeltaNetForCausalLM(
    TriadicGatedDeltaNetPreTrainedModel, FLAForCausalLM, FLAUnsupportedCacheGenerationMixin,
):

    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = TriadicGatedDeltaNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()
