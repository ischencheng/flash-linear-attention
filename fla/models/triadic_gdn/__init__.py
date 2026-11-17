# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.triadic_gdn.configuration_triadic_gdn import TriadicGatedDeltaNetConfig
from fla.models.triadic_gdn.modeling_triadic_gdn import TriadicGatedDeltaNetForCausalLM, TriadicGatedDeltaNetModel

AutoConfig.register(TriadicGatedDeltaNetConfig.model_type, TriadicGatedDeltaNetConfig, exist_ok=True)
AutoModel.register(TriadicGatedDeltaNetConfig, TriadicGatedDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(TriadicGatedDeltaNetConfig, TriadicGatedDeltaNetForCausalLM, exist_ok=True)

__all__ = ['TriadicGatedDeltaNetConfig', 'TriadicGatedDeltaNetForCausalLM', 'TriadicGatedDeltaNetModel']
