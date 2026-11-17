# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fla.layers.triadic_gdn import TriadicGatedDeltaNet
from fla.models import TriadicGatedDeltaNetConfig

from .utils import (
    check_forward,
    check_generate,
    check_generation,
    create_model_and_config,
)


@pytest.mark.parametrize('varlen', [False, True], ids=['dense', 'varlen'])
@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'key_factors', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-E{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64, (4,), torch.bfloat16),
            (4, 4, 1024, 4, 64, (2, 2), torch.bfloat16),
            (2, 2, 1000, 4, 64, (2, 3, 2), torch.bfloat16),
        ]
    ],
)
def test_modeling(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    key_factors: tuple[int, ...],
    dtype: torch.dtype,
    varlen: bool,
):
    check_forward(
        L,
        B,
        T,
        H,
        D,
        TriadicGatedDeltaNetConfig,
        use_l2warp=False,
        dtype=dtype,
        varlen=varlen,
        head_dim=D,
        key_factors=key_factors,
    )


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'key_factors', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-E{}-{}".format(*test))
        for test in [
            (2, 4, 2000, 8, 64, (4,), torch.float16),
            (2, 4, 2000, 8, 64, (2, 2), torch.float16),
        ]
    ],
)
def test_generation(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    key_factors: tuple[int, ...],
    dtype: torch.dtype,
):
    model, config = create_model_and_config(
        TriadicGatedDeltaNetConfig, L, H, D, dtype, head_dim=D, key_factors=key_factors,
    )
    check_generation(L, B, T, H, D, TriadicGatedDeltaNetConfig, dtype, model=model, config=config)


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (2, 2, 64, 8, 64, torch.float32),
        ]
    ],
)
def test_generate_prefill(
    L: int,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
):
    check_generate(L, B, T, H, D, TriadicGatedDeltaNetConfig, dtype)


def test_state_shape():
    layer = TriadicGatedDeltaNet(hidden_size=64, head_dim=16, num_heads=2, key_factors=(2, 3), use_short_conv=False)
    assert layer.num_slices == 6
    assert layer.k2_proj.out_features == layer.q2_proj.out_features == layer.a2_proj.out_features == 2 * 5
    assert layer.A_log.shape == layer.dt_bias.shape == (2 * 5,)
    with pytest.raises(ValueError, match='key_factors'):
        TriadicGatedDeltaNet(hidden_size=64, head_dim=16, num_heads=2, key_factors=(2, 0))
    with pytest.raises(ValueError, match='key_factors'):
        TriadicGatedDeltaNetConfig(key_factors=[])


def test_initialization(make_model):
    torch.manual_seed(42)
    model = make_model(TriadicGatedDeltaNetConfig)
    with torch.device('meta'):
        layer = TriadicGatedDeltaNet(hidden_size=64, head_dim=16, num_heads=4, key_factors=(2, 2), use_short_conv=False)
    model._init_weights(layer)
    assert all(parameter.is_meta for parameter in layer.parameters())
    layer.to_empty(device='cpu')
    for name in ('A_log', 'dt_bias'):
        setattr(layer, name, nn.Parameter(torch.full_like(getattr(layer, name), float('nan'))))
    parameters = (layer.A_log, layer.dt_bias)
    model._init_weights(layer)
    for parameter, actual in zip(parameters, (layer.A_log, layer.dt_bias), strict=True):
        assert actual is parameter
        assert torch.isfinite(actual).all()
        assert actual._no_weight_decay
    assert ((layer.A_log.exp() > 0) & (layer.A_log.exp() <= 16)).all()
    dt = F.softplus(layer.dt_bias)
    assert ((dt >= 0.001) & (dt <= 0.1)).all()
    for parameter in parameters:
        parameter._is_hf_initialized = True
    expected = [parameter.detach().clone() for parameter in parameters]
    model._init_weights(layer)
    for actual, reference in zip(parameters, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
