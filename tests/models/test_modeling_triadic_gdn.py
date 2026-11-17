# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from fla.layers.triadic_gdn import TriadicGatedDeltaNet
from fla.models import TriadicGatedDeltaNetConfig
from fla.ops.triadic_gdn import fuse_key_factors, naive_recurrent_triadic_gdn
from fla.utils import assert_close, device

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
            (2, 4, 512, 8, 64, (2, 3, 2), torch.float16),
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


@pytest.mark.parametrize('key_factors', [(4,), (2, 3)], ids=lambda e: f"E{e}")
@torch.no_grad()
def test_layer_reference(key_factors: tuple[int, ...]):
    # rebuild the layer's inputs to the op from its weights and run the recurrent reference
    torch.manual_seed(42)
    B, T, H, D, C = 2, 100, 2, 32, 64
    layer = TriadicGatedDeltaNet(hidden_size=C, head_dim=D, num_heads=H, key_factors=key_factors, use_short_conv=False)
    for module in layer.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.2)
    layer = layer.to(device)
    x = torch.randn(B, T, C, device=device)
    o, _, _ = layer(x)

    E = sum(key_factors)
    bounds = list(accumulate(key_factors, initial=0))

    def split(t):
        return [t[..., start:end] for start, end in zip(bounds[:-1], bounds[1:])]

    q = F.normalize(F.silu(layer.q_proj(x)).view(B, T, H, D), dim=-1)
    k = F.normalize(F.silu(layer.k_proj(x)).view(B, T, H, D), dim=-1)
    v = F.silu(layer.v_proj(x)).view(B, T, H, D)
    k2s = [F.normalize(F.softplus(t), dim=-1) for t in split(layer.k2_proj(x).view(B, T, H, E))]
    q2s = [F.normalize(F.softplus(t), dim=-1) for t in split(layer.q2_proj(x).view(B, T, H, E))]
    g = -layer.A_log.view(H, E).exp() * F.softplus(layer.a2_proj(x).view(B, T, H, E) + layer.dt_bias.view(H, E))
    k2, q2, g = fuse_key_factors(k2s, q2s, split(g))
    ref, _ = naive_recurrent_triadic_gdn(q=q, k=k, v=v, k2=k2, q2=q2, g=g, beta=layer.b_proj(x).sigmoid())
    ref = layer.o_norm(ref.to(x.dtype), layer.g_proj(x).view(B, T, H, D))
    ref = layer.o_proj(ref.flatten(-2))
    assert_close('o', ref, o, 0.005)


def test_state_shape():
    layer = TriadicGatedDeltaNet(hidden_size=64, head_dim=16, num_heads=2, key_factors=(2, 3), use_short_conv=False)
    assert layer.num_slices == 6
    assert layer.k2_proj.out_features == layer.q2_proj.out_features == layer.a2_proj.out_features == 2 * 5
    assert layer.A_log.shape == layer.dt_bias.shape == (2 * 5,)
    assert TriadicGatedDeltaNet(hidden_size=64, head_dim=16, num_heads=2, key_factors=4).key_factors == (4,)
    assert TriadicGatedDeltaNetConfig(key_factors=4).key_factors == [4]
    with pytest.raises(ValueError, match='key_factors'):
        TriadicGatedDeltaNet(hidden_size=64, head_dim=16, num_heads=2, key_factors=(2, 0))
    with pytest.raises(ValueError, match='key_factors'):
        TriadicGatedDeltaNetConfig(key_factors=[])
    with pytest.raises(ValueError, match='output_gate_activation'):
        TriadicGatedDeltaNetConfig(output_gate_activation='relu')


def test_initialization(init_layout, make_model):
    torch.manual_seed(42)
    model = make_model(TriadicGatedDeltaNetConfig)
    with torch.device('meta'):
        layer = TriadicGatedDeltaNet(hidden_size=64, head_dim=16, num_heads=4, key_factors=(2, 2), use_short_conv=False)
    model._init_weights(layer)
    assert all(parameter.is_meta for parameter in layer.parameters())
    layer.to_empty(device='cpu')
    for name in ('A_log', 'dt_bias'):
        value = torch.full_like(getattr(layer, name), float('nan'))
        if init_layout is not None:
            mesh, placements = init_layout
            value = distribute_tensor(value, mesh, placements)
        setattr(layer, name, nn.Parameter(value))
    parameters = (layer.A_log, layer.dt_bias)
    model._init_weights(layer)
    values = []
    for parameter, actual in zip(parameters, (layer.A_log, layer.dt_bias), strict=True):
        assert actual is parameter
        assert actual._no_weight_decay
        if init_layout is not None:
            assert isinstance(actual, DTensor)
            assert actual.device_mesh == mesh
            assert actual.placements == placements
            actual = actual.to_local()
        assert torch.isfinite(actual).all()
        values.append(actual)
    assert ((values[0].exp() > 0) & (values[0].exp() <= 16)).all()
    dt = F.softplus(values[1])
    assert ((dt >= 0.001) & (dt <= 0.1)).all()
    for parameter in parameters:
        parameter._is_hf_initialized = True
    expected = [parameter.detach().clone() for parameter in parameters]
    model._init_weights(layer)
    for actual, reference in zip(parameters, expected, strict=True):
        if init_layout is not None:
            actual, reference = actual.to_local(), reference.to_local()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
