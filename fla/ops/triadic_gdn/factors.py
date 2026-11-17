# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def fuse_key_factors(
    k2s: list[torch.Tensor],
    q2s: list[torch.Tensor],
    gs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Fuse the key factors of an n-way state into the second key, query and decay of a triadic op.

    With `m` second keys `k_a` of size `E_a`, the joint key `k_1 ⊗ ... ⊗ k_m ⊗ k` is a triadic key with `E = E_1 * ... * E_m`.
    Slice `(e_1, ..., e_m)` decays by `exp(g_1[e_1] + ... + g_m[e_m])`.
    The fused slice index is row-major, so a state of shape `[N, H, E, K, V]` views as `[N, H, E_1, ..., E_m, K, V]`.

    Args:
        k2s (list[torch.Tensor]):
            Second keys, each of shape `[B, T, H, E_a]`.
        q2s (list[torch.Tensor]):
            Second queries, each of shape `[B, T, H, E_a]`.
        gs (list[torch.Tensor]):
            Per-axis decays in log space, each of shape `[B, T, H, E_a]`.
            All factors share `B`, `T` and `H`.

    Returns:
        k2 (torch.Tensor):
            Fused second keys of shape `[B, T, H, E_1 * ... * E_m]`.
        q2 (torch.Tensor):
            Fused second queries of shape `[B, T, H, E_1 * ... * E_m]`.
        g (torch.Tensor):
            Fused decays in log space of shape `[B, T, H, E_1 * ... * E_m]`.
    """
    if not (len(k2s) == len(q2s) == len(gs)) or not k2s:
        raise ValueError(f"expected the same nonzero number of factors, got {len(k2s)}, {len(q2s)} and {len(gs)}.")
    for a, (k2, q2, g) in enumerate(zip(k2s, q2s, gs)):
        if not (k2.shape == q2.shape == g.shape):
            raise ValueError(f"factor {a}: k2, q2 and g shapes differ: {list(k2.shape)}, {list(q2.shape)}, {list(g.shape)}.")
        if k2.shape[:-1] != k2s[0].shape[:-1]:
            raise ValueError(
                f"factor {a}: expected leading dims {list(k2s[0].shape[:-1])} as in factor 0, got {list(k2.shape[:-1])}.",
            )
    k2, q2, g = k2s[0], q2s[0], gs[0]
    for k2_a, q2_a, g_a in zip(k2s[1:], q2s[1:], gs[1:]):
        k2 = (k2[..., :, None] * k2_a[..., None, :]).flatten(-2)
        q2 = (q2[..., :, None] * q2_a[..., None, :]).flatten(-2)
        g = (g[..., :, None] + g_a[..., None, :]).flatten(-2)
    return k2, q2, g
