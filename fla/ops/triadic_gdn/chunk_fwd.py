# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets, solve_tril
from fla.ops.utils.op import exp


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_fwd_kernel_intra(
    q,
    k,
    k2,
    q2,
    g,
    beta,
    A,
    Aqk,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    m_t = o_t < T
    m_A = (o_i[:, None] >= o_i[None, :]) & m_t[:, None] & m_t[None, :]
    p_A = (bos + o_t[:, None]) * H*BT + i_h * BT + o_i[None, :]
    p_e = (bos + o_t) * H*E + i_h * E

    # A = beta * (k @ k^T) * R, strictly lower, with R[i, j] = sum_e k2[i, e] * k2[j, e] * exp(g[i, e] - g[j, e])
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k = k + (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0)
        b_A += tl.dot(b_k, tl.trans(b_k))
    b_R = tl.zeros([BT, BT], dtype=tl.float32)
    for i_e in range(E):
        b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        # gate differences are formed before exp so that no factor can overflow
        b_R += tl.where(m_A, b_k2[:, None] * b_k2[None, :] * exp(tl.where(m_A, b_g[:, None] - b_g[None, :], 0)), 0)
    b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0).to(tl.float32)
    b_A = tl.where(o_i[:, None] > o_i[None, :], b_beta[:, None] * b_A * b_R, 0)
    tl.store(A + p_A, b_A.to(A.dtype.element_ty), mask=m_t[:, None])

    # Aqk = scale * (q @ k^T) * Rq, lower, with Rq[i, j] = sum_e q2[i, e] * k2[j, e] * exp(g[i, e] - g[j, e])
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_q = q + (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        p_k = k + (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & (o_k < K)[None, :], other=0)
        b_k = tl.load(p_k, mask=m_t[:, None] & (o_k < K)[None, :], other=0)
        b_A += tl.dot(b_q, tl.trans(b_k))
    b_R = tl.zeros([BT, BT], dtype=tl.float32)
    for i_e in range(E):
        b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_q2 = tl.load(q2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_R += tl.where(m_A, b_q2[:, None] * b_k2[None, :] * exp(tl.where(m_A, b_g[:, None] - b_g[None, :], 0)), 0)
    b_A = b_A * b_R * scale
    tl.store(Aqk + p_A, b_A.to(Aqk.dtype.element_ty), mask=m_t[:, None])


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_fwd_kernel_h(
    k,
    v,
    k2,
    g,
    beta,
    Ai,
    v_new,
    h,
    s,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Runs the chunk recurrence for one value block of one head.

    `s` holds the running state `[N, H, E, K, V]` in float32 and is updated in place;
    each program owns its value block of all `E` slices, so slices are streamed through it one at a time.
    """
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    p_h = o_k[:, None] * V + o_v[None, :]
    s += i_nh * E*K*V

    for i_t in range(NT):
        o_t = i_t * BT + o_i
        m_t = o_t < T
        p_k = k + (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
        p_v = v + (bos + o_t[:, None]) * H*V + i_h * V + o_v[None, :]
        p_A = Ai + (bos + o_t[:, None]) * H*BT + i_h * BT + o_i[None, :]
        p_e = (bos + o_t) * H*E + i_h * E
        # [BT, BK]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k[None, :], other=0)
        b_beta = tl.load(beta + (bos + o_t) * H + i_h, mask=m_t, other=0).to(tl.float32)
        # [BT, BV]
        b_u = b_beta[:, None] * tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0).to(tl.float32)

        # u = (I + A)^-1 (beta * v - sum_e (beta * k2_e * exp(g_e)) * (k @ S_e))
        p_ht = h + ((boh + i_t) * H + i_h) * E*K*V
        for i_e in range(E):
            b_h = tl.load(s + i_e * K*V + p_h, mask=m_h, other=0)
            tl.store(p_ht + i_e * K*V + p_h, b_h.to(p_ht.dtype.element_ty), mask=m_h)
            b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_u -= (b_beta * b_k2 * exp(b_g))[:, None] * tl.dot(b_k, b_h.to(b_k.dtype))
        b_u = tl.dot(tl.load(p_A, mask=m_t[:, None], other=0), b_u.to(b_k.dtype))
        p_vn = v_new + (bos + o_t[:, None]) * H*V + i_h * V + o_v[None, :]
        tl.store(p_vn, b_u.to(p_vn.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])

        # S_e <- exp(g_last_e) * S_e + k^T @ (k2_e * exp(g_last_e - g_e) * u)
        p_last = (bos + tl.minimum(i_t * BT + BT, T) - 1) * H*E + i_h * E
        for i_e in range(E):
            b_k2 = tl.load(k2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
            b_g_last = tl.load(g + p_last + i_e).to(tl.float32)
            b_w = (b_k2 * exp(b_g_last - b_g))[:, None] * b_u
            b_h = tl.load(s + i_e * K*V + p_h, mask=m_h, other=0) * exp(b_g_last)
            b_h += tl.dot(tl.trans(b_k), b_w.to(b_k.dtype))
            tl.store(s + i_e * K*V + p_h, b_h, mask=m_h)
        # the next chunk reads the state written by other threads of this program
        tl.debug_barrier()


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def chunk_triadic_gdn_fwd_kernel_o(
    q,
    q2,
    g,
    Aqk,
    v_new,
    h,
    o,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int64), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t
    else:
        bos, eos = i_b * T, i_b * T + T
        i_tg = i_b * tl.cdiv(T, BT) + i_t

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_k = o_k < K
    m_v = o_v < V

    p_q = q + (bos + o_t[:, None]) * H*K + i_h * K + o_k[None, :]
    p_h = h + (i_tg * H + i_h) * E*K*V + o_k[:, None] * V + o_v[None, :]
    p_e = (bos + o_t) * H*E + i_h * E
    # [BT, BK]
    b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0)
    # o = scale * sum_e (q2_e * exp(g_e)) * (q @ S_e) + Aqk @ v_new
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_e in range(E):
        b_h = tl.load(p_h + i_e * K*V, mask=m_k[:, None] & m_v[None, :], other=0)
        b_q2 = tl.load(q2 + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_g = tl.load(g + p_e + i_e, mask=m_t, other=0).to(tl.float32)
        b_o += (b_q2 * exp(b_g))[:, None] * tl.dot(b_q, b_h)
    b_o *= scale

    p_A = Aqk + (bos + o_t[:, None]) * H*BT + i_h * BT + o_i[None, :]
    p_vn = v_new + (bos + o_t[:, None]) * H*V + i_h * V + o_v[None, :]
    b_vn = tl.load(p_vn, mask=m_t[:, None] & m_v[None, :], other=0)
    b_o += tl.dot(tl.load(p_A, mask=m_t[:, None], other=0).to(b_vn.dtype), b_vn)

    p_o = o + (bos + o_t[:, None]) * H*V + i_h * V + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


def chunk_triadic_gdn_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, E = *k.shape, k2.shape[-1]
    BT = chunk_size
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = k.new_empty(B, T, H, BT, dtype=torch.float32)
    Aqk = torch.empty_like(A)
    chunk_triadic_gdn_fwd_kernel_intra[(NT, B * H)](
        q=q,
        k=k,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        A=A,
        Aqk=Aqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        E=E,
        BT=BT,
        BK=min(64, triton.next_power_of_2(K)),
        num_warps=8,
        num_stages=1,
    )
    Ai = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
    return Ai, Aqk


def chunk_triadic_gdn_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Ai: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V, E = *k.shape, v.shape[-1], k2.shape[-1]
    BT = chunk_size
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    BV = 64 if K <= 128 else 32

    h = k.new_empty(B, NT, H, E, K, V)
    v_new = torch.empty_like(v)
    # the running state, which ends as the final state
    s = k.new_zeros(N, H, E, K, V, dtype=torch.float32) if initial_state is None else initial_state.float().clone()
    chunk_triadic_gdn_fwd_kernel_h[(triton.cdiv(V, BV), N * H)](
        k=k,
        v=v,
        k2=k2,
        g=g,
        beta=beta,
        Ai=Ai,
        v_new=v_new,
        h=h,
        s=s,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        E=E,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=BV,
        num_warps=4,
        # the slice loops read and write the running state in place; pipelining their loads faults at K=256 (Triton 3.7)
        num_stages=1,
    )
    return h, v_new, s


def chunk_triadic_gdn_fwd_o(
    q: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    Aqk: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K, V, E = *q.shape, v_new.shape[-1], q2.shape[-1]
    BT = chunk_size
    if cu_seqlens is None:
        NT, chunk_offsets = triton.cdiv(T, BT), None
    else:
        NT, chunk_offsets = len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    BV = 64 if K <= 128 else 32

    o = torch.empty_like(v_new)
    chunk_triadic_gdn_fwd_kernel_o[(triton.cdiv(V, BV), NT, B * H)](
        q=q,
        q2=q2,
        g=g,
        Aqk=Aqk,
        v_new=v_new,
        h=h,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        E=E,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=BV,
        num_warps=4,
    )
    return o


def chunk_triadic_gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2: torch.Tensor,
    q2: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    Ai, Aqk = chunk_triadic_gdn_fwd_intra(
        q=q,
        k=k,
        k2=k2,
        q2=q2,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    h, v_new, final_state = chunk_triadic_gdn_fwd_h(
        k=k,
        v=v,
        k2=k2,
        g=g,
        beta=beta,
        Ai=Ai,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    o = chunk_triadic_gdn_fwd_o(
        q=q,
        q2=q2,
        g=g,
        Aqk=Aqk,
        v_new=v_new,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return g, Ai, Aqk, o, final_state if output_final_state else None
