"""Kernels Triton del camino de decodificación (pocas filas por paso).

Por capa: rms -> qkv (1 GEMV int8) -> rope+caché -> atención (longitud leída de la GPU)
          -> o (+residual) -> rms -> w1|w3 (1 GEMV) -> silu*mul -> w2 (+residual).
Sin máscaras ni tramos: un solo CUDA graph sirve para toda la canción."""
import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------ GEMV int8 (+ residual opcional)
@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=s)
        for bn in (32, 64, 128) for bk in (128, 256, 512) for w in (4, 8) for s in (3, 4)
    ],
    key=["N", "K", "MP", "RES"],
    cache_results=True,
)
@triton.jit
def _gemv8(x_ptr, w_ptr, s_ptr, r_ptr, y_ptr, M, N, K, MP: tl.constexpr, RES: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rm = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[None, :] * K + rk[:, None], mask=(rn[None, :] < N) & (rk[:, None] < K), other=0)
        acc += tl.dot(x, w.to(x.dtype))
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    msk = (rm[:, None] < M) & (rn[None, :] < N)
    y = (acc * s[None, :]).to(y_ptr.dtype.element_ty)
    if RES:
        y = y + tl.load(r_ptr + rm[:, None] * N + rn[None, :], mask=msk, other=0.0)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], y, mask=msk)


def int8_mm(x, w, s, res=None):
    """x [..., K] fp16/bf16, w [N, K] int8, s [N] -> [..., N] (+ res)."""
    shp = x.shape
    x = x.reshape(-1, shp[-1]).contiguous()
    M, K = x.shape
    N = w.shape[0]
    if M > 16:
        y = x @ (w.to(x.dtype) * s[:, None]).t()
        y = y if res is None else y + res.reshape(M, N)
        return y.reshape(*shp[:-1], N)
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]),)
    r = res.reshape(M, N).contiguous() if res is not None else y
    _gemv8[grid](x, w, s, r, y, M, N, K, MP=16, RES=res is not None)
    return y.reshape(*shp[:-1], N)


# ------------------------------------------------------------------ RMSNorm
@triton.jit
def _rms(x_ptr, w_ptr, y_ptr, D, eps, BD: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BD)
    m = d < D
    x = tl.load(x_ptr + r * D + d, mask=m, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + eps)
    w = tl.load(w_ptr + d, mask=m, other=0.0)
    y = (x * inv).to(w.dtype) * w
    tl.store(y_ptr + r * D + d, y, mask=m)


def rms(x, w, eps=1e-5):
    R, D = x.shape
    y = torch.empty_like(x)
    _rms[(R,)](x, w, y, D, eps, BD=triton.next_power_of_2(D), num_warps=8)
    return y


# ------------------------------------------------------------------ RoPE + escritura en caché
@triton.jit
def _rope_cache(qkv_ptr, rope_ptr, pos_ptr, qo_ptr, kc_ptr, vc_ptr, S, Hq, Hk, L,
                HD: tl.constexpr, HALF: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    b = r // S
    p = tl.load(pos_ptr + r)
    W = (Hq + 2 * Hk) * HD
    i = tl.arange(0, HALF)
    mi = i < HD // 2
    c = tl.load(rope_ptr + p * HD + 2 * i, mask=mi, other=0.0)
    sn = tl.load(rope_ptr + p * HD + 2 * i + 1, mask=mi, other=0.0)
    if h < Hq:
        base = qkv_ptr + r * W + h * HD
        x0 = tl.load(base + 2 * i, mask=mi, other=0.0).to(tl.float32)
        x1 = tl.load(base + 2 * i + 1, mask=mi, other=0.0).to(tl.float32)
        o = qo_ptr + r * Hq * HD + h * HD
        tl.store(o + 2 * i, (x0 * c - x1 * sn).to(qo_ptr.dtype.element_ty), mask=mi)
        tl.store(o + 2 * i + 1, (x1 * c + x0 * sn).to(qo_ptr.dtype.element_ty), mask=mi)
    else:
        kh = h - Hq
        base = qkv_ptr + r * W + Hq * HD + kh * HD
        x0 = tl.load(base + 2 * i, mask=mi, other=0.0).to(tl.float32)
        x1 = tl.load(base + 2 * i + 1, mask=mi, other=0.0).to(tl.float32)
        dst = (b * Hk + kh) * L * HD + p * HD
        tl.store(kc_ptr + dst + 2 * i, (x0 * c - x1 * sn).to(kc_ptr.dtype.element_ty), mask=mi)
        tl.store(kc_ptr + dst + 2 * i + 1, (x1 * c + x0 * sn).to(kc_ptr.dtype.element_ty), mask=mi)
        vb = qkv_ptr + r * W + (Hq + Hk) * HD + kh * HD
        tl.store(vc_ptr + dst + 2 * i, tl.load(vb + 2 * i, mask=mi, other=0.0), mask=mi)
        tl.store(vc_ptr + dst + 2 * i + 1, tl.load(vb + 2 * i + 1, mask=mi, other=0.0), mask=mi)


# ------------------------------------------------------------------ atención de decodificación (flash-decoding)
@triton.jit
def _attn(q_ptr, k_ptr, v_ptr, pos_ptr, o_ptr, pm_ptr, pl_ptr, pa_ptr, S, Hq, Hk, L, scale,
          HD: tl.constexpr, HDP: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr, NSPLIT: tl.constexpr):
    r = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    b = r // S
    ln = tl.load(pos_ptr + r) + 1
    chunk = tl.cdiv(tl.cdiv(ln, NSPLIT), BLK) * BLK
    lo = sp * chunk
    hi = tl.minimum(ln, lo + chunk)
    gi = tl.arange(0, 16)
    d = tl.arange(0, HDP)
    md = d < HD
    qh = kh * G + gi
    mq = (gi[:, None] < G) & md[None, :]
    q = tl.load(q_ptr + r * Hq * HD + qh[:, None] * HD + d[None, :], mask=mq, other=0.0)
    m_i = tl.full((16,), float("-inf"), tl.float32)
    l_i = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, HDP), tl.float32)
    kb = k_ptr + (b * Hk + kh) * L * HD
    vb = v_ptr + (b * Hk + kh) * L * HD
    for s0 in range(lo, hi, BLK):
        t = s0 + tl.arange(0, BLK)
        tm = t < hi
        k = tl.load(kb + t[:, None] * HD + d[None, :], mask=tm[:, None] & md[None, :], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(tm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp(m_i - m_new)
        pe = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pe, 1)
        v = tl.load(vb + t[:, None] * HD + d[None, :], mask=tm[:, None] & md[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(pe.to(v.dtype), v)
        m_i = m_new
    if NSPLIT == 1:
        o = acc / l_i[:, None]
        tl.store(o_ptr + r * Hq * HD + qh[:, None] * HD + d[None, :], o.to(o_ptr.dtype.element_ty), mask=mq)
    else:
        pidx = (r * Hq + qh) * NSPLIT + sp
        tl.store(pm_ptr + pidx, m_i, mask=gi < G)
        tl.store(pl_ptr + pidx, l_i, mask=gi < G)
        tl.store(pa_ptr + pidx[:, None] * HD + d[None, :], acc, mask=mq)


@triton.jit
def _attn_combine(pm_ptr, pl_ptr, pa_ptr, o_ptr, HD: tl.constexpr, HDP: tl.constexpr, NSPLIT: tl.constexpr):
    rh = tl.program_id(0)
    s = tl.arange(0, NSPLIT)
    d = tl.arange(0, HDP)
    m = tl.load(pm_ptr + rh * NSPLIT + s)
    l = tl.load(pl_ptr + rh * NSPLIT + s)
    a = tl.load(pa_ptr + (rh * NSPLIT + s)[:, None] * HD + d[None, :], mask=d[None, :] < HD, other=0.0)
    M = tl.max(m, 0)
    w = tl.where(m > float("-inf"), tl.exp(m - M), 0.0)
    o = tl.sum(a * w[:, None], 0) / tl.sum(l * w, 0)
    tl.store(o_ptr + rh * HD + d, o.to(o_ptr.dtype.element_ty), mask=d < HD)


# ------------------------------------------------------------------ SiLU(a) * b
@triton.jit
def _silu_mul(x_ptr, y_ptr, F_, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < F_
    a = tl.load(x_ptr + r * 2 * F_ + c, mask=m, other=0.0)
    bb = tl.load(x_ptr + r * 2 * F_ + F_ + c, mask=m, other=0.0)
    af = a.to(tl.float32)
    s = (af / (1.0 + tl.exp(-af))).to(a.dtype)
    tl.store(y_ptr + r * F_ + c, s * bb, mask=m)


