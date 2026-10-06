"""GEMV/GEMM pequeño con pesos int8 por canal (y = x @ (w*s).T), para M <= 16."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=s)
        for bn in (32, 64, 128) for bk in (128, 256, 512) for w in (4, 8) for s in (3, 4)
    ],
    key=["N", "K", "MP"],
    cache_results=True,
)
@triton.jit
def _int8_mm(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K, MP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    acc = acc * s[None, :]
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def int8_mm(x, w, s):
    """x [..., K] bf16/fp16, w [N, K] int8, s [N] -> [..., N] del tipo de x."""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).contiguous()
    M, K = x2.shape
    N = w.shape[0]
    if M > 16:
        return (x2 @ (w.to(x.dtype) * s[:, None]).t()).reshape(*shp[:-1], N)
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]),)
    _int8_mm[grid](x2, w, s.to(x.dtype), y, M, N, K, MP=16)
    return y.reshape(*shp[:-1], N)
