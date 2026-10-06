"""Motor ligero de HeartMuLa: pesos int4/bf16, caché KV estática y CUDA graphs.

Reimplementa la inferencia de HeartMuLa (backbone Llama 3B + decoder 300M) sin
mmgp ni transformers, con la misma matemática que el original.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

AUDIO_VOCAB = 8197
N_CB = 8
MUQ_DIM = 512
DIM = 3072
TEXT_BOS, TEXT_EOS, AUDIO_EOS, EMPTY_ID = 128000, 128001, 8193, 0
FRAME_MS = 80
INT4_GROUP = 128

BACKBONE = dict(layers=28, heads=24, kv_heads=8, max_seq=8192)
DECODER = dict(layers=3, heads=8, kv_heads=4, max_seq=2048)


# ---------------------------------------------------------------- RoPE (Llama3 escalado, estilo torchtune)
def rope_cache(head_dim: int, max_seq: int, base=500_000, scale=32, low=1, high=4, old=8192):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    lo_w, hi_w = old / low, old / high
    out = []
    for f in freqs.tolist():
        wl = 2 * math.pi / f
        if wl < hi_w:
            out.append(f)
        elif wl > lo_w:
            out.append(f / scale)
        else:
            s = (old / wl - low) / (high - low)
            out.append((1 - s) * f / scale + s * f)
    theta = torch.tensor(out, dtype=torch.float32)
    idx = torch.einsum("i,j->ij", torch.arange(max_seq, dtype=torch.float32), theta)
    return torch.stack([idx.cos(), idx.sin()], -1)  # [S, D/2, 2]


def apply_rope(x, cache):
    # x: [B, S, H, D]; cache: [B, S, D/2, 2]
    xs = x.float().reshape(*x.shape[:-1], -1, 2)
    c = cache.unsqueeze(2)
    out = torch.stack(
        [xs[..., 0] * c[..., 0] - xs[..., 1] * c[..., 1],
         xs[..., 1] * c[..., 0] + xs[..., 0] * c[..., 1]], -1)
    return out.flatten(3).type_as(x)


def rms(x, w, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).type_as(x) * w


# ---------------------------------------------------------------- capas lineales
class Linear:
    """bf16 o int4 (kernel tinygemm de PyTorch, sm80+)."""

    def __init__(self, w=None, q=None, sz=None, g=INT4_GROUP, q8=None, s8=None):
        self.w, self.q, self.sz, self.g, self.q8, self.s8 = w, q, sz, g, q8, s8

    def __call__(self, x):
        if self.w is not None:
            return F.linear(x, self.w)
        if self.q8 is not None:
            from int8_kernel import int8_mm
            return int8_mm(x, self.q8, self.s8)
        shp = x.shape
        y = torch._weight_int4pack_mm(x.reshape(-1, shp[-1]), self.q, self.g, self.sz)
        return y.reshape(*shp[:-1], y.shape[-1])

    @property
    def nbytes(self):
        if self.w is not None:
            return self.w.numel() * self.w.element_size()
        if self.q8 is not None:
            return self.q8.numel() + self.s8.numel() * 2
        return self.q.numel() * self.q.element_size() + self.sz.numel() * 2


def quantize_int4(w: torch.Tensor, g=INT4_GROUP, iters=0):
    """w [N,K] float -> (nibbles uint8 [N,K/2], scale_zero bf16 [K/G,N,2]). Formato portable.
    iters>0: optimiza escala/mínimo por grupo (mínimos cuadrados alternados)."""
    N, K = w.shape
    wf = w.float().reshape(N, K // g, g)
    mn, mx = wf.amin(-1, keepdim=True), wf.amax(-1, keepdim=True)
    scale = (mx - mn).clamp(min=1e-8) / 15
    for _ in range(iters):
        q = ((wf - mn) / scale).round().clamp(0, 15)
        # ajuste lineal w ~ scale*q + mn por grupo
        qm, wm = q.mean(-1, keepdim=True), wf.mean(-1, keepdim=True)
        var = ((q - qm) ** 2).sum(-1, keepdim=True)
        new_scale = ((q - qm) * (wf - wm)).sum(-1, keepdim=True) / var.clamp(min=1e-8)
        new_scale = torch.where(var > 0, new_scale, scale).clamp(min=1e-8)
        scale, mn = new_scale, wm - new_scale * qm
    q = ((wf - mn) / scale).round().clamp(0, 15).to(torch.uint8).reshape(N, K)
    nib = (q[:, ::2] << 4) | q[:, 1::2]
    zero = mn + scale * 8
    sz = torch.cat([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16)
    return nib.contiguous(), sz


def dequant_int4(nib, sz, g=INT4_GROUP):
    hi, lo = (nib >> 4).to(torch.float32), (nib & 15).to(torch.float32)
    q = torch.stack([hi, lo], -1).reshape(nib.shape[0], -1)
    N, K = q.shape
    s = sz.float().transpose(0, 1)  # [N, K/G, 2]
    q = q.reshape(N, K // g, g)
    return ((q - 8) * s[..., :1] + s[..., 1:]).reshape(N, K)


class Emb:
    """Tabla de embeddings bf16 o int8 por fila."""

    def __init__(self, w=None, q8=None, s8=None):
        self.w, self.q8, self.s8 = w, q8, s8

    def __call__(self, idx):
        if self.w is not None:
            return F.embedding(idx, self.w)
        return F.embedding(idx, self.q8).to(self.s8.dtype) * F.embedding(idx, self.s8[:, None])

    @property
    def nbytes(self):
        if self.w is not None:
            return self.w.numel() * 2
        return self.q8.numel() + self.s8.numel() * 2


# ---------------------------------------------------------------- transformer con caché estática
class Block:
    def __init__(self, p, pre, heads, kv_heads):
        self.q, self.k, self.v, self.o = (p(pre + f"attn.{n}_proj.weight") for n in ("q", "k", "v", "output"))
        self.w1, self.w2, self.w3 = (p(pre + f"mlp.w{i}.weight") for i in (1, 2, 3))
        self.n1, self.n2 = p(pre + "sa_norm.scale", norm=True), p(pre + "mlp_norm.scale", norm=True)
        self.h, self.kvh = heads, kv_heads
        self.hd = DIM // heads
        self.kc = self.vc = None

    def alloc(self, B, L, dev, dtype):
        self.kc = torch.zeros(B, self.kvh, L, self.hd, device=dev, dtype=dtype)
        self.vc = torch.zeros_like(self.kc)

    def __call__(self, x, pos, rc, attn_len, mask):
        """x [B,S,D]; pos [B,S] posiciones absolutas; rc [B,S,D/2,2].
        attn_len: longitud (int) de caché que se atiende; mask: [B,1,S,attn_len] bool o None (causal prefill)."""
        B, S, _ = x.shape
        h = rms(x, self.n1)
        q = apply_rope(self.q(h).view(B, S, self.h, self.hd), rc)
        k = apply_rope(self.k(h).view(B, S, self.kvh, self.hd), rc)
        v = self.v(h).view(B, S, self.kvh, self.hd)
        # escribe en la caché (posiciones iguales en todo el batch)
        p0 = pos[0]
        self.kc.index_copy_(2, p0, k.transpose(1, 2))
        self.vc.index_copy_(2, p0, v.transpose(1, 2))
        kk, vv = self.kc[:, :, :attn_len], self.vc[:, :, :attn_len]
        g = self.h // self.kvh
        if S == 1 and g > 1:
            # GQA sin copiar la caché: los g heads de query de cada head KV van como "secuencia"
            qh = q.view(B, self.kvh, g, self.hd)
            o = F.scaled_dot_product_attention(qh, kk, vv, attn_mask=mask)
            o = o.reshape(B, 1, -1)
        else:
            qh = q.transpose(1, 2)  # [B,H,S,D]
            if g > 1:
                kk = kk.repeat_interleave(g, 1)
                vv = vv.repeat_interleave(g, 1)
            o = F.scaled_dot_product_attention(qh, kk, vv, attn_mask=mask, is_causal=mask is None and S > 1)
            o = o.transpose(1, 2).reshape(B, S, -1)
        x = x + self.o(o)
        h = rms(x, self.n2)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Stack:
    def __init__(self, p, prefix, cfg):
        self.blocks = [Block(p, f"{prefix}.layers.{i}.", cfg["heads"], cfg["kv_heads"]) for i in range(cfg["layers"])]
        self.norm = p(f"{prefix}.norm.scale", norm=True)
        self.rope = None
        self.hd = DIM // cfg["heads"]
        self.max_seq = cfg["max_seq"]

    def alloc(self, B, L, dev, dtype):
        self.rope = rope_cache(self.hd, max(L, 16)).to(dev)
        for b in self.blocks:
            b.alloc(B, L, dev, dtype)

    def free(self):
        for b in self.blocks:
            b.kc = b.vc = None

    def __call__(self, x, pos, attn_len, mask):
        rc = self.rope[pos]
        for b in self.blocks:
            x = b(x, pos, rc, attn_len, mask)
        return rms(x, self.norm)


# ---------------------------------------------------------------- modelo
class HeartMuLaLite:
    def __init__(self, weights: str | Path, device="cuda", dtype=None):
        self.dev = torch.device(device)
        major = torch.cuda.get_device_capability(self.dev)[0] if self.dev.type == "cuda" else 0
        # fp16 en GPU: medido, queda más cerca del original que bf16 (KL 0,0015 frente a 0,0023)
        # y funciona también en Turing (RTX 20xx / GTX 16xx), que no tiene bf16
        if dtype is None:
            dtype = torch.float16 if self.dev.type == "cuda" else torch.bfloat16
        self.dt = dtype
        self.tinygemm = major >= 8 and dtype == torch.bfloat16
        try:
            import triton  # noqa: F401
            self.fast8 = major >= 7 and self.dev.type == "cuda"
        except Exception:
            self.fast8 = False
        dev_load = str(self.dev) if self.dev.type == "cuda" else "cpu"
        f = safe_open(str(weights), "pt", device=dev_load)  # directo a la GPU: sin copia completa en RAM
        fc = safe_open(str(weights), "pt", device="cpu")
        keys = set(f.keys())
        self.meta = json.loads(f.metadata().get("lite", "{}"))

        def p(name, norm=False):
            if name in keys:
                t = f.get_tensor(name).to(dtype)
                return t if norm else Linear(w=t)
            if name + ".q8" in keys:
                q8, s8 = f.get_tensor(name + ".q8"), f.get_tensor(name + ".s8")
                if self.fast8:
                    return Linear(q8=q8, s8=s8.to(dtype))
                return Linear(w=(q8.float() * s8.float()[:, None]).to(dtype))
            nib, sz = f.get_tensor(name + ".q4"), f.get_tensor(name + ".sz")
            g = nib.shape[1] * 2 // sz.shape[0]
            if self.tinygemm:
                return Linear(q=torch._convert_weight_to_int4pack(nib, 8), sz=sz, g=g)
            return Linear(w=dequant_int4(nib, sz, g).to(dtype))

        def emb(name, src):
            if name in keys:
                return Emb(w=src.get_tensor(name).to(dtype))
            return Emb(q8=src.get_tensor(name + ".q8"), s8=src.get_tensor(name + ".s8").to(dtype))

        self.backbone = Stack(p, "backbone", BACKBONE)
        self.decoder = Stack(p, "decoder", DECODER)
        # Embeddings de texto en CPU: solo se usan para el prompt (ahorra VRAM)
        self.text_emb = emb("text_embeddings.weight", fc)
        self.audio_emb = emb("audio_embeddings.weight", f)
        self.uncond = f.get_tensor("unconditional_text_embedding.weight").to(dtype)
        self.projection = p("projection.weight")
        self.c0_head = p("codebook0_head.weight")
        if "audio_head" in keys:
            ah = f.get_tensor("audio_head").to(dtype)  # [7, 3072, V]
            self.audio_head = [Linear(w=ah[i].t().contiguous()) for i in range(N_CB - 1)]
        else:
            self.audio_head = [p(f"audio_head.{i}") for i in range(N_CB - 1)]
        self.muq_b = f.get_tensor("muq_linear.bias").to(dtype)
        self.cb_off = (torch.arange(N_CB, device=self.dev) * AUDIO_VOCAB)
        self.graphs = {}
        self.dec_mask = torch.tril(torch.ones(N_CB, N_CB, dtype=torch.bool, device=self.dev))

    def to(self, device):
        """Mueve los pesos (no el embedding de texto, que vive en CPU). Para turnarse la VRAM con el códec."""
        device = torch.device(device)

        def mv(o):
            for k, v in list(vars(o).items()):
                if torch.is_tensor(v):
                    setattr(o, k, v.to(device, non_blocking=True))
                elif isinstance(v, (Linear, Emb, Block, Stack)):
                    mv(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, (Linear, Block)):
                            mv(x)
        for k in ("backbone", "decoder", "audio_emb", "projection", "c0_head"):
            mv(getattr(self, k))
        for l in self.audio_head:
            mv(l)
        self.uncond, self.muq_b, self.cb_off, self.dec_mask = (
            t.to(device) for t in (self.uncond, self.muq_b, self.cb_off, self.dec_mask))
        torch.cuda.synchronize()
        return self

    # -------------------------------------------------- sampling (igual que el original)
    @staticmethod
    def _sample(logits, topk, temp):
        logits = logits / temp
        thr = torch.topk(logits, topk)[0][..., -1, None]
        logits = logits.masked_fill(logits < thr, -float("inf"))
        probs = torch.softmax(torch.log_softmax(logits, -1), -1)
        qe = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / qe, -1, keepdim=True)

    def _guided(self, logits, cfg, topk, temp):
        N = self.N
        if self.B == 2 * N:
            logits = logits[N:] + (logits[:N] - logits[N:]) * cfg
        s = self._sample(logits.float(), topk, temp)
        return torch.cat([s, s], 0) if self.B == 2 * N else s

    def _embed_audio(self, cb, tok):
        return self.audio_emb(tok + cb * AUDIO_VOCAB)

    # -------------------------------------------------- un frame (backbone + 7 pasos del decoder)
    def _frame_from_h(self, last_h, cfg, topk, temp):
        c0 = self._guided(self.c0_head(last_h), cfg, topk, temp)  # [B,1]
        samples = [c0]
        cur = torch.cat([last_h.unsqueeze(1), self._embed_audio(0, c0)], 1)  # [B,2,D]
        pos = torch.arange(2, device=self.dev).unsqueeze(0).expand(self.B, 2)
        for i in range(1, N_CB):
            S = cur.shape[1]
            mask = self.dec_mask[pos[0]][None, None, :, :N_CB]
            h = self.decoder(self.projection(cur), pos, N_CB, mask.expand(self.B, 1, S, N_CB))
            ci = self._guided(self.audio_head[i - 1](h[:, -1]), cfg, topk, temp)
            samples.append(ci)
            cur = self._embed_audio(i, ci)
            pos = pos[:, -1:] + 1
        return torch.cat(samples, 1)  # [B,8]

    def _step(self, tok, pos, bucket, cfg, topk, temp):
        """tok [B,8] frame anterior; pos [B,1]."""
        h = self.audio_emb(tok + self.cb_off).sum(1, dtype=self.dt).unsqueeze(1)
        ar = self._ar[:bucket]
        mask = (ar[None, :] <= pos[0][:, None])[None, None].expand(self.B, 1, 1, bucket)
        h = self.backbone(h, pos, bucket, mask)
        return self._frame_from_h(h[:, -1], cfg, topk, temp)

    # -------------------------------------------------- prompt
    def _prompt(self, lyrics, tags, tokenizer):
        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = "<tag>" + tags
        if not tags.endswith("</tag>"):
            tags = tags + "</tag>"
        t_ids = tokenizer.encode(tags).ids
        l_ids = tokenizer.encode(lyrics.lower()).ids
        for ids in (t_ids, l_ids):
            if ids[0] != TEXT_BOS:
                ids.insert(0, TEXT_BOS)
            if ids[-1] != TEXT_EOS:
                ids.append(TEXT_EOS)
        muq_idx = len(t_ids)
        text = t_ids + [0] + l_ids
        return torch.tensor(text), muq_idx

    @torch.inference_mode()
    def generate(self, lyrics, tags, tokenizer, *, max_ms=60_000, temperature=1.0, topk=50,
                 cfg_scale=1.5, seed=None, use_graphs=True, progress=None, abort=None, n=1):
        """Genera n candidatos a la vez (comparten la lectura de pesos). Devuelve [8,T] si n=1, lista si n>1."""
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)
        self.N = N = n
        self.B = B = 2 * N if cfg_scale != 1.0 else N
        text, muq_idx = self._prompt(lyrics, tags, tokenizer)
        P = len(text)
        n_frames = max_ms // FRAME_MS
        L = P + n_frames + 2
        self._ar = torch.arange(L + 256, device=self.dev)
        self.backbone.alloc(B, L + 256, self.dev, self.dt)
        self.decoder.alloc(B, N_CB, self.dev, self.dt)

        # prefill: embeddings de texto calculados en CPU
        te = self.text_emb(text).to(self.dev)  # [P,D]
        h = te.unsqueeze(0).repeat(B, 1, 1)
        # posición muq: muq_linear(ceros) = bias; filas incondicionales = embedding incondicional
        h[:N, muq_idx] = self.muq_b
        if B == 2 * N:
            h[N:] = self.uncond[0]
        pos = torch.arange(P, device=self.dev).unsqueeze(0).expand(B, P)
        hb = self.backbone(h, pos, P, None)
        tok = self._frame_from_h(hb[:, -1], cfg_scale, topk, temperature)

        out = torch.empty(n_frames + 1, N, N_CB, dtype=torch.long, device=self.dev)
        out[0] = tok[:N]
        cur_pos = P
        ends = [None] * N  # índice del primer frame EOS de cada candidato
        bucket_size = 256
        static = None
        CHECK = 25
        last = n_frames
        for i in range(n_frames):
            if abort and abort():
                self.release()
                return None
            bucket = min(((cur_pos + 1) // bucket_size + 1) * bucket_size, L + 256)
            if use_graphs and self.dev.type == "cuda":
                key = (B, bucket, topk, float(temperature), float(cfg_scale))
                if static is None:
                    static = dict(tok=tok.clone(), pos=torch.zeros(B, 1, dtype=torch.long, device=self.dev))
                static["tok"].copy_(tok)
                static["pos"].fill_(cur_pos)
                g = self.graphs.get(key)
                if g is None:
                    g = self._capture(static, bucket, cfg_scale, topk, temperature)
                    self.graphs[key] = g
                g[0].replay()
                tok = g[1]
            else:
                p_t = torch.full((B, 1), cur_pos, dtype=torch.long, device=self.dev)
                tok = self._step(tok, p_t, bucket, cfg_scale, topk, temperature)
            out[i + 1] = tok[:N]
            cur_pos += 1
            if (i + 1) % CHECK == 0 or i == n_frames - 1:
                a = max(1, i + 2 - CHECK)
                eos = (out[a: i + 2] >= AUDIO_EOS).any(-1).cpu()  # [t, N]
                for c in range(N):
                    if ends[c] is None and eos[:, c].any():
                        ends[c] = a + int(eos[:, c].nonzero()[0, 0])
                if all(e is not None for e in ends):
                    last = i + 1
                    break
                if progress:
                    progress((i + 1) * FRAME_MS / 1000, max_ms / 1000)
        res = []
        for c in range(N):
            T = ends[c] if ends[c] is not None else last + 1
            res.append(out[:T, c].t().clone())
        self.release()
        return res[0] if n == 1 else res

    def _capture(self, static, bucket, cfg, topk, temp):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        # warmup fuera del grafo sin tocar la caché real: la escritura es en pos actual, se reescribe igualmente
        with torch.cuda.stream(s):
            self._step(static["tok"], static["pos"], bucket, cfg, topk, temp)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=getattr(self, "_pool", None)):
            out = self._step(static["tok"], static["pos"], bucket, cfg, topk, temp)
        self._pool = g.pool()
        return g, out

    def release(self):
        # los grafos apuntan a la caché: se liberan juntos
        self.graphs.clear()
        self._pool = None
        self.backbone.free()
        self.decoder.free()
        torch.cuda.empty_cache()

    def vram_weights(self):
        tot = 0
        for st in (self.backbone, self.decoder):
            for b in st.blocks:
                tot += sum(l.nbytes for l in (b.q, b.k, b.v, b.o, b.w1, b.w2, b.w3))
        tot += self.audio_emb.nbytes + sum(l.nbytes for l in self.audio_head) + self.c0_head.nbytes + self.projection.nbytes
        return tot
