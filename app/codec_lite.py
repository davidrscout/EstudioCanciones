"""Carga y decodificación del HeartCodec sin mmgp (bf16 por defecto)."""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from heartcodec.configuration_heartcodec import HeartCodecConfig  # noqa: E402
from heartcodec.modeling_heartcodec import HeartCodec  # noqa: E402


def load_codec(path, dtype=torch.float16, device="cuda", low_vram=None):
    path = Path(path)
    if path.is_dir():
        cfg = json.load(open(path / "config.json"))
        files = sorted(path.glob("model*.safetensors"))
    else:
        cfg = json.load(open(path.with_suffix(".json")))
        files = [path]
    from accelerate import init_empty_weights
    with init_empty_weights():
        codec = HeartCodec(HeartCodecConfig(**cfg))
    sd = {}
    for f in files:
        sd.update(load_file(str(f), device=str(device)))
    sd = {k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items() if not k.startswith("scalar_model.encoder")}
    del codec.scalar_model.encoder  # solo se usa para codificar audio, no para generar
    missing, unexpected = codec.load_state_dict(sd, strict=False, assign=True)
    assert not missing and not unexpected, (missing, unexpected)
    codec = codec.to(device).eval()
    _chunk_waveform_decoder(codec.scalar_model, chunk=None if low_vram is None else (186 if low_vram else 10_000))
    return codec


def _chunk_waveform_decoder(sm, chunk=None, left=32, right=16, hop=1920):
    """El decodificador de onda se come ~2,4 GB por ventana de 30 s. Por trozos con contexto
    (32 frames a la izquierda, 16 a la derecha) da lo mismo (dif. del orden del ruido bf16) con ~0,3 GB."""
    full = sm.decode
    if chunk is None:
        # trozos grandes si sobra VRAM (más rápido); pequeños en gráficas justas
        ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
        tot = torch.cuda.get_device_properties(0).total_memory / 1024**3 if ok else 0
        chunk = 10_000 if tot >= 12 else 186

    def decode(x):
        B, _, T = x.shape
        if chunk >= T:
            return full(x)
        outs = []
        for b in range(B):
            parts = []
            for s in range(0, T, chunk):
                a, e = max(0, s - left), min(T, s + chunk + right)
                y = full(x[b:b + 1, :, a:e])
                parts.append(y[..., (s - a) * hop:(s - a + min(chunk, T - s)) * hop])
            outs.append(torch.cat(parts, -1))
        return torch.cat(outs, 0)
    sm.decode = decode


@torch.inference_mode()
def decode(codec, frames, steps=10, guidance=1.25):
    # acumulación fp16 en las matmul del códec: x1,6 en GeForce. Medido en 22 pares (misma semilla):
    # estética Audiobox ±0,002 (otra semilla cambia 0,028), sin NaN; el códec ya es estocástico
    mm = torch.backends.cuda.matmul
    fast = codec.device.type == "cuda" and codec.dtype == torch.float16 and hasattr(mm, "allow_fp16_accumulation")
    if fast:
        mm.allow_fp16_accumulation = True
    try:
        return codec.detokenize(frames.clamp(0, 8191).to(codec.device), num_steps=steps,
                                guidance_scale=guidance, disable_progress=True)
    finally:
        if fast:
            mm.allow_fp16_accumulation = False
