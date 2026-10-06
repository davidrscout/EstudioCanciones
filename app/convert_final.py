"""Convierte los pesos oficiales (fp32) al formato de distribución.

uso: python convert_final.py <dir_HeartMuLa> <dir_HeartCodec> <dir_salida> [--keep-emb-bf16]
  mula_int8.safetensors : capas lineales, cabezas y embeddings en int8 por canal/fila; normas en bf16
  codec_bf16.safetensors: códec en bf16 sin el encoder (no se usa para generar)
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def q8_rows(t):
    """int8 simétrico por fila (canal de salida / fila de embedding)."""
    tf = t.float().cuda()
    s = tf.abs().amax(1).clamp(min=1e-8) / 127
    return (tf / s[:, None]).round().clamp(-127, 127).to(torch.int8).cpu(), s.to(torch.bfloat16).cpu()


def shards(d):
    d = Path(d)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return [d / s for s in sorted(set(json.load(open(idx))["weight_map"].values()))]
    return [d / "model.safetensors"]


def convert_mula(src, dst, keep_emb=False):
    out = {}
    for sh in shards(src):
        with safe_open(str(sh), "pt") as f:
            for k in f.keys():
                if "pos_embeddings" in k or k == "muq_linear.weight":
                    continue  # rope se recalcula; muq de entrada siempre es cero -> solo cuenta el bias
                t = f.get_tensor(k)
                if k == "audio_head":
                    for i in range(t.shape[0]):
                        out[f"audio_head.{i}.q8"], out[f"audio_head.{i}.s8"] = q8_rows(t[i].t().contiguous())
                elif t.dim() == 2 and k != "unconditional_text_embedding.weight" and not (keep_emb and "embeddings" in k):
                    out[k + ".q8"], out[k + ".s8"] = q8_rows(t)
                else:
                    out[k] = t.to(torch.bfloat16).contiguous()
    save_file(out, str(dst), metadata={"lite": json.dumps({"mode": "int8"})})


def convert_codec(src, dst):
    out = {}
    for sh in shards(src):
        with safe_open(str(sh), "pt") as f:
            for k in f.keys():
                if k.startswith("scalar_model.encoder"):
                    continue
                t = f.get_tensor(k)
                out[k] = t.to(torch.bfloat16).contiguous() if t.is_floating_point() else t
    save_file(out, str(dst))
    shutil.copy(Path(src) / "config.json", Path(dst).with_suffix(".json"))


if __name__ == "__main__":
    mula, codec, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    convert_mula(mula, outdir / "mula_int8.safetensors", "--keep-emb-bf16" in sys.argv)
    convert_codec(codec, outdir / "codec_bf16.safetensors")
    for p in sorted(outdir.glob("*.safetensors")):
        print(p.name, round(p.stat().st_size / 1e9, 2), "GB")
