"""Descarga los modelos. Tres caminos, en este orden:
1. Release de GitHub con los pesos ya convertidos (~7 GB, ~9 GB con el pack de calidad).
2. Repo de Hugging Face con los pesos convertidos: CANCIONES_REPO=usuario/repo
3. Si falla lo anterior: baja los oficiales de HeartMuLa (~22 GB), los convierte aquí y borra los originales.
uso: python descargar.py <dir_modelos> <dir_temporal> [--calidad]
"""
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

models, tmp = Path(sys.argv[1]), Path(sys.argv[2])
calidad = "--calidad" in sys.argv
repo = os.environ.get("CANCIONES_REPO", "").strip()
models.mkdir(parents=True, exist_ok=True)
BASE = ["mula_int8.safetensors", "codec_bf16.safetensors", "codec_bf16.json", "tokenizer.json"]


RELEASE = os.environ.get("CANCIONES_RELEASE",
                         "https://github.com/davidrscout/EstudioCanciones/releases/download/modelos-v1")


def have(*names):
    return all((models / n).exists() for n in names)


def _get(url, dst):
    """Descarga con reanudación (si se corta, continúa donde iba)."""
    import urllib.request
    done = dst.stat().st_size if dst.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # ya estaba completo
            return
        raise
    total = done + int(r.headers.get("Content-Length", 0))
    mode = "ab" if done and r.status == 206 else "wb"
    done = done if mode == "ab" else 0
    last = 0
    with open(dst, mode) as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if done - last > 200 << 20:
                last = done
                print(f"   {dst.name}: {done / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)


def from_release():
    import hashlib
    import json
    import urllib.request
    man = json.loads(urllib.request.urlopen(RELEASE + "/manifest.json", timeout=60).read())
    tmp.mkdir(parents=True, exist_ok=True)
    for it in man:
        if it["pack"] == "calidad" and not calidad:
            continue
        dst = models / it["path"]
        if dst.exists() and dst.stat().st_size == it["size"]:
            continue
        print(f"Descargando {it['path']} ({it['size'] / 1e9:.2f} GB)…", flush=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        part_tmp = dst.with_name(dst.name + ".descargando")
        with open(part_tmp, "wb") as out:
            for p in it["parts"]:
                pf = tmp / p
                _get(f"{RELEASE}/{p}", pf)
                with open(pf, "rb") as f:
                    while True:
                        b = f.read(1 << 24)
                        if not b:
                            break
                        h.update(b)
                        out.write(b)
        if h.hexdigest() != it["sha256"]:
            part_tmp.unlink()
            for p in it["parts"]:
                (tmp / p).unlink(missing_ok=True)
            raise RuntimeError(f"{it['path']} llegó corrupto; vuelve a ejecutar el instalador")
        os.replace(part_tmp, dst)
        for p in it["parts"]:
            (tmp / p).unlink(missing_ok=True)


ok = False
if not repo and RELEASE:
    try:
        from_release()
        ok = True
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo usar la descarga rápida ({e}); uso la oficial.", flush=True)

if ok:
    pass
elif repo:
    pats = BASE + (["judges/*"] if calidad else [])
    snapshot_download(repo, local_dir=str(models), allow_patterns=pats)
else:
    if not have(*BASE):
        from convert_final import convert_codec, convert_mula
        print("Descargando el modelo oficial (unos 22 GB)…", flush=True)
        mula = tmp / "HeartMuLa"
        codec = tmp / "HeartCodec"
        snapshot_download("HeartMuLa/HeartMuLa-oss-3B-happy-new-year", local_dir=str(mula))
        snapshot_download("HeartMuLa/HeartCodec-oss-20260123", local_dir=str(codec))
        hf_hub_download("HeartMuLa/HeartMuLaGen", "tokenizer.json", local_dir=str(models))
        print("Convirtiendo a formato ligero…", flush=True)
        convert_mula(mula, models / "mula_int8.safetensors")
        convert_codec(codec, models / "codec_bf16.safetensors")
        shutil.rmtree(mula, ignore_errors=True)
        shutil.rmtree(codec, ignore_errors=True)
    if calidad and not have("judges/aesthetics.pt", "judges/HeartTranscriptor-oss/model.safetensors"):
        import torch
        from safetensors.torch import load_file, save_file
        j = models / "judges"
        tr = j / "HeartTranscriptor-oss"
        snapshot_download("HeartMuLa/HeartTranscriptor-oss", local_dir=str(tr))
        sd = {k: (v.half() if v.is_floating_point() else v).clone()
              for k, v in load_file(str(tr / "model.safetensors")).items()}
        save_file(sd, str(tr / "model16.tmp"), metadata={"format": "pt"})
        del sd
        os.replace(tr / "model16.tmp", tr / "model.safetensors")
        p = hf_hub_download("facebook/audiobox-aesthetics", "checkpoint.pt", local_dir=str(tmp / "aes"))
        shutil.move(p, j / "aesthetics.pt")
        shutil.rmtree(tmp / "aes", ignore_errors=True)
print("Modelos listos.", flush=True)
