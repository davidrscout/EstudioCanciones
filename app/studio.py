"""Núcleo de la app: genera canciones con HeartMuLa ligero y, opcionalmente, elige la mejor de N tomas."""
from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

from engine import HeartMuLaLite
import codec_lite

SR = 48000


def vram_gb():
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def to_mp3(wav: torch.Tensor, path: str, kbps=256):
    import lameenc
    x = wav.float().clamp(-1, 1).numpy()
    if x.shape[0] == 1:
        x = np.repeat(x, 2, 0)
    pcm = (x.T * 32767).astype(np.int16).tobytes()
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps)
    enc.set_in_sample_rate(SR)
    enc.set_channels(2)
    enc.set_quality(2)
    with open(path, "wb") as f:
        f.write(enc.encode(pcm) + enc.flush())


class Studio:
    def __init__(self, models_dir: str | Path, log=print):
        self.dir = Path(models_dir)
        self.log = log
        self.lock = threading.Lock()
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.vram = vram_gb()
        self.cpu = not (torch.cuda.is_available() and torch.cuda.device_count() > 0)
        # con <12 GB de VRAM no caben a la vez transformer y códec: se turnan (el que espera, en RAM)
        self.swap = not self.cpu and (self.vram < 12 or os.environ.get("CANCIONES_SWAP") == "1")
        t = time.time()
        self.mula = HeartMuLaLite(self.dir / "mula_int8.safetensors", device="cpu" if self.cpu else "cuda")
        # códec en fp16 en GPU: más cerca del fp32 original que bf16 (medido) y válido en RTX 20xx.
        # En CPU, bf16 (en CPU es más rápido que fp32 y fp16).
        self.codec = codec_lite.load_codec(self.dir / "codec_bf16.safetensors",
                                           torch.bfloat16 if self.cpu else torch.float16,
                                           device="cpu" if (self.swap or self.cpu) else "cuda",
                                           low_vram=self.swap)
        self.judges = None
        self._on_gpu = "mula"
        # medido en modo turnos: 1 toma ~4,2 GB, 2 tomas ~4,4 GB, 4 tomas ~5,8 GB de pico
        self.max_takes = 2 if (not self.cpu and self.vram < 7.5) else 4
        if self.cpu:
            torch.set_num_threads(os.cpu_count() or 4)
        # con VRAM de sobra, los jueces se cargan y calientan en segundo plano mientras se escribe
        # la letra: la primera canción no paga sus ~13 s de arranque
        self._jt = None
        if self.has_judges() and not self.cpu and not self.swap:
            self._jt = threading.Thread(target=self._preload_judges, daemon=True)
            self._jt.start()
        log(f"Modelos cargados en {time.time() - t:.0f} s ("
            + ("SIN GPU NVIDIA: modo CPU, irá lento" if self.cpu else f"GPU {self.vram:.0f} GB, turnos={'sí' if self.swap else 'no'}") + ")")

    def _preload_judges(self):
        try:
            g = torch.Generator().manual_seed(0)
            self._judges().score(torch.randn(2, SR * 5, generator=g) * 0.1, SR, "hola")
        except Exception as e:  # noqa: BLE001  se reintentará al puntuar
            self.log(f"No se pudieron precargar los jueces: {e}")
            self.judges = None

    def has_judges(self):
        return (self.dir / "judges").exists()

    def _judges(self):
        if self.judges is None:
            from judges import Judges
            j = self.dir / "judges"
            self.judges = Judges(transcriptor=str(j / "HeartTranscriptor-oss"), aes=str(j / "aesthetics.pt"))
            self.judges.dev = "cpu" if self.cpu else "cuda"
            if self.swap:
                self.judges.to("cpu")
        return self.judges

    def _gpu(self, who: str):
        """En GPUs pequeñas solo un modelo ocupa la VRAM a la vez; los demás esperan en RAM."""
        if not self.swap or self._on_gpu == who:
            return
        parts = {"mula": self.mula, "codec": self.codec, "judges": self.judges}
        for name, m in parts.items():
            if m is not None and name != who:
                m.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        parts[who].to("cuda")
        self._on_gpu = who

    def generate(self, lyrics, tags, seconds=120, takes=1, seed=-1, cfg=1.5, temperature=1.0, topk=50,
                 progress=lambda frac, msg: None, abort=None):
        takes = min(takes, self.max_takes)
        with self.lock:
            # la precarga de jueces usa la GPU: termina antes de capturar los CUDA graphs
            if self._jt is not None:
                self._jt.join()
                self._jt = None
            seed = seed if seed is not None and seed >= 0 else int(time.time() * 1000) % 2**31
            progress(0.0, "Componiendo…")
            self._gpu("mula")
            frames = self.mula.generate(
                lyrics, tags, self.tok, max_ms=int(seconds * 1000), temperature=temperature, topk=topk,
                cfg_scale=cfg, seed=seed, n=takes, abort=abort,
                progress=lambda s, tot: progress(0.75 * s / tot, f"Componiendo… {int(s)} s de {int(tot)} s"))
            if frames is None:
                return None
            frames = frames if isinstance(frames, list) else [frames]
            self._gpu("codec")
            wavs = []
            for i, fr in enumerate(frames):
                if abort and abort():
                    return None
                progress(0.75 + 0.15 * i / len(frames), f"Produciendo audio {i + 1}/{len(frames)}…")
                torch.manual_seed(seed + i)
                wavs.append(codec_lite.decode(self.codec, fr).float().cpu())
            scores = [None] * len(wavs)
            best = 0
            if len(wavs) > 1 and self.has_judges():
                try:
                    J = self._judges()
                    self._gpu("judges")
                    scores = J.score_many(wavs, SR, lyrics, progress=lambda f, msg: progress(0.9 + 0.1 * f, msg))
                    for s in scores:
                        s["total"] = round(s["CE"] + s["CU"] + s["PQ"] - 10 * s["WER"], 3)
                    best = max(range(len(wavs)), key=lambda i: scores[i]["total"])
                except Exception as e:  # noqa: BLE001  si los jueces fallan, se entregan las tomas sin ordenar
                    self.log(f"No se pudieron puntuar las tomas: {e}")
                    scores, best = [None] * len(wavs), 0
            progress(1.0, "¡Lista!")
            return dict(wavs=wavs, scores=scores, best=best, seed=seed)
