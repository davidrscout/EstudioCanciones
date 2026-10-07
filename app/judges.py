"""Jueces automáticos de calidad: letra inteligible (HeartTranscriptor + Demucs) y estética (Audiobox)."""
import re
import unicodedata

import torch
import torchaudio

M = "modelos"


def _norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"\[[^\]]*\]", " ", t)
    return re.sub(r"[^\w\s]", " ", t).split()


class Judges:
    def __init__(self, transcriptor=M + r"\HeartTranscriptor-oss", aes=M + r"\audiobox-aesthetics\checkpoint.pt"):
        from transformers import pipeline
        from demucs.pretrained import get_model
        from audiobox_aesthetics.infer import AesPredictor
        dev = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
        self.asr = pipeline("automatic-speech-recognition", model=transcriptor,
                            torch_dtype=torch.float16 if dev == "cuda" else torch.float32, device=dev)
        self.sep = get_model("htdemucs").to(dev).eval()
        self.aes = AesPredictor(checkpoint_pth=aes)

    def to(self, device):
        """Para GPUs pequeñas: los jueces solo ocupan VRAM mientras escuchan."""
        self.asr.model.to(device)
        self.asr.device = torch.device(device)
        self.sep.to(device)
        self.aes.model.to(device)
        self.aes.device = torch.device(device)
        self.dev = device
        return self

    @torch.inference_mode()
    def vocals(self, wav, sr, fp16=False):
        from demucs.apply import apply_model
        w = torchaudio.functional.resample(wav.float(), sr, self.sep.samplerate)
        if w.shape[0] == 1:
            w = w.repeat(2, 1)
        dev = getattr(self, "dev", "cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and str(dev).startswith("cuda")):
            out = apply_model(self.sep, w[None].to(dev), split=True, overlap=0.1)[0].float()
        v = out[self.sep.sources.index("vocals")].mean(0)
        return torchaudio.functional.resample(v.cpu(), self.sep.samplerate, 16000)

    @torch.inference_mode()
    def aesthetics(self, wav, sr):
        r = self.aes.forward([{"path": wav.float().cpu(), "sample_rate": sr}])[0]
        return {k: round(float(v), 3) for k, v in r.items()}

    def score(self, wav, sr, lyrics):
        return self.score_many([wav], sr, lyrics)[0]

    @torch.inference_mode()
    def score_many(self, wavs, sr, lyrics, progress=lambda frac, msg: None, fp16_sep=True):
        """Puntúa varias tomas de la misma letra. La transcripción va en un solo lote: Whisper
        genera token a token (limitado por latencia), así que 4 tomas cuestan casi como 1."""
        import jiwer
        n = len(wavs)
        res = [self.aesthetics(w, sr) for w in wavs]
        voc = []
        for i, w in enumerate(wavs):
            progress(0.4 * i / n, f"Separando la voz de la toma {i + 1}/{n}…")
            voc.append({"raw": self.vocals(w, sr, fp16_sep).numpy(), "sampling_rate": 16000})
        progress(0.4, "Escuchando la letra de " + ("la toma…" if n == 1 else f"las {n} tomas…"))
        outs = self.asr(voc, chunk_length_s=30, batch_size=8,
                        generate_kwargs=dict(task="transcribe", num_beams=2, condition_on_prev_tokens=False))
        ref = _norm(lyrics)
        for a, o in zip(res, outs):
            txt = o["text"]
            hyp = _norm(txt)
            if not hyp:
                wer = 1.0
            else:
                # (sustituciones + omisiones) / palabras pedidas: no penaliza lo que se oiga en el tramo instrumental
                m = jiwer.process_words(" ".join(ref), " ".join(hyp))
                wer = (m.substitutions + m.deletions) / max(1, len(ref))
            a["WER"] = round(wer, 3)
            a["texto"] = txt
        return res
