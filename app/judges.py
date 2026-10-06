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
    def vocals(self, wav, sr):
        from demucs.apply import apply_model
        w = torchaudio.functional.resample(wav.float(), sr, self.sep.samplerate)
        if w.shape[0] == 1:
            w = w.repeat(2, 1)
        out = apply_model(self.sep, w[None].to(getattr(self, 'dev', 'cuda')), split=True, overlap=0.1)[0]
        v = out[self.sep.sources.index("vocals")].mean(0)
        return torchaudio.functional.resample(v.cpu(), self.sep.samplerate, 16000)

    @torch.inference_mode()
    def lyrics_wer(self, wav, sr, lyrics):
        """Error de palabras (0 = perfecto) entre la letra pedida y la transcrita."""
        import jiwer
        v = self.vocals(wav, sr).numpy()
        txt = self.asr({"raw": v, "sampling_rate": 16000}, chunk_length_s=30, batch_size=4,
                       generate_kwargs=dict(task="transcribe", num_beams=2, condition_on_prev_tokens=False))["text"]
        ref, hyp = _norm(lyrics), _norm(txt)
        if not hyp:
            return 1.0, txt
        # (sustituciones + omisiones) / palabras pedidas: no penaliza lo que se oiga en el tramo instrumental
        o = jiwer.process_words(" ".join(ref), " ".join(hyp))
        return (o.substitutions + o.deletions) / max(1, len(ref)), txt

    @torch.inference_mode()
    def aesthetics(self, wav, sr):
        r = self.aes.forward([{"path": wav.float().cpu(), "sample_rate": sr}])[0]
        return {k: round(float(v), 3) for k, v in r.items()}

    def score(self, wav, sr, lyrics):
        a = self.aesthetics(wav, sr)
        wer, txt = self.lyrics_wer(wav, sr, lyrics)
        a["WER"] = round(wer, 3)
        a["texto"] = txt
        return a
