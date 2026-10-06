"""Servidor local: abre http://127.0.0.1:7860 y crea canciones."""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
HOME = Path(os.environ.get("CANCIONES_HOME", ROOT.parent))
MODELS = HOME / "modelos"
OUT = HOME / "canciones"
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("CANCIONES_PORT", "7860"))

state = {"status": "cargando", "msg": "Cargando modelos…", "error": None}
jobs: dict[str, dict] = {}
work: "queue.Queue[str]" = queue.Queue()
studio = None


def slug(t):
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:40] or "cancion"


def loader():
    global studio
    try:
        sys.path.insert(0, str(ROOT))
        from studio import Studio
        studio = Studio(MODELS, log=lambda m: print(m, flush=True))
        state.update(status="listo", msg="Listo")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        state.update(status="error", msg="No se pudieron cargar los modelos", error=str(e))


def worker():
    while True:
        jid = work.get()
        job = jobs[jid]
        if job["status"] == "cancelado":
            continue
        while studio is None and state["status"] == "cargando":
            time.sleep(0.5)
        if studio is None:
            job.update(status="error", msg=state.get("error") or "Modelos no cargados")
            continue
        job.update(status="generando", t0=time.time())
        try:
            def prog(frac, msg):
                job.update(progress=round(frac, 3), msg=msg)
            r = studio.generate(job["lyrics"], job["tags"], seconds=job["seconds"], takes=job["takes"],
                                seed=job["seed"], cfg=job["cfg"], temperature=job["temperature"],
                                progress=prog, abort=lambda: job["status"] == "cancelado")
            if r is None:
                job.update(status="cancelado", msg="Cancelada")
                continue
            from studio import to_mp3
            base = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(job['title'] or job['tags'])}"
            files = []
            order = [r["best"]] + [i for i in range(len(r["wavs"])) if i != r["best"]]
            for rank, i in enumerate(order):
                name = f"{base}.mp3" if rank == 0 else f"{base}-toma{rank + 1}.mp3"
                to_mp3(r["wavs"][i], str(OUT / name))
                sc = r["scores"][i]
                files.append(dict(file=name, best=rank == 0,
                                  score=None if sc is None else {k: sc[k] for k in ("CE", "CU", "PC", "PQ", "WER")},
                                  seconds=round(r["wavs"][i].shape[-1] / 48000, 1)))
            (OUT / f"{base}.txt").write_text(
                f"Estilo: {job['tags']}\nSemilla: {r['seed']}\n\n{job['lyrics']}\n", encoding="utf-8")
            job.update(status="hecho", progress=1, msg="¡Lista!", files=files, seed=r["seed"],
                       took=round(time.time() - job["t0"], 1))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            msg = str(e)
            if "out of memory" in msg.lower():
                msg = "La tarjeta gráfica se ha quedado sin memoria. Prueba con menos duración o 1 toma."
            job.update(status="error", msg=msg)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (ROOT / "ui.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            info = dict(state)
            if studio is not None:
                info.update(vram=round(studio.vram, 1), judges=studio.has_judges(), cpu=studio.cpu, max_takes=studio.max_takes)
            info["queue"] = sum(1 for j in jobs.values() if j["status"] in ("en cola", "generando"))
            return self._json(info)
        if u.path.startswith("/api/job/"):
            j = jobs.get(u.path.rsplit("/", 1)[-1])
            if not j:
                return self._json({"error": "no existe"}, 404)
            return self._json({k: v for k, v in j.items() if k not in ("lyrics",)})
        if u.path == "/api/songs":
            items = []
            for p in sorted(OUT.glob("*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
                if "-toma" in p.stem:
                    continue
                txt = OUT / (p.stem + ".txt")
                items.append(dict(file=p.name, info=txt.read_text(encoding="utf-8")[:400] if txt.exists() else ""))
            return self._json(items)
        if u.path.startswith("/canciones/"):
            p = (OUT / unquote(u.path.split("/", 2)[2])).resolve()
            if p.parent != OUT.resolve() or not p.exists():
                return self._json({"error": "no existe"}, 404)
            return self._send(200, p.read_bytes(), "audio/mpeg")
        self._json({"error": "no existe"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/generate":
            lyrics = (data.get("lyrics") or "").strip()
            tags = (data.get("tags") or "").strip()
            if not lyrics or not tags:
                return self._json({"error": "Falta la letra o el estilo"}, 400)
            jid = uuid.uuid4().hex[:10]
            jobs[jid] = dict(id=jid, status="en cola", progress=0, msg="En cola…", lyrics=lyrics, tags=tags,
                             title=(data.get("title") or "").strip(),
                             seconds=max(10, min(240, int(data.get("seconds", 120)))),
                             takes=max(1, min(4, int(data.get("takes", 1)))),
                             seed=int(data.get("seed", -1)), cfg=float(data.get("cfg", 1.5)),
                             temperature=float(data.get("temperature", 1.0)))
            work.put(jid)
            return self._json({"id": jid})
        if u.path.startswith("/api/cancel/"):
            j = jobs.get(u.path.rsplit("/", 1)[-1])
            if j and j["status"] in ("en cola", "generando"):
                j["status"] = "cancelado"
            return self._json({"ok": True})
        self._json({"error": "no existe"}, 404)


def main():
    threading.Thread(target=loader, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Abre {url} en el navegador (Ctrl+C para cerrar)", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
