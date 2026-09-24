import asyncio
import threading
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from scanner import scan
from ws_scan_events import WebSocketScanEvents
from scan_events import ScanEvents
from config import (
    DEFAULT_FREQ_START, DEFAULT_FREQ_END, DEFAULT_FREQ_STEP,
    DEFAULT_SAMPLE_RATE, DEFAULT_GAIN, DEFAULT_PPM,
    DEFAULT_SNR_THRESHOLD, DEFAULT_RECORD_SECS, DEFAULT_OUTPUT_DIR,
    DEFAULT_N_AVG, DEFAULT_DUPLICATE_HZ, DEFAULT_DETECTOR,
    DEFAULT_RTL_TCP_HOST, DEFAULT_RTL_TCP_PORT, DEFAULT_TCP_TIMEOUT,
)

app = FastAPI(title="IRAS Scanner API")

# Créer et monter le répertoire de sortie pour servir l'audio et les images
output_path = Path(DEFAULT_OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=DEFAULT_OUTPUT_DIR), name="static")

# Dashboard web (webui/index.html) — monté sur un préfixe dédié pour ne pas
# entrer en conflit avec /static ci-dessus (qui sert le dossier de sortie).
webui_path = Path(__file__).resolve().parent / "webui"
if webui_path.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(webui_path), html=True), name="dashboard")


@app.get("/")
async def root():
    """Redirige la racine vers le dashboard, pour un accès direct à http://<host>:<port>/."""
    return RedirectResponse(url="/dashboard/")

# ─────────────────────────────────────────────
#  ÉTAT PARTAGÉ (Thread-safe)
# ─────────────────────────────────────────────
_ws_lock = threading.Lock()

_scan_state = {
    "thread": None,
    "stop_flag": None,
    "connected_ws": set(),
}


class ScanConfig(BaseModel):
    mode: str = "rtltcp"  # "usb", "rtltcp" ou "replay"
    rtl_tcp_host: str = DEFAULT_RTL_TCP_HOST
    rtl_tcp_port: int = DEFAULT_RTL_TCP_PORT
    rtl_tcp_timeout: float = DEFAULT_TCP_TIMEOUT

    # Rejeu fichier (mode "replay" uniquement — pas de SDR requis)
    replay_dir: Optional[str] = None
    replay_freq_tolerance_hz: Optional[float] = None

    # Import auto d'IQ externe (SDR++/SDR#/GQRX...) : chemin(s) serveur vers
    # des WAV bruts, convertis à la volée puis rejoués (force mode="replay",
    # inutile de fournir replay_dir ni de lancer import_iq_recording.py à la main).
    iq_file: Optional[List[str]] = None

    freq_start: float = DEFAULT_FREQ_START
    freq_end: float = DEFAULT_FREQ_END
    freq_step: float = DEFAULT_FREQ_STEP
    sample_rate: float = DEFAULT_SAMPLE_RATE

    gain: float = DEFAULT_GAIN
    ppm: int = DEFAULT_PPM
    offset_tuning: bool = False

    snr_threshold: float = DEFAULT_SNR_THRESHOLD
    record_secs: float = DEFAULT_RECORD_SECS
    min_separation_hz: float = DEFAULT_DUPLICATE_HZ
    detector: str = DEFAULT_DETECTOR
    save_hop_iq: bool = False

    n_avg: int = DEFAULT_N_AVG
    output_dir: str = DEFAULT_OUTPUT_DIR


def _config_to_args(cfg: ScanConfig) -> Namespace:
    d = cfg.model_dump()
    d.pop("mode")
    return Namespace(**d)


class BroadcastScanEvents(ScanEvents):
    """Diffuse chaque événement à tous les WebSockets actuellement connectés."""

    def __init__(self, connected_ws: set, loop: asyncio.AbstractEventLoop, lock: threading.Lock):
        self._connected_ws = connected_ws
        self._loop = loop
        self._lock = lock

    def _broadcast(self, method_name, *a, **kw):
        with self._lock:
            active_sockets = list(self._connected_ws)

        for ws in active_sockets:
            getattr(WebSocketScanEvents(ws, self._loop), method_name)(*a, **kw)

    def on_scan_start(self, info):
        self._broadcast("on_scan_start", info)

    def on_hop_complete(self, *a, **kw):
        self._broadcast("on_hop_complete", *a, **kw)

    def on_signal_extracted(self, *a, **kw):
        self._broadcast("on_signal_extracted", *a, **kw)

    def on_signal_duplicate(self, *a, **kw):
        self._broadcast("on_signal_duplicate", *a, **kw)

    def on_hop_error(self, *a, **kw):
        self._broadcast("on_hop_error", *a, **kw)

    def on_scan_end(self, *a, **kw):
        self._broadcast("on_scan_end", *a, **kw)

    def on_scan_interrupted(self):
        self._broadcast("on_scan_interrupted")


# ─────────────────────────────────────────────
#  ENDPOINTS REST
# ─────────────────────────────────────────────

@app.post("/scan/start")
async def start_scan(cfg: ScanConfig):
    with _ws_lock:
        already_running = _scan_state["thread"] is not None and _scan_state["thread"].is_alive()
    if already_running:
        return {"status": "error", "message": "Un scan est déjà en cours."}

    mode = "replay" if cfg.iq_file else cfg.mode

    if mode == "replay" and not cfg.replay_dir and not cfg.iq_file:
        return {"status": "error", "message": "mode='replay' nécessite replay_dir ou iq_file."}

    args = _config_to_args(cfg)
    stop_flag = threading.Event()
    loop = asyncio.get_running_loop()
    events = BroadcastScanEvents(_scan_state["connected_ws"], loop, _ws_lock)

    def run_scan():
        try:
            scan(args, mode, events=events, stop_flag=stop_flag)
        finally:
            with _ws_lock:
                _scan_state["thread"] = None
                _scan_state["stop_flag"] = None

    thread = threading.Thread(target=run_scan, daemon=True)
    with _ws_lock:
        _scan_state["thread"] = thread
        _scan_state["stop_flag"] = stop_flag
    thread.start()

    return {"status": "started"}


@app.post("/scan/stop")
async def stop_scan():
    with _ws_lock:
        stop_flag = _scan_state["stop_flag"]
    if stop_flag is None:
        return {"status": "error", "message": "Aucun scan en cours."}
    stop_flag.set()
    return {"status": "stop_requested"}


@app.get("/scan/status")
async def scan_status():
    with _ws_lock:
        thread = _scan_state["thread"]
    running = thread is not None and thread.is_alive()
    return {"running": running}


# ─────────────────────────────────────────────
#  WEBSOCKET
# ─────────────────────────────────────────────

@app.websocket("/scan/events")
async def scan_events_ws(websocket: WebSocket):
    await websocket.accept()
    with _ws_lock:
        _scan_state["connected_ws"].add(websocket)
    try:
        while True:
            # Attend les déconnexions sans bloquer
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            _scan_state["connected_ws"].discard(websocket)