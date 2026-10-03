import asyncio
import os
from scan_events import ScanEvents
from config import DEFAULT_OUTPUT_DIR

class WebSocketScanEvents(ScanEvents):
    def __init__(self, websocket, loop):
        self._ws = websocket
        self._loop = loop

    def _send(self, payload: dict):
        try:
            asyncio.run_coroutine_threadsafe(
                self._ws.send_json(payload), self._loop
            )
        except Exception:
            pass

    def _to_static_url(self, file_path: str) -> str:
        """Convertit un chemin serveur (ex: ./dataset/20260724_.../signal.wav)
        en URL statique (/static/20260724_.../signal.wav), cohérent avec
        app.mount("/static", StaticFiles(directory=DEFAULT_OUTPUT_DIR)) dans api.py.
        """
        if not file_path:
            return ""
        output_root = os.path.normpath(DEFAULT_OUTPUT_DIR)
        norm_path = os.path.normpath(file_path)
        try:
            rel_path = os.path.relpath(norm_path, output_root)
        except ValueError:
            # chemins sur des lecteurs différents (Windows) ou hors output_root
            return file_path
        return "/static/" + rel_path.replace("\\", "/")

    def on_scan_start(self, info: dict):
        self._send({"type": "scan_start", **info})

    def on_hop_complete(self, hop_index, total_hops, freq_hz, floor_db,
                         best_snr_db, signals, saturation, hop_plot_path):
        self._send({
            "type": "hop_complete",
            "hop_index": hop_index,
            "total_hops": total_hops,
            "freq_hz": float(freq_hz),
            "floor_db": float(floor_db),
            "best_snr_db": float(best_snr_db),
            "saturation": bool(saturation),
            "signal_count": len(signals),
            "hop_plot_url": self._to_static_url(hop_plot_path),
            "signals": [
                {
                    "freq_hz": float(s["freq_hz"]),
                    "snr_db": float(s["snr_db"]),
                    "bw_hz": float(s["bw_hz"]),
                }
                for s in signals
            ],
        })

    def on_signal_extracted(self, freq_hz, snr_db, bw_hz, wav_path, debug_plot_path):
        self._send({
            "type": "signal_extracted",
            "freq_hz": float(freq_hz),
            "snr_db": float(snr_db),
            "bw_hz": float(bw_hz),
            "wav_url": self._to_static_url(wav_path),
            "debug_plot_url": self._to_static_url(debug_plot_path),
        })

    def on_signal_duplicate(self, freq_hz):
        self._send({"type": "signal_duplicate", "freq_hz": float(freq_hz)})

    def on_hop_error(self, hop_index, total_hops, freq_hz, error):
        self._send({
            "type": "hop_error",
            "hop_index": hop_index,
            "total_hops": total_hops,
            "freq_hz": float(freq_hz),
            "error": str(error),
        })

    def on_scan_end(self, detected_count, total_hops, output_dir):
        self._send({
            "type": "scan_end",
            "detected_count": detected_count,
            "total_hops": total_hops,
            "output_dir": output_dir,
        })

    def on_scan_interrupted(self):
        self._send({"type": "scan_interrupted"})