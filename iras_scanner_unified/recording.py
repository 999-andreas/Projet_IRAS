"""
Enregistrement IQ, sauvegarde en fichiers, et threading producteur/consommateur.
"""

import os
import json
import numpy as np
import threading
import wave
import time
from datetime import datetime
from config import SETTLE_SKIP_SAMPLES
from visualization import save_debug_plot

# ─────────────────────────────────────────────
#  [2] THREAD PRODUCTEUR
# ─────────────────────────────────────────────

#plus utilisé car débile 
class IQProducer(threading.Thread):
    """
    Lit les samples en continu et les pousse dans une queue.
    Pendant que le thread principal traite la FFT, l'acquisition continue.
    """

    def __init__(self, sdr, buf_queue, n_samples, n_buffers=5):
        super().__init__(daemon=True)
        self._sdr       = sdr
        self._queue     = buf_queue
        self._n_samples = n_samples
        self._n_buffers = n_buffers
        self._stop_evt  = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        chunk = max(256, (self._n_samples // 256) * 256)
        while not self._stop_evt.is_set():
            try:
                samples = self._sdr.read_samples(chunk)
                self._queue.put(samples, timeout=2.0)
            except Exception:
                if not self._stop_evt.is_set():
                    break


# ─────────────────────────────────────────────
#  ENREGISTREMENT IQ
# ─────────────────────────────────────────────

def record_iq(sdr, duration_sec, sample_rate):
    """Enregistre duration_sec secondes d'IQ de manière synchrone."""
    
    # Purge initiale du transitoire / PLL
    time.sleep(0.03)
    if SETTLE_SKIP_SAMPLES > 0:
        _ = sdr.read_samples(SETTLE_SKIP_SAMPLES)

    num_samples = int(sample_rate * duration_sec)
    chunk_size  = 16 * 1024
    chunks      = []
    collected   = 0

    while collected < num_samples:
        to_read = min(chunk_size, num_samples - collected)
        to_read = max(256, (to_read // 256) * 256)
        samples = sdr.read_samples(to_read)
        chunks.append(samples)
        collected += len(samples)

    return np.concatenate(chunks)[:num_samples]


# ─────────────────────────────────────────────
#  SAUVEGARDE FICHIERS
# ─────────────────────────────────────────────

def save_sample(iq, freq_hz, snr_db, sample_rate, output_dir, bw_hz=None):
    """
    Sauvegarde les données IQ en format WAV stéréo (I/Q) et métadonnées JSON.
    """
    # output_dir is expected to be the run-specific folder created by the scanner
    os.makedirs(output_dir, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_mhz = freq_hz / 1e6
    base     = f"signal_{freq_mhz:.3f}MHz_{ts}"
    wav_path = os.path.join(output_dir, base + ".wav")
    jpath    = os.path.join(output_dir, base + ".json")

    # Normalise et convertit en int16
    iq_normalized = iq / (np.max(np.abs(iq)) + 1e-12)
    i_data = (iq_normalized.real * 32767).astype(np.int16)
    q_data = (iq_normalized.imag * 32767).astype(np.int16)

    # Entrelace I et Q en stéréo (I=gauche, Q=droite)
    stereo = np.empty(len(iq) * 2, dtype=np.int16)
    stereo[0::2] = i_data
    stereo[1::2] = q_data

    # Écrit le fichier WAV
    with wave.open(wav_path, 'w') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(stereo.tobytes())

    # Sauvegarde les métadonnées JSON
    meta = {
        "frequency_hz"  : freq_hz,
        "frequency_mhz" : round(freq_mhz, 3),
        "sample_rate"   : sample_rate,
        "timestamp"     : ts,
        "num_samples"   : len(iq),
        "snr_db"        : round(float(snr_db), 2),
    }
    if bw_hz is not None:
        meta["estimated_bandwidth_hz"] = float(bw_hz)

    with open(jpath, "w") as f:
        json.dump(meta, f, indent=2)

    return wav_path