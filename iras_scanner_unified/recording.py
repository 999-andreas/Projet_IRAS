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


# ─────────────────────────────────────────────
#  REJEU : HOP BRUT (pour sdr.FileReplaySdrClient)
# ─────────────────────────────────────────────
#
# Contrairement à save_sample() (qui sauvegarde un signal DÉJÀ isolé et
# normalisé pour l'écoute/le debug), les fonctions ci-dessous sauvegardent
# et relisent le buffer IQ BRUT d'un hop ENTIER, tel que capturé par
# sdr.read_samples() avant toute détection. C'est ce qui permet de rejouer
# un scan complet (même pipeline de détection/extraction) sans RTL-SDR :
# voir scanner.scan() (option --save-hop-iq) et sdr.FileReplaySdrClient.

def save_hop_iq(iq, freq_hz, sample_rate, output_dir, hop_index):
    """
    Sauvegarde le buffer IQ brut d'un hop (pré-détection) en WAV stéréo I/Q
    + métadonnées JSON, nommé pour correspondre au hop_XXXX_*.png produit
    par visualization.save_hop_plot() (même base de nom).

    Pas de renormalisation par le max (contrairement à save_sample) : les
    échantillons du RTL-SDR sont déjà dans [-1, 1], et on veut conserver
    l'échelle réelle du hop (plancher de bruit, saturation) pour que le
    rejeu se comporte comme un vrai scan.
    """
    os.makedirs(output_dir, exist_ok=True)

    base     = f"hop_{hop_index:04d}_{freq_hz/1e6:.3f}MHz"
    wav_path = os.path.join(output_dir, base + ".iq.wav")
    jpath    = os.path.join(output_dir, base + ".iq.json")

    i_data = np.clip(iq.real * 32767, -32768, 32767).astype(np.int16)
    q_data = np.clip(iq.imag * 32767, -32768, 32767).astype(np.int16)

    stereo = np.empty(len(iq) * 2, dtype=np.int16)
    stereo[0::2] = i_data
    stereo[1::2] = q_data

    with wave.open(wav_path, 'w') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(stereo.tobytes())

    meta = {
        "hop_index"    : hop_index,
        "frequency_hz" : freq_hz,
        "frequency_mhz": round(freq_hz / 1e6, 3),
        "sample_rate"  : sample_rate,
        "timestamp"    : datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_samples"  : len(iq),
    }
    with open(jpath, "w") as f:
        json.dump(meta, f, indent=2)

    return wav_path


def load_hop_iq(wav_path):
    """Relit un fichier .iq.wav écrit par save_hop_iq() → (iq complex64, sample_rate)."""
    with wave.open(wav_path, 'r') as f:
        sample_rate = f.getframerate()
        n_frames    = f.getnframes()
        raw         = f.readframes(n_frames)

    stereo = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    iq = (stereo[0::2] + 1j * stereo[1::2]).astype(np.complex64)
    return iq, sample_rate


def list_hop_recordings(directory):
    """
    Scanne `directory` pour les paires hop_XXXX_*.iq.wav / .iq.json produites
    par save_hop_iq(), et retourne une liste de dicts triée par fréquence :

        {hop_index, freq_hz, sample_rate, num_samples, wav_path, json_path}

    Utilisé par sdr.FileReplaySdrClient pour construire son index de rejeu.
    Retourne une liste vide si le dossier n'existe pas ou ne contient aucun
    hop enregistré (ex: dossier issu d'un scan sans --save-hop-iq).
    """
    entries = []
    if not os.path.isdir(directory):
        return entries

    for name in sorted(os.listdir(directory)):
        if not name.endswith(".iq.json"):
            continue
        jpath = os.path.join(directory, name)
        try:
            with open(jpath) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        wav_path = os.path.join(directory, name[:-len(".json")] + ".wav")
        if not os.path.isfile(wav_path):
            continue

        entries.append({
            "hop_index"  : meta.get("hop_index"),
            "freq_hz"    : meta["frequency_hz"],
            "sample_rate": meta["sample_rate"],
            "num_samples": meta.get("num_samples"),
            "wav_path"   : wav_path,
            "json_path"  : jpath,
        })

    entries.sort(key=lambda e: e["freq_hz"])
    return entries