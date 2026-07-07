#!/usr/bin/env python3
"""
IRAS Scanner v2 — RTL-SDR automatic signal scanner & IQ dataset collector
Détecte les signaux, les classe par modulation (heuristiques) et enregistre en .npy

Améliorations vs v1 :
  [1] Moyennage FFT    — N frames avec 50 % d'overlap (gain SNR ≈ √N)
  [2] Threading        — producer/consumer via queue pour acquisition continue
  [3] Pleine bande     — sample_rate 2.4 MHz, step 2 MHz → 10× moins de hops
  [4] DC spike         — masquage des bins centraux + option offset-tuning
  [5] Détection NumPy  — scipy.ndimage.label remplace la boucle Python pure
  [6] Bruit floor      — 10e percentile au lieu de la médiane (robuste aux signaux denses)
  [7] Correction PPM   — rtlsdr_set_freq_correction() exposé en paramètre CLI
  [8] Stabilisation    — lire+jeter le premier bloc après retune (vide le buffer USB)
  [9] Fenêtre Blackman — conservée (déjà meilleure que rtl_power C original)
  [10] Gain adaptatif  — avertissement de saturation + support du mode AGC automatique
"""

import numpy as np
import json
import time
import os
import queue
import threading
import argparse
from datetime import datetime
from scipy.ndimage import label as ndimage_label
from rtlsdr import RtlSdr


# ─────────────────────────────────────────────
#  CONFIG PAR DÉFAUT
# ─────────────────────────────────────────────

DEFAULT_FREQ_START      = 88e6      # Hz
DEFAULT_FREQ_END        = 108e6     # Hz
DEFAULT_FREQ_STEP       = 2_000_000 # Hz  — [3] pleine BW par hop (≈ sample_rate - guard)
DEFAULT_SAMPLE_RATE     = 2_400_000 # Hz  — [3] BW max RTL-SDR stable
DEFAULT_GAIN            = 0        # dB  — [10] réduit pour limiter la saturation
DEFAULT_PPM             = 0         # ppm — [7] correction cristal oscillateur
DEFAULT_SNR_THRESHOLD   = 20        # dB au dessus du plancher de bruit
DEFAULT_RECORD_SECS     = 3         # secondes d'enregistrement IQ
DEFAULT_OUTPUT_DIR      = "./dataset"
DEFAULT_FFT_SIZE        = 2048
DEFAULT_N_AVG           = 64        # [1] nombre de frames à moyenner
DEFAULT_DUPLICATE_HZ    = 300_000   # Hz
DEFAULT_DC_WIDTH_BINS   = None      # None = auto (1 % des bins)
SETTLE_SKIP_SAMPLES     = 8_192     # [8] samples jetés après retune
SATURATION_THRESHOLD_DB = -3.0      # [10] niveau max accepté avant avertissement


# ─────────────────────────────────────────────
#  [1] FFT MOYENNÉE (Welch simplifié, 50 % overlap)
# ─────────────────────────────────────────────

def compute_averaged_psd(samples, fft_size=DEFAULT_FFT_SIZE, n_avg=DEFAULT_N_AVG):
    """
    Densité spectrale de puissance moyennée sur N frames avec 50 % d'overlap.

    Accumulation en linéaire (pas en dB) pour que la moyenne soit statistiquement
    correcte, puis conversion finale en dBFS.  Le gain SNR théorique est ≈ √N :
    pour n_avg=32 → +7.5 dB de réjection du bruit.
    """
    window   = np.blackman(fft_size)            # [9] Blackman conservé
    norm     = np.sum(window ** 2)              # normalisation de puissance
    total    = np.zeros(fft_size, dtype=np.float64)
    step     = fft_size // 2                    # 50 % overlap
    count    = 0

    for start in range(0, len(samples) - fft_size, step):
        frame   = samples[start : start + fft_size]
        fft_out = np.fft.fftshift(np.fft.fft(frame * window, fft_size))
        total  += np.abs(fft_out) ** 2
        count  += 1
        if count >= n_avg:
            break

    if count == 0:
        # Pas assez de samples pour même une frame — retourne un spectre plat
        return np.full(fft_size, -120.0)

    psd = 10 * np.log10(total / (count * norm) + 1e-20)
    return psd


# ─────────────────────────────────────────────
#  [4] MASQUAGE DU DC SPIKE
# ─────────────────────────────────────────────

def mask_dc_spike(psd, fft_size=DEFAULT_FFT_SIZE, dc_width_bins=DEFAULT_DC_WIDTH_BINS):
    """
    Remplace les bins du DC spike par le minimum du spectre.
    dc_width_bins=None → automatique : 1 % de la taille FFT (≥ 4 bins).
    """
    if dc_width_bins is None:
        dc_width_bins = max(4, fft_size // 100)
    center = fft_size // 2
    psd    = psd.copy()
    lo     = max(0, center - dc_width_bins)
    hi     = min(fft_size, center + dc_width_bins)
    psd[lo:hi] = np.min(psd)
    return psd


# ─────────────────────────────────────────────
#  [5] + [6] DÉTECTION DE PIC (NumPy + percentile)
# ─────────────────────────────────────────────

def noise_floor(psd):
    """
    [6] Estime le plancher de bruit avec le 10e percentile.
    Plus robuste que la médiane quand le spectre contient de nombreux signaux :
    la médiane monte avec la densité de signaux, le 10e percentile non.
    """
    return float(np.percentile(psd, 10))


def peak_power(psd, min_bins=10, above_db=6):
    """
    [5] Détecte le pic le plus puissant sur au moins min_bins bins consécutifs.

    Utilise scipy.ndimage.label pour étiqueter les régions connexes au dessus du seuil —
    remplace la boucle Python pure de la v1 par des opérations vectorielles C.
    """
    floor  = noise_floor(psd)
    above  = psd > (floor + above_db)

    labeled, n_regions = ndimage_label(above)
    if n_regions == 0:
        return floor

    best = floor
    for region_id in range(1, n_regions + 1):
        mask = labeled == region_id
        if np.sum(mask) >= min_bins:
            best = max(best, float(np.max(psd[mask])))
    return best


def snr_value(psd):
    """Retourne le SNR (pwr_pic - plancher_bruit) en dB."""
    floor = noise_floor(psd)
    pwr   = peak_power(psd)
    return pwr - floor


# ─────────────────────────────────────────────
#  [10] VÉRIFICATION DE SATURATION
# ─────────────────────────────────────────────

def check_saturation(psd, threshold_db=SATURATION_THRESHOLD_DB):
    """
    Retourne True si le maximum du spectre dépasse le seuil de saturation.
    Au dessus de ~-3 dBFS le convertisseur RTL-SDR écrête et génère des harmoniques.
    """
    return float(np.max(psd)) >= threshold_db


# ─────────────────────────────────────────────
#  CLASSIFICATION PAR HEURISTIQUES (inchangée mais utilise la PSD moyennée)
# ─────────────────────────────────────────────

def estimate_bandwidth(samples, sample_rate, fft_size=DEFAULT_FFT_SIZE,
                       n_avg=DEFAULT_N_AVG, threshold_db=10):
    """Estime la largeur de bande occupée en Hz."""
    psd     = compute_averaged_psd(samples, fft_size, n_avg)
    psd     = mask_dc_spike(psd, fft_size)
    floor   = noise_floor(psd)
    mask    = psd > (floor + threshold_db)
    bw_bins = int(np.sum(mask))
    bw_hz   = bw_bins * (sample_rate / fft_size)
    return bw_hz


def am_fm_ratio(samples):
    """
    Ratio AM/FM heuristique.
    Retourne (var_amplitude, var_phase).
    """
    amplitude = np.abs(samples)
    phase     = np.unwrap(np.angle(samples))
    var_amp   = float(np.var(amplitude / (np.mean(amplitude) + 1e-12)))
    var_phase = float(np.var(np.diff(phase)))
    return var_amp, var_phase


def is_duplicate_signal(freq_hz, detected_freqs, min_separation_hz):
    """Retourne True si un signal trop proche a déjà été enregistré."""
    return any(abs(freq_hz - d) <= min_separation_hz for d in detected_freqs)


# ─────────────────────────────────────────────
#  ENREGISTREMENT IQ
# ─────────────────────────────────────────────

def record_iq(sdr, duration_sec, sample_rate):
    """Enregistre duration_sec secondes d'IQ en chunks sûrs pour l'USB."""
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


def save_sample(iq, freq_hz, snr_db, sample_rate, output_dir):
    """Sauvegarde les données IQ (.npy) et les métadonnées (.json)."""
    os.makedirs(output_dir, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_mhz = freq_hz / 1e6
    base     = f"signal_{freq_mhz:.3f}MHz_{ts}"
    np_path  = os.path.join(output_dir, base + ".npy")
    jpath    = os.path.join(output_dir, base + ".json")

    np.save(np_path, iq)

    with open(jpath, "w") as f:
        json.dump({
            "frequency_hz"  : freq_hz,
            "frequency_mhz" : round(freq_mhz, 3),
            "sample_rate"   : sample_rate,
            "timestamp"     : ts,
            "num_samples"   : len(iq),
            "snr_db"        : round(float(snr_db), 2),
        }, f, indent=2)

    return np_path


# ─────────────────────────────────────────────
#  [2] THREAD PRODUCTEUR — acquisition continue
# ─────────────────────────────────────────────

class IQProducer(threading.Thread):
    """
    Lit les samples RTL-SDR en continu dans un thread dédié et les pousse
    dans une queue.  Pendant que le thread principal traite la FFT, l'USB
    continue à drainer le buffer du dongle sans interruption.
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
                # Bloque si la queue est pleine (backpressure naturelle)
                self._queue.put(samples, timeout=2.0)
            except Exception:
                if not self._stop_evt.is_set():
                    break


# ─────────────────────────────────────────────
#  SCANNER PRINCIPAL
# ─────────────────────────────────────────────

def scan(args):
    sdr             = RtlSdr()
    sdr.sample_rate = args.sample_rate
    sdr.gain        = args.gain          # [10] gain configurable

    # [7] Correction PPM du cristal oscillateur
    if args.ppm != 0:
        sdr.set_freq_correction(args.ppm)

    # [4] Mode offset-tuning pour déplacer le DC spike hors bande (optionnel)
    if args.offset_tuning:
        try:
            sdr.set_offset_tuning(True)
            print("  Offset tuning activé (DC spike déplacé hors bande).")
        except AttributeError:
            print("  ⚠  Offset tuning non supporté par cette version de pyrtlsdr — ignoré.")

    freqs          = np.arange(args.freq_start, args.freq_end, args.freq_step)
    total          = len(freqs)
    detected       = 0
    detected_freqs = []

    # Taille du snapshot de détection : assez pour n_avg frames avec 50 % overlap
    snap_size = DEFAULT_FFT_SIZE * (args.n_avg + 1)

    print(f"\n{'─'*65}")
    print(f"  IRAS Scanner v2")
    print(f"  Plage       : {args.freq_start/1e6:.1f} – {args.freq_end/1e6:.1f} MHz")
    print(f"  Pas         : {args.freq_step/1e6:.2f} MHz  (BW captée : {args.sample_rate/1e6:.1f} MHz)")
    print(f"  Hops        : {total}  (vs {int((args.freq_end - args.freq_start) / 200e3)} avec pas 200 kHz)")
    print(f"  Seuil SNR   : +{args.snr_threshold} dB")
    print(f"  Moyennage   : {args.n_avg} frames FFT (overlap 50 %)")
    print(f"  PPM corr.   : {args.ppm:+d} ppm")
    print(f"  Gain        : {args.gain} dB{' (auto)' if args.gain == 0 else ''}")
    print(f"  Output      : {args.output_dir}")
    print(f"{'─'*65}\n")

    # [2] Queue partagée producteur/consommateur
    buf_queue = queue.Queue(maxsize=5)

    try:
        for i, freq in enumerate(freqs):
            sdr.center_freq = freq

            # [8] Stabilisation : lecture+rejet du buffer résiduel après retune
            _ = sdr.read_samples(max(256, (SETTLE_SKIP_SAMPLES // 256) * 256))
            time.sleep(0.02)

            # [2] Lancer le thread producteur pour ce hop
            producer = IQProducer(sdr, buf_queue, snap_size)
            producer.start()

            # Collecter les samples depuis la queue (consommateur)
            collected = []
            while sum(len(c) for c in collected) < snap_size:
                try:
                    chunk = buf_queue.get(timeout=2.0)
                    collected.append(chunk)
                except queue.Empty:
                    break

            producer.stop()
            producer.join(timeout=1.0)
            # Vider la queue pour le prochain hop
            while not buf_queue.empty():
                try:
                    buf_queue.get_nowait()
                except queue.Empty:
                    break

            if not collected:
                print(f"[{i+1:>4}/{total}] {freq/1e6:>9.3f} MHz  ← TIMEOUT acquisition, hop ignoré")
                continue

            snap = np.concatenate(collected)[:snap_size]

            # [1] PSD moyennée + [4] masquage DC
            psd  = compute_averaged_psd(snap, DEFAULT_FFT_SIZE, args.n_avg)
            psd  = mask_dc_spike(psd, DEFAULT_FFT_SIZE)

            # [6] SNR avec plancher 10e percentile
            floor   = noise_floor(psd)
            pwr     = peak_power(psd)
            snr_val = pwr - floor

            # [10] Avertissement saturation
            sat_warn = ""
            if check_saturation(psd):
                sat_warn = "  ⚠ SATURATION"

            print(f"[{i+1:>4}/{total}] {freq/1e6:>9.3f} MHz  |  "
                  f"floor={floor:>+6.1f} dBFS  SNR={snr_val:>+5.1f} dB{sat_warn}", end="")

            if snr_val >= args.snr_threshold:
                if is_duplicate_signal(freq, detected_freqs, args.min_separation_hz):
                    print("  ← SIGNAL DÉJÀ ENREGISTRÉ (duplication ignorée)")
                    continue

                print(f"  ← SIGNAL DÉTECTÉ, enregistrement {args.record_secs}s ...",
                      end="", flush=True)

                iq   = record_iq(sdr, args.record_secs, args.sample_rate)
                path = save_sample(iq, freq, snr_val, args.sample_rate, args.output_dir)

                detected += 1
                detected_freqs.append(freq)
                print(f"\n           → {path}")
            else:
                print()

    except KeyboardInterrupt:
        print("\n\nScan interrompu.")

    finally:
        sdr.close()

    print(f"\n{'─'*65}")
    print(f"  Scan terminé : {detected} signal(s) enregistré(s) sur {total} fréquences")
    print(f"  Dataset : {args.output_dir}")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IRAS Scanner v2 — RTL-SDR automatic signal collector (optimisé)"
    )
    parser.add_argument("--freq-start",    type=float, default=DEFAULT_FREQ_START,
                        help=f"Fréquence de départ en Hz (défaut: {DEFAULT_FREQ_START/1e6}MHz)")
    parser.add_argument("--freq-end",      type=float, default=DEFAULT_FREQ_END,
                        help=f"Fréquence de fin en Hz (défaut: {DEFAULT_FREQ_END/1e6}MHz)")
    parser.add_argument("--freq-step",     type=float, default=DEFAULT_FREQ_STEP,
                        help=f"Pas en Hz (défaut: {DEFAULT_FREQ_STEP/1e6}MHz — pleine BW)")
    parser.add_argument("--sample-rate",   type=float, default=DEFAULT_SAMPLE_RATE,
                        help=f"Sample rate en Hz (défaut: {DEFAULT_SAMPLE_RATE/1e6}MHz)")
    # [10] Gain = 0 pour activer le mode AGC automatique du RTL-SDR
    parser.add_argument("--gain",          type=float, default=DEFAULT_GAIN,
                        help="Gain en dB. Utiliser 0 pour le mode AGC automatique.")
    # [7] Correction PPM
    parser.add_argument("--ppm",           type=int,   default=DEFAULT_PPM,
                        help="Correction de fréquence en ppm (trouver avec rtl_test -p).")
    parser.add_argument("--snr-threshold", type=float, default=DEFAULT_SNR_THRESHOLD,
                        help=f"Seuil SNR en dB (défaut: {DEFAULT_SNR_THRESHOLD}dB)")
    parser.add_argument("--n-avg",         type=int,   default=DEFAULT_N_AVG,
                        help=f"Nombre de frames FFT à moyenner (défaut: {DEFAULT_N_AVG})")
    parser.add_argument("--record-secs",   type=float, default=DEFAULT_RECORD_SECS,
                        help=f"Durée d'enregistrement en secondes (défaut: {DEFAULT_RECORD_SECS}s)")
    parser.add_argument("--output-dir",    type=str,   default=DEFAULT_OUTPUT_DIR,
                        help=f"Dossier de sortie (défaut: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--min-separation-hz", type=float, default=DEFAULT_DUPLICATE_HZ,
                        help=f"Seuil de séparation pour filtrer les duplications (défaut: {DEFAULT_DUPLICATE_HZ}Hz)")
    # [4] Offset tuning optionnel
    parser.add_argument("--offset-tuning", action="store_true", default=False,
                        help="Active l'offset tuning pour déplacer le DC spike hors bande.")

    args = parser.parse_args()
    scan(args)


if __name__ == "__main__":
    main()