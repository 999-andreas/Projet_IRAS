#!/usr/bin/env python3
"""
IRAS Scanner — RTL-SDR automatic signal scanner & IQ dataset collector
Détecte les signaux, les classe par modulation (heuristiques) et enregistre en .npy
"""

import numpy as np
import json
import time
import os
import argparse
from datetime import datetime
from rtlsdr import RtlSdr

# ─────────────────────────────────────────────
#  CONFIG PAR DÉFAUT
# ─────────────────────────────────────────────

DEFAULT_FREQ_START    = 88e6    # Hz
DEFAULT_FREQ_END      = 108e6   # Hz
DEFAULT_FREQ_STEP     = 200e3   # Hz  (pas du scanner)
DEFAULT_SAMPLE_RATE   = 1e6     # Hz
DEFAULT_GAIN          = 40      # dB
DEFAULT_SNR_THRESHOLD = 30      # dB au dessus du plancher de bruit
DEFAULT_RECORD_SECS   = 5       # secondes d'enregistrement IQ
DEFAULT_OUTPUT_DIR    = "./dataset"
DEFAULT_FFT_SIZE      = 2048

# ─────────────────────────────────────────────
#  UTILITAIRES FFT
# ─────────────────────────────────────────────
def compute_psd(samples, fft_size=DEFAULT_FFT_SIZE):
    """Calcule la densité spectrale de puissance en dBFS."""
    window  = np.blackman(fft_size)
    fft_out = np.fft.fftshift(np.fft.fft(samples[:fft_size] * window, fft_size))
    psd     = 20 * np.log10(np.abs(fft_out) / fft_size + 1e-12)
    return psd


def peak_power(samples, fft_size=DEFAULT_FFT_SIZE, min_bins=10):
    """
    Retourne la puissance du pic le plus significatif.
    Ignore les pics parasites narrow (< min_bins consécutifs au dessus du bruit).
    """
    psd   = compute_psd(samples, fft_size)
    floor = np.median(psd)

    # Cherche les régions au dessus du bruit
    above = psd > (floor + 6)  # 6 dB au dessus du bruit

    # Trouve les groupes de bins consécutifs
    best_power = floor
    count = 0
    for i, val in enumerate(above):
        if val:
            count += 1
        else:
            if count >= min_bins:
                # Région valide — prend le max de cette région
                region_max = np.max(psd[i - count:i])
                best_power = max(best_power, region_max)
            count = 0

    # Vérifie la dernière région
    if count >= min_bins:
        best_power = max(best_power, np.max(psd[-count:]))

    return best_power


def noise_floor(samples, fft_size=DEFAULT_FFT_SIZE):
    """Estime le plancher de bruit (médiane du spectre)."""
    return np.median(compute_psd(samples, fft_size))


# ─────────────────────────────────────────────
#  CLASSIFICATION PAR HEURISTIQUES
# ─────────────────────────────────────────────
def estimate_bandwidth(samples, sample_rate, fft_size=DEFAULT_FFT_SIZE, threshold_db=10):
    """
    Estime la largeur de bande occupée en Hz.
    threshold_db : nb de dB au dessus du plancher de bruit pour considérer un bin 'occupé'
    """
    psd     = compute_psd(samples, fft_size)
    floor   = noise_floor(samples, fft_size)
    mask    = psd > (floor + threshold_db)
    bw_bins = np.sum(mask)
    bw_hz   = bw_bins * (sample_rate / fft_size)
    return bw_hz


def am_fm_ratio(samples):
    """
    Ratio AM/FM heuristique.
    - FM : l'information est dans les variations de phase → variance de phase élevée
    - AM : l'information est dans l'enveloppe → variance d'amplitude élevée, phase stable
    Retourne (var_amplitude, var_phase)
    """
    amplitude = np.abs(samples)
    phase     = np.unwrap(np.angle(samples))
    var_amp   = np.var(amplitude / (np.mean(amplitude) + 1e-12))
    var_phase = np.var(np.diff(phase))
    return var_amp, var_phase


# ─────────────────────────────────────────────
#  ENREGISTREMENT IQ
# ─────────────────────────────────────────────
def record_iq(sdr, duration_sec, sample_rate):
    """Enregistre duration_sec secondes d'échantillons IQ en chunks."""
    num_samples = int(sample_rate * duration_sec)
    chunk_size  = 16 * 1024  # réduit pour plus de stabilité USB
    chunks      = []
    collected   = 0

    while collected < num_samples:
        to_read = min(chunk_size, num_samples - collected)
        # RTL-SDR exige des multiples de 256 samples
        to_read = max(256, (to_read // 256) * 256)
        samples = sdr.read_samples(to_read)
        chunks.append(samples)
        collected += len(samples)

    iq = np.concatenate(chunks)[:num_samples]
    return iq


def save_sample(iq, freq_hz, snr_db, sample_rate, output_dir):
    """Sauvegarde les données IQ et les métadonnées."""
    os.makedirs(output_dir, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_mhz = freq_hz / 1e6

    base      = f"signal_{freq_mhz:.3f}MHz_{ts}"
    np_path   = os.path.join(output_dir, base + ".npy")
    json_path = os.path.join(output_dir, base + ".json")

    np.save(np_path, iq)

    meta_full = {
        "frequency_hz"  : freq_hz,
        "frequency_mhz" : round(freq_mhz, 3),
        "sample_rate"   : sample_rate,
        "timestamp"     : ts,
        "num_samples"   : len(iq),
        "snr_db"        : round(float(snr_db), 2),
    }
    with open(json_path, "w") as f:
        json.dump(meta_full, f, indent=2)

    return np_path


# ─────────────────────────────────────────────
#  SCANNER PRINCIPAL
# ─────────────────────────────────────────────
def scan(args):
    sdr = RtlSdr()
    sdr.sample_rate = args.sample_rate
    sdr.gain        = args.gain

    freqs    = np.arange(args.freq_start, args.freq_end, args.freq_step)
    total    = len(freqs)
    detected = 0

    print(f"\n{'─'*60}")
    print(f"  IRAS Scanner")
    print(f"  Plage  : {args.freq_start/1e6:.1f} – {args.freq_end/1e6:.1f} MHz")
    print(f"  Pas    : {args.freq_step/1e3:.0f} kHz")
    print(f"  Seuil  : +{args.snr_threshold} dB au dessus du bruit")
    print(f"  Output : {args.output_dir}")
    print(f"{'─'*60}\n")

    try:
        for i, freq in enumerate(freqs):
            sdr.center_freq = freq
            time.sleep(0.05)  # laisse le tuner se stabiliser

            # Snapshot rapide pour la détection
            snap    = sdr.read_samples(DEFAULT_FFT_SIZE)
            floor   = noise_floor(snap)
            pwr     = peak_power(snap)
            snr_val = pwr - floor

            print(f"[{i+1:>4}/{total}] {freq/1e6:>9.3f} MHz  |  "
                  f"floor={floor:>+6.1f} dBFS  SNR={snr_val:>+5.1f} dB", end="")

            if snr_val >= args.snr_threshold:
                print(f"  ← SIGNAL DÉTECTÉ, enregistrement {args.record_secs}s ...", end="", flush=True)

                iq   = record_iq(sdr, args.record_secs, args.sample_rate)
                path = save_sample(iq, freq, snr_val, args.sample_rate, args.output_dir)

                detected += 1
                print(f"\n           → {path}")
            else:
                print()

    except KeyboardInterrupt:
        print("\n\nScan interrompu.")

    finally:
        sdr.close()

    print(f"\n{'─'*60}")
    print(f"  Scan terminé : {detected} signal(s) enregistré(s) sur {total} fréquences")
    print(f"  Dataset : {args.output_dir}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="IRAS Scanner — RTL-SDR automatic signal collector"
    )
    parser.add_argument("--freq-start",    type=float, default=DEFAULT_FREQ_START,
                        help=f"Fréquence de départ en Hz (défaut: {DEFAULT_FREQ_START/1e6}MHz)")
    parser.add_argument("--freq-end",      type=float, default=DEFAULT_FREQ_END,
                        help=f"Fréquence de fin en Hz (défaut: {DEFAULT_FREQ_END/1e6}MHz)")
    parser.add_argument("--freq-step",     type=float, default=DEFAULT_FREQ_STEP,
                        help=f"Pas en Hz (défaut: {DEFAULT_FREQ_STEP/1e3}kHz)")
    parser.add_argument("--sample-rate",   type=float, default=DEFAULT_SAMPLE_RATE,
                        help=f"Sample rate en Hz (défaut: {DEFAULT_SAMPLE_RATE/1e3}kHz)")
    parser.add_argument("--gain",          type=float, default=DEFAULT_GAIN,
                        help=f"Gain en dB (défaut: {DEFAULT_GAIN})")
    parser.add_argument("--snr-threshold", type=float, default=DEFAULT_SNR_THRESHOLD,
                        help=f"Seuil SNR en dB au dessus du bruit (défaut: {DEFAULT_SNR_THRESHOLD}dB)")
    parser.add_argument("--record-secs",   type=float, default=DEFAULT_RECORD_SECS,
                        help=f"Durée d'enregistrement en secondes (défaut: {DEFAULT_RECORD_SECS}s)")
    parser.add_argument("--output-dir",    type=str,   default=DEFAULT_OUTPUT_DIR,
                        help=f"Dossier de sortie (défaut: {DEFAULT_OUTPUT_DIR})")

    args = parser.parse_args()
    scan(args)


if __name__ == "__main__":
    main()