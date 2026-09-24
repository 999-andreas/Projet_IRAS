#!/usr/bin/env python3
"""
Importe des enregistrements IQ externes (SDR++, SDR#, GQRX, GNU Radio...)
vers le format hop_XXXX_*.iq.wav / .iq.json attendu par le mode replay
(--mode replay --replay-dir ..., voir sdr.FileReplaySdrClient).

--replay-dir attend un DOSSIER contenant des paires hop_XXXX_<freq>MHz.iq.wav
/ .iq.json produites par recording.save_hop_iq() — un simple fichier .wav de
SDR++ ne suffit pas tel quel (pas de métadonnées hop_index/fréquence dans le
format attendu, même si les octets I/Q bruts sont là). Ce script fait la
conversion : il lit le(s) WAV source(s) (SDR++/SDR# : PCM 16 bits OU IEEE
float32, mono ou stéréo I/Q — auto-détecté), en déduit la fréquence centrale
depuis le nom de fichier (convention SDR++/SDR#/GQRX '..._<freq>Hz_...') sauf
si --freq-hz est fourni, et écrit un hop réutilisable par FileReplaySdrClient
via la même fonction save_hop_iq() que le scanner utilise pour ses propres
captures — donc format garanti identique, aucune divergence possible.

Exemples :

    # Un seul fichier, fréquence auto-détectée dans le nom
    python import_iq_recording.py \\
        ~/.config/sdrpp/recordings/baseband_99958223Hz_18-34-05_04-09-2026.wav \\
        --output-dir ./dataset/imported_99958223Hz

    # Plusieurs fichiers -> plusieurs hops dans le même dossier de rejeu
    python import_iq_recording.py \\
        ~/.config/sdrpp/recordings/baseband_97515000Hz_*.wav \\
        ~/.config/sdrpp/recordings/baseband_99958223Hz_*.wav \\
        --output-dir ./dataset/imported_multi

Puis :

    python iras_scanner.py --mode replay --replay-dir ./dataset/imported_99958223Hz \\
        --freq-start 99.9e6 --freq-end 100.1e6 --freq-step 0.2e6 --sample-rate 2.4e6
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recording import save_hop_iq  # noqa: E402  (même fonction que le scanner)

FREQ_IN_NAME_RE = re.compile(r"(\d+)Hz")


def guess_freq_hz(path: str):
    """Convention SDR++ / SDR# / beaucoup d'outils : '..._<freq_hz>Hz_...'."""
    m = FREQ_IN_NAME_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def load_wav_iq(path: str):
    """
    Lit un WAV I/Q (PCM 8/16/24/32 bits OU IEEE float32), mono ou stéréo,
    et retourne (iq: complex64 dans [-1, 1] environ, sample_rate: int).

    scipy.io.wavfile est utilisé plutôt que le module 'wave' de la stdlib
    car SDR++ écrit souvent en IEEE float32 (format WAV 3), que 'wave' ne
    sait pas lire.
    """
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(path)

    if data.ndim == 1:
        print(f"  ⚠  '{os.path.basename(path)}' est MONO (1 canal) : ce n'est "
              f"probablement PAS un enregistrement I/Q brut (plutôt de l'audio "
              f"démodulé). Traité comme Q=0 — les résultats de détection seront "
              f"probablement inutilisables.")
        real = data.astype(np.float32)
        imag = np.zeros_like(real)
    else:
        real = data[:, 0].astype(np.float32)
        imag = data[:, 1].astype(np.float32)

    # Ramène vers [-1, 1] selon le type d'origine, sans renormaliser par le
    # max (on veut garder l'échelle réelle, comme save_hop_iq).
    if np.issubdtype(data.dtype, np.integer):
        full_scale = float(np.iinfo(data.dtype).max)
        real, imag = real / full_scale, imag / full_scale
    # si déjà float (IEEE float32 WAV), on suppose que la source (SDR++, etc.)
    # a déjà écrit dans une plage raisonnable ([-1, 1] typiquement) : rien à faire.

    iq = (real + 1j * imag).astype(np.complex64)
    return iq, int(sample_rate)


def import_one(path, output_dir, hop_index, freq_hz_override, sample_rate_override, max_secs):
    freq_hz = freq_hz_override or guess_freq_hz(path)
    if freq_hz is None:
        raise ValueError(
            f"Impossible de déduire la fréquence centrale de '{path}' depuis "
            f"son nom (attendu : '..._<freq_en_Hz>Hz_...', convention SDR++/SDR#). "
            f"Précisez --freq-hz explicitement pour ce fichier."
        )

    iq, sample_rate = load_wav_iq(path)
    if sample_rate_override:
        sample_rate = sample_rate_override

    if max_secs is not None:
        n_max = int(sample_rate * max_secs)
        if len(iq) > n_max:
            iq = iq[:n_max]

    wav_path = save_hop_iq(iq, freq_hz, sample_rate, output_dir, hop_index)
    duration_s = len(iq) / sample_rate
    print(f"  [{hop_index:04d}] {os.path.basename(path)} → "
          f"{os.path.basename(wav_path)}  "
          f"({freq_hz/1e6:.3f} MHz, {sample_rate/1e6:.3f} Msps, {duration_s:.2f} s)")


def build_parser():
    p = argparse.ArgumentParser(
        description="Importe des enregistrements IQ externes (SDR++, SDR#, GQRX...) "
                     "vers le format hop_*.iq.wav attendu par --mode replay."
    )
    p.add_argument("inputs", nargs="+", help="Fichier(s) WAV source (I/Q, stéréo).")
    p.add_argument("--output-dir", required=True,
                    help="Dossier de sortie (à donner ensuite à --replay-dir).")
    p.add_argument("--freq-hz", type=float, default=None,
                    help="Fréquence centrale en Hz. Sinon déduite du nom de fichier "
                         "(un seul fichier en entrée uniquement).")
    p.add_argument("--sample-rate", type=float, default=None,
                    help="Force le sample rate (Hz). Par défaut : celui du WAV source.")
    p.add_argument("--max-secs", type=float, default=None,
                    help="Tronque chaque hop importé à cette durée max (s). "
                         "Par défaut : tout le fichier est importé.")
    return p


def main():
    args = build_parser().parse_args()

    if args.freq_hz is not None and len(args.inputs) > 1:
        print("  ⚠  --freq-hz fourni avec plusieurs fichiers : appliqué à TOUS. "
              "Si vos fichiers sont à des fréquences différentes, importez-les "
              "un par un plutôt.")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, path in enumerate(args.inputs, start=1):
        import_one(
            path, args.output_dir, hop_index=i,
            freq_hz_override=int(args.freq_hz) if args.freq_hz else None,
            sample_rate_override=int(args.sample_rate) if args.sample_rate else None,
            max_secs=args.max_secs,
        )

    print(f"\n  Terminé → {args.output_dir}")
    print(f"  Rejeu   → python iras_scanner.py --mode replay --replay-dir {args.output_dir} ...")


if __name__ == "__main__":
    main()
