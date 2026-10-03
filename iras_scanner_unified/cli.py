"""
Interface CLI : menu et arguments.
"""

import argparse
from config import (
    DEFAULT_FREQ_START, DEFAULT_FREQ_END, DEFAULT_FREQ_STEP,
    DEFAULT_SAMPLE_RATE, DEFAULT_GAIN, DEFAULT_PPM,
    DEFAULT_SNR_THRESHOLD, DEFAULT_RECORD_SECS, DEFAULT_OUTPUT_DIR,
    DEFAULT_N_AVG, DEFAULT_DUPLICATE_HZ, DEFAULT_DETECTOR,
    DEFAULT_RTL_TCP_HOST, DEFAULT_RTL_TCP_PORT, DEFAULT_TCP_TIMEOUT
)


def select_connection_mode():
    """Menu interactif pour choisir le mode de connexion."""
    print("\n" + "="*65)
    print("  IRAS Scanner — Sélection du mode de connexion")
    print("="*65)
    print("\n  [1] USB      — RTL-SDR branché directement (local)")
    print("  [2] RTL-TCP  — Connexion via serveur rtl_tcp (réseau)")
    print("  [3] Replay   — Rejoue des hops IQ déjà enregistrés (sans SDR)")
    print()

    while True:
        try:
            choice = input("  Choix [1, 2 ou 3] : ").strip()
            if choice == "1":
                return "usb"
            elif choice == "2":
                return "rtltcp"
            elif choice == "3":
                return "replay"
            else:
                print("entrer 1, 2 ou 3.")
        except KeyboardInterrupt:
            print("\n  Annulation.")
            exit(1)


def build_argument_parser():
    """Configure argparse avec tous les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="IRAS Scanner v2 — Unified Edition (USB ou RTL-TCP)"
    )

    # ── Mode de connexion (optionnel) ──────────
    # Si fourni, bypass le menu interactif select_connection_mode()
    # de cli.py — indispensable pour un lancement non-interactif (Docker,
    # script automatisé, cron...), où un input() bloquerait indéfiniment
    # faute de terminal attaché.
    parser.add_argument("--mode", type=str, default=None, choices=["usb", "rtltcp", "replay"],
                        help="Mode de connexion SDR. Si fourni, saute le menu interactif "
                             "(nécessaire pour un lancement non-interactif, ex: Docker).")

    # ── Paramètres RTL-TCP (optionnels) ────────
    tcp = parser.add_argument_group("RTL-TCP (optionnel, pour mode réseau)")
    tcp.add_argument("--rtl-tcp-host",    type=str,   default=DEFAULT_RTL_TCP_HOST,
                     help=f"Hôte du serveur rtl_tcp (défaut: {DEFAULT_RTL_TCP_HOST})")
    tcp.add_argument("--rtl-tcp-port",    type=int,   default=DEFAULT_RTL_TCP_PORT,
                     help=f"Port du serveur rtl_tcp (défaut: {DEFAULT_RTL_TCP_PORT})")
    tcp.add_argument("--rtl-tcp-timeout", type=float, default=DEFAULT_TCP_TIMEOUT,
                     help=f"Timeout socket en secondes (défaut: {DEFAULT_TCP_TIMEOUT}s)")

    # ── Paramètres Replay (optionnels, pour mode fichier) ──
    replay = parser.add_argument_group("Replay (optionnel, pour mode fichier — pas de SDR)")
    replay.add_argument("--replay-dir",              type=str,   default=None,
                       help="Dossier contenant des hops enregistrés avec --save-hop-iq "
                            "(ex: un dossier ./dataset/AAAAMMJJ_HHMMSS d'un run précédent). "
                            "Requis en --mode replay.")
    replay.add_argument("--replay-freq-tolerance-hz", type=float, default=None,
                       help="Écart max (Hz) toléré entre la fréquence demandée et la "
                            "fréquence du hop enregistré le plus proche. Par défaut : pas "
                            "de limite (on rejoue toujours le hop le plus proche).")
    replay.add_argument("--iq-file", type=str, nargs="+", default=None,
                       help="Fichier(s) WAV I/Q externe(s) (SDR++, SDR#, GQRX...) à rejouer "
                            "directement, sans passer par import_iq_recording.py ni --replay-dir : "
                            "converti(s) automatiquement puis rejoué(s) comme --mode replay. "
                            "Fréquence déduite du nom de fichier ('..._<freq>Hz_...') si présente.")

    # ── Paramètres scanner ────────────────────
    scanner = parser.add_argument_group("Scanner")
    scanner.add_argument("--freq-start",         type=float, default=DEFAULT_FREQ_START,
                        help=f"Fréquence de départ en Hz (défaut: {DEFAULT_FREQ_START/1e6}MHz)")
    scanner.add_argument("--freq-end",           type=float, default=DEFAULT_FREQ_END,
                        help=f"Fréquence de fin en Hz (défaut: {DEFAULT_FREQ_END/1e6}MHz)")
    scanner.add_argument("--freq-step",          type=float, default=DEFAULT_FREQ_STEP,
                        help=f"Pas en Hz (défaut: {DEFAULT_FREQ_STEP/1e6}MHz)")
    scanner.add_argument("--sample-rate",        type=float, default=DEFAULT_SAMPLE_RATE,
                        help=f"Sample rate en Hz (défaut: {DEFAULT_SAMPLE_RATE/1e6}MHz)")

    # ── Gain et correction ────────────────────
    gain = parser.add_argument_group("Gain et correction")
    gain.add_argument("--gain",               type=float, default=DEFAULT_GAIN,
                     help="Gain en dB. Utiliser 0 pour le mode AGC automatique.")
    gain.add_argument("--ppm",                type=int,   default=DEFAULT_PPM,
                     help="Correction de fréquence en ppm.")

    # ── Détection et enregistrement ───────────
    detection = parser.add_argument_group("Détection et enregistrement")
    detection.add_argument("--snr-threshold",      type=float, default=DEFAULT_SNR_THRESHOLD,
                          help=f"Seuil SNR en dB (défaut: {DEFAULT_SNR_THRESHOLD}dB)")
    detection.add_argument("--record-secs",        type=float, default=DEFAULT_RECORD_SECS,
                          help=f"Durée d'enregistrement en secondes (défaut: {DEFAULT_RECORD_SECS}s)")
    detection.add_argument("--min-separation-hz",  type=float, default=DEFAULT_DUPLICATE_HZ,
                          help=f"Seuil anti-duplication en Hz (défaut: {DEFAULT_DUPLICATE_HZ}Hz)")
    detection.add_argument("--detector",           type=str,   default=DEFAULT_DETECTOR,
                          choices=["threshold", "cfar"],
                          help=f"Méthode de détection des signaux : 'threshold' (seuil fixe "
                               f"vs plancher de bruit global) ou 'cfar' (SO-CFAR, seuil adaptatif "
                               f"local, meilleur pour signaux proches / bande non homogène) "
                               f"(défaut: {DEFAULT_DETECTOR})")
    detection.add_argument("--save-hop-iq",         action="store_true", default=False,
                          help="Sauvegarde le buffer IQ brut de chaque hop (hop_XXXX_*.iq.wav) "
                               "en plus des signaux extraits, pour pouvoir rejouer ce run plus "
                               "tard avec --mode replay --replay-dir <run_folder>. "
                               "Attention : fichiers volumineux (≈ sample_rate × record_secs × 4 "
                               "octets par hop).")

    # ── FFT et traitement ─────────────────────
    processing = parser.add_argument_group("Traitement du signal")
    processing.add_argument("--n-avg",              type=int,   default=DEFAULT_N_AVG,
                           help=f"Nombre de frames FFT à moyenner (défaut: {DEFAULT_N_AVG})")
    processing.add_argument("--offset-tuning",      action="store_true", default=False,
                           help="Active l'offset tuning pour déplacer le DC spike hors bande.")

    # ── Sortie ───────────────────────────────
    output = parser.add_argument_group("Sortie")
    output.add_argument("--output-dir",         type=str,   default=DEFAULT_OUTPUT_DIR,
                       help=f"Dossier de sortie (défaut: {DEFAULT_OUTPUT_DIR})")

    return parser