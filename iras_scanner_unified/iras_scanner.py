#!/usr/bin/env python3
"""
IRAS Scanner v2 - Unified Edition

Scanner de signaux RF automatique avec support USB et RTL-TCP.

Modes de connexion :
  [USB]     - RTL-SDR branché directement sur le PC (connexion locale)
  [RTL-TCP] - RTL-SDR accédé via un serveur rtl_tcp (réseau)
  [REPLAY]  - Rejoue des hops IQ déjà enregistrés (--save-hop-iq), sans SDR

Améliorations v2 :
  [1] Moyennage FFT    - N frames avec 50 % d'overlap (gain SNR ≈ √N)
  [2] Threading        - producer/consumer via queue pour acquisition continue
  [3] Pleine bande     - step 2 MHz 
  [4] DC spike         - masquage des bins centraux
  [5] Détection NumPy  - scipy.ndimage.label remplace la boucle Python pure
  [6] Bruit floor      - 10e percentile au lieu de la médiane
  [7] Correction PPM   - compensation de la dérive du cristal oscillateur(pour le fun)
  [8] Stabilisation    - lecture+rejet du premier bloc après retune
  [9] Fenêtre Blackman - apodisation pour réduire les lobes secondaires de la FFT
  [10] Gain adaptatif  - avertissement saturation + mode AGC(pour le fun)
  [11] Graphiques      - plots de diagnostic et spectrogrammes
"""

from cli import select_connection_mode, build_argument_parser
from scanner import scan


def main():
    """Point d'entrée principal."""
    # Parsing des arguments D'ABORD : on a besoin de savoir si --mode a
    # été fourni avant de décider s'il faut afficher le menu interactif.
    parser = build_argument_parser()
    args = parser.parse_args()

    # Si --mode est explicite (ex: lancement Docker/script automatisé),
    # on saute select_connection_mode()
    if args.iq_file:
        # --iq-file bypass le menu et --mode : import auto + rejeu direct.
        mode = "replay"
    elif args.mode is not None:
        mode = args.mode
    else:
        mode = select_connection_mode()

    if mode == "replay" and not args.replay_dir and not args.iq_file:
        parser.error("--mode replay nécessite --replay-dir <dossier> ou --iq-file <fichier(s)>")

    # Lancer le scan
    scan(args, mode)


if __name__ == "__main__":
    main()