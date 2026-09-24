"""
Configuration et constantes par défaut pour IRAS Scanner.
"""

# ─────────────────────────────────────────────
#  FRÉQUENCES ET DÉBIT
# ─────────────────────────────────────────────

DEFAULT_FREQ_START      = 9.4e6       # 17.5 MHz
DEFAULT_FREQ_END        = 10.0e6      # Hz
DEFAULT_FREQ_STEP       = 200_000 # Hz
DEFAULT_SAMPLE_RATE     = 1_200_000 # Hz peut être mettre  1.2GHz c'est suffisant 


# ─────────────────────────────────────────────
#  PARAMÈTRES DE GAIN ET CORRECTION
# ─────────────────────────────────────────────

DEFAULT_GAIN            = 12       # dB (0 = AGC automatique)
DEFAULT_PPM             = 0         # ppm - correction cristal oscillateur


# ─────────────────────────────────────────────
#  DÉTECTION ET ENREGISTREMENT
# ─────────────────────────────────────────────

DEFAULT_SNR_THRESHOLD   = 10        # dB au dessus du plancher de bruit
DEFAULT_RECORD_SECS     = 3         # secondes d'enregistrement IQ
DEFAULT_OUTPUT_DIR      = "./dataset"


# ─────────────────────────────────────────────
#  FFT ET TRAITEMENT DU SIGNAL
# ─────────────────────────────────────────────

DEFAULT_FFT_SIZE        = 2048      # taille de la FFT (doit être une puissance de 2)
DEFAULT_N_AVG           = 64        # nombre de frames à moyenner
DEFAULT_DUPLICATE_HZ    = 15_000    # Hz - seuil anti-duplication
DEFAULT_DC_WIDTH_BINS   = None      # None = auto (1 % des bins
DEFAULT_BW_THRESHOLD_DB = 10        # Seuil unique et partagé pour la mesure de BW)
SETTLE_SKIP_SAMPLES     = 15000        # samples jetés après retune (pour laisser le temps au signal de se stabiliser)
SATURATION_THRESHOLD_DB = -3.0      # niveau max avant avertissement

# ─────────────────────────────────────────────
#  DÉTECTION CFAR (SO-CFAR : Smallest-Of)
# ─────────────────────────────────────────────

DEFAULT_DETECTOR          = "cfar"  # "threshold" (méthode historique) ou "cfar"
DEFAULT_CFAR_GUARD_CELLS  = 25    # cellules de garde de chaque côté du bin testé (plus le signal est large, plus il faut de cellules de garde)
DEFAULT_CFAR_TRAIN_CELLS  = 20    # cellules d'entraînement de chaque côté (fenêtre de bruit)
DEFAULT_CFAR_PFA          = 1e-3  # probabilité de fausse alarme cible (plus petit = plus strict) 

# ─────────────────────────────────────────────
#  RTL-TCP (réseau)
# ─────────────────────────────────────────────

DEFAULT_RTL_TCP_HOST    = "192.168.1.89"
DEFAULT_RTL_TCP_PORT    = 1234
DEFAULT_TCP_TIMEOUT     = 10.0      # secondes
