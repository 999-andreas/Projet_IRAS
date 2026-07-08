"""
Configuration et constantes par défaut pour IRAS Scanner.
"""

# ─────────────────────────────────────────────
#  FRÉQUENCES ET DÉBIT
# ─────────────────────────────────────────────

DEFAULT_FREQ_START      = 88e6      # 17.5 MHz
DEFAULT_FREQ_END        = 108e6     # Hz
DEFAULT_FREQ_STEP       = 2_000_000 # Hz
DEFAULT_SAMPLE_RATE     = 1_400_000 # Hz peut être mettre  1.2GHz c'est suffisant 


# ─────────────────────────────────────────────
#  PARAMÈTRES DE GAIN ET CORRECTION
# ─────────────────────────────────────────────

DEFAULT_GAIN            = 20        # dB (0 = AGC automatique)
DEFAULT_PPM             = 0         # ppm - correction cristal oscillateur


# ─────────────────────────────────────────────
#  DÉTECTION ET ENREGISTREMENT
# ─────────────────────────────────────────────

DEFAULT_SNR_THRESHOLD   = 20        # dB au dessus du plancher de bruit
DEFAULT_RECORD_SECS     = 3         # secondes d'enregistrement IQ
DEFAULT_OUTPUT_DIR      = "./dataset"


# ─────────────────────────────────────────────
#  FFT ET TRAITEMENT DU SIGNAL
# ─────────────────────────────────────────────

DEFAULT_FFT_SIZE        = 2048      # taille de la FFT (doit être une puissance de 2)
DEFAULT_N_AVG           = 64        # nombre de frames à moyenner 
DEFAULT_DUPLICATE_HZ    = 200_000    # Hz - seuil anti-duplication
DEFAULT_DC_WIDTH_BINS   = None      # None = auto (1 % des bins
DEFAULT_BW_THRESHOLD_DB = 5        # Seuil de détection de la bande passante
SETTLE_SKIP_SAMPLES     = 0         # samples jetés après retune (utile si avait pas des sleep dans le code)
SATURATION_THRESHOLD_DB = -3.0      # niveau max avant avertissement (purement informatif)


# ─────────────────────────────────────────────
#  RTL-TCP (réseau)
# ─────────────────────────────────────────────

DEFAULT_RTL_TCP_HOST    = "192.168.1.12"
DEFAULT_RTL_TCP_PORT    = 1234
DEFAULT_TCP_TIMEOUT     = 10.0      # secondes
