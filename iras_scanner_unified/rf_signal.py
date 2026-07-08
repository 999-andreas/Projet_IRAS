"""
Traitement du signal : FFT, PSD, détection de pics, SNR, etc.
"""

import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.signal import iirfilter, lfilter
from config import (
    DEFAULT_FFT_SIZE, DEFAULT_N_AVG, DEFAULT_DC_WIDTH_BINS, DEFAULT_SNR_THRESHOLD,
    SATURATION_THRESHOLD_DB, DEFAULT_BW_THRESHOLD_DB
)

'''pas utilisé finalement, potienellement utile pour le numérique
def estimate_accurate_bw_over_time(iq, sample_rate, fft_size=512, overlap=256):
    """
    Calcule la bande passante en suivant son évolution bloc par bloc dans le temps,
    puis extrait une valeur moyenne robuste en éliminant les silences.
    """
    step = fft_size - overlap
    window = np.blackman(fft_size)
    
    instantaneous_bws = []
    center_bin = fft_size // 2
    search_width = max(5, int(fft_size * 0.05)) # ±5% autour du centre
    
    # Parcours de l'enregistrement dans le temps
    for start in range(0, len(iq) - fft_size, step):
        frame = iq[start : start + fft_size]
        
        # FFT du bloc temporel courant
        fft_out = np.fft.fftshift(np.fft.fft(frame * window, fft_size))
        psd_block = 10 * np.log10(np.abs(fft_out) ** 2 + 1e-20)
        
        # On cherche le pic central de CE bloc précisément
        zone_centrale = psd_block[center_bin - search_width : center_bin + search_width]
        peak_bin = int(np.argmax(zone_centrale) + (center_bin - search_width))
        
        # On mesure la BW de ce bloc via le moteur robuste
        bw_hz, _, _ = compute_robust_bandwidth(
            psd=psd_block,
            peak_bin=peak_bin,
            sample_rate=sample_rate,
            fft_size=fft_size,
            threshold_db=DEFAULT_BW_THRESHOLD_DB,
            reference_is_floor=False
        )
        
        # On n'enregistre la BW que si le bloc contient vraiment du signal
        # (on évite de mesurer le bruit pendant les silences des bursts)
        floor = np.percentile(psd_block, 10)
        if psd_block[peak_bin] > (floor + 8): # Le pic doit dépasser le bruit du bloc
            instantaneous_bws.append(bw_hz)
            
    if not instantaneous_bws:
        return sample_rate * 0.1 # Valeur de secours si tout est vide
        
    # --- Agrégation robuste ---
    # Au lieu d'une moyenne pure qui peut être tirée vers le bas par des blocs de transition,
    # on utilise le 75ème ou 90ème percentile. Ça donne la largeur "maximale" occupée 
    # par le signal lorsqu'il est pleinement actif.
    accurate_bw_hz = np.percentile(instantaneous_bws, 75)
    
    return float(accurate_bw_hz)
'''
    

def isolate_central_signal(iq, current_sr, target_bw):
    """
    Filtre le signal IQ pour ne garder que la bande utile centrale.
    (Conservé ici au cas où d'autres modules y font référence).
    """
    decim_factor = int(current_sr / (target_bw * 1.5))
    if decim_factor <= 1:
        return iq, current_sr
    
    nyquist = current_sr / 2
    cutoff = (target_bw * 1.2) / 2
    b, a = iirfilter(4, cutoff / nyquist, btype='lowpass', ftype='butter')
    
    iq_filtered = lfilter(b, a, iq)
    iq_zoomed = iq_filtered[::decim_factor]
    new_sr = current_sr / decim_factor
    
    return iq_zoomed, new_sr
# ─────────────────────────────────────────────
#  [1] FFT MOYENNÉE (Welch simplifié, 50 % overlap); 
# ─────────────────────────────────────────────

def compute_averaged_psd(samples, fft_size=DEFAULT_FFT_SIZE, n_avg=DEFAULT_N_AVG):
    """
    Densité spectrale de puissance moyennée sur N frames avec 50 % d'overlap.
    """
    window   = np.blackman(fft_size)
    norm     = np.sum(window ** 2)
    total    = np.zeros(fft_size, dtype=np.float64)
    step     = fft_size // 2
    count    = 0

    for start in range(0, len(samples) - fft_size, step):
        frame   = samples[start : start + fft_size]
        fft_out = np.fft.fftshift(np.fft.fft(frame * window, fft_size))
        total  += np.abs(fft_out) ** 2
        count  += 1
        if count >= n_avg:
            break

    if count == 0:
        return np.full(fft_size, -120.0)

    psd = 10 * np.log10(total / (count * norm) + 1e-20)
    return psd


# ─────────────────────────────────────────────
#  [4] MASQUAGE DU DC SPIKE; le dc spike est un pic artificiel à 0 Hz (centre du spectre) causé par le convertisseur A/N.
# ─────────────────────────────────────────────

def mask_dc_spike(psd, fft_size=DEFAULT_FFT_SIZE, dc_width_bins=DEFAULT_DC_WIDTH_BINS):
    """
    Remplace les bins du DC spike par le minimum du spectre.
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
#  [6] PLANCHER DE BRUIT (10e percentile)
# ─────────────────────────────────────────────

def noise_floor(psd):
    """
    Estime le plancher de bruit avec le 10e percentile.
    Plus robuste que la médiane en présence de nombreux signaux.
    """
    return float(np.percentile(psd, 10))


# ─────────────────────────────────────────────
#  [5] DÉTECTION DE PIC (NumPy + scipy.ndimage)
# ─────────────────────────────────────────────

def peak_power(psd, min_bins=10, above_db=6):
    """
    Détecte le pic le plus puissant sur au moins min_bins bins consécutifs.
    Utilise scipy.ndimage.label pour étiqueter les régions connexes.
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
#sert un peu à rien pour l'instant, c'est juste une info de debug
def check_saturation(psd, threshold_db=SATURATION_THRESHOLD_DB):
    """
    Retourne True si le maximum du spectre dépasse le seuil.
    Au dessus de ~-3 dBFS le convertisseur RTL-SDR écrête.
    """
    return float(np.max(psd)) >= threshold_db


# ─────────────────────────────────────────────
#  HEURISTIQUES DE CLASSIFICATION
# ─────────────────────────────────────────────

def estimate_bandwidth(samples, sample_rate, fft_size=DEFAULT_FFT_SIZE,
                        n_avg=DEFAULT_N_AVG, threshold_db=DEFAULT_BW_THRESHOLD_DB):
    """Estime la largeur de bande globale occupée en Hz."""
    psd = compute_averaged_psd(samples, fft_size, n_avg)
    psd = mask_dc_spike(psd, fft_size)
    
    peak_bin = int(np.argmax(psd))
    
    bw_hz, _, _ = compute_robust_bandwidth(
        psd, peak_bin, sample_rate, fft_size, 
        threshold_db=threshold_db, reference_is_floor=True
    )
    return bw_hz

'''
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
'''

def is_duplicate_signal(freq_hz, detected_freqs, min_separation_hz):
    """Retourne True si un signal trop proche a déjà été enregistré."""
    return any(abs(freq_hz - d) <= min_separation_hz for d in detected_freqs)


# ─────────────────────────────────────────────
#  LOCALISATION PRÉCISE DU SIGNAL
# ─────────────────────────────────────────────

def peak_freq_hz(psd, center_freq_hz, sample_rate, fft_size=DEFAULT_FFT_SIZE):
    """
    Retourne la fréquence absolue (Hz) du bin le plus puissant dans le PSD.
    Utilise une interpolation parabolique pour sub-bin accuracy.
    """
    # Fréquences correspondant à chaque bin
    freqs = center_freq_hz + np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / sample_rate)
    )

    peak_bin = int(np.argmax(psd))

    # Interpolation parabolique sur les 3 bins autour du pic
    if 0 < peak_bin < fft_size - 1:
        alpha = psd[peak_bin - 1]
        beta  = psd[peak_bin]
        gamma = psd[peak_bin + 1]
        denom = (alpha - 2 * beta + gamma)
        if denom != 0:
            offset_bins = 0.5 * (alpha - gamma) / denom
        else:
            offset_bins = 0.0
        bin_width = sample_rate / fft_size
        peak_hz = freqs[peak_bin] + offset_bins * bin_width
    else:
        peak_hz = freqs[peak_bin]

    return float(peak_hz)

def compute_robust_bandwidth(psd, peak_bin, sample_rate, fft_size=DEFAULT_FFT_SIZE, 
                             threshold_db=DEFAULT_BW_THRESHOLD_DB, reference_is_floor=False):
    """
    Calcule la largeur de bande occupée de manière robuste autour d'un pic donné.
    Gère les creux (vallées d'atténuation AM) grâce à une tolérance proportionnelle à la FFT (1%).
    
    Si reference_is_floor=True  : Le seuil est absolu (Plancher de bruit + threshold_db). -> (Pour détection globale/multi-signaux)
    Si reference_is_floor=False : Le seuil est relatif (Valeur du pic - threshold_db).     -> (Pour occupied_bandwidth -N dB)
    """
    peak_val = psd[peak_bin]
    
    if reference_is_floor:
        cutoff = noise_floor(psd) + threshold_db
        # Si le pic lui-même n'atteint pas le seuil requis
        if peak_val < cutoff:
            return 0.0, 0.0, 0.0
    else:
        cutoff = peak_val - threshold_db

    # Tolérance aux creux : 1% de la taille de la FFT (ex: 20 bins pour une FFT 2048)
    gap_tolerance_bins = max(1, int(fft_size * 0.01))

    # --- Extension vers la GAUCHE ---
    lo = peak_bin
    consecutive_gaps = 0
    last_valid_lo = peak_bin
    
    while lo > 0:
        lo -= 1
        if psd[lo] >= cutoff:
            consecutive_gaps = 0
            last_valid_lo = lo
        else:
            consecutive_gaps += 1
            if consecutive_gaps > gap_tolerance_bins:
                break
    lo = last_valid_lo

    # --- Extension vers la DROITE ---
    hi = peak_bin
    consecutive_gaps = 0
    last_valid_hi = peak_bin
    
    while hi < fft_size - 1:
        hi += 1
        if psd[hi] >= cutoff:
            consecutive_gaps = 0
            last_valid_hi = hi
        else:
            consecutive_gaps += 1
            if consecutive_gaps > gap_tolerance_bins:
                break
    hi = last_valid_hi

    # Conversion des bins en fréquences relatives (-Fs/2 à Fs/2)
    bin_width_hz = sample_rate / fft_size
    freq_lo_rel = (lo - fft_size // 2) * bin_width_hz
    freq_hi_rel = (hi - fft_size // 2) * bin_width_hz
    bw_hz = freq_hi_rel - freq_lo_rel

    return float(bw_hz), float(freq_lo_rel), float(freq_hi_rel)

'''
def occupied_bandwidth(psd, center_freq_hz, sample_rate,
                       fft_size=DEFAULT_FFT_SIZE, threshold_db=10,
                       min_bw_hz=50_000, max_bw_hz=500_000):
    """
    Estime la largeur de bande occupée autour du pic (méthode -N dB).

    Retourne (bw_hz, freq_lo_hz, freq_hi_hz) :
      - bw_hz     : largeur totale en Hz
      - freq_lo   : bord bas en Hz
      - freq_hi   : bord haut en Hz

    On part du bin du pic et on s'étale des deux côtés jusqu'à tomber
    sous (peak - threshold_db), puis on applique des bornes min/max
    pour éviter les enregistrements trop étroits ou trop larges.
    """
    freqs    = center_freq_hz + np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / sample_rate)
    )
    peak_bin = int(np.argmax(psd))
    peak_val = psd[peak_bin]
    cutoff   = peak_val - threshold_db

    # Chercher le bord bas
    lo = peak_bin
    while lo > 0 and psd[lo] >= cutoff:
        lo -= 1

    # Chercher le bord haut
    hi = peak_bin
    while hi < fft_size - 1 and psd[hi] >= cutoff:
        hi += 1

    freq_lo = float(freqs[lo])
    freq_hi = float(freqs[hi])
    bw_hz   = freq_hi - freq_lo

    # Appliquer les bornes min/max
    if bw_hz < min_bw_hz:
        half = min_bw_hz / 2
        mid  = (freq_lo + freq_hi) / 2
        freq_lo, freq_hi = mid - half, mid + half
        bw_hz = min_bw_hz

    if bw_hz > max_bw_hz:
        half = max_bw_hz / 2
        mid  = (freq_lo + freq_hi) / 2
        freq_lo, freq_hi = mid - half, mid + half
        bw_hz = max_bw_hz

    return float(bw_hz), float(freq_lo), float(freq_hi)
'''

# ─────────────────────────────────────────────
#  DÉTECTION MULTI-SIGNAUX PAR HOP
# ─────────────────────────────────────────────

def detect_all_signals(psd, center_freq_hz, sample_rate,
                       fft_size=DEFAULT_FFT_SIZE,
                       snr_threshold_db=DEFAULT_SNR_THRESHOLD,
                       min_bins=5,
                       above_db=6,
                       min_bw_hz=100,
                       max_bw_hz=500_000):
    """
    Détecte TOUS les signaux au-dessus du seuil SNR dans un PSD.

    Retourne une liste de dicts, triée par SNR décroissant :
      [
        {
          'freq_hz'  : fréquence exacte du pic (Hz),
          'snr_db'   : SNR individuel (dB),
          'peak_db'  : puissance du pic (dBFS),
          'floor_db' : plancher de bruit local (dBFS),
          'bw_hz'    : largeur de bande occupée (Hz),
          'freq_lo'  : bord bas (Hz),
          'freq_hi'  : bord haut (Hz),
        },
        ...
      ]
    """
    freqs    = center_freq_hz + np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    floor    = noise_floor(psd)
    above    = psd > (floor + above_db)

    labeled, n_regions = ndimage_label(above)
    signals = []

    for region_id in range(1, n_regions + 1):
        mask = labeled == region_id
        if np.sum(mask) < min_bins:
            continue

        # Puissance et position du pic dans la région
        region_psd  = psd.copy()
        region_psd[~mask] = -999
        peak_bin    = int(np.argmax(region_psd))
        peak_val    = float(psd[peak_bin])
        snr_db      = peak_val - floor

        if snr_db < snr_threshold_db:
            continue

        # Fréquence exacte avec interpolation parabolique
        if 0 < peak_bin < fft_size - 1:
            alpha = psd[peak_bin - 1]
            beta  = psd[peak_bin]
            gamma = psd[peak_bin + 1]
            denom = alpha - 2 * beta + gamma
            offset_bins = 0.5 * (alpha - gamma) / denom if denom != 0 else 0.0
            bin_width   = sample_rate / fft_size
            exact_freq  = float(freqs[peak_bin]) + offset_bins * bin_width
        else:
            exact_freq = float(freqs[peak_bin])

        # logique unifiée (basée sur le plancher de bruit pour le multi-signal)
        bw_hz, freq_lo_rel, freq_hi_rel = compute_robust_bandwidth(
            psd, peak_bin, sample_rate, fft_size, 
            threshold_db=DEFAULT_BW_THRESHOLD_DB, reference_is_floor=True
        )

        freq_lo = center_freq_hz + freq_lo_rel
        freq_hi = center_freq_hz + freq_hi_rel

        if bw_hz < min_bw_hz:
            half    = min_bw_hz / 2
            mid     = (freq_lo + freq_hi) / 2
            freq_lo, freq_hi = mid - half, mid + half
            bw_hz   = min_bw_hz
        if bw_hz > max_bw_hz:
            half    = max_bw_hz / 2
            mid     = (freq_lo + freq_hi) / 2
            freq_lo, freq_hi = mid - half, mid + half
            bw_hz   = max_bw_hz

        signals.append({
            'freq_hz'  : exact_freq,
            'snr_db'   : snr_db,
            'peak_db'  : peak_val,
            'floor_db' : floor,
            'bw_hz'    : bw_hz,
            'freq_lo'  : freq_lo,
            'freq_hi'  : freq_hi,
        })

    # Trier par SNR décroissant
    signals.sort(key=lambda s: s['snr_db'], reverse=True)
    return signals