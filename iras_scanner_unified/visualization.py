"""
Visualisation et plots de diagnostic.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def make_daily_output_dir(root_dir):
    """Retourne le sous-dossier journalier de sortie et le crée."""
    date_dir = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(root_dir, date_dir)
    os.makedirs(path, exist_ok=True)
    return path


def save_debug_plot(iq, freq_hz, sample_rate, output_dir, base_name, snr_threshold):
    """
    Génère un graphique de diagnostic multi-panels avec :
    - Spectrum (PSD)
    - Waterfall (spectrogramme)
    - Constellation (I/Q)
    - Histogramme FSK
    - Enveloppe temporelle
    """
    plt.figure(figsize=(15, 12))
    plt.suptitle(f"Analyse du Signal : {freq_hz/1e6:.3f} MHz", fontsize=16)

   # --- 1. Spectrum (PSD) ---
    plt.subplot(3, 2, 1)
    fft_size = 2048
    window = np.blackman(fft_size)
    from scipy.signal import welch
    from signal import compute_robust_bandwidth
    from config import DEFAULT_BW_THRESHOLD_DB

    # Calcul de la PSD (en échelle linéaire d'abord pour welch)
    f, psd = welch(iq, fs=sample_rate/1e6, window=window, nperseg=fft_size,
                   scaling='density', detrend=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-20)

    # 1. Verrouillage de la recherche au centre du spectre (±5% des bins de la FFT)
    center_bin = fft_size // 2
    search_width = max(5, int(fft_size * 0.05)) 
    zone_centrale = psd_db[center_bin - search_width : center_bin + search_width]
    
    # Index absolu du pic utile dans le tableau de la FFT
    peak_bin = int(np.argmax(zone_centrale) + (center_bin - search_width))
    peak_freq_mhz = f[peak_bin]  # Fréquence exacte du pic pour le tracé

    # 2. Détermination de la BW via le moteur robuste unifié (seuil relatif au pic)
    bw_hz, freq_lo_rel, freq_hi_rel = compute_robust_bandwidth(
        psd=psd_db,
        peak_bin=peak_bin,
        sample_rate=sample_rate,
        fft_size=fft_size,
        threshold_db=DEFAULT_BW_THRESHOLD_DB,
        reference_is_floor=True
    )
    
    # Conversion des fréquences relatives renvoyées par le moteur en MHz (offset du graphique)
    bw_lo_mhz = freq_lo_rel / 1e6
    bw_hi_mhz = freq_hi_rel / 1e6
    bw_mhz = bw_hz / 1e6

    # Tracé du spectre
    plt.plot(f, psd_db, color='cyan', linewidth=0.8)
    plt.fill_between(f, psd_db, -140, color='cyan', alpha=0.2)

    # Lignes de repère horizontales (Plancher et Seuil de détection global)
    floor = np.percentile(psd_db, 10)
    plt.axhline(floor, color='yellow', linestyle=':', label='Noise floor')
    plt.axhline(floor + snr_threshold, color='red', linestyle='--', label='Seuil Détection')

    # 3. Trait vertical rouge indiquant le pic du signal ciblé
    plt.axvline(peak_freq_mhz, color='red', linestyle='-', linewidth=0.5, label='Pic Signal')

    # 4. Surlignage de la zone de bande passante estimée (Look-ahead AM pris en compte)
    plt.axvspan(bw_lo_mhz, bw_hi_mhz, color='yellow', alpha=0.12,
                label=f'BW estimée (-{DEFAULT_BW_THRESHOLD_DB} dB)')

    # Configuration des axes et légendes
    plt.title("Spectre de Puissance (PSD)")
    plt.ylabel("Puissance (dBFS)")
    plt.xlabel("Offset Fréquence (MHz)")
    plt.ylim(floor - 10, max(psd_db) + 10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Affichage du texte informatif
    plt.text(
        0.02, 0.98,
        f"Floor: {floor:.1f} dBFS\nBW estimée: {bw_mhz*1e3:.0f} kHz",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # --- 2. Waterfall (Spectrogramme) ---
    plt.subplot(3, 2, 2)
    plt.specgram(iq, NFFT=1024, Fs=sample_rate/1e3, noverlap=512, cmap='magma')
    plt.title("Waterfall (Spectrogramme)")
    plt.ylabel("kHz")

    # --- 3. Constellation (I/Q) ---
    plt.subplot(3, 2, 3)
    sub_iq = iq[:10000]
    plt.scatter(sub_iq.real, sub_iq.imag, s=0.5, alpha=0.3, color='lime')
    plt.axhline(0, color='white', lw=0.5)
    plt.axvline(0, color='white', lw=0.5)
    plt.title("Constellation")
    plt.gca().set_facecolor('black')
    plt.axis('equal')

    # --- 4. Histogramme FSK ---
    plt.subplot(3, 2, 4)
    freq_inst = np.diff(np.unwrap(np.angle(iq)))
    plt.hist(freq_inst, bins=128, color='orange', alpha=0.8)
    plt.title("Histogramme de Déviation (FSK Discriminator)")

    # --- 5. Enveloppe Temporelle ---
    plt.subplot(3, 2, (5, 6))
    plt.plot(np.abs(iq[:3000]), color='white', linewidth=0.5)
    plt.title("Enveloppe d'Amplitude (Domaine temporel)")
    plt.gca().set_facecolor('black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Use run-specific folder created by scanner
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, base_name + ".png")
    plt.savefig(plot_path, facecolor='#2e2e2e')
    plt.close()

    return plot_path
