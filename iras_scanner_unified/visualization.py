"""
Visualisation et plots de diagnostic.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirfilter, lfilter
from scipy.signal import welch
from rf_signal import compute_robust_bandwidth
from config import DEFAULT_BW_THRESHOLD_DB


def save_debug_plot(iq, freq_hz, sample_rate, output_dir, base_name, snr_threshold):
    """
    Génère un graphique de diagnostic multi-panels.
    Prend en entrée un signal IQ DÉJÀ centré et filtré (zoomé numérique effectué en amont).
    """
    plt.figure(figsize=(15, 12))
    plt.suptitle(f"Analyse du Signal : {freq_hz/1e6:.3f} MHz", fontsize=16)

    # --- PANNEAU 1 : Spectrum (PSD) ---
    plt.subplot(3, 2, 1)
    fft_size = 2048
    window = np.blackman(fft_size)
    
    # Calcul de la PSD sur le flux déjà isolé et adapté
    f, psd = welch(iq, fs=sample_rate/1e6, window=window, nperseg=fft_size,
                   scaling='density', detrend=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + 1e-20)

    # Masquage du DC Spike par interpolation linéaire locale en dB
    center_bin = fft_size // 2
    dc_width = 4  # Largeur de la zone à gommer
    if center_bin - dc_width > 0 and center_bin + dc_width < fft_size:
        val_left = psd_db[center_bin - dc_width]
        val_right = psd_db[center_bin + dc_width]
        psd_db[center_bin - dc_width : center_bin + dc_width] = np.linspace(val_left, val_right, dc_width * 2)

    # Recherche dynamique du pic sur le spectre isolé
    peak_bin = int(np.argmax(psd_db))
    peak_freq_mhz = f[peak_bin]

    # Calcul de la bande passante occupée réelle sur ce spectre zoomé
    bw_hz_global, flo_clean, fhi_clean = compute_robust_bandwidth(
        psd=psd_db, 
        peak_bin=peak_bin, 
        sample_rate=sample_rate, 
        fft_size=fft_size,
        threshold_db=DEFAULT_BW_THRESHOLD_DB, 
        reference_is_floor=True  # Mesure de la cloche complète depuis le bruit
    )
    
    bw_lo_mhz = flo_clean / 1e6
    bw_hi_mhz = fhi_clean / 1e6

    # Tracé du spectre
    plt.plot(f, psd_db, color='cyan', linewidth=0.8)
    plt.fill_between(f, psd_db, -140, color='cyan', alpha=0.2)

    # Lignes de repère
    floor = np.percentile(psd_db, 10)
    plt.axhline(floor, color='yellow', linestyle=':', label='Noise floor')
    plt.axhline(floor + snr_threshold, color='red', linestyle='--', label='Seuil Détection')
    plt.axvline(peak_freq_mhz, color='red', linestyle='-', linewidth=0.5, label='Pic Signal')

    # Dessin du rectangle de la bande passante
    if bw_hz_global > 0:
        plt.axvspan(bw_lo_mhz, bw_hi_mhz, color='yellow', alpha=0.12,
                    label=f'BW estimée (-{DEFAULT_BW_THRESHOLD_DB} dB)')

    plt.title("Spectre de Puissance (PSD)")
    plt.ylabel("Puissance (dBFS)")
    plt.xlabel("Offset Fréquence (MHz)")
    plt.ylim(floor - 5, max(psd_db) + 5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Affichage du texte de debug synchronisé
    plt.text(
        0.02, 0.98,
        f"Floor: {floor:.1f} dBFS\nBW estimée: {bw_hz_global*1e-3:.0f} kHz",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # --- PANNEAU 2 : Waterfall (Spectrogramme) ---
    plt.subplot(3, 2, 2)
    plt.specgram(iq, NFFT=1024, Fs=sample_rate/1e3, noverlap=512, cmap='magma')
    plt.title("Waterfall (Spectrogramme)")
    plt.ylabel("kHz")

    # --- PANNEAU 3 : Constellation (I/Q) ---
    plt.subplot(3, 2, 3)
    sub_iq = iq[:50000]
    plt.hexbin(sub_iq.real, sub_iq.imag, gridsize=50, cmap='magma', mincnt=1)
    plt.axhline(0, color='white', lw=0.5)
    plt.axvline(0, color='white', lw=0.5)
    plt.title("Constellation (Densité)")
    plt.gca().set_facecolor('black')
    plt.axis('equal')

    # --- PANNEAU 4 : Histogramme FSK ---
    plt.subplot(3, 2, 4)
    # Calcule la vraie déviation instantanée convertie directement en Hertz
    freq_inst = np.diff(np.unwrap(np.angle(iq))) * (sample_rate / (2 * np.pi))
    plt.hist(freq_inst, bins=128, color='orange', alpha=0.8)
    plt.title("Histogramme de Déviation (Hz)")
    plt.grid(True, alpha=0.2)

    # --- PANNEAU 5 : Enveloppe Temporelle ---
    plt.subplot(3, 2, (5, 6))
    plt.plot(np.abs(iq[:3000]), color='white', linewidth=0.5)
    plt.title("Enveloppe d'Amplitude (Domaine temporel)")
    plt.gca().set_facecolor('black')

    # Sauvegarde et fermeture
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, base_name + ".png")
    plt.savefig(plot_path, facecolor='#2e2e2e')
    plt.close()

    return plot_path