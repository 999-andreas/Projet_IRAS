"""
Boucle principale de scan.
"""

import os
import time
import numpy as np
from datetime import datetime
from config import (
    DEFAULT_FFT_SIZE, DEFAULT_N_AVG, SETTLE_SKIP_SAMPLES,
    DEFAULT_CFAR_GUARD_CELLS, DEFAULT_CFAR_TRAIN_CELLS, DEFAULT_CFAR_PFA,
)
from sdr import UsbSdrClient, RtlTcpClient
from rf_signal import (
    compute_averaged_psd, mask_dc_spike, noise_floor, peak_power,
    check_saturation, is_duplicate_signal, dedupe_signals_intra_hop,
    detect_all_signals, detect_all_signals_cfar, extract_and_center_signal
)
from recording import save_sample
from visualization import save_debug_plot, save_hop_plot


def scan(args, mode):
    """
    Boucle principale de scan :
    - Crée le client SDR (USB ou RTL-TCP)
    - Configure le hardware
    - Balaye les fréquences
    - Détecte, enregistre et trace les signaux
    """

    # ── Connexion SDR ──────────────────────────
    if mode == "usb":
        print(f"\n  Connexion au RTL-SDR USB…")
        sdr = UsbSdrClient()
    else:  # rtltcp
        print(f"\n  Connexion au serveur rtl_tcp → {args.rtl_tcp_host}:{args.rtl_tcp_port} …")
        sdr = RtlTcpClient(
            host    = args.rtl_tcp_host,
            port    = args.rtl_tcp_port,
            timeout = args.rtl_tcp_timeout,
        )

    sdr.sample_rate = args.sample_rate
    sdr.gain        = args.gain

    if args.ppm != 0:
        sdr.set_freq_correction(args.ppm)

    if args.offset_tuning:
        try:
            sdr.set_offset_tuning(True)
            print("  Offset tuning activé (DC spike déplacé hors bande).")
        except Exception as e:
            print(f"  ⚠  Offset tuning refusé : {e}")

    # ── Préparation scan ──────────────────────
    # ── Avertissement si le pas crée trop de chevauchement ──
    overlap_ratio = 1.0 - (args.freq_step / args.sample_rate)
    if overlap_ratio > 0.5:
        print(f"  ⚠  Pas de scan ({args.freq_step/1e6:.2f} MHz) < 50 % de la BW "
              f"({args.sample_rate/1e6:.1f} MHz) → chevauchement {overlap_ratio*100:.0f} %.")
        print(f"     Conseil : utilisez --freq-step {args.sample_rate*0.8/1e6:.1f}e6 "
              f"pour éviter les doublons. La déduplication s'en chargera quand même.")

    freqs          = np.arange(args.freq_start, args.freq_end, args.freq_step)
    total          = len(freqs)
    detected       = 0
    detected_freqs = []

    # Taille du buffer de DÉTECTION (n_avg trames FFT, comme avant).
    snap_size = DEFAULT_FFT_SIZE * (args.n_avg + 1)

    # Taille du buffer COMPLET du hop, capturé UNE SEULE FOIS et utilisé
    # à la fois pour la détection (ses n_avg premières trames, via
    # snap_size ci-dessus) ET pour l'extraction de chaque signal détecté
    # (extract_and_center_signal, plus bas). Fini le second enregistrement
    # séparé par signal avec retune matériel : plus aucun retune, plus
    # aucun changement de sample_rate pendant le hop, donc plus de
    # transitoire de PLL ni de buffer TCP à repurger à ce moment-là  
    # le SDR reste simplement fixe sur (freq, args.sample_rate) pendant toute
    # la durée du hop. Comportement strictement identique en USB et
    # RTL-TCP
    hop_capture_samples = max(snap_size, int(args.sample_rate * args.record_secs))

    mode_str = "USB" if mode == "usb" else "RTL-TCP"
    print(f"\n{'─'*65}")
    print(f"  IRAS Scanner v2 — Unified Edition ({mode_str})")
    if mode == "rtltcp":
        print(f"  Serveur     : {args.rtl_tcp_host}:{args.rtl_tcp_port}")
    print(f"  Plage       : {args.freq_start/1e6:.1f} – {args.freq_end/1e6:.1f} MHz")
    print(f"  Pas         : {args.freq_step/1e6:.2f} MHz  (BW captée : {args.sample_rate/1e6:.1f} MHz)")
    print(f"  Hops        : {total}")
    print(f"  Détecteur   : {'SO-CFAR' if args.detector == 'cfar' else 'Seuil fixe (threshold)'}")
    if args.detector == "cfar":
        print(f"  Seuil SNR   : (ignoré en mode CFAR — décision purement locale, voir config.py)")
    else:
        print(f"  Seuil SNR   : +{args.snr_threshold} dB")
    print(f"  Moyennage   : {args.n_avg} frames FFT (overlap 50 %)")
    print(f"  Capture/hop : {hop_capture_samples/args.sample_rate:.2f} s "
          f"({hop_capture_samples} échantillons, détection + extraction "
          f"signaux sur le même buffer — aucun retune pendant le hop)")
    print(f"  PPM corr.   : {args.ppm:+d} ppm")
    print(f"  Gain        : {args.gain} dB{' (AGC auto)' if args.gain == 0 else ''}")
    print(f"  Output root : {args.output_dir}")
    # Créer un dossier unique pour cette exécution du script
    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Run folder  : {run_dir}")
    print(f"{'─'*65}\n")

    # code synchrone...
    try:
        for i, freq in enumerate(freqs):
            sdr.center_freq = freq

            # Stabilisation & Purge du transitoire de retune (crucial pour RTL-TCP !)
            _ = sdr.read_samples(max(256, (SETTLE_SKIP_SAMPLES // 256) * 256))
            time.sleep(0.02)

            # Capture UNIQUE et COMPLÈTE du hop : sert à la fois à la
            # détection (ses premières trames) et à l'extraction de
            # chaque signal détecté plus bas
            try:
                hop_iq = sdr.read_samples(hop_capture_samples)
            except Exception as e:
                print(f"[{i+1:>4}/{total}] {freq/1e6:>9.3f} MHz  ← ERREUR acquisition ({e}), hop ignoré")
                continue

            # ── Traitement du signal (détection sur le DÉBUT du buffer) ──
            snap  = hop_iq[:snap_size]
            psd   = compute_averaged_psd(snap, DEFAULT_FFT_SIZE, args.n_avg)
            floor = noise_floor(psd)
            #psd   = mask_dc_spike(psd, DEFAULT_FFT_SIZE)
            

            # ── Avertissement saturation ──
            sat_warn = "  ⚠ SATURATION" if check_saturation(psd) else ""

            # ── Détection multi-signaux (threshold classique ou SO-CFAR) ──
            if args.detector == "cfar":
                signals = detect_all_signals_cfar(
                    psd, freq, args.sample_rate,
                    fft_size=DEFAULT_FFT_SIZE
                )
            else:
                signals = detect_all_signals(
                    psd, freq, args.sample_rate,
                    DEFAULT_FFT_SIZE, args.snr_threshold
                )

            best_snr = signals[0]['snr_db'] if signals else (peak_power(psd) - floor)
            print(f"[{i+1:>4}/{total}] {freq/1e6:>9.3f} MHz  |  "
                  f"floor={floor:>+6.1f} dBFS  "
                  f"SNR={best_snr:>+5.1f} dB  "
                  f"signaux={len(signals)}{sat_warn}")

            #  Déduplication INTRA-HOP : un même signal large (porteuse FM,
            #    etc.) peut ressortir en plusieurs pics/régions proches
            n_before_dedupe = len(signals)
            signals = dedupe_signals_intra_hop(signals, args.min_separation_hz)
            if len(signals) < n_before_dedupe:
                print(f"           ↳ {n_before_dedupe - len(signals)} doublon(s) intra-hop fusionné(s) "
                      f"({n_before_dedupe} → {len(signals)} signaux distincts)")

            #  Plot du hop : montre exactement comment la détection a décidé
            #    sur CE spectre (même seuil que celui réellement appliqué)
            hop_plot_path = save_hop_plot(
                psd, freq, args.sample_rate, signals, run_dir, i + 1,
                fft_size=DEFAULT_FFT_SIZE,
                detector=args.detector,
                snr_threshold=args.snr_threshold,
                cfar_guard_cells=DEFAULT_CFAR_GUARD_CELLS,
                cfar_train_cells=DEFAULT_CFAR_TRAIN_CELLS,
                cfar_pfa=DEFAULT_CFAR_PFA,
            )

            for sig in signals:
                exact_freq = sig['freq_hz']
                bw_hz      = sig['bw_hz']
                snr_val    = sig['snr_db']

                # Déduplication sur la fréquence exacte
                if is_duplicate_signal(exact_freq, detected_freqs, args.min_separation_hz):
                    print(f"           ↳ DOUBLON {exact_freq/1e6:.4f} MHz — ignoré")
                    continue

                # Largeur du filtre d'extraction : marge selon que le
                # signal est large (FM) ou étroit (AM/PMR), inchangé
                if bw_hz > 100_000:
                    decim_bw = max(bw_hz * 1.2, 150_000)
                else:
                    decim_bw = max(bw_hz * 1.3, 12_000)

                print(f"           ↳ EXTRACTION {exact_freq/1e6:.4f} MHz  "
                      f"SNR={snr_val:>+5.1f} dB  BW≈{bw_hz/1e3:.0f} kHz ...",
                      end="", flush=True)

                # Extraction 100% numérique depuis le buffer déjà capturé :
                # mixage complexe pour recentrer exact_freq sur 0 Hz, puis
                # filtre passe-bas + décimation vers la bande utile.
                # AUCUN retune, AUCUN changement de sample_rate ici
                iq_clean, sample_rate_clean = extract_and_center_signal(
                    hop_iq, freq, exact_freq, args.sample_rate, decim_bw
                )

                path      = save_sample(iq_clean, exact_freq, snr_val, sample_rate_clean, run_dir, bw_hz)
                base_name = os.path.basename(path).replace(".wav", "")
                path_png  = save_debug_plot(iq_clean, exact_freq, sample_rate_clean, run_dir,
                                            base_name, args.snr_threshold)

                detected += 1
                detected_freqs.append(exact_freq)
                print(f"\n           → Audio: {path}")
                print(f"           → Debug: {path_png}")

    except KeyboardInterrupt:
        print("\n\nScan interrompu.")

    finally:
        sdr.close()

    # ── Résumé ──
    print(f"\n{'─'*65}")
    print(f"  Scan terminé : {detected} signal(s) enregistré(s) sur {total} fréquences")
    print(f"  Dataset : {args.output_dir}")
    print(f"{'─'*65}\n")