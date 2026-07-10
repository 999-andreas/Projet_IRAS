"""
Boucle principale de scan.
"""

import os
import time
import queue
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
    detect_all_signals, detect_all_signals_cfar, isolate_central_signal
)
from recording import IQProducer, record_iq, save_sample
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

    snap_size = DEFAULT_FFT_SIZE * (args.n_avg + 1)

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
    print(f"  PPM corr.   : {args.ppm:+d} ppm")
    print(f"  Gain        : {args.gain} dB{' (AGC auto)' if args.gain == 0 else ''}")
    print(f"  Output root : {args.output_dir}")
    # Créer un dossier unique pour cette exécution du script
    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Run folder  : {run_dir}")
    print(f"{'─'*65}\n")

    buf_queue = queue.Queue(maxsize=5)

    try:
        for i, freq in enumerate(freqs):
            sdr.center_freq = freq

            # [8] Stabilisation : rejet du buffer résiduel après retune
            _ = sdr.read_samples(max(256, (SETTLE_SKIP_SAMPLES // 256) * 256))
            time.sleep(0.02)

            # [2] Lancer le thread producteur
            producer = IQProducer(sdr, buf_queue, snap_size)
            producer.start()

            # Collecter les samples depuis la queue
            collected = []
            while sum(len(c) for c in collected) < snap_size:
                try:
                    chunk = buf_queue.get(timeout=2.0)
                    collected.append(chunk)
                except queue.Empty:
                    break

            producer.stop()
            producer.join(timeout=1.0)
            # Vider la queue pour le prochain hop
            while not buf_queue.empty():
                try:
                    buf_queue.get_nowait()
                except queue.Empty:
                    break

            if not collected:
                print(f"[{i+1:>4}/{total}] {freq/1e6:>9.3f} MHz  ← TIMEOUT acquisition, hop ignoré")
                continue

            snap = np.concatenate(collected)[:snap_size]

            # ── Traitement du signal ──
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

                # Si le signal est étroit, on serre le filtre pour chasser les voisins.
                if bw_hz > 100_000:
                    decim_bw = max(bw_hz * 1.2, 150_000)  # Marge de sécurité pour la déviation FM
                else:
                    decim_bw = max(bw_hz * 1.3, 12_000)   # Filtre minime (minimum 12 kHz)

                # Le théorème de Nyquist-Shannon + du rab un ratio d'environ 2.5x.
                target_sr = decim_bw * 2.5
                
                # Le driver RTL-SDR rejette les valeurs entre 300 kHz et 900 kHz.
                # On choisit donc le palier de 256 kHz si ça passe, sinon on saute direct à 900 kHz.
                if target_sr <= 256_000:
                    record_sr = 256_000
                else:
                    record_sr = 900_010
                
                # Sécurité : On ne dépasse jamais le sample rate maximum demandé par l'utilisateur
                record_sr = min(record_sr, args.sample_rate)
                print(f"           ↳ ENREGISTREMENT {exact_freq/1e6:.4f} MHz  "
                      f"SNR={snr_val:>+5.1f} dB  BW≈{bw_hz/1e3:.0f} kHz  "
                      f"SR={record_sr/1e3:.0f} kHz ...", end="", flush=True)

                # Retuner + stabiliser sur le signal exact
                sdr.center_freq = exact_freq
                _ = sdr.read_samples(max(256, (SETTLE_SKIP_SAMPLES // 256) * 256))
                sdr.sample_rate = record_sr

                iq        = record_iq(sdr, args.record_secs, record_sr)
                iq_clean, sample_rate_clean = isolate_central_signal(iq, record_sr, decim_bw) #on isole le signal central pour éviter les voisins
                path      = save_sample(iq_clean, exact_freq, snr_val, sample_rate_clean, run_dir, bw_hz)
                base_name = os.path.basename(path).replace(".wav", "")
                path_png  = save_debug_plot(iq_clean, exact_freq, sample_rate_clean, run_dir,
                                            base_name, args.snr_threshold)

                # Restaurer le SR de scan
                sdr.sample_rate = args.sample_rate

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