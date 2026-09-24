"""
Boucle principale de scan.
"""

from asyncio import events
import os
import time
import numpy as np
from datetime import datetime
from config import (
    DEFAULT_FFT_SIZE, DEFAULT_N_AVG, SETTLE_SKIP_SAMPLES,
    DEFAULT_CFAR_GUARD_CELLS, DEFAULT_CFAR_TRAIN_CELLS, DEFAULT_CFAR_PFA,
)
from sdr import UsbSdrClient, RtlTcpClient, FileReplaySdrClient
from rf_signal import (
    compute_averaged_psd, mask_dc_spike, noise_floor, peak_power,
    check_saturation, is_duplicate_signal, dedupe_signals_intra_hop,
    detect_all_signals, detect_all_signals_cfar, extract_and_center_signal
)
from recording import save_sample, save_hop_iq
from visualization import save_debug_plot, save_hop_plot
from scan_events import ScanEvents

def scan(args, mode, events=None, stop_flag=None):

    """
        Boucle principale de scan :
        - Crée le client SDR (USB, RTL-TCP, ou rejeu fichier)
        - Configure le hardware (ignoré en rejeu)
        - Balaye les fréquences
        - Détecte, enregistre et trace les signaux
    """

    # Initialisation neutre : si events est None, on utilise l'objet de base (qui ne fait rien)
    events = events or ScanEvents()

    # ── Import auto d'IQ externe (--iq-file) ──────────────────
    # Convertit à la volée le(s) WAV externe(s) (SDR++/SDR#/GQRX...) vers un
    # dossier de hops rejouables, puis bascule en mode replay dessus — évite
    # de lancer import_iq_recording.py à la main avant chaque scan.
    iq_files = getattr(args, "iq_file", None)
    if iq_files:
        from import_iq_recording import import_one
        import_dir = os.path.join(args.output_dir, "_imported_iq", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(import_dir, exist_ok=True)
        for i, path in enumerate(iq_files, start=1):
            import_one(path, import_dir, hop_index=i,
                       freq_hz_override=None, sample_rate_override=None, max_secs=None)
        args.replay_dir = import_dir
        mode = "replay"

    # ── Connexion SDR ──────────────────────────
    if mode == "usb":
        print(f"\n  Connexion au RTL-SDR USB…")
        sdr = UsbSdrClient()
    elif mode == "rtltcp":
        print(f"\n  Connexion au serveur rtl_tcp → {args.rtl_tcp_host}:{args.rtl_tcp_port} …")
        sdr = RtlTcpClient(
            host    = args.rtl_tcp_host,
            port    = args.rtl_tcp_port,
            timeout = args.rtl_tcp_timeout,
        )
    else:  # replay — aucun matériel, on rejoue des hops déjà enregistrés
        print(f"\n  Rejeu de hops enregistrés → {args.replay_dir}")
        sdr = FileReplaySdrClient(
            replay_dir        = args.replay_dir,
            freq_tolerance_hz = getattr(args, "replay_freq_tolerance_hz", None),
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

    mode_str = {"usb": "USB", "rtltcp": "RTL-TCP", "replay": "Replay (fichier)"}.get(mode, mode)
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


    # Notifier le début du scan
    events.on_scan_start({
        "mode": mode,
        "freq_start": args.freq_start,
        "freq_end": args.freq_end,
        "freq_step": args.freq_step,
        "sample_rate": args.sample_rate,
        "snr_threshold": args.snr_threshold,
        "detector": args.detector,
        "total_hops": total,
        "run_dir": run_dir,
        "output_dir": args.output_dir,
    })

    # code synchrone...
    try:
        for i, freq in enumerate(freqs):
            # ── 1. Vérification du bouton STOP ──
            if stop_flag is not None and stop_flag.is_set():
                events.on_scan_interrupted()
                break

            sdr.center_freq = freq
            _ = sdr.read_samples(max(256, (SETTLE_SKIP_SAMPLES // 256) * 256))
            time.sleep(0.02)

            # ── 2. Capture avec gestion des erreurs sur le Hop ──
            try:
                hop_iq = sdr.read_samples(hop_capture_samples)
            except Exception as e:
                events.on_hop_error(i + 1, total, freq, e)
                continue

            # Sauvegarde du hop BRUT (pré-détection) pour rejeu ultérieur
            # sans RTL-SDR — voir recording.save_hop_iq / sdr.FileReplaySdrClient.
            # Inutile (et redondant) en mode replay lui-même.
            if getattr(args, "save_hop_iq", False) and mode != "replay":
                save_hop_iq(hop_iq, freq, args.sample_rate, run_dir, i + 1)

            # Traitement FFT & Détection
            snap  = hop_iq[:snap_size]
            psd   = compute_averaged_psd(snap, DEFAULT_FFT_SIZE, args.n_avg)
            floor = noise_floor(psd)
            is_sat = check_saturation(psd)

            if args.detector == "cfar":
                signals = detect_all_signals_cfar(psd, freq, args.sample_rate, fft_size=DEFAULT_FFT_SIZE)
            else:
                signals = detect_all_signals(psd, freq, args.sample_rate, DEFAULT_FFT_SIZE, args.snr_threshold)

            best_snr = signals[0]['snr_db'] if signals else (peak_power(psd) - floor)

            # Déduplication intra-hop
            signals = dedupe_signals_intra_hop(signals, args.min_separation_hz)

            # Sauvegarde graphique du hop
            hop_plot_path = save_hop_plot(
                psd, freq, args.sample_rate, signals, run_dir, i + 1,
                fft_size=DEFAULT_FFT_SIZE,
                detector=args.detector,
                snr_threshold=args.snr_threshold,
                cfar_guard_cells=DEFAULT_CFAR_GUARD_CELLS,
                cfar_train_cells=DEFAULT_CFAR_TRAIN_CELLS,
                cfar_pfa=DEFAULT_CFAR_PFA,
            )

            # ── 3. Notification du Hop Terminé ──
            events.on_hop_complete(
                hop_index=i + 1,
                total_hops=total,
                freq_hz=freq,
                floor_db=floor,
                best_snr_db=best_snr,
                signals=signals,
                saturation=is_sat,
                hop_plot_path=hop_plot_path
            )

            # Traitement des signaux trouvés
            for sig in signals:
                exact_freq = sig['freq_hz']
                bw_hz      = sig['bw_hz']
                snr_val    = sig['snr_db']

                if is_duplicate_signal(exact_freq, detected_freqs, args.min_separation_hz):
                    # ── 4. Notification Doublon ──
                    events.on_signal_duplicate(exact_freq)
                    continue

                if bw_hz > 100_000:
                    decim_bw = max(bw_hz * 1.2, 150_000)
                else:
                    decim_bw = max(bw_hz * 1.3, 12_000)

                iq_clean, sample_rate_clean = extract_and_center_signal(
                    hop_iq, freq, exact_freq, args.sample_rate, decim_bw
                )

                path     = save_sample(iq_clean, exact_freq, snr_val, sample_rate_clean, run_dir, bw_hz)
                base_name = os.path.basename(path).replace(".wav", "")
                path_png  = save_debug_plot(iq_clean, exact_freq, sample_rate_clean, run_dir,
                                            base_name, args.snr_threshold)

                detected += 1
                detected_freqs.append(exact_freq)

                # ── 5. Notification Signal Extrait ──
                events.on_signal_extracted(
                    freq_hz=exact_freq,
                    snr_db=snr_val,
                    bw_hz=bw_hz,
                    wav_path=path,
                    debug_plot_path=path_png
                )

    except KeyboardInterrupt:
        events.on_scan_interrupted()

    finally:
        sdr.close()

    # ── 6. Notification de Fin de Scan ──
    events.on_scan_end(detected, total, args.output_dir)