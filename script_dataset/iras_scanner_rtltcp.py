#!/usr/bin/env python3
"""
IRAS Scanner v2 — RTL-TCP edition
Source IQ : serveur rtl_tcp (réseau) au lieu du dongle USB local.

Toutes les améliorations v2 sont conservées :
  [1] Moyennage FFT    — N frames avec 50 % d'overlap (gain SNR ≈ √N)
  [2] Threading        — producer/consumer via queue pour acquisition continue
  [3] Pleine bande     — sample_rate 2.4 MHz, step 2 MHz → 10× moins de hops
  [4] DC spike         — masquage des bins centraux + option offset-tuning
  [5] Détection NumPy  — scipy.ndimage.label remplace la boucle Python pure
  [6] Bruit floor      — 10e percentile au lieu de la médiane
  [7] Correction PPM   — envoyée comme commande TCP au serveur
  [8] Stabilisation    — lecture+rejet du premier bloc après retune
  [9] Fenêtre Blackman — conservée
  [10] Gain adaptatif  — avertissement saturation + mode AGC

Nouveau :
  [11] RtlTcpClient    — client TCP pur (socket) compatible avec l'interface
                         pyrtlsdr : center_freq, sample_rate, gain,
                         read_samples(), close().
                         Protocole rtl_tcp : commandes 5 octets (cmd + uint32 BE)
                         + stream I/Q uint8 interleaved centré sur 127.5.

Lancer le serveur côté SDR :
    rtl_tcp -a 0.0.0.0 -p 1234

Puis ce script :
    python iras_scanner2_rtltcp.py --rtl-tcp-host 192.168.1.10
"""

import numpy as np
import json
import time
import os
import queue
import socket
import struct
import threading
import argparse
import wave
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import label as ndimage_label



# ─────────────────────────────────────────────
#  CONFIG PAR DÉFAUT
# ─────────────────────────────────────────────

DEFAULT_FREQ_START      = 88e6      # Hz
DEFAULT_FREQ_END        = 108e6     # Hz
DEFAULT_FREQ_STEP       = 2_000_000 # Hz
DEFAULT_SAMPLE_RATE     = 2_400_000 # Hz
DEFAULT_GAIN            = 20         # dB  (0 = AGC automatique)
DEFAULT_PPM             = 0         # ppm
DEFAULT_SNR_THRESHOLD   = 20        # dB au dessus du plancher de bruit
DEFAULT_RECORD_SECS     = 3         # secondes d'enregistrement IQ
DEFAULT_OUTPUT_DIR      = "./dataset"
DEFAULT_FFT_SIZE        = 2048
DEFAULT_N_AVG           = 64
DEFAULT_DUPLICATE_HZ    = 300_000   # Hz
DEFAULT_DC_WIDTH_BINS   = None      # None = auto (1 % des bins)
SETTLE_SKIP_SAMPLES     = 8_192     # samples jetés après retune
SATURATION_THRESHOLD_DB = -3.0      # niveau max accepté avant avertissement

# RTL-TCP
DEFAULT_RTL_TCP_HOST    = "192.168.1.12"
DEFAULT_RTL_TCP_PORT    = 1234
DEFAULT_TCP_TIMEOUT     = 10.0      # secondes


# ─────────────────────────────────────────────
#  [11] CLIENT RTL-TCP
# ─────────────────────────────────────────────

class RtlTcpClient:
    """
    Client réseau pour un serveur rtl_tcp.

    Protocole rtl_tcp :
      - À la connexion, le serveur envoie un magic header de 12 octets :
          b"RTL0" + uint32 tuner_type + uint32 tuner_gain_count
      - Chaque commande depuis le client = 5 octets :
          1 octet  : identifiant de commande (CMD_*)
          4 octets : paramètre en big-endian uint32
      - Flux de données : I/Q uint8 interleaved, centré sur 127.5
          → samples complexes = (I - 127.5)/127.5 + j*(Q - 127.5)/127.5

    Interface identique à pyrtlsdr.RtlSdr pour que scan() n'ait pas à changer :
      .center_freq   (property r/w)
      .sample_rate   (property r/w)
      .gain          (property r/w)
      .read_samples(n_samples) → np.ndarray complexe
      .set_freq_correction(ppm)
      .set_offset_tuning(enabled)
      .close()
    """

    # Identifiants de commandes RTL-TCP
    CMD_SET_FREQ         = 0x01
    CMD_SET_SAMPLE_RATE  = 0x02
    CMD_SET_GAIN_MODE    = 0x03   # 0 = AGC automatique, 1 = gain manuel
    CMD_SET_GAIN         = 0x04   # valeur en dixièmes de dB
    CMD_SET_FREQ_CORR    = 0x05   # ppm (entier signé)
    CMD_SET_OFFSET_TUNING= 0x09   # 0 ou 1

    def __init__(self, host=DEFAULT_RTL_TCP_HOST, port=DEFAULT_RTL_TCP_PORT,
                 timeout=DEFAULT_TCP_TIMEOUT):
        self._host    = host
        self._port    = port
        self._timeout = timeout
        self._sock    = None

        # État interne (reflète ce qu'on a envoyé au serveur)
        self._center_freq   = 100_000_000
        self._sample_rate   = DEFAULT_SAMPLE_RATE
        self._gain          = 0

        self._connect()

    # ── Connexion & handshake ──────────────────

    def _connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        try:
            self._sock.connect((self._host, self._port))
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(
                f"Impossible de se connecter au serveur rtl_tcp à "
                f"{self._host}:{self._port} — {e}"
            ) from e

        # Lire le magic header (12 octets)
        header = self._recv_exact(12)
        if header[:4] != b"RTL0":
            raise ValueError(
                f"Réponse inattendue du serveur rtl_tcp : {header!r} "
                f"(attendu magic b'RTL0')"
            )
        tuner_type  = struct.unpack(">I", header[4:8])[0]
        gain_count  = struct.unpack(">I", header[8:12])[0]
        print(f"  [rtl_tcp] Connecté — tuner type={tuner_type}, "
              f"gain steps={gain_count}")

    # ── Envoi de commandes ─────────────────────

    def _send_cmd(self, cmd: int, param: int):
        """Envoie une commande rtl_tcp (1 octet cmd + 4 octets param big-endian)."""
        # param est interprété comme uint32 ; les valeurs négatives (ppm signé)
        # sont recodées en two's complement sur 32 bits.
        param_u32 = param & 0xFFFFFFFF
        pkt = struct.pack(">BI", cmd, param_u32)
        self._sock.sendall(pkt)

    # ── Réception de données ───────────────────

    def _recv_exact(self, n_bytes: int) -> bytes:
        """Lit exactement n_bytes depuis le socket, en gérant les reads partiels."""
        buf = bytearray()
        while len(buf) < n_bytes:
            chunk = self._sock.recv(n_bytes - len(buf))
            if not chunk:
                raise EOFError("Connexion rtl_tcp fermée prématurément.")
            buf.extend(chunk)
        return bytes(buf)

    def _recv_iq_bytes(self, n_samples: int) -> np.ndarray:
        """
        Lit n_samples paires IQ (= 2*n_samples octets) et retourne un tableau
        complexe normalisé dans [-1, 1].
        """
        raw = self._recv_exact(2 * n_samples)
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        # Dé-centrage et normalisation
        data = (data - 127.5) / 127.5
        return data[0::2] + 1j * data[1::2]

    # ── Interface pyrtlsdr-compatible ─────────

    @property
    def center_freq(self) -> float:
        return self._center_freq

    @center_freq.setter
    def center_freq(self, freq_hz: float):
        self._center_freq = int(freq_hz)
        self._send_cmd(self.CMD_SET_FREQ, self._center_freq)

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate_hz: float):
        self._sample_rate = int(rate_hz)
        self._send_cmd(self.CMD_SET_SAMPLE_RATE, self._sample_rate)

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, gain_db: float):
        self._gain = gain_db
        if gain_db == 0:
            # Activer l'AGC automatique
            self._send_cmd(self.CMD_SET_GAIN_MODE, 0)
        else:
            # Passer en gain manuel puis régler
            self._send_cmd(self.CMD_SET_GAIN_MODE, 1)
            self._send_cmd(self.CMD_SET_GAIN, int(gain_db * 10))

    def set_freq_correction(self, ppm: int):
        """[7] Correction fréquence en ppm (entier signé → uint32 two's complement)."""
        self._send_cmd(self.CMD_SET_FREQ_CORR, int(ppm))

    def set_offset_tuning(self, enabled: bool):
        """[4] Active/désactive l'offset tuning côté serveur."""
        self._send_cmd(self.CMD_SET_OFFSET_TUNING, 1 if enabled else 0)

    def read_samples(self, n_samples: int) -> np.ndarray:
        """
        Lit n_samples échantillons IQ complexes depuis le serveur rtl_tcp.
        n_samples est arrondi à un multiple de 256 (contrainte RTL-SDR).
        """
        n = max(256, (int(n_samples) // 256) * 256)
        return self._recv_iq_bytes(n)

    def close(self):
        """Ferme la connexion TCP."""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ─────────────────────────────────────────────
#  visualisation de diagnostic (optionnelle, pour aider a la classification future)
# ─────────────────────────────────────────────

def save_debug_plot(iq, freq_hz, sample_rate, output_dir, base_name, snr_threshold):
    """
    Génère un graphique de diagnostic avec vue PSD (type SDR++).
    """
    plt.figure(figsize=(15, 12))
    plt.suptitle(f"Analyse du Signal : {freq_hz/1e6:.3f} MHz", fontsize=16)

    # --- 1. Vue Puissance (Spectrum) — Style SDR++ ---
    plt.subplot(3, 2, 1)
    # Calcul de la PSD
    fft_size = 2048
    window = np.blackman(fft_size)
    # On utilise Welch pour un tracé lisse
    from scipy.signal import welch
    f, psd = welch(iq, fs=sample_rate/1e6, window=window, nperseg=fft_size, scaling='density', detrend=False)
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(10 * np.log10(psd + 1e-20))
    
    plt.plot(f, psd, color='cyan', linewidth=0.8)
    plt.fill_between(f, psd, -140, color='cyan', alpha=0.2)
    
    # Ligne de seuil
    floor = np.percentile(psd, 10)
    plt.axhline(floor + snr_threshold, color='red', linestyle='--', label='Seuil Détection')
    
    plt.title("Spectre de Puissance (PSD)")
    plt.ylabel("Puissance (dBFS)")
    plt.xlabel("Offset Fréquence (MHz)")
    plt.ylim(floor - 10, max(psd) + 10)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- 2. Spectrogramme (Waterfall) ---
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
    plt.gca().set_facecolor('black') # Style pro
    plt.axis('equal')

    # --- 4. Histogramme des fréquences (Pour FSK) ---
    plt.subplot(3, 2, 4)
    freq_inst = np.diff(np.unwrap(np.angle(iq)))
    plt.hist(freq_inst, bins=128, color='orange', alpha=0.8)
    plt.title("Histogramme de Déviation (FSK Discriminator)")

    # --- 5. Zoom Enveloppe Temporelle (AM) ---
    plt.subplot(3, 2, (5, 6)) # Prend toute la largeur en bas
    plt.plot(np.abs(iq[:3000]), color='white', linewidth=0.5)
    plt.title("Enveloppe d'Amplitude (Domaine temporel)")
    plt.gca().set_facecolor('black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, base_name + ".png")
    plt.savefig(plot_path, facecolor='#2e2e2e') # Fond sombre
    plt.close()

# ─────────────────────────────────────────────
#  [1] FFT MOYENNÉE (Welch simplifié, 50 % overlap)
# ─────────────────────────────────────────────

def compute_averaged_psd(samples, fft_size=DEFAULT_FFT_SIZE, n_avg=DEFAULT_N_AVG):
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
#  [4] MASQUAGE DU DC SPIKE
# ─────────────────────────────────────────────

def mask_dc_spike(psd, fft_size=DEFAULT_FFT_SIZE, dc_width_bins=DEFAULT_DC_WIDTH_BINS):
    if dc_width_bins is None:
        dc_width_bins = max(4, fft_size // 100)
    center = fft_size // 2
    psd    = psd.copy()
    lo     = max(0, center - dc_width_bins)
    hi     = min(fft_size, center + dc_width_bins)
    psd[lo:hi] = np.min(psd)
    return psd


# ─────────────────────────────────────────────
#  [5] + [6] DÉTECTION DE PIC
# ─────────────────────────────────────────────

def noise_floor(psd):
    return float(np.percentile(psd, 10))


def peak_power(psd, min_bins=10, above_db=6):
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
    return peak_power(psd) - noise_floor(psd)


# ─────────────────────────────────────────────
#  [10] VÉRIFICATION DE SATURATION
# ─────────────────────────────────────────────

def check_saturation(psd, threshold_db=SATURATION_THRESHOLD_DB):
    return float(np.max(psd)) >= threshold_db


# ─────────────────────────────────────────────
#  CLASSIFICATION PAR HEURISTIQUES
# ─────────────────────────────────────────────

def estimate_bandwidth(samples, sample_rate, fft_size=DEFAULT_FFT_SIZE,
                       n_avg=DEFAULT_N_AVG, threshold_db=10):
    psd     = compute_averaged_psd(samples, fft_size, n_avg)
    psd     = mask_dc_spike(psd, fft_size)
    floor   = noise_floor(psd)
    mask    = psd > (floor + threshold_db)
    bw_bins = int(np.sum(mask))
    bw_hz   = bw_bins * (sample_rate / fft_size)
    return bw_hz


def am_fm_ratio(samples):
    amplitude = np.abs(samples)
    phase     = np.unwrap(np.angle(samples))
    var_amp   = float(np.var(amplitude / (np.mean(amplitude) + 1e-12)))
    var_phase = float(np.var(np.diff(phase)))
    return var_amp, var_phase


def is_duplicate_signal(freq_hz, detected_freqs, min_separation_hz):
    return any(abs(freq_hz - d) <= min_separation_hz for d in detected_freqs)


# ─────────────────────────────────────────────
#  ENREGISTREMENT IQ
# ─────────────────────────────────────────────

def record_iq(sdr, duration_sec, sample_rate):
    """Enregistre duration_sec secondes d'IQ en chunks."""
    num_samples = int(sample_rate * duration_sec)
    chunk_size  = 16 * 1024
    chunks      = []
    collected   = 0

    while collected < num_samples:
        to_read = min(chunk_size, num_samples - collected)
        to_read = max(256, (to_read // 256) * 256)
        samples = sdr.read_samples(to_read)
        chunks.append(samples)
        collected += len(samples)

    return np.concatenate(chunks)[:num_samples]


def save_sample(iq, freq_hz, snr_db, sample_rate, output_dir):
    """Sauvegarde les données IQ en format WAV stéréo (I/Q) et les métadonnées (.json)."""
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_mhz = freq_hz / 1e6
    base     = f"signal_{freq_mhz:.3f}MHz_{ts}"
    wav_path = os.path.join(output_dir, base + ".wav")
    jpath    = os.path.join(output_dir, base + ".json")

    # Normalise et convertit en int16
    iq_normalized = iq / (np.max(np.abs(iq)) + 1e-12)
    i_data = (iq_normalized.real * 32767).astype(np.int16)
    q_data = (iq_normalized.imag * 32767).astype(np.int16)
    
    # Entrelace I et Q en stéréo (I=gauche, Q=droite)
    stereo = np.empty(len(iq) * 2, dtype=np.int16)
    stereo[0::2] = i_data
    stereo[1::2] = q_data
    
    # Écrit le fichier WAV
    with wave.open(wav_path, 'w') as f:
        f.setnchannels(2)       # stéréo = I/Q
        f.setsampwidth(2)       # 16 bits
        f.setframerate(int(sample_rate))
        f.writeframes(stereo.tobytes())

    # Sauvegarde les métadonnées JSON
    with open(jpath, "w") as f:
        json.dump({
            "frequency_hz"  : freq_hz,
            "frequency_mhz" : round(freq_mhz, 3),
            "sample_rate"   : sample_rate,
            "timestamp"     : ts,
            "num_samples"   : len(iq),
            "snr_db"        : round(float(snr_db), 2),
        }, f, indent=2)

    return wav_path


# ─────────────────────────────────────────────
#  [2] THREAD PRODUCTEUR — acquisition continue
# ─────────────────────────────────────────────

class IQProducer(threading.Thread):
    """
    Lit les samples RTL-TCP en continu et les pousse dans une queue.
    Pendant que le thread principal traite la FFT, le flux TCP continue
    à être drainé sans interruption.
    """

    def __init__(self, sdr, buf_queue, n_samples, n_buffers=5):
        super().__init__(daemon=True)
        self._sdr       = sdr
        self._queue     = buf_queue
        self._n_samples = n_samples
        self._n_buffers = n_buffers
        self._stop_evt  = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        chunk = max(256, (self._n_samples // 256) * 256)
        while not self._stop_evt.is_set():
            try:
                samples = self._sdr.read_samples(chunk)
                self._queue.put(samples, timeout=2.0)
            except Exception:
                if not self._stop_evt.is_set():
                    break


# ─────────────────────────────────────────────
#  SCANNER PRINCIPAL
# ─────────────────────────────────────────────

def scan(args):
    # [11] Connexion au serveur RTL-TCP (remplace RtlSdr())
    print(f"\n  Connexion au serveur rtl_tcp → {args.rtl_tcp_host}:{args.rtl_tcp_port} …")
    sdr = RtlTcpClient(
        host    = args.rtl_tcp_host,
        port    = args.rtl_tcp_port,
        timeout = args.rtl_tcp_timeout,
    )

    sdr.sample_rate = args.sample_rate
    sdr.gain        = args.gain   # 0 = AGC automatique

    # [7] Correction PPM
    if args.ppm != 0:
        sdr.set_freq_correction(args.ppm)

    # [4] Offset tuning optionnel
    if args.offset_tuning:
        try:
            sdr.set_offset_tuning(True)
            print("  Offset tuning activé (DC spike déplacé hors bande).")
        except Exception as e:
            print(f"  ⚠  Offset tuning refusé par le serveur : {e}")

    freqs          = np.arange(args.freq_start, args.freq_end, args.freq_step)
    total          = len(freqs)
    detected       = 0
    detected_freqs = []

    snap_size = DEFAULT_FFT_SIZE * (args.n_avg + 1)

    print(f"\n{'─'*65}")
    print(f"  IRAS Scanner v2 — RTL-TCP edition")
    print(f"  Serveur     : {args.rtl_tcp_host}:{args.rtl_tcp_port}")
    print(f"  Plage       : {args.freq_start/1e6:.1f} – {args.freq_end/1e6:.1f} MHz")
    print(f"  Pas         : {args.freq_step/1e6:.2f} MHz  (BW captée : {args.sample_rate/1e6:.1f} MHz)")
    print(f"  Hops        : {total}")
    print(f"  Seuil SNR   : +{args.snr_threshold} dB")
    print(f"  Moyennage   : {args.n_avg} frames FFT (overlap 50 %)")
    print(f"  PPM corr.   : {args.ppm:+d} ppm")
    print(f"  Gain        : {args.gain} dB{' (AGC auto)' if args.gain == 0 else ''}")
    print(f"  Output      : {args.output_dir}")
    print(f"{'─'*65}\n")

    buf_queue = queue.Queue(maxsize=5)

    try:
        for i, freq in enumerate(freqs):
            sdr.center_freq = freq

            # [8] Stabilisation après retune
            _ = sdr.read_samples(max(256, (SETTLE_SKIP_SAMPLES // 256) * 256))
            time.sleep(0.02)

            # [2] Thread producteur
            producer = IQProducer(sdr, buf_queue, snap_size)
            producer.start()

            collected = []
            while sum(len(c) for c in collected) < snap_size:
                try:
                    chunk = buf_queue.get(timeout=2.0)
                    collected.append(chunk)
                except queue.Empty:
                    break

            producer.stop()
            producer.join(timeout=1.0)
            while not buf_queue.empty():
                try:
                    buf_queue.get_nowait()
                except queue.Empty:
                    break

            if not collected:
                print(f"[{i+1:>4}/{total}] {freq/1e6:>9.3f} MHz  ← TIMEOUT acquisition, hop ignoré")
                continue

            snap = np.concatenate(collected)[:snap_size]

            # [1] PSD moyennée + [4] masquage DC
            psd  = compute_averaged_psd(snap, DEFAULT_FFT_SIZE, args.n_avg)
            psd  = mask_dc_spike(psd, DEFAULT_FFT_SIZE)

            # [6] SNR avec plancher 10e percentile
            floor   = noise_floor(psd)
            pwr     = peak_power(psd)
            snr_val = pwr - floor

            # [10] Avertissement saturation
            sat_warn = ""
            if check_saturation(psd):
                sat_warn = "  ⚠ SATURATION"

            print(f"[{i+1:>4}/{total}] {freq/1e6:>9.3f} MHz  |  "
                  f"floor={floor:>+6.1f} dBFS  SNR={snr_val:>+5.1f} dB{sat_warn}", end="")

            if snr_val >= args.snr_threshold:
                if is_duplicate_signal(freq, detected_freqs, args.min_separation_hz):
                    print("  ← SIGNAL DÉJÀ ENREGISTRÉ (duplication ignorée)")
                    continue

                print(f"  ← SIGNAL DÉTECTÉ, enregistrement {args.record_secs}s ...",
                      end="", flush=True)

                iq   = record_iq(sdr, args.record_secs, args.sample_rate)
                path = save_sample(iq, freq, snr_val, args.sample_rate, args.output_dir)
                base_name = os.path.basename(path).replace(".wav", "")
                path_png = save_debug_plot(iq, freq, args.sample_rate, args.output_dir, base_name, args.snr_threshold)
                
                detected += 1
                detected_freqs.append(freq)
                print(f"\n           → Audio: {path}")
                print(f"           → Debug: {path_png}")
            else:
                print()

    except KeyboardInterrupt:
        print("\n\nScan interrompu.")

    finally:
        sdr.close()

    print(f"\n{'─'*65}")
    print(f"  Scan terminé : {detected} signal(s) enregistré(s) sur {total} fréquences")
    print(f"  Dataset : {args.output_dir}")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IRAS Scanner v2 — RTL-TCP edition (source IQ via réseau)"
    )

    # ── Paramètres RTL-TCP [11] ──────────────
    tcp = parser.add_argument_group("RTL-TCP (source IQ réseau)")
    tcp.add_argument("--rtl-tcp-host",    type=str,   default=DEFAULT_RTL_TCP_HOST,
                     help=f"Hôte du serveur rtl_tcp (défaut: {DEFAULT_RTL_TCP_HOST})")
    tcp.add_argument("--rtl-tcp-port",    type=int,   default=DEFAULT_RTL_TCP_PORT,
                     help=f"Port du serveur rtl_tcp (défaut: {DEFAULT_RTL_TCP_PORT})")
    tcp.add_argument("--rtl-tcp-timeout", type=float, default=DEFAULT_TCP_TIMEOUT,
                     help=f"Timeout socket en secondes (défaut: {DEFAULT_TCP_TIMEOUT}s)")

    # ── Paramètres scanner ───────────────────
    parser.add_argument("--freq-start",         type=float, default=DEFAULT_FREQ_START,
                        help=f"Fréquence de départ en Hz (défaut: {DEFAULT_FREQ_START/1e6}MHz)")
    parser.add_argument("--freq-end",           type=float, default=DEFAULT_FREQ_END,
                        help=f"Fréquence de fin en Hz (défaut: {DEFAULT_FREQ_END/1e6}MHz)")
    parser.add_argument("--freq-step",          type=float, default=DEFAULT_FREQ_STEP,
                        help=f"Pas en Hz (défaut: {DEFAULT_FREQ_STEP/1e6}MHz)")
    parser.add_argument("--sample-rate",        type=float, default=DEFAULT_SAMPLE_RATE,
                        help=f"Sample rate en Hz (défaut: {DEFAULT_SAMPLE_RATE/1e6}MHz)")
    parser.add_argument("--gain",               type=float, default=DEFAULT_GAIN,
                        help="Gain en dB. Utiliser 0 pour le mode AGC automatique.")
    parser.add_argument("--ppm",                type=int,   default=DEFAULT_PPM,
                        help="Correction de fréquence en ppm.")
    parser.add_argument("--snr-threshold",      type=float, default=DEFAULT_SNR_THRESHOLD,
                        help=f"Seuil SNR en dB (défaut: {DEFAULT_SNR_THRESHOLD}dB)")
    parser.add_argument("--n-avg",              type=int,   default=DEFAULT_N_AVG,
                        help=f"Nombre de frames FFT à moyenner (défaut: {DEFAULT_N_AVG})")
    parser.add_argument("--record-secs",        type=float, default=DEFAULT_RECORD_SECS,
                        help=f"Durée d'enregistrement en secondes (défaut: {DEFAULT_RECORD_SECS}s)")
    parser.add_argument("--output-dir",         type=str,   default=DEFAULT_OUTPUT_DIR,
                        help=f"Dossier de sortie (défaut: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--min-separation-hz",  type=float, default=DEFAULT_DUPLICATE_HZ,
                        help=f"Seuil anti-duplication en Hz (défaut: {DEFAULT_DUPLICATE_HZ}Hz)")
    parser.add_argument("--offset-tuning",      action="store_true", default=False,
                        help="Active l'offset tuning (commande envoyée au serveur rtl_tcp).")

    args = parser.parse_args()
    scan(args)


if __name__ == "__main__":
    main()