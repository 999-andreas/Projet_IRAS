"""
Interface SDR et implémentations (USB et RTL-TCP).
"""

import time
import socket
import struct
import numpy as np
from config import DEFAULT_RTL_TCP_HOST, DEFAULT_RTL_TCP_PORT, DEFAULT_TCP_TIMEOUT


# ─────────────────────────────────────────────
#  INTERFACE ABSTRAITE
# ─────────────────────────────────────────────

class SdrInterface:
    """Interface commune pour USB et RTL-TCP."""

    def __init__(self):
        # Attributs de secours pour que les setters des sous-classes
        # puissent stocker les valeurs avant que le device soit prêt
        self._center_freq = 100_000_000
        self._sample_rate = 2_400_000
        self._gain = 20
        self.center_freq = self._center_freq
        self.sample_rate = self._sample_rate
        self.gain = self._gain

    def set_freq_correction(self, ppm):
        raise NotImplementedError

    def set_offset_tuning(self, enabled):
        raise NotImplementedError

    def read_samples(self, n_samples):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


# ─────────────────────────────────────────────
#  CLIENT RTL-TCP (réseau)
# ─────────────────────────────────────────────

class RtlTcpClient(SdrInterface):
    """Client TCP pour un serveur rtl_tcp."""

    CMD_SET_FREQ         = 0x01
    CMD_SET_SAMPLE_RATE  = 0x02
    CMD_SET_GAIN_MODE    = 0x03
    CMD_SET_GAIN         = 0x04
    CMD_SET_FREQ_CORR    = 0x05
    CMD_SET_OFFSET_TUNING= 0x09

    def __init__(self, host=DEFAULT_RTL_TCP_HOST, port=DEFAULT_RTL_TCP_PORT,
                 timeout=DEFAULT_TCP_TIMEOUT):
        self._host    = host
        self._port    = port
        self._timeout = timeout
        self._sock    = None
        self._connect()
        super().__init__()

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

        header = self._recv_exact(12)
        if header[:4] != b"RTL0":
            raise ValueError(
                f"Réponse inattendue du serveur rtl_tcp : {header!r}"
            )
        tuner_type  = struct.unpack(">I", header[4:8])[0]
        gain_count  = struct.unpack(">I", header[8:12])[0]
        print(f"  [rtl_tcp] Connecté — tuner type={tuner_type}, "
              f"gain steps={gain_count}")

    def _flush_socket_buffer(self):
        """
        Vide activement le buffer de réception du socket TCP.
        Passe temporairement le socket en mode non-bloquant pour consommer
        tous les échantillons obsolètes déjà poussés par le serveur rtl_tcp.
        """
        if self._sock is None:
            return

        self._sock.setblocking(False)
        try:
            while True:
                data = self._sock.recv(4096)
                if not data:
                    break
        except (BlockingIOError, OSError):
            pass  # Buffer vidé avec succès
        finally:
            self._sock.setblocking(True)
            self._sock.settimeout(self._timeout)

    def _send_cmd(self, cmd: int, param: int):
        param_u32 = param & 0xFFFFFFFF
        pkt = struct.pack(">BI", cmd, param_u32)
        self._sock.sendall(pkt)

    def _recv_exact(self, n_bytes: int) -> bytes:
        buf = bytearray()
        while len(buf) < n_bytes:
            chunk = self._sock.recv(n_bytes - len(buf))
            if not chunk:
                raise EOFError("Connexion rtl_tcp fermée prématurément.")
            buf.extend(chunk)
        return bytes(buf)

    def _recv_iq_bytes(self, n_samples: int) -> np.ndarray:
        raw = self._recv_exact(2 * n_samples)
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 127.5) / 127.5
        return data[0::2] + 1j * data[1::2]

    @property
    def center_freq(self) -> float:
        return self._center_freq

    @center_freq.setter
    def center_freq(self, freq_hz: float):
        self._center_freq = int(freq_hz)
        self._send_cmd(self.CMD_SET_FREQ, self._center_freq)
        
        # 1. Purge immédiate du buffer accumulé avant/pendant l'envoi de la commande
        self._flush_socket_buffer()
        
        # 2. Pause pour laisser le tuner/PLL physique se stabiliser
        time.sleep(0.01)
        
        # 3. Seconde purge des échantillons de transition arrivés pendant le rétablissement
        self._flush_socket_buffer()

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate_hz: float):
        self._sample_rate = int(rate_hz)
        self._send_cmd(self.CMD_SET_SAMPLE_RATE, self._sample_rate)
        
        # Purge du buffer réseau pour éliminer les échantillons à l'ancien SR
        self._flush_socket_buffer()
        time.sleep(0.01)
        self._flush_socket_buffer()

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, gain_db: float):
        self._gain = gain_db
        if gain_db == 0:
            self._send_cmd(self.CMD_SET_GAIN_MODE, 0)
        else:
            self._send_cmd(self.CMD_SET_GAIN_MODE, 1)
            self._send_cmd(self.CMD_SET_GAIN, int(gain_db * 10))
        self._flush_socket_buffer()

    def set_freq_correction(self, ppm: int):
        self._send_cmd(self.CMD_SET_FREQ_CORR, int(ppm))
        self._flush_socket_buffer()

    def set_offset_tuning(self, enabled: bool):
        self._send_cmd(self.CMD_SET_OFFSET_TUNING, 1 if enabled else 0)
        self._flush_socket_buffer()

    def read_samples(self, n_samples: int) -> np.ndarray:
        n = max(256, (int(n_samples) // 256) * 256)
        return self._recv_iq_bytes(n)

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ─────────────────────────────────────────────
#  CLIENT USB (local)
# ─────────────────────────────────────────────

class UsbSdrClient(SdrInterface):
    """Client USB pour RTL-SDR branché localement."""

    def __init__(self):
        self._sdr = None          # doit être défini AVANT super().__init__()
        super().__init__()        # les setters appellent _require_sdr()
        try:
            from rtlsdr import RtlSdr
            self._sdr = RtlSdr()
        except ImportError as e:
            raise ImportError(
                "Bibliothèque 'rtlsdr' non trouvée. "
                "Installez-la avec: pip install pyrtlsdr"
            ) from e
        except Exception as e:
            self._sdr = None
            raise ConnectionError(
                f"Impossible de se connecter au RTL-SDR USB : {e}"
            ) from e

        # Applique les valeurs mémorisées durant super().__init__()
        self._sdr.center_freq = self._center_freq
        self._sdr.sample_rate = self._sample_rate
        self._sdr.gain = self._gain
        print("  [USB] Connecté au RTL-SDR (indice 0)")

    def _require_sdr(self):
        if self._sdr is None:
            raise ConnectionError("Le RTL-SDR USB n'est pas initialisé.")
        return self._sdr

    @property
    def center_freq(self) -> float:
        if self._sdr is None:
            return self._center_freq
        return self._sdr.center_freq

    @center_freq.setter
    def center_freq(self, freq_hz: float):
        self._center_freq = int(freq_hz)
        if self._sdr is not None:
            self._sdr.center_freq = self._center_freq

    @property
    def sample_rate(self) -> float:
        if self._sdr is None:
            return self._sample_rate
        return self._sdr.sample_rate

    @sample_rate.setter
    def sample_rate(self, rate_hz: float):
        self._sample_rate = int(rate_hz)
        if self._sdr is not None:
            self._sdr.sample_rate = self._sample_rate

    @property
    def gain(self) -> float:
        if self._sdr is None:
            return self._gain
        return self._sdr.gain

    @gain.setter
    def gain(self, gain_db: float):
        self._gain = gain_db
        if self._sdr is not None:
            self._sdr.gain = gain_db

    def set_freq_correction(self, ppm: int):
        self._require_sdr().set_freq_correction(int(ppm))

    def set_offset_tuning(self, enabled: bool):
        try:
            self._require_sdr().set_offset_tuning(enabled)
        except AttributeError as e:
            raise AttributeError("Offset tuning non supporté par cette version de pyrtlsdr") from e

    def read_samples(self, n_samples: int) -> np.ndarray:
        n = max(256, (int(n_samples) // 256) * 256)
        return self._require_sdr().read_samples(n)

    def close(self):
        if self._sdr is not None:
            try:
                self._sdr.close()
            except Exception:
                pass
            self._sdr = None


# ─────────────────────────────────────────────
#  CLIENT REJEU FICHIER (aucun matériel requis)
# ─────────────────────────────────────────────

class FileReplaySdrClient(SdrInterface):
    """
    Rejoue des hops préalablement enregistrés (scan lancé avec
    --save-hop-iq, voir recording.save_hop_iq) à la place d'un vrai
    RTL-SDR. Permet de faire tourner scanner.scan() exactement comme en
    live (mêmes détection/extraction/tracés), mais en lisant les
    échantillons IQ depuis des fichiers au lieu du matériel — pratique
    pour itérer sur la pipeline sans ressortir le RTL-SDR à chaque fois.

    Fonctionnement :
      - Au chargement, indexe tous les hops enregistrés dans `replay_dir`
        via recording.list_hop_recordings(), triés par fréquence.
      - Quand scanner.py fait `sdr.center_freq = freq` pour un hop, on
        cherche le hop enregistré dont la fréquence centrale est la plus
        proche (optionnellement borné par `freq_tolerance_hz`) et on
        charge son buffer IQ (mis en cache pour les rejeux suivants).
      - read_samples() reproduit le pattern à deux appels de scan() :
        le premier appel après un changement de fréquence correspond au
        "settle skip" (de toute façon jeté par l'appelant) → on renvoie
        des zéros sans toucher au buffer ; les appels suivants consomment
        le buffer réel via un curseur interne, avec zero-padding si on
        demande plus d'échantillons qu'il n'y en a d'enregistrés.
    """

    def __init__(self, replay_dir, freq_tolerance_hz=None):
        from recording import list_hop_recordings, load_hop_iq
        self._load_hop_iq = load_hop_iq

        self._hops = list_hop_recordings(replay_dir)
        if not self._hops:
            raise FileNotFoundError(
                f"Aucun hop rejouable trouvé dans '{replay_dir}'. "
                f"Il faut d'abord lancer un scan (USB ou RTL-TCP) avec "
                f"--save-hop-iq pour générer des fichiers hop_*.iq.wav."
            )

        self._replay_dir        = replay_dir
        self._freq_tolerance_hz = freq_tolerance_hz
        self._buffer            = None
        self._cursor            = 0
        self._settle_pending    = False
        self._cache             = {}

        freqs_mhz = ", ".join(f"{h['freq_hz']/1e6:.3f}" for h in self._hops)
        print(f"  [Replay] {len(self._hops)} hop(s) indexé(s) dans "
              f"'{replay_dir}' : {freqs_mhz} MHz")

        super().__init__()

    def _nearest_hop(self, freq_hz):
        best = min(self._hops, key=lambda h: abs(h["freq_hz"] - freq_hz))
        tol = self._freq_tolerance_hz
        if tol is not None and abs(best["freq_hz"] - freq_hz) > tol:
            return None
        return best

    @property
    def center_freq(self) -> float:
        return self._center_freq

    @center_freq.setter
    def center_freq(self, freq_hz: float):
        self._center_freq = int(freq_hz)
        hop = self._nearest_hop(self._center_freq)

        if hop is None:
            print(f"  [Replay] ⚠  Aucun hop enregistré proche de "
                  f"{self._center_freq/1e6:.3f} MHz (tolérance dépassée) "
                  f"→ hop vide (zéros).")
            self._buffer = np.zeros(0, dtype=np.complex64)
        else:
            key = hop["wav_path"]
            if key not in self._cache:
                iq, sr = self._load_hop_iq(key)
                self._cache[key] = iq
                if sr != self._sample_rate:
                    print(f"  [Replay] ⚠  Sample rate enregistré ({sr/1e6:.3f} MHz) "
                          f"≠ sample rate demandé ({self._sample_rate/1e6:.3f} MHz) "
                          f"pour {key} — rejoué tel quel (pas de rééchantillonnage).")
            self._buffer = self._cache[key]

        self._cursor         = 0
        self._settle_pending = True

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate_hz: float):
        self._sample_rate = int(rate_hz)

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, gain_db: float):
        self._gain = gain_db  # no-op matériel : conservé pour compat d'API

    def set_freq_correction(self, ppm):
        pass  # sans objet en rejeu fichier

    def set_offset_tuning(self, enabled):
        pass  # sans objet en rejeu fichier

    def read_samples(self, n_samples: int) -> np.ndarray:
        n = max(256, (int(n_samples) // 256) * 256)

        # Le premier read après un retune == le "settle skip" de scan(),
        # de toute façon jeté par l'appelant : on ne consomme pas le buffer.
        if self._settle_pending:
            self._settle_pending = False
            return np.zeros(n, dtype=np.complex64)

        buf = self._buffer if self._buffer is not None else np.zeros(0, dtype=np.complex64)
        end = self._cursor + n
        chunk = buf[self._cursor:end]

        if len(chunk) < n:
            if len(buf) > 0:
                print(f"  [Replay] ⚠  Hop épuisé ({len(buf)} échantillons "
                      f"enregistrés < {n} demandés) — complété par des zéros.")
            chunk = np.concatenate([chunk, np.zeros(n - len(chunk), dtype=np.complex64)])

        self._cursor = end
        return chunk

    def close(self):
        self._cache.clear()