"""
Interface SDR et implémentations (USB et RTL-TCP).
"""

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
            self._send_cmd(self.CMD_SET_GAIN_MODE, 0)
        else:
            self._send_cmd(self.CMD_SET_GAIN_MODE, 1)
            self._send_cmd(self.CMD_SET_GAIN, int(gain_db * 10))

    def set_freq_correction(self, ppm: int):
        self._send_cmd(self.CMD_SET_FREQ_CORR, int(ppm))

    def set_offset_tuning(self, enabled: bool):
        self._send_cmd(self.CMD_SET_OFFSET_TUNING, 1 if enabled else 0)

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