#!/usr/bin/env python3
"""Convertit un .npy IQ en .wav IQ lisible par SDR++"""
import numpy as np
import wave
import struct
import sys

def convert(npy_path, sample_rate=1000000):
    iq = np.load(npy_path)
    
    # Normalise et convertit en int16
    iq = iq / (np.max(np.abs(iq)) + 1e-12)
    i = (iq.real * 32767).astype(np.int16)
    q = (iq.imag * 32767).astype(np.int16)
    
    # Interleave I et Q en stéréo (I=gauche, Q=droite)
    stereo = np.empty(len(iq) * 2, dtype=np.int16)
    stereo[0::2] = i
    stereo[1::2] = q
    
    out_path = npy_path.replace(".npy", ".wav")
    with wave.open(out_path, 'w') as f:
        f.setnchannels(2)       # stéréo = I/Q
        f.setsampwidth(2)       # 16 bits
        f.setframerate(sample_rate)
        f.writeframes(stereo.tobytes())
    
    print(f"Converti : {out_path}  ({sample_rate/1e6} MHz, {len(iq)} samples)")

if __name__ == "__main__":
    npy_path    = sys.argv[1]
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
    convert(npy_path, sample_rate)