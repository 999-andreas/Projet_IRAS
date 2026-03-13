import numpy as np
import scipy.io.wavfile as wav
from iras_cnn1 import build_model

RADIO_CLASSES = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK',
    '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
    '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC',
    'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]

# ── Chargement du .wav ──────────────────────────────────────────
path_sample = str(input("Chemin du fichier.wav : "))
samp_rate, data = wav.read(path_sample)
print(f"Sample rate : {samp_rate} Hz")
print(f"Shape brut  : {data.shape}")
print(f"dtype       : {data.dtype}")

# ── Conversion en I/Q float32 ───────────────────────────────────
# SDR++ exporte en int16 stéréo → normaliser entre -1 et 1
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0
elif data.dtype == np.int8:
    data = data.astype(np.float32) / 128.0

I = data[:, 0]  # canal gauche = I
Q = data[:, 1]  # canal droit  = Q

print(f"Samples I/Q disponibles : {len(I)}")

# ── Découper en chunks de 1024 et classifier ────────────────────
model = build_model((1024, 2), 24)
model.load_weights('../../IRASv1/best_model.h5')

CHUNK = 1024
nb_chunks = len(I) // CHUNK
print(f"Nombre de chunks à analyser : {nb_chunks}\n")

resultats = {}

for i in range(nb_chunks):
    debut = i * CHUNK
    fin   = debut + CHUNK

    iq = np.column_stack([
        I[debut:fin],
        Q[debut:fin]
    ]).astype(np.float32)

    # Normalisation par chunk
    iq = (iq - iq.mean()) / (iq.std() + 1e-8)

    tensor = iq[np.newaxis, ...]  # (1, 1024, 2)
    probs  = model.predict(tensor, verbose=0)[0]

    classe    = RADIO_CLASSES[np.argmax(probs)]
    confiance = float(np.max(probs)) * 100

    resultats[classe] = resultats.get(classe, 0) + 1
    print(f"Chunk {i+1:04d}/{nb_chunks} → {classe} ({confiance:.1f}%)")

# ── Résumé ──────────────────────────────────────────────────────
print("\n" + "="*40)
print("RÉSUMÉ DE LA CAPTURE")
print("="*40)
for classe, count in sorted(resultats.items(), key=lambda x: -x[1]):
    pct = count / nb_chunks * 100
    print(f"{classe:<15} : {count:4d} chunks ({pct:.1f}%)")
