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
if data.dtype == np.int16:
    data = data.astype(np.float32) / 32768.0
elif data.dtype == np.int8:
    data = data.astype(np.float32) / 128.0

I = data[:, 0]
Q = data[:, 1]
print(f"Samples I/Q disponibles : {len(I)}")

# ── Modèle ──────────────────────────────────────────────────────
model = build_model((1024, 2), 24)
model.load_weights('../../IRASv1/model_final.h5')

# ── Préparer tous les chunks d'un coup ──────────────────────────
CHUNK = 1024
nb_chunks = len(I) // CHUNK
print(f"Nombre de chunks : {nb_chunks}")

all_chunks = np.zeros((nb_chunks, CHUNK, 2), dtype=np.float32)
for i in range(nb_chunks):
    debut = i * CHUNK
    iq = np.column_stack([I[debut:debut+CHUNK], Q[debut:debut+CHUNK]])
    iq = (iq - iq.mean()) / (iq.std() + 1e-8)
    all_chunks[i] = iq

# ── Inférence en un seul appel ──────────────────────────────────
print("Inférence en cours...")
all_probs = model.predict(all_chunks, batch_size=512, verbose=1)

# ── Résumé ──────────────────────────────────────────────────────
resultats = {}
for probs in all_probs:
    classe = RADIO_CLASSES[np.argmax(probs)]
    resultats[classe] = resultats.get(classe, 0) + 1

print("\n" + "="*40)
print("RÉSUMÉ DE LA CAPTURE")
print("="*40)
for classe, count in sorted(resultats.items(), key=lambda x: -x[1]):
    pct = count / nb_chunks * 100
    print(f"{classe:<15} : {count:4d} chunks ({pct:.1f}%)")
