import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from iras_cnn1 import build_model

# ── Chargement ──────────────────────────────────────────────────
with h5py.File('/home/cawps/Bureau/IRAS/Datasets/GOLD_XYZ_OSC.0001_1024.hdf5', 'r') as f:
    X = f['X'][:]  # (N, 1024, 2) — déjà au bon format !
    Y = f['Y'][:]  # (N, 24) — one-hot sur 24 classes
    Z = f['Z'][:]  # (N, 1) — SNR en dB

# ── Filtrer par SNR (optionnel) ─────────────────────────────────
snr_mask = Z[:, 0] >= 0  # garder SNR >= 0 dB
X = X[snr_mask]
Y = Y[snr_mask]

# ── Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ── Modèle ──────────────────────────────────────────────────────
model = build_model((1024, 2), Y.shape[1])  # 24 classes auto
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1),
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
]

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=callbacks
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nPrécision test : {acc*100:.2f}%")
