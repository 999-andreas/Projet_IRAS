import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from iras_cnn1 import build_model

# ── Paramètres ──────────────────────────────────────────────────
HDF5_PATH = '/home/cawps/Bureau/IRAS/Datasets/IRAS_11092026.hdf5'
BATCH_SIZE = 256
SNR_MIN = 5

SHAPE = 4096

# ── Récupérer les indices valides (SNR >= 0) ─────────────────────
with h5py.File(HDF5_PATH, 'r') as f:
    Z = f['snr'][:]
    total = f['iq'].shape[0]

valid_idx = np.where(Z >= SNR_MIN)[0]
#valid_idx = valid_idx[:len(valid_idx)//10]
np.random.shuffle(valid_idx)

split = int(len(valid_idx) * 0.8)
train_idx = valid_idx[:split]
test_idx  = valid_idx[split:]

# ── Générateur ──────────────────────────────────────────────────
def make_generator(indices, batch_size):
    def generator():
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i+batch_size]
            batch_sorted = np.sort(batch)  # h5py impose l'ordre croissant
            with h5py.File(HDF5_PATH, 'r') as f:
                X = f['iq'][batch_sorted]
                Y = f['label'][batch_sorted]
            
            #conversion one-hot
            Y = to_categorical(Y, num_classes=2)

            yield X.astype(np.float32), Y.astype(np.float32)
    return generator

num_classes = 2
dataset_train = tf.data.Dataset.from_generator(
    make_generator(train_idx, BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(None, SHAPE, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

dataset_test = tf.data.Dataset.from_generator(
    make_generator(test_idx, BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(None, SHAPE, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

# ── Modèle ──────────────────────────────────────────────────────
model = build_model((SHAPE, 2), num_classes)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1),
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
]

steps_train = len(train_idx) // BATCH_SIZE
steps_test  = len(test_idx)  // BATCH_SIZE

model.fit(
    dataset_train,
    epochs=5,
    steps_per_epoch=steps_train,
    validation_data=dataset_test,
    validation_steps=steps_test // 5,
    callbacks=callbacks
)

loss, acc = model.evaluate(dataset_test, steps=steps_test)
print(f"\nPrécision test : {acc*100:.2f}%")
