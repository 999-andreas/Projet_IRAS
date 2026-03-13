import numpy as np
from tensorflow.keras import layers, models, Input
from tensorflow.keras.regularizers import l2
from iras_cnn1 import build_model

obs_layers = input("Voir les couches [Y/N] : ")
radio_classes = ['AM', 'FM', 'BPSK', 'QPSK', 'QAM16']
model = build_model((128, 2), len(radio_classes))
model.summary() if obs_layers != 'N' else 0
# Simuler une entrée comme GNU Radio en produirait
# Shape : (batch_size, 128 samples, 2 canaux = I/Q)
fake_signal = np.random.randn(1, 128, 2).astype(np.float32)

predictions = model.predict(fake_signal)

#Display
print("-"*40)
[print(radio_classes[i]," = " + str(float(predictions[0][i]*100)) + "%", sep="\t") for i in range(len(predictions[0]))]
print("Classe prédite :", radio_classes[np.argmax(predictions)])
print("-"*40)
