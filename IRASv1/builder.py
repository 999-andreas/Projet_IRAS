import numpy as np
from tensorflow.keras import layers, models, Input
from tensorflow.keras.regularizers import l2
from iras_cnn1 import build_model

obs_layers = input("Voir les couches [Y/N] : ")
radio_classes = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK',
    '8PSK', '16PSK', '32PSK', '16APSK', '32APSK',
    '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC',
    'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]
model = build_model((1024, 2), len(radio_classes))
model.load_weights("best_model.h5")

model.summary() if obs_layers != 'N' else 0

fake_signal = np.random.randn(1, 128, 2).astype(np.float32)
predictions = model.predict(fake_signal)

#Display
print("-"*40)
[print(radio_classes[i]," = " + str(float(predictions[0][i]*100)) + "%", sep="\t") for i in range(len(predictions[0]))]
print("Classe prédite :", radio_classes[np.argmax(predictions)])
print("-"*40)

