# predict.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import sys
import os

# =========================
# LOAD MODEL
# =========================

model = load_model("model/best_resnet_model.h5")

# =========================
# CLASS LABELS
# =========================

class_names = [
    "Charminar",
    "Gateway of India",
    "Qutub Minar",
    "Sun Temple Konark",
    "Taj Mahal"
]

# =========================
# GET IMAGE PATH
# =========================

img_path = sys.argv[1]

# =========================
# LOAD AND PREPROCESS IMAGE
# =========================

img = image.load_img(img_path, target_size=(224, 224))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = preprocess_input(img_array)

# =========================
# PREDICT
# =========================

predictions = model.predict(img_array, verbose=0)
predicted_index = np.argmax(predictions)

confidence = float(np.max(predictions)) * 100

predicted_class = class_names[predicted_index]

# =========================
# OUTPUT
# =========================

print(f"{predicted_class}|{confidence:.2f}")