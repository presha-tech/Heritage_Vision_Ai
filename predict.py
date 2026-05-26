# predict.py

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

import sys

# =========================
# LOAD TFLITE MODEL
# =========================

interpreter = tf.lite.Interpreter(
    model_path="model/heritage_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

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
# LOAD IMAGE
# =========================

img = image.load_img(

    img_path,

    target_size=(224,224)

)

img_array = image.img_to_array(

    img

)

img_array = np.expand_dims(

    img_array,

    axis=0

)

img_array = preprocess_input(

    img_array

)

img_array = img_array.astype(

    np.float32

)

# =========================
# RUN TFLITE INFERENCE
# =========================

interpreter.set_tensor(

    input_details[0]["index"],

    img_array

)

interpreter.invoke()

predictions = interpreter.get_tensor(

    output_details[0]["index"]

)

# =========================
# GET RESULT
# =========================

predicted_index = int(

    np.argmax(predictions)

)

confidence = float(

    np.max(predictions)

) * 100

predicted_class = class_names[

    predicted_index

]

# =========================
# OUTPUT
# =========================

print(

f"{predicted_class}|{confidence:.2f}"

)