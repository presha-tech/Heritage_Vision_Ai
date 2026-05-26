# app.py

import os
import sqlite3

# ─────────────────────────────────────────────────
# TENSORFLOW ENV SETTINGS
# ─────────────────────────────────────────────────

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request

# Reduce memory usage on Render

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ─────────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────────

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# ─────────────────────────────────────────────────
# LOAD TFLITE MODEL
# ─────────────────────────────────────────────────

interpreter = tf.lite.Interpreter(
    model_path="model/heritage_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

# ─────────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────────

CLASS_NAMES = [

    "Charminar",

    "Gateway of India",

    "Qutub Minar",

    "Sun Temple Konark",

    "Taj Mahal"

]

# ─────────────────────────────────────────────────
# MONUMENT PREDICTION
# ─────────────────────────────────────────────────

def predict_monument(filepath):

    img = keras_image.load_img(
        filepath,
        target_size=(224,224)
    )

    arr = keras_image.img_to_array(
        img
    )

    arr = np.expand_dims(
        arr,
        axis=0
    )

    arr = preprocess_input(
        arr
    )

    arr = arr.astype(
        np.float32
    )

    interpreter.set_tensor(

        input_details[0]["index"],

        arr

    )

    interpreter.invoke()

    preds = interpreter.get_tensor(

        output_details[0]["index"]

    )

    idx = int(
        np.argmax(preds)
    )

    confidence = float(
        np.max(preds)
    ) * 100

    monument = CLASS_NAMES[idx]

    return monument, confidence


# ─────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if "image" not in request.files:

            return render_template(

                "index.html",

                error="No image uploaded."

            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(

                "index.html",

                error="No image selected."

            )

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            filename

        )

        file.save(filepath)

        monument, confidence = \
        predict_monument(filepath)

        db_path = os.path.join(

            os.path.dirname(__file__),

            "monuments.db"

        )

        conn = sqlite3.connect(
            db_path
        )

        cursor = conn.cursor()

        cursor.execute(
        """
        SELECT

        history,
        dynasty,
        construction_period,
        architecture,
        unesco_status,
        tourism_facts

        FROM monuments

        WHERE name = ?

        """,
        (monument,)
        )

        row = cursor.fetchone()

        conn.close()

        NA = "Information not available."

        history,\
        dynasty,\
        construction_period,\
        architecture,\
        unesco_status,\
        tourism_facts = (

        row

        if row

        else

        (NA,NA,NA,NA,NA,NA)

        )

        return render_template(

            "result.html",

            monument=monument,

            confidence=f"{confidence:.2f}",

            image_path=filepath,

            history=history,

            dynasty=dynasty,

            construction_period=construction_period,

            architecture=architecture,

            unesco_status=unesco_status,

            tourism_facts=tourism_facts

        )

    except Exception as e:

        return str(e),500


# ─────────────────────────────────────────────────
# LOCAL ENTRY
# ─────────────────────────────────────────────────

if __name__=="__main__":

    port = int(

        os.environ.get(

            "PORT",

            5000

        )

    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )