# app.py

import os
import sqlite3

# ─────────────────────────────────────────────────────────────
# TENSORFLOW ENV LIMITS
# MUST BE BEFORE TENSORFLOW IMPORT
# ─────────────────────────────────────────────────────────────

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─────────────────────────────────────────────────────────────
# TENSORFLOW IMPORT
# ─────────────────────────────────────────────────────────────

print("STEP 0: Importing TensorFlow...")

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request

print("STEP 0 COMPLETE: TensorFlow imported")

# Extra thread limiting

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# UPLOADS
# ─────────────────────────────────────────────────────────────

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

print("STEP 1: Upload folder ready")

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

MODEL_PATH = "model/best_resnet_model.h5"

print("STEP 2: Loading model...")

_model = load_model(MODEL_PATH)

print("STEP 2 COMPLETE: Model loaded")

# Warm up model once

print("STEP 2.5: Warmup inference")

_dummy = np.zeros(
    (1,224,224,3),
    dtype=np.float32
)

_dummy = preprocess_input(_dummy)

_ = _model(
    _dummy,
    training=False
)

print("STEP 2.5 COMPLETE")

# ─────────────────────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = [

    "Charminar",

    "Gateway of India",

    "Qutub Minar",

    "Sun Temple Konark",

    "Taj Mahal"

]

print("STEP 3: Labels loaded")

# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_monument(filepath):

    print(
        "STEP P1: Loading image",
        flush=True
    )

    img = keras_image.load_img(
        filepath,
        target_size=(224,224)
    )

    print(
        "STEP P2: Converting",
        flush=True
    )

    arr = keras_image.img_to_array(
        img
    )

    print(
        "STEP P3: Expanding",
        flush=True
    )

    arr = np.expand_dims(
        arr,
        axis=0
    )

    print(
        "STEP P4: Preprocessing",
        flush=True
    )

    arr = preprocess_input(
        arr
    )

    print(
        "STEP P5: Running inference",
        flush=True
    )

    preds = _model(
        arr,
        training=False
    ).numpy()

    print(
        "STEP P6: Inference complete",
        flush=True
    )

    idx = int(
        np.argmax(preds)
    )

    confidence = float(
        np.max(preds)
    ) * 100

    monument = CLASS_NAMES[idx]

    print(
        f"STEP P7: {monument} {confidence:.2f}",
        flush=True
    )

    return monument,confidence

# ─────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────

@app.route("/")
def home():

    print(
        "HOME PAGE OPENED",
        flush=True
    )

    return render_template(
        "index.html"
    )

# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    print(
        "STEP 4: Predict route entered",
        flush=True
    )

    try:

        if "image" not in request.files:

            print(
                "NO IMAGE",
                flush=True
            )

            return render_template(
                "index.html",
                error="No image uploaded."
            )

        file = request.files["image"]

        print(
            "STEP 5: Image received",
            flush=True
        )

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

        print(
            f"STEP 6: Saving {filepath}",
            flush=True
        )

        file.save(filepath)

        print(
            "STEP 7: File saved",
            flush=True
        )

        monument,confidence = \
        predict_monument(filepath)

        print(
            "STEP 8: Prediction done",
            flush=True
        )

        db_path = os.path.join(
            os.path.dirname(__file__),
            "monuments.db"
        )

        print(
            "STEP 9: DB open",
            flush=True
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

        WHERE name=?

        """,
        (monument,)
        )

        row = cursor.fetchone()

        conn.close()

        print(
            "STEP 10: DB fetched",
            flush=True
        )

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

        print(
            "STEP 11: Rendering",
            flush=True
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

        print(
            f"ERROR: {str(e)}",
            flush=True
        )

        return str(e),500

# ─────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────

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