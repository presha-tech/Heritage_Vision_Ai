# app.py

import os
import sqlite3
import urllib.request

# ─────────────────────────────────────────────────────────────
# TENSORFLOW — import at module level so Gunicorn loads it
# during the boot phase (before any request timeout clock starts).
# Importing inside the request handler causes a WORKER TIMEOUT.
# ─────────────────────────────────────────────────────────────
print("Importing TensorFlow (this takes ~20s on CPU)...")
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
from werkzeug.utils import secure_filename
print("TensorFlow imported successfully.")

from flask import Flask, render_template, request

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# UPLOAD FOLDER
# ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MODEL DOWNLOAD  (runs once at startup, not per-request)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "model/best_resnet_model.h5"

# ─────────────────────────────────────────────────────────────
# LOAD MODEL  (also at startup — not inside a request handler)
# ─────────────────────────────────────────────────────────────
print("Loading model weights...")
_model = load_model(MODEL_PATH)
print("Model ready.")

# ─────────────────────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Charminar",
    "Gateway of India",
    "Qutub Minar",
    "Sun Temple Konark",
    "Taj Mahal",
]

# ─────────────────────────────────────────────────────────────
# PREDICTION  (fast — model already loaded)
# ─────────────────────────────────────────────────────────────
def predict_monument(filepath):
    img        = keras_image.load_img(filepath, target_size=(224, 224))
    arr        = keras_image.img_to_array(img)
    arr        = preprocess_input(arr[None, ...])
    preds      = _model.predict(arr, verbose=0)
    idx        = int(preds.argmax())
    confidence = float(preds.max()) * 100
    return CLASS_NAMES[idx], confidence


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return render_template("index.html", error="No image uploaded.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No image selected.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        monument, confidence = predict_monument(filepath)

        # Fetch monument info from DB
        db_path = os.path.join(os.path.dirname(__file__), "monuments.db")
        conn    = sqlite3.connect(db_path)
        cursor  = conn.cursor()
        cursor.execute(
            """
            SELECT history, dynasty, construction_period,
                   architecture, unesco_status, tourism_facts
            FROM monuments
            WHERE name = ?
            """,
            (monument,),
        )
        row = cursor.fetchone()
        conn.close()

        NA = "Information not available."
        history, dynasty, construction_period, architecture, unesco_status, tourism_facts = (
            row if row else (NA, NA, NA, NA, NA, NA)
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
            tourism_facts=tourism_facts,
        )

    except Exception as e:
        return f"ERROR: {str(e)}", 500


# ─────────────────────────────────────────────────────────────
# LOCAL DEV ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
