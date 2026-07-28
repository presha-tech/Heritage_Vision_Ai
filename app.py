# app.py

import os
import sqlite3

# ─────────────────────────────────────────────────
# TENSORFLOW ENV SETTINGS
# ─────────────────────────────────────────────────

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.preprocessing import image as keras_image
# from tensorflow.keras.applications.resnet50 import preprocess_input

from werkzeug.utils import secure_filename
from flask import Flask, render_template, request

# ─────────────────────────────────────────────────
# GEMINI (DIRECT SDK) SETTINGS
# ─────────────────────────────────────────────────
# The Flask app calls the Gemini API directly using google-genai.
# The API key is read from the GEMINI_API_KEY environment variable —
# it is never hardcoded in this file.

import base64
import json
import mimetypes
import re

from google import genai

try:
    # Loads variables from a local .env file into os.environ.
    # In production (Render, etc.) GEMINI_API_KEY is normally set directly
    # as a platform environment variable, so this is a no-op there.
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Reduce memory usage on Render

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

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

# interpreter = tf.lite.Interpreter(
#     model_path="model/heritage_model.tflite"
# )

# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()

# output_details = interpreter.get_output_details()

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

# def predict_monument(filepath):
#
#     img = keras_image.load_img(
#         filepath,
#         target_size=(224,224)
#     )
#
#     arr = keras_image.img_to_array(
#         img
#     )
#
#     arr = np.expand_dims(
#         arr,
#         axis=0
#     )
#
#     arr = preprocess_input(
#         arr
#     )
#
#     arr = arr.astype(
#         np.float32
#     )
#
#     interpreter.set_tensor(
#
#         input_details[0]["index"],
#
#         arr
#
#     )
#
#     interpreter.invoke()
#
#     preds = interpreter.get_tensor(
#
#         output_details[0]["index"]
#
#     )
#
#     idx = int(
#         np.argmax(preds)
#     )
#
#     confidence = float(
#         np.max(preds)
#     ) * 100
#
#     monument = CLASS_NAMES[idx]
#
#     return monument, confidence


GEMINI_MODEL = "gemini-3.5-flash-lite"


def get_gemini_model():
    """
    Fixed to gemini-3.5-flash-lite.
    """
    return GEMINI_MODEL


def predict_monument(filepath):

    if _client is None:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    with open(filepath, "rb") as f:
        image_bytes = f.read()

    mime_type = mimetypes.guess_type(filepath)[0] or "image/jpeg"

    monument_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(CLASS_NAMES))

    system_prompt = (
        "You are a monument classification system.\n"
        "The image can only belong to one of these five Indian monuments:\n"
        f"{monument_list}\n"
        "Analyze the image carefully.\n"
        "Return ONLY valid JSON, with no markdown formatting and no extra "
        "text, in exactly this shape:\n"
        "{\n"
        '  "monument": "one of the five names exactly as listed above",\n'
        '  "confidence": <number from 0 to 100, representing percent certainty>,\n'
        '  "reason": "short visual explanation"\n'
        "}"
    )

    model_name = get_gemini_model()

    response = _client.models.generate_content(
        model=model_name,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": system_prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_bytes,
                        }
                    },
                ],
            }
        ],
        config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    text = (response.text or "").strip()

    # Defensive parsing in case the model wraps the JSON in markdown
    # fences despite the response_mime_type constraint.
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except Exception:
        data = {}

    monument = data.get("monument", "Unknown")

    # Normalize to the canonical class name (case/whitespace tolerant)
    for name in CLASS_NAMES:
        if name.lower() == str(monument).strip().lower():
            monument = name
            break

    confidence = float(data.get("confidence", 0) or 0)

    reason = data.get("reason", "")

    return monument, confidence, reason


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

        monument, confidence, ai_reason = \
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

            tourism_facts=tourism_facts,

            ai_reason=ai_reason

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