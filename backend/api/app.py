import os, io, base64, tempfile
import requests
import numpy as np
from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Where we’ll store the downloaded model on the function’s disk
MODEL_PATH = os.path.join(tempfile.gettempdir(), "best.pt")

def load_model():
    if not os.path.exists(MODEL_PATH):
        url = os.getenv("MODEL_URL")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)
    return YOLO(MODEL_PATH)

# Load once per cold-start
model = load_model()

@app.route("/", methods=["GET", "OPTIONS"])
def index():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    return jsonify({"message": "Server is running!"})

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files["image"].stream).convert("RGB")
    arr = np.array(img)

    results = model(arr, conf=0.3, iou=0.4)[0]
    annotated = results.plot()
    annotated_rgb = Image.fromarray(annotated[..., ::-1])

    buf = io.BytesIO()
    annotated_rgb.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    count = len(results.boxes)
    return jsonify({
        "detection":    count > 0,
        "message":      "Fractures detected." if count else "No fractures detected.",
        "image_base64": img_b64
    })

# Vercel will import and call `app` directly
