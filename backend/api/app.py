from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load your trained YOLOv8 model at cold start
model = YOLO('models/best.pt')

@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    # CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    return jsonify({'message': 'Server is running!'})

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    # CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)

    # run inference
    results = model(image_np, conf=0.3, iou=0.4)
    result = results[0]

    # draw boxes and convert BGR→RGB
    annotated = result.plot()
    annotated_rgb = Image.fromarray(annotated[..., ::-1])

    # buffer → base64
    buf = io.BytesIO()
    annotated_rgb.save(buf, format='JPEG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    num_boxes = len(result.boxes)
    return jsonify({
        'detection':     num_boxes > 0,
        'message':       'Fractures detected.' if num_boxes > 0 else 'No fractures detected.',
        'image_base64':  img_b64
    })

# Note: Vercel will invoke the Flask app directly; no need for an __main__ guard.
