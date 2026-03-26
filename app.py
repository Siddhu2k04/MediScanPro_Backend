from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)
CORS(app)

# =========================
# MODEL SETUP
# =========================

MODEL_PATH = "ct_scan_model.h5"

# 🔴 IMPORTANT: Replace with your actual Google Drive FILE ID
FILE_ID = "YOUR_FILE_ID"
GDRIVE_URL = f"https://drive.google.com/uc?id=1cSg_HR6ZjWY1IdBo_I0ouTJa1O06ZwSp"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class labels (CHANGE THESE)
class_labels = ["class1", "class2", "class3"]

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "MediScan Backend Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Read image
        img = Image.open(file).convert("RGB")
        img = img.resize((128, 128))

        # Preprocess
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)
        pred_class = int(np.argmax(pred))

        result = class_labels[pred_class]
        confidence = float(np.max(pred))

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)