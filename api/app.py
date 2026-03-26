from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

# ✅ NO direct loading here
model = None

def load_model_once():
    global model
    if model is None:
        print("Loading model...")  # debug log
        model = tf.keras.models.load_model(
            "ct_scan_model.h5",
            compile=False
        )

class_labels = ["class1", "class2", "class3"]

@app.route("/")
def home():
    return "API is running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    load_model_once()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = "temp.png"
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        pred_class = int(np.argmax(pred))

        return jsonify({
            "prediction": class_labels[pred_class],
            "confidence": float(np.max(pred))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
