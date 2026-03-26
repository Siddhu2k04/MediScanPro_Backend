from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

# ✅ DO NOT load model here
model = None

def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model(
            "ct_scan_model.h5",
            compile=False
        )

# ✅ Your class labels (change if needed)
class_labels = ["class1", "class2", "class3"]

# ✅ Home route (for testing)
@app.route("/")
def home():
    return "API is running ✅"

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    load_model_once()  # ✅ load model here

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save temporary file
    filepath = "temp.png"
    file.save(filepath)

    try:
        # Preprocess image
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
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

    finally:
        # Clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)

# ✅ Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
