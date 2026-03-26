from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

# Load model lazily (IMPORTANT)
model = None

def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model(
            "ct_scan_model.h5",
            compile=False   # ✅ FIX
        )

# Class labels
class_labels = ["class1", "class2", "class3"]

@app.route("/")
def home():
    return "API is running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    load_model_once()  # ✅ load when needed

    file = request.files["file"]

    filepath = "temp.png"
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred_class = np.argmax(pred)

    result = class_labels[pred_class]
    confidence = float(np.max(pred))

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
