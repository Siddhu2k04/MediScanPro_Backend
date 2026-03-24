from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("ct_scan_model.h5")

# Class labels (IMPORTANT)
class_labels = ["class1", "class2", "class3"]  # replace with your actual classes

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # Save image temporarily
    filepath = "temp.png"
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)

    result = class_labels[pred_class]
    confidence = float(np.max(pred))

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })

 if __name__ == "__main__":
    app.run()
