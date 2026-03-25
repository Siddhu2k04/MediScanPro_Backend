from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)
CORS(app)

# ================= MODEL SETUP =================
MODEL_PATH = "ct_scan_model.h5"
FILE_ID = "1cSg_HR6ZjWY1IdBo_I0ouTJa1O06ZwSp"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ✅ UPDATE according to your training dataset
class_labels = [
    "Lung Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
]

print("CLASS LABELS:", class_labels)

# ================= ROUTES =================
@app.route("/")
def home():
    return "Backend Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Correct order
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # ================= IMAGE =================
        img = Image.open(file).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ================= PREDICTION =================
        pred = model.predict(img_array)

        print("RAW PRED:", pred)   # 🔥 IMPORTANT DEBUG

        # Handle both cases (binary / multiclass)
        if pred.shape[1] == 1:
            # Binary model
            value = pred[0][0]
            if value > 0.5:
                result = "Cancer"
                confidence = value
            else:
                result = "Normal"
                confidence = 1 - value
        else:
            # Multiclass model
            pred_class = int(np.argmax(pred))
            result = class_labels[pred_class]
            confidence = np.max(pred)

        confidence = round(float(confidence) * 100, 2)

        print("FINAL:", result, confidence)

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)