from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        print("File received:", file.filename)

        return jsonify({"prediction": "Pneumonia"})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == "__main__":
    app.run(debug=True)