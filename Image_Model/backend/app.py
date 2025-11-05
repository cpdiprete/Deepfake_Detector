import os
import io
import pickle
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model

BASE_D = os.path.dirname(os.path.abspath(__file__))
MODEL_D = os.path.join(BASE_D, os.pardir, "models", "saved_models")
DOCS_D  = os.path.join(BASE_D, os.pardir, "docs")


scaler = pickle.load(open(os.path.join(MODEL_D, "scaler.pkl"), "rb"))

pca_map = {
    v: pickle.load(open(os.path.join(MODEL_D, f"pca_{v}.pkl"), "rb"))
    for v in ("60", "95")
}

cnn_map = {
    v: load_model(os.path.join(MODEL_D, f"cnn_{v}.keras"), compile=False)
    for v in ("60", "95")
}

CLASS_NAMES = ["Real", "Fake"]

app = Flask(__name__, static_folder=DOCS_D, static_url_path="")

@app.route("/", methods=["GET"])
def index():
    return app.send_static_file("app.html")

def preprocess(img: Image.Image, var: str) -> np.ndarray:

    img = img.convert("L").resize((32, 32))
    arr = np.array(img).flatten().reshape(1, -1)

    return pca_map[var].transform(scaler.transform(arr))

@app.route("/predict", methods=["POST"])
def classify():
    image_file = request.files["image"]
    type    = request.form.get("type")

    img   = Image.open(io.BytesIO(image_file.read()))
    feat  = preprocess(img, type)
    probs = cnn_map[type].predict(feat)[0]
    idx   = np.argmax(probs)

    return jsonify(type=type, prediction=CLASS_NAMES[idx])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
