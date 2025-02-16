from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
from flask_cors import CORS
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)  # Enable CORS to handle requests from different origins

# Limit GPU Memory Usage
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load Trained Model
MODEL_PATH = r"C:\Users\ashta\AI Doctor\lung_cancer_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define Class Labels
CLASS_NAMES = ["Cancerous", "Non-Cancerous"]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read Image File Directly into Memory
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))  # Resize for Model
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch

        # Make Prediction
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[1] if prediction[0][0] < 0.5 else CLASS_NAMES[0]

        return jsonify({
            "prediction": predicted_class,
            "confidence": float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run Flask App on Custom Host & Port
    public_url = ngrok.connect(5000).public_url  # Expose with ngrok
    print(f"Public URL: {public_url}")
    app.run(host="0.0.0.0", port=5000, debug=True)
