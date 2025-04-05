import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, AutoImageProcessor
from flask import Flask, request, jsonify
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__)  # Fixed _name_ to __name__

# Load model and processor
model_path = "brain_tumor_final_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model or processor: {str(e)}")
    raise

# Class labels
class_labels = ["glioma", "meningioma", "notumor", "pituitary"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        
        # Validate file
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Process image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()

        return jsonify({
            "predicted_class": class_labels[pred_class],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':  # Fixed _name_ and _main_ to __name__ and __main__
    # Added port configuration as environment variable with default
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)