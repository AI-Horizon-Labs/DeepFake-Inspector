import os
import json
import numpy as np
from tensorflow import keras
from PIL import Image

MODEL_PATH = "/workspace/models/patience-10/deepfake_detector_model.keras"
THRESHOLD_PATH = "/workspace/models/patience-10/best_threshold.json"

IMAGE_PATH = "/workspace/datasets/Dataset/Test/Real/real_7.jpg"
IMAGE_SIZE = (256, 256)

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as f:
        best_threshold = json.load(f)["best_threshold"]
else:
    best_threshold = 0.5  # fallback
print(f"Loaded best threshold: {best_threshold:.4f}")

def preprocess_image(image_path, target_size=(256, 256)):
    """Reads and prepares an image for the model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

print(f"Evaluating image: {IMAGE_PATH}")
img = preprocess_image(IMAGE_PATH, IMAGE_SIZE)
pred = model.predict(img, verbose=0)

score = float(pred[0][0])
label = "Deepfake" if score < best_threshold else "Real"
confidence = (1 - score) * 100 if label == "Deepfake" else score * 100

print("\n===== Result =====")
print(f"Imagem: {IMAGE_PATH}")
print(f"Score (sigmoid): {score:.4f}")
print(f"Threshold used: {best_threshold:.4f}")
print(f"Classification: {label}")
print(f"Confidence: {confidence:.2f}%")
print("=====================\n")
