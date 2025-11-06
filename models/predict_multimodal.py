import sys
import numpy as np
import tensorflow as tf
import joblib
from models.feature_extraction import extract_image_features

# Paths
MODEL_PATH = "food_calorie_multimodal.pkl"

# Check command-line arguments
if len(sys.argv) < 2:
    raise ValueError("Usage: python predict_multimodal.py <image_file> [weight protein carbs fats fiber sugars sodium]")

# Image path
image_file = sys.argv[1]

# Optional metadata inputs
# Default values if not provided
default_meta = [100, 3, 36, 10, 2, 16, 120]  # Example: weight, protein, carbs, fats, fiber, sugars, sodium
meta_input = sys.argv[2:]  # List of strings

if meta_input:
    if len(meta_input) != len(default_meta):
        raise ValueError(f"Please provide {len(default_meta)} metadata values or none for defaults.")
    meta_values = np.array([float(x) for x in meta_input]).reshape(1, -1)
else:
    meta_values = np.array(default_meta).reshape(1, -1)

# Load model
model = joblib.load(MODEL_PATH)

# Extract image features
img_features = extract_image_features(image_file).reshape(1, -1)

# Combine image + metadata features
X = np.hstack((img_features, meta_values))

# Predict
calories = model.predict(X)
print(f"Estimated Calories: {calories[0]:.2f} kcal")
