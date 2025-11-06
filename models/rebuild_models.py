"""
Rebuild and save clean versions of both models:
1. ResNet50 feature extractor (512-dim output)
2. Random Forest regressor (trained on dummy data)
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import joblib

def build_and_save_resnet(save_path):
    """Build a clean ResNet50 feature extractor that outputs 512-dim features."""
    # Base ResNet50 without top layers
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Input layer that includes preprocessing
    inp = tf.keras.layers.Input(shape=(None, None, 3), name="input_image")
    x = tf.keras.applications.resnet.preprocess_input(inp)
    
    # Get ResNet features
    x = base(x)
    
    # Project to 512 dimensions
    x = tf.keras.layers.Dense(512, activation="linear", name="proj_512")(x)
    
    # Create and save model
    model = tf.keras.Model(inputs=inp, outputs=x, name="resnet50_512")
    model.save(save_path, save_format='h5')
    return model

def generate_dummy_data(n_samples=600):
    """Generate dummy training data."""
    X = np.random.normal(size=(n_samples, 512)).astype(np.float32)
    y = (np.abs(X).sum(axis=1) * 2.0) + np.random.randn(n_samples) * 10 + 100
    return X, y.astype(np.float32)

def train_and_save_rf(save_path):
    """Train RF on dummy data and save it."""
    X, y = generate_dummy_data()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, save_path)
    return rf

def main():
    # Ensure models directory exists
    models_dir = os.path.dirname(__file__)
    os.makedirs(models_dir, exist_ok=True)
    
    # Paths for saved models
    resnet_path = os.path.join(models_dir, "resnet_feature_extractor.h5")
    rf_path = os.path.join(models_dir, "rf_model.pkl")
    
    print("Building and saving ResNet feature extractor...")
    build_and_save_resnet(resnet_path)
    print(f"ResNet saved to: {resnet_path}")
    
    print("\nTraining and saving Random Forest...")
    train_and_save_rf(rf_path)
    print(f"Random Forest saved to: {rf_path}")
    
    print("\nDone! Both models rebuilt and saved.")

if __name__ == "__main__":
    main()