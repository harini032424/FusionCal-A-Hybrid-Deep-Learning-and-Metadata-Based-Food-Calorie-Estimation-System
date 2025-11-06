# models/train_rf.py
"""
Training script:
- Loads precomputed features from features_coll (GridFS bytes saved earlier)
- Joins with metadata (e.g., portion_size, categorical info)
- Trains a RandomForestRegressor
- Saves model to disk using joblib
"""
import numpy as np
import io, os, joblib
from sklearn.ensemble import RandomForestRegressor
from db.db_connection import features_coll, images_coll
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_PATH = os.getenv("RF_MODEL_PATH", "models/rf_model.pkl")

def load_feature_bytes(doc):
    bio = io.BytesIO(doc["feature_bytes"])
    bio.seek(0)
    arr = np.load(bio, allow_pickle=False)
    return arr

def prepare_dataset(limit=None):
    # join features with metadata documents
    X_feats = []
    X_meta = []
    y = []
    cursor = features_coll.find({}, limit=limit)
    for fdoc in cursor:
        fname = fdoc["image_filename"]
        meta = images_coll.find_one({"filename": fname})
        if not meta or "calories" not in meta:
            continue  # skip unlabeled
        feat = load_feature_bytes(fdoc)
        X_feats.append(feat)
        # example metadata features: portion_size (float), maybe category encoded
        # We'll just take portion_size as numeric; extend with one-hot enc for categories
        portion = float(meta.get("portion_size", 1.0))
        X_meta.append([portion])
        y.append(float(meta["calories"]))
    X = np.hstack([np.array(X_feats), np.array(X_meta)])
    return X, np.array(y)

def train_rf(limit=1000):
    X, y = prepare_dataset(limit=limit)
    # Split data 80/20 for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest model
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions and evaluate
    train_preds = rf.predict(X_train)
    test_preds = rf.predict(X_test)
    
    # Print metrics for both training and test sets
    print("\nModel Performance Metrics:")
    print("-" * 30)
    print("Training Set:")
    print(f"MAE: {mean_absolute_error(y_train, train_preds):.2f}")
    print(f"RMSE: {mean_squared_error(y_train, train_preds, squared=False):.2f}")
    print("\nTest Set:")
    print(f"MAE: {mean_absolute_error(y_test, test_preds):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, test_preds, squared=False):.2f}")
    
    # Save the model
    joblib.dump(rf, MODEL_PATH)
    print(f"\nSaved RF model to {MODEL_PATH}")
    return rf, (X_test, y_test)  # Return test set for further evaluation if needed

if __name__ == "__main__":
    train_rf()
