import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

import tensorflow as tf

from pymongo import MongoClient, errors

def get_features_collection():
    """Try to connect to a real MongoDB on localhost:27017; if that fails, use mongomock in-memory DB."""
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # attempt to force connection
        client.admin.command('ping')
        db = client["food_calorie_db"]
        print("✅ Connected to MongoDB successfully!")
        return db["features"]
    except Exception as e:
        print("Could not connect to MongoDB (will use in-memory mongomock). Error:", e)
        try:
            import mongomock
        except Exception:
            raise RuntimeError("mongomock not installed — please install it in the venv to proceed or start mongod locally")
        mclient = mongomock.MongoClient()
        mdb = mclient["food_calorie_db"]
        return mdb["features"]

# features_coll will be created at runtime
features_coll = get_features_collection()


def ensure_models_dir():
    models_dir = os.path.join(os.path.dirname(__file__))
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    return models_dir


def generate_dummy_features(n_samples=500, dim=512):
    # Features: random floats, labels: calories between 50 and 1000
    X = np.random.normal(size=(n_samples, dim)).astype(np.float32)
    # Create a synthetic relationship: sum of features scaled + noise
    y = (np.abs(X).sum(axis=1) * 2.0) + np.random.randn(n_samples) * 10 + 100
    y = y.astype(np.float32)
    return X, y


def store_dummy_to_mongo(n_samples=500, dim=512):
    docs = []
    X, y = generate_dummy_features(n_samples=n_samples, dim=dim)
    for i in range(n_samples):
        docs.append({
            "feature": X[i].tolist(),
            "calories": float(y[i])
        })
    if docs:
        features_coll.insert_many(docs)


def load_features_from_mongo(limit=None):
    cursor = features_coll.find({}, {"feature": 1, "calories": 1})
    if limit:
        cursor = cursor.limit(limit)
    feats = []
    labels = []
    for d in cursor:
        feats.append(d.get("feature"))
        labels.append(d.get("calories"))
    if not feats:
        return None, None
    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y


def build_and_save_resnet(models_dir, out_path=None):
    # Build ResNet50 (pretrained) and add a Dense projection to 512 dims
    base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg")
    inp = tf.keras.Input(shape=(None, None, 3), name="input_image")
    x = tf.keras.applications.resnet.preprocess_input(inp)
    x = base(x)
    x = tf.keras.layers.Dense(512, activation="linear", name="proj_512")(x)
    model = tf.keras.Model(inputs=inp, outputs=x, name="resnet50_512")
    if out_path is None:
        out_path = os.path.join(models_dir, "resnet_feature_extractor.h5")
    model.save(out_path)
    return out_path


def train_and_save_rf(X, y, models_dir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    out_path = os.path.join(models_dir, "rf_model.pkl")
    joblib.dump(rf, out_path)
    return out_path, r2, mae


def main():
    print("-- Preparing models directory")
    models_dir = ensure_models_dir()

    # Check Mongo collection
    try:
        count = features_coll.count_documents({})
    except Exception as e:
        print("Error accessing MongoDB features collection:", e)
        return

    print(f"Features docs in MongoDB: {count}")
    if count == 0:
        print("No features found in MongoDB — generating dummy data and inserting...")
        store_dummy_to_mongo(n_samples=600, dim=512)
        print("Inserted dummy feature documents into MongoDB")

    X, y = load_features_from_mongo()
    if X is None:
        print("Failed to load features from MongoDB — aborting")
        return

    print(f"Loaded features shape: {X.shape}, labels shape: {y.shape}")

    print("Training RandomForestRegressor...")
    rf_path, r2, mae = train_and_save_rf(X, y, models_dir)
    print(f"Model saved to: {rf_path}")
    print(f"RandomForest R2: {r2:.4f}, MAE: {mae:.4f}")

    resnet_path = os.path.join(models_dir, "resnet_feature_extractor.h5")
    if not os.path.exists(resnet_path):
        print("Building and saving ResNet-based 512-dim extractor (this may take a while)...")
        try:
            out = build_and_save_resnet(models_dir, resnet_path)
            print(f"ResNet feature extractor saved to: {out}")
        except Exception as e:
            print("Failed to build/save ResNet extractor:", e)
    else:
        print(f"ResNet extractor already exists at: {resnet_path}")


if __name__ == "__main__":
    main()
