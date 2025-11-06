from pymongo import MongoClient, errors, ASCENDING, DESCENDING
import os
from datetime import datetime
import mongomock


def get_db_clients():
    """Attempt to connect to a real MongoDB; otherwise fall back to mongomock in-memory DB."""
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB_NAME = os.getenv("DB_NAME", "food_calorie_db")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command('ping')
        db = client[DB_NAME]
        
        # Create indexes for better query performance
        db.predictions.create_index([("timestamp", DESCENDING)])
        db.predictions.create_index([("category", ASCENDING)])
        db.features.create_index([("calories", ASCENDING)])
        
        print("✅ Connected to MongoDB successfully!")
        using_mongomock = False
        # Ensure canonical collection name `food_data` exists. If there is an
        # older `predictions` collection, copy its documents into `food_data` to
        # preserve existing records (non-destructive: only if `food_data` missing).
        try:
            existing = db.list_collection_names()
            if 'food_data' not in existing:
                if 'predictions' in existing:
                    # create food_data and copy docs
                    db.create_collection('food_data')
                    docs = list(db['predictions'].find({}))
                    if docs:
                        db['food_data'].insert_many(docs)
                else:
                    # just create an empty collection
                    db.create_collection('food_data')
        except Exception:
            # If collection creation fails (permissions), continue; app can still write later.
            pass

        return db, using_mongomock
    except Exception as e:
        print("""
⚠️ Could not connect to MongoDB, using in-memory mongomock.
To use a real MongoDB database:
1. Install MongoDB Server locally (recommended for development)
2. Use MongoDB Atlas (for production)
3. Continue using mongomock for testing

Error details: %s
""" % str(e))
        client = mongomock.MongoClient()
        db = client[DB_NAME]
        using_mongomock = True
        return db, using_mongomock


# initialize
db, _USING_MOCK = get_db_clients()

# collections
features_coll = db["features"]
images_coll = db["images"]
# Use `food_data` as the primary collection for storing prediction results (per BDA requirements)
predictions_coll = db["food_data"]


def store_image_file(b: bytes, filename: str = None):
    """Stores raw image bytes into images_coll. Returns the inserted id."""
    doc = {"data": b, "filename": filename}
    res = images_coll.insert_one(doc)
    return res.inserted_id


def get_image_file(doc_id):
    """Retrieve image data by document id."""
    d = images_coll.find_one({"_id": doc_id})
    if not d:
        return None
    return d.get("data")


def save_metadata_doc(doc: dict):
    """Save metadata about an image to the database."""
    return images_coll.insert_one(doc)


def save_feature(image_filename: str, feature_bytes: bytes, shape=None, dtype: str = None, calories: float = None):
    """Save extracted feature bytes along with metadata.

    This signature is compatible with `models/feature_extraction.batch_extract_and_store`
    which calls `save_feature(filename, bytes, shape, dtype)`.

    Stored document fields:
    - image_filename: original filename
    - feature_bytes: binary blob (np.save serialized)
    - shape: tuple describing feature shape
    - dtype: dtype string
    - calories: optional ground-truth calories
    """
    doc = {
        "image_filename": image_filename,
        "feature_bytes": feature_bytes,
        "shape": tuple(shape) if shape is not None else None,
        "dtype": dtype,
        "calories": float(calories) if calories is not None else None,
    }
    return features_coll.insert_one(doc)


def save_prediction_log(doc: dict):
    """Save prediction with timestamp and category."""
    if 'timestamp' not in doc:
        doc['timestamp'] = datetime.utcnow()
    
    # Add category based on calories
    if 'predicted_calories' in doc:
        calories = doc['predicted_calories']
        if calories < 200:
            doc['category'] = 'Low'
        elif calories < 400:
            doc['category'] = 'Medium'
        elif calories < 700:
            doc['category'] = 'High'
        else:
            doc['category'] = 'Very High'
    
    # Ensure minimal fields expected by the app/analytics
    # Normalize keys: predicted_calories, predicted_food (optional), filename
    if 'predicted_food' not in doc and 'predicted_label' in doc:
        doc['predicted_food'] = doc.get('predicted_label')

    if 'filename' not in doc and 'image_name' in doc:
        doc['filename'] = doc.get('image_name')

    return predictions_coll.insert_one(doc)

def get_analytics_data():
    """Get aggregated analytics data for dashboard."""
    pipeline = [
        {
            '$facet': {
                'total_predictions': [{'$count': 'count'}],
                'avg_calories': [{'$group': {'_id': None, 'avg': {'$avg': '$predicted_calories'}}}],
                'category_distribution': [
                    {'$group': {'_id': '$category', 'count': {'$sum': 1}}}
                ],
                'top_calories': [
                    {'$sort': {'predicted_calories': -1}},
                    {'$limit': 5}
                ],
                'lowest_calories': [
                    {'$sort': {'predicted_calories': 1}},
                    {'$limit': 1}
                ]
            }
        }
    ]
    
    return predictions_coll.aggregate(pipeline)


def get_recent_predictions(limit: int = 10):
    """Return recent prediction documents sorted by timestamp descending."""
    cursor = predictions_coll.find().sort("timestamp", DESCENDING).limit(limit)
    return list(cursor)


def get_image_by_filename(filename: str):
    """Retrieve raw image bytes by filename from images collection."""
    d = images_coll.find_one({"filename": filename})
    if not d:
        return None
    return d.get("data")

