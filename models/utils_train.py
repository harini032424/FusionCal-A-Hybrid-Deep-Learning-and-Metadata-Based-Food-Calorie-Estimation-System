# models/utils_train.py
import io, numpy as np
from db.db_connection import features_coll, images_coll

def iter_features_with_meta(limit=None):
    cursor = features_coll.find({}, limit=limit)
    for fdoc in cursor:
        fname = fdoc["image_filename"]
        meta = images_coll.find_one({"filename": fname})
        if not meta:
            continue
        bio = io.BytesIO(fdoc["feature_bytes"])
        bio.seek(0)
        feat = np.load(bio, allow_pickle=False)
        yield fname, feat, meta
