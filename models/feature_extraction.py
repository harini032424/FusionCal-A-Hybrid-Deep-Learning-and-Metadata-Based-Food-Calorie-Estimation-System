# models/feature_extractor.py
import numpy as np
import io
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from db.db_connection import store_image_file, save_feature, images_coll, fs, get_image_file
from PIL import Image
import numpy as np

# create model once
_base = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # global average pooling
model = _base  # ready to call model.predict on preprocessed batch (shape (N,224,224,3))


def preprocess_pil(img_pil, target_size=(224,224)):
    img = img_pil.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32)
    return preprocess_input(arr)


def extract_features_from_bytes(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    arr = preprocess_pil(img)
    return arr  # not batched


def batch_extract_and_store(filename_to_bytes_iter, batch_size=32):
    """
    filename_to_bytes_iter: iterable of (filename, bytes)
    For each batch: extract features, np.save to bytes, store in features collection
    """
    import numpy as np, io
    batch_names = []
    batch_imgs = []
    for fname, b in filename_to_bytes_iter:
        batch_names.append(fname)
        batch_imgs.append(extract_features_from_bytes(b))
        if len(batch_names) == batch_size:
            arr = np.stack(batch_imgs, axis=0)
            feats = model.predict(arr, verbose=0)
            # save each feature individually to DB (serialized)
            for n, f in zip(batch_names, feats):
                bio = io.BytesIO()
                np.save(bio, f, allow_pickle=False)
                bio.seek(0)
                save_feature(n, bio.read(), f.shape, str(f.dtype))
            batch_names = []
            batch_imgs = []
    # tail
    if batch_names:
        arr = np.stack(batch_imgs, axis=0)
        feats = model.predict(arr, verbose=0)
        for n, f in zip(batch_names, feats):
            bio = io.BytesIO()
            np.save(bio, f, allow_pickle=False)
            bio.seek(0)
            save_feature(n, bio.read(), f.shape, str(f.dtype))
