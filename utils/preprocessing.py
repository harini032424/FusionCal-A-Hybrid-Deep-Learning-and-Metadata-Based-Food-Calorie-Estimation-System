# utils/preprocessing.py
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_and_preprocess(path_or_bytes, target=(224,224)):
    if isinstance(path_or_bytes, bytes):
        img = Image.open(io.BytesIO(path_or_bytes))
    else:
        img = Image.open(path_or_bytes)
    img = img.convert("RGB").resize(target)
    arr = np.array(img).astype(np.float32)
    return preprocess_input(arr)
