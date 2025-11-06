# data_ingest/ingest_to_mongo.py
import os, io
from db.db_connection import store_image_file, save_metadata_doc
from tqdm import tqdm

def ingest_from_folder(root_dir, cal_label_map=None):
    """
    root_dir: folder structure root_dir/<label>/*.jpg
    cal_label_map: optional mapping label -> typical_calorie (float) for seed labels
    """
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in tqdm(os.listdir(label_dir), desc=label):
            p = os.path.join(label_dir, fname)
            with open(p, "rb") as f:
                b = f.read()
            gfid = store_image_file(b, filename=f"{label}/{fname}")
            doc = {
                "filename": f"{label}/{fname}",
                "original_path": p,
                "label": label,
                "gridfs_id": gfid,
                # default placeholders — you should fill real portion_size and calories if available
                "portion_size": 1.0,
                "calories": float(cal_label_map.get(label, 0.0)) if cal_label_map else None
            }
            save_metadata_doc(doc)

if __name__ == "__main__":
    ingest_from_folder("/path/to/food-101/images")
