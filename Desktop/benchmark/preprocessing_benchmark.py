import os
import cv2
import time
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Define resolutions
RESOLUTIONS = {
    "original": None,
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160)
}

# Setup logger
logging.basicConfig(filename='preprocessing_log.txt',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

def preprocess_images(input_folder, resize="720p"):
    face_data = []
    resize_dim = RESOLUTIONS.get(resize, None)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)  # use -1 for CPU

    logging.info(f"Starting preprocessing with resolution: {resize}")
    print(f"\n🧠 Preprocessing images at {resize}...")

    for fname in tqdm(os.listdir(input_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_folder, fname)
        img = cv2.imread(path)
        if img is None:
            logging.warning(f"❌ Could not read {fname}")
            continue

        if resize_dim:
            img = cv2.resize(img, resize_dim)

        faces = app.get(img)
        for face in faces:
            entry = {
                "filename": fname,
                "bbox": face.bbox,
                "embedding": face.embedding
            }
            face_data.append(entry)

    return face_data

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Preprocessing Benchmark Tool")
    parser.add_argument("--input", required=True, help="Path to folder containing images")
    parser.add_argument("--output", default="processed_data.pkl", help="Output .pkl filename")
    parser.add_argument("--resize", default="720p", choices=RESOLUTIONS.keys(), help="Resize resolution")

    args = parser.parse_args()

    start_time = time.time()
    data = preprocess_images(args.input, resize=args.resize)

    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    end_time = time.time()
    duration = end_time - start_time

    # Summary
    print(f"\n✅ Done! Processed {len(data)} faces from '{args.input}'.")
    print(f"📁 Data saved to: {args.output}")
    print(f"🕒 Total time taken: {duration:.2f} seconds")

    logging.info(f"✅ Processed {len(data)} faces from '{args.input}' in {duration:.2f} seconds.")
    logging.info(f"Output saved to: {args.output}")
