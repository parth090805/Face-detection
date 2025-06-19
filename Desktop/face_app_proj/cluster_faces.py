import os
import time
import pickle
import json
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from insightface.app import FaceAnalysis
from tqdm import tqdm

# Step 1: Load previously saved embeddings
with open("processed_data.pkl", "rb") as f:
    face_data = pickle.load(f)

print(f"Loaded {len(face_data)} face entries.")

# Step 2: Extract all embeddings
embeddings = np.array([entry["embedding"] for entry in face_data])

# Step 3: Apply DBSCAN clustering (groups similar faces)
clustering = DBSCAN(eps=0.45, min_samples=1, metric='cosine')
labels = clustering.fit_predict(embeddings)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"✅ Found {n_clusters} unique people (clusters)")

# Step 4: Group data by cluster label
clusters = defaultdict(list)
for entry, label in zip(face_data, labels):
    clusters[label].append(entry)

# Step 5: Save photo filenames per person (cluster)
person_photos = defaultdict(set)

for label, faces in clusters.items():
    for face in faces:
        person_photos[label].add(face["filename"])

# Save this data for later UI use
with open("person_to_photos.json", "w") as f:
    json.dump({str(k): list(v) for k, v in person_photos.items()}, f, indent=2)

print("✅ Saved person-to-photo mapping as person_to_photos.json")

# Step 6: Save one cropped face per person for UI thumbnails
print("📸 Saving thumbnails...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

thumbnail_dir = "thumbnails"
os.makedirs(thumbnail_dir, exist_ok=True)

skipped = 0

for label, faces in tqdm(clusters.items()):
    face = faces[0]  # pick first face of the person
    img_path = os.path.join("downloaded_images", face["filename"])
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Could not load image: {face['filename']}")
        skipped += 1
        continue

    x1, y1, x2, y2 = map(int, face["bbox"])

    # Clip coordinates to image boundaries
    x1 = max(0, min(x1, img.shape[1]))
    x2 = max(0, min(x2, img.shape[1]))
    y1 = max(0, min(y1, img.shape[0]))
    y2 = max(0, min(y2, img.shape[0]))

    if x2 <= x1 or y2 <= y1:
        print(f"⚠️ Invalid bbox for {face['filename']} -> Skipping")
        skipped += 1
        continue

    face_crop = img[y1:y2, x1:x2]

    if face_crop is None or face_crop.size == 0:
        print(f"⚠️ Empty face crop for {face['filename']} -> Skipping")
        skipped += 1
        continue

    save_path = os.path.join(thumbnail_dir, f"person_{label}.jpg")
    cv2.imwrite(save_path, face_crop)

print(f"✅ Thumbnails saved in 'thumbnails/' folder")
print(f"❗ Skipped {skipped} problematic faces during thumbnail generation")

# Step 7: Save one representative embedding per cluster (for face search)
print("📦 Saving cluster representative embeddings...")

cluster_reps = {}

for label, faces in clusters.items():
    # Use first face's embedding as representative
    rep_embedding = faces[0]["embedding"]
    cluster_reps[label] = rep_embedding

with open("cluster_representatives.pkl", "wb") as f:
    pickle.dump(cluster_reps, f)

print("✅ Saved cluster representatives to 'cluster_representatives.pkl'")
