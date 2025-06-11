import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ---------- SETUP ----------
input_folder = "img_variants"   # Folder with all resolution variants
output_csv = "focused_benchmark_log.csv"

# Init face detection model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # use -1 for CPU

def get_image_size_mb(path):
    return round(os.path.getsize(path) / (1024 * 1024), 2)

# ---------- PROCESS ----------
results = []

print(f"🚀 Benchmarking resolution variants in: {input_folder}")
for fname in tqdm(os.listdir(input_folder)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(input_folder, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    # Detect resolution label from filename (e.g., "_720p" in "pic_720p.jpg")
    resolution = "unknown"
    for res in ["4k", "1080p", "720p", "480p", "original"]:
        if f"_{res}" in fname.lower():
            resolution = res
            break

    img_mb = get_image_size_mb(path)
    h, w = img.shape[:2]

    start_time = time.time()
    faces = app.get(img)
    duration = round(time.time() - start_time, 4)

    results.append({
        "filename": fname,
        "resolution": resolution,
        "dimensions": f"{w}x{h}",
        "image_MB": img_mb,
        "faces_detected": len(faces),
        "time_sec": duration
    })

# ---------- SAVE RESULTS ----------
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

# ---------- APPEND AVERAGE TIME ----------
avg_times = (
    df.groupby("resolution")["time_sec"]
    .mean()
    .reset_index()
    .rename(columns={"time_sec": "avg_time_sec"})
)

with open(output_csv, "a") as f:
    f.write("\n\n# Average time per resolution\n")
    avg_times.to_csv(f, index=False)

print(f"\n✅ Done! Logged {len(df)} entries")
print(f"📁 Results saved to: {output_csv}")
