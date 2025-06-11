import os
import cv2

# Input/output folders
input_dir = "4k_images"
output_dir = "img_variants"
os.makedirs(output_dir, exist_ok=True)

# Target resolutions
resolutions = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (640, 480)
}

# Process all images
for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(input_dir, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠️ Could not read {fname}")
        continue

    base, ext = os.path.splitext(fname)

    for label, size in resolutions.items():
        resized = cv2.resize(img, size)
        out_name = f"{base}_{label}{ext}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, resized)

    # Copy original as "photo_original.jpg"
    out_name = f"{base}_original{ext}"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, img)

print(f"✅ Resized images saved to: {output_dir}")
