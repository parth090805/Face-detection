# import os
# import cv2
# import pickle
# import numpy as np
# from tqdm import tqdm
# from insightface.app import FaceAnalysis

# # Configuration
# IMAGE_DIR = "downloaded_images"
# OUTPUT_FILE = "processed_data.pkl"

# # Initialize InsightFace
# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=0)  # Set to -1 if you want CPU-only

# # Helper function to extract all faces from all images
# def preprocess_images(image_dir):
#     face_data = []  # Will hold dicts with filename, bbox, embedding

#     for filename in tqdm(os.listdir(image_dir)):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue

#         image_path = os.path.join(image_dir, filename)
#         img = cv2.imread(image_path)

#         if img is None:
#             print(f"Failed to read {filename}")
#             continue

#         faces = app.get(img)

#         for face in faces:
#             entry = {
#                 "filename": filename,
#                 "bbox": face.bbox,  # [x1, y1, x2, y2]
#                 "embedding": face.embedding
#             }
#             face_data.append(entry)

#     return face_data

# # Run preprocessing and save
# if __name__ == "__main__":
#     print("Starting face preprocessing...")
#     data = preprocess_images(IMAGE_DIR)
    
#     with open(OUTPUT_FILE, "wb") as f:
#         pickle.dump(data, f)

#     print(f"Done! Processed {len(data)} faces from {IMAGE_DIR}.")
#     print(f"Saved to {OUTPUT_FILE}")

# import os
# import cv2
# import pickle
# import numpy as np
# from tqdm import tqdm
# from insightface.app import FaceAnalysis

# # Configuration
# IMAGE_DIR = "downloaded_images"
# OUTPUT_FILE = "processed_data.pkl"

# # Initialize InsightFace
# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=0)  # Set to -1 if you want CPU-only

# # Helper function to extract all faces from all images in subfolders
# def preprocess_images(image_dir):
#     face_data = []  # Will hold dicts with filename, bbox, embedding

#     for root, _, files in os.walk(image_dir):
#         for filename in files:
#             if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue

#             image_path = os.path.join(root, filename)
#             img = cv2.imread(image_path)

#             if img is None:
#                 print(f"Failed to read {image_path}")
#                 continue

#             faces = app.get(img)

#             for face in faces:
#                 entry = {
#                     "filename": filename,        # Same as original script (no folder path)
#                     "bbox": face.bbox,
#                     "embedding": face.embedding
#                 }
#                 face_data.append(entry)

#     return face_data

# # Run preprocessing and save
# if __name__ == "__main__":
#     print("Starting face preprocessing...")
#     data = preprocess_images(IMAGE_DIR)

#     with open(OUTPUT_FILE, "wb") as f:
#         pickle.dump(data, f)

#     print(f"Done! Processed {len(data)} faces from {IMAGE_DIR}.")
#     print(f"Saved to {OUTPUT_FILE}")


import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Configuration
IMAGE_DIR = "downloaded_images"
OUTPUT_FILE = "processed_data.pkl"

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # Use -1 for CPU-only

# Helper function to extract all faces from all images in subfolders
def preprocess_images(image_dir):
    face_data = []

    # Gather all image paths recursively
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, filename))

    print(f"Found {len(image_paths)} images to process.")

    # Process each image with progress bar
    for image_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.basename(image_path)
        img = cv2.imread(image_path)

        if img is None:
            print(f"❌ Failed to read: {image_path}")
            continue

        faces = app.get(img)
        for face in faces:
            entry = {
                "filename": filename,        # Same structure as original
                "bbox": face.bbox,
                "embedding": face.embedding
            }
            face_data.append(entry)

    return face_data

# Run preprocessing and save
if __name__ == "__main__":
    print("Starting face preprocessing...")
    data = preprocess_images(IMAGE_DIR)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Done! Processed {len(data)} faces from {IMAGE_DIR}.")
    print(f"📁 Saved to {OUTPUT_FILE}")
