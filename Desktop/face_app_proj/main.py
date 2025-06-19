# from fastapi import FastAPI
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import uvicorn
# import pickle
# import json
# import os
# import numpy as np
# import cv2
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------- Load face analysis model ----------
# face_app = FaceAnalysis(name="buffalo_l")
# face_app.prepare(ctx_id=0)  # use -1 for CPU

# # ---------- Load clustering data ----------
# with open("cluster_representatives.pkl", "rb") as f:
#     cluster_embeddings = pickle.load(f)

# with open("person_to_photos.json", "r") as f:
#     person_to_photos = json.load(f)

# # ---------- Load 480p → 4K Drive mapping ----------
# with open("drive_file_map.json", "r") as f:
#     filename_to_drive_id = json.load(f)

# def get_drive_url(filename):
#     file_id = filename_to_drive_id.get(filename)
#     if file_id:
#         return f"https://drive.google.com/file/d/{file_id}/view"
#     return None

# # ---------- FastAPI App ----------
# app = FastAPI(title="Face Photo Match API")

# class MatchRequest(BaseModel):
#     request_id: str
#     path_face: str  # local path to uploaded face image
#     name: str
#     phone_number: int


# @app.post("/api/v1/get-similar-photos")
# async def get_similar_photos(request: MatchRequest):
#     image_path = request.path_face

#     if not os.path.exists(image_path):
#         return JSONResponse(status_code=400, content={
#             "error": f"File '{image_path}' does not exist.",
#             "request_id": request.request_id
#         })

#     img = cv2.imread(image_path)
#     if img is None:
#         return JSONResponse(status_code=400, content={
#             "error": "Could not read image or unsupported format.",
#             "request_id": request.request_id
#         })

#     faces = face_app.get(img)
#     if not faces:
#         return JSONResponse(status_code=400, content={
#             "error": "No face detected in the image.",
#             "request_id": request.request_id
#         })

#     uploaded_embedding = faces[0].embedding

#     # Compute similarity with each cluster representative
#     similarities = {
#         str(pid): cosine_similarity([uploaded_embedding], [emb])[0][0]
#         for pid, emb in cluster_embeddings.items()
#     }

#     best_match_id = max(similarities, key=similarities.get)
#     image_paths = person_to_photos.get(best_match_id, [])

#     # Map to Google Drive 4K URLs
#     image_urls_4k = []
#     for path in image_paths:
#         fname = os.path.basename(path)
#         drive_url = get_drive_url(fname)
#         if drive_url:
#             image_urls_4k.append(drive_url)

#     return {
#         "request_id": request.request_id,
#         "face_id": best_match_id,
#         "imagePaths": image_urls_4k
#     }


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import json
import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Initialize Face Analysis ----------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)  # use -1 for CPU, 0 for GPU

# ---------- Load Clustering and Mapping Data ----------
with open("cluster_representatives.pkl", "rb") as f:
    cluster_embeddings = pickle.load(f)

with open("person_to_photos.json", "r") as f:
    person_to_photos = json.load(f)

with open("drive_file_map.json", "r") as f:
    filename_to_drive_id = json.load(f)

def get_drive_url(filename):
    file_id = filename_to_drive_id.get(filename)
    if file_id:
        return f"https://lh3.googleusercontent.com/d/{file_id}=w600"
    return None


# https://lh3.googleusercontent.com/d/1RStarVhCtarEPNFTmBQdlfxWd8daSU_M=w800

# ---------- FastAPI App ----------
app = FastAPI(title="Face Photo Match API")

# ---------- Enable CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://localhost:5500"] or similar in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- POST Endpoint ----------
@app.post("/api/v1/get-similar-photos")
async def get_similar_photos(
    request_id: str = Form(...),
    name: str = Form(...),
    phone_number: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Save uploaded image to disk
        contents = await image.read()
        os.makedirs("temp_faces", exist_ok=True)
        image_path = f"temp_faces/{request_id}.jpg"
        with open(image_path, "wb") as f:
            f.write(contents)

        # Read and validate image
        img = cv2.imread(image_path)
        if img is None:
            return JSONResponse(status_code=400, content={
                "error": "Could not read image or unsupported format.",
                "request_id": request_id
            })

        # Detect face and extract embedding
        faces = face_app.get(img)
        if not faces:
            return JSONResponse(status_code=400, content={
                "error": "No face detected in the image.",
                "request_id": request_id
            })

        uploaded_embedding = faces[0].embedding

        # Match with cluster embeddings
        similarities = {
            str(pid): cosine_similarity([uploaded_embedding], [emb])[0][0]
            for pid, emb in cluster_embeddings.items()
        }

        best_match_id = max(similarities, key=similarities.get)
        matched_files = person_to_photos.get(best_match_id, [])

        # Map to Drive URLs
        image_urls_4k = []
        for path in matched_files:
            fname = os.path.basename(path)
            drive_url = get_drive_url(fname)
            if drive_url:
                image_urls_4k.append(drive_url)

        return {
            "request_id": request_id,
            "face_id": best_match_id,
            "imagePaths": image_urls_4k
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "request_id": request_id
        })

# ---------- Launch ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
