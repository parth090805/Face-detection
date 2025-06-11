from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import pickle
import json
import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Load model and cluster data once on startup
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

with open("cluster_representatives.pkl", "rb") as f:
    cluster_embeddings = pickle.load(f)

with open("person_to_photos.json", "r") as f:
    person_to_photos = json.load(f)

app = FastAPI(title="Face Photo Match API")


class MatchRequest(BaseModel):
    request_id: str
    path_face: str  # path to the image file on disk
    name: str
    phone_number: int


@app.post("/api/v1/get-similar-photos")
async def get_similar_photos(request: MatchRequest):  #parses the request body into a MatchRequest object.
    image_path = request.path_face

    if not os.path.exists(image_path):
        return JSONResponse(status_code=400, content={
            "error": f"File '{image_path}' does not exist.",
            "request_id": request.request_id
        })

    img = cv2.imread(image_path)
    if img is None:
        return JSONResponse(status_code=400, content={
            "error": "Could not read image or unsupported format.",
            "request_id": request.request_id
        })

    faces = face_app.get(img)
    if not faces:
        return JSONResponse(status_code=400, content={
            "error": "No face detected in the image.",
            "request_id": request.request_id
        })

    uploaded_embedding = faces[0].embedding

    # Compute similarity with each cluster representative
    similarities = {
        str(pid): cosine_similarity([uploaded_embedding], [emb])[0][0]
        for pid, emb in cluster_embeddings.items()
    }

    best_match_id = max(similarities, key=similarities.get)
    image_paths = person_to_photos.get(best_match_id, [])

    return {
        "request_id": request.request_id,
        "face_id": best_match_id,
        "imagePaths": image_paths
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
