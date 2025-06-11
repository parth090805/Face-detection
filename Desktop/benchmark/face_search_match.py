import streamlit as st
import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tempfile

# Load clustered face data
with open("clusters.json", "rb") as f:
    clusters = pickle.load(f)

with open("processed_data.pkl", "rb") as f:
    face_data = pickle.load(f)

# Prepare FaceAnalysis model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

def extract_embedding(image):
    img = np.array(image)
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].embedding

def find_best_cluster(embedding, clusters):
    max_sim = -1
    best_cluster_id = -1
    for cluster_id, entries in clusters.items():
        cluster_embeddings = [face_data[idx]['embedding'] for idx in entries]
        avg_embedding = np.mean(cluster_embeddings, axis=0)
        sim = cosine_similarity([embedding], [avg_embedding])[0][0]
        if sim > max_sim:
            max_sim = sim
            best_cluster_id = cluster_id
    return best_cluster_id, max_sim

st.title("🔍 Find Yourself in the Photo Collection")
uploaded_file = st.file_uploader("Upload your face photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing your face..."):
        embedding = extract_embedding(image)

        if embedding is None:
            st.error("No face detected. Please upload a clear face photo.")
        else:
            cluster_id, similarity = find_best_cluster(embedding, clusters)
            st.success(f"Match found in cluster: Person {int(cluster_id)} (Similarity: {similarity:.4f})")
