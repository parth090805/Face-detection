import streamlit as st
import json
import os
import pickle
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# ------------------- Load Data -------------------
with open("person_to_photos.json", "r") as f:
    person_to_photos = json.load(f)

with open("cluster_representatives.pkl", "rb") as f:
    cluster_embeddings = pickle.load(f)

st.set_page_config(page_title="Face Gallery", layout="wide")
st.title("🧠 AI-Powered Face Photo Gallery")

# ------------------- UI: Browse by Face -------------------
st.markdown("### 👥 Choose a Person:")

person_ids = sorted(person_to_photos.keys(), key=lambda x: int(x))
selected_person = st.session_state.get("selected_person", None)

cols = st.columns(5)
for idx, person_id in enumerate(person_ids):
    with cols[idx % 5]:
        thumbnail_path = os.path.join("thumbnails", f"person_{person_id}.jpg")
        if os.path.exists(thumbnail_path):
            image = Image.open(thumbnail_path)
            st.image(image, caption=f"Person {person_id}", use_container_width=True)
            if st.button(f"View Person {person_id}", key=f"btn_{person_id}"):
                st.session_state.selected_person = person_id
                selected_person = person_id

# ------------------- Display Matching Photos -------------------
if selected_person:
    st.markdown(f"### 📸 All Photos Containing Person {selected_person}")
    image_list = person_to_photos[selected_person]
    photo_cols = st.columns(3)
    for i, fname in enumerate(image_list):
        img_path = os.path.join("images", fname)
        if os.path.exists(img_path):
            with photo_cols[i % 3]:
                img = Image.open(img_path)
                st.image(img, caption=fname, use_container_width=True)

# ------------------- Upload & Match a Face -------------------
st.markdown("---")
st.markdown("## 🔍 Upload a Face to Find Matching Person")
uploaded_file = st.file_uploader("Upload a photo of your face", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if query_img is not None:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0)  # use -1 for CPU

        faces = app.get(query_img)

        if not faces:
            st.error("❌ No face detected in the uploaded image.")
        else:
            uploaded_embedding = faces[0].embedding

            # Compute similarity to each cluster representative
            similarities = {
                str(pid): cosine_similarity([uploaded_embedding], [emb])[0][0]
                for pid, emb in cluster_embeddings.items()
            }

            # Sort and pick best match
            best_match_id = max(similarities, key=similarities.get)
            best_score = similarities[best_match_id]

            st.success(f"✅ This face matches **Person {best_match_id}** with similarity `{best_score:.4f}`")

            # Show photos for matched person
            st.markdown(f"### 📂 Photos Containing Person {best_match_id}")
            image_list = person_to_photos[str(best_match_id)]

            match_cols = st.columns(3)
            for i, fname in enumerate(image_list):
                img_path = os.path.join("images", fname)
                if os.path.exists(img_path):
                    with match_cols[i % 3]:
                        img = Image.open(img_path)
                        st.image(img, caption=fname, use_container_width=True)
