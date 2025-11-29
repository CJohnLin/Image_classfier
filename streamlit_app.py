import streamlit as st
import os
import torch
from model.predict import predict_single
from gdrive_downloader import download_file_from_google_drive

# Google Drive model URL
MODEL_URL = "https://drive.google.com/uc?export=download&id=1fe85t5UJhNYCCQBgpGSKCXnW4BcQvVkj"
MODEL_PATH = "model/best_model.pt"

# Auto download model
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Downloading from Google Drive...")
    download_file_from_google_drive(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded successfully.")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Streamlit UI
st.title("Flower Classifier with Grad-CAM")
st.write("Upload an image to run prediction.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

use_cam = st.checkbox("Enable Grad-CAM visualization", value=True)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    if st.button("Run Prediction"):
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        pred, conf, cam_path = predict_single(
            MODEL_PATH,
            "data/class_names.txt",
            temp_path,
            device,
            use_cam,
            return_cam_path=True,
        )

        st.subheader(f"Prediction: {pred} ({conf * 100:.2f}%)")

        if use_cam and cam_path:
            st.image(cam_path, caption="Grad-CAM Heatmap", use_column_width=True)
