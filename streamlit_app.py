import streamlit as st
import os
import torch
from PIL import Image
import numpy as np

from model.predict import predict_single, load_model
from gdrive_downloader import download_file_from_google_drive

# ===========================
#       PAGE TITLE
# ===========================
st.set_page_config(page_title="Flower Classifier", layout="wide")

st.title("🌸 Flower Classification with Grad-CAM")
st.write("Upload an image and let the model classify it with optional Grad-CAM visualization.")

# ===========================
#       CONSTANTS
# ===========================
MODEL_PATH = "model/best_model.pt"
GOOGLE_DRIVE_FILE_ID = "1fe85t5UJhNYCCQBgpGSKCXnW4BcQvVkj"  # 你給我的模型 ID
CLASSES = list(range(102))  # Oxford 102 Flowers dataset


# ===========================
#  DOWNLOAD MODEL IF NEEDED
# ===========================
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        st.warning("模型不存在，正在從 Google Drive 下載...")
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
        st.success("模型下載完成！")


ensure_model_exists()


# ===========================
#       LOAD MODEL
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH, device)

st.success(f"Model loaded on **{device}**")


# ===========================
#       IMAGE UPLOAD
# ===========================
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

use_cam = st.checkbox("Enable Grad-CAM visualization", value=True)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Predicting..."):
            pred, conf, cam_overlay = predict_single(
                model,
                CLASSES,
                img,
                device,
                use_cam=use_cam
            )

        st.subheader(f"Prediction: **{pred} ({conf*100:.2f}%)**")

        if use_cam and cam_overlay is not None:
            st.image(cam_overlay, caption="Grad-CAM Result", use_container_width=True)
