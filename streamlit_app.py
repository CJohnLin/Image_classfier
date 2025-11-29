import streamlit as st
import torch
from model.model_def import create_model
from model.predict import predict_single

st.set_page_config(page_title="Image Classifier + GradCAM", layout="wide")

st.title("🌸 Flower Image Classifier with Grad-CAM")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "png"])
use_cam = st.checkbox("Enable Grad-CAM visualization", True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Load model checkpoint -----
ckpt = torch.load("model/best_model.pt", map_location=device)

classes = ckpt["classes"]                # 類別名稱
model = create_model(num_classes=len(classes))
model.load_state_dict(ckpt["model"])     # 正確載入 state_dict

model.to(device)
model.eval()


if uploaded:
    temp_path = "temp_uploaded.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    st.image(uploaded, caption="Uploaded image", use_column_width=True)

    pred, conf, cam_path = predict_single(
        model, classes, temp_path, device, use_cam
    )

    st.subheader(f"Prediction: **{pred}** ({conf*100:.2f}%)")

    if use_cam and cam_path:
        st.image(cam_path, caption="Grad-CAM heatmap", use_column_width=True)
