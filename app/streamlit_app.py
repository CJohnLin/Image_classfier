# app/streamlit_app.py
import streamlit as st
from PIL import Image
import io
import torch
from torchvision import transforms, models
import numpy as np
import cv2
import os
import base64
from model.gradcam import GradCAM, preprocess_image, overlay_cam_on_image

@st.cache_resource
def load_model(model_path="model.pt", device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", None)
    model = models.resnet18(pretrained=False)
    n_features = model.fc.in_features
    model.fc = torch.nn.Linear(n_features, len(classes) if classes else 102)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, classes

def predict(image: Image.Image, model, classes, device="cpu", topk=3):
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        topk_probs, topk_idx = torch.topk(probs, topk)
        topk_probs = topk_probs.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()
        labels = [classes[i] for i in topk_idx]
    return labels, topk_probs, input_tensor

def cv2_im_from_pil(pil):
    rgb_np = np.array(pil.convert("RGB"))
    return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

def download_link(obj, filename, text):
    b64 = base64.b64encode(obj).decode()
    href = f'<a href="data:file/zip;base64,{b64}" download="{filename}">{text}</a>'
    return href

st.set_page_config(page_title="Flowers102 Classifier", layout="centered")
st.title("Flowers102 — Image Classifier (ResNet18 + Grad-CAM)")

st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model path", value="model.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(model_path):
    st.sidebar.warning("Put model.pt in repo root or specify path. You can train using model/train.py")
else:
    model, classes = load_model(model_path, device=device)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)
    if st.button("Classify"):
        with st.spinner("Predicting..."):
            labels, probs, input_tensor = predict(image, model, classes, device=device)
            st.success("Done")
            st.subheader("Top Predictions")
            for lab, p in zip(labels, probs):
                st.write(f"- **{lab}**: {p*100:.2f}%")
            # Grad-CAM
            # target layer: model.layer4[-1] for ResNet18
            target_layer = model.layer4[-1]
            grad = GradCAM(model, target_layer)
            cam = grad(input_tensor, class_idx=None)
            img_cv = cv2_im_from_pil(image)
            overlay = overlay_cam_on_image(img_cv, cam, alpha=0.5)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            st.subheader("Grad-CAM")
            st.image(overlay_rgb, use_column_width=True)
            # offer download of overlay
            _, img_buf = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            st.markdown(download_link(img_buf.tobytes(), "gradcam.jpg", "Download Grad-CAM image"), unsafe_allow_html=True)
