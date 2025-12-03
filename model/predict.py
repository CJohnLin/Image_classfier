import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from .model_def import create_model
from .gradcam import GradCAM


def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32)
    img = img.unsqueeze(0)
    return img


def load_model(model_path, device):
    model = create_model()
    state = torch.load(model_path, map_location=device)

    if "model" in state:
        # fine-tuned 格式
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def apply_colormap_on_image(org_img, activation_map):
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(org_img, 0.5, heatmap, 0.5, 0)
    return overlay


def predict_single(model, classes, img_pil, device, use_cam=False):
    """
    回傳：
    pred_class (int)
    conf       (float)
    cam_overlay (np.array) or None
    """
    inputs = preprocess_image(img_pil).to(device)

    # Forward
    outputs = model(inputs)
    probs = F.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, dim=1)

    pred_class = pred.item()
    confidence = float(conf.item())

    # ====== Grad-CAM ======
    cam_overlay = None
    if use_cam:
        target_layer = "layer4"
        cam = GradCAM(model, target_layer)
        activation_map = cam.generate_cam(inputs, pred_class)

        activation_map = cv2.resize(activation_map, (224, 224))

        orig = np.array(img_pil.resize((224, 224)))
        if orig.dtype != np.uint8:
            orig = orig.astype(np.uint8)

        activation_map = (activation_map * 255).astype(np.uint8)
        cam_overlay = apply_colormap_on_image(orig, activation_map)

    return pred_class, confidence, cam_overlay
