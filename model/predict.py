import torch
import numpy as np
from PIL import Image
import cv2
import os
from .gradcam import GradCAM

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    orig_img = img.copy()

    img = img.resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)

    tensor = torch.tensor(arr).float().unsqueeze(0)
    return tensor, orig_img


def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):

    # ---- force activation_map shape ----
    activation_map = np.squeeze(activation_map)

    if activation_map.ndim != 2:
        raise ValueError(f"activation_map must be 2D (H,W), got shape {activation_map.shape}")

    # normalize to 0~1
    activation_map = activation_map - activation_map.min()
    activation_map = activation_map / (activation_map.max() + 1e-7)

    heatmap = np.uint8(activation_map * 255)

    # now heatmap MUST be (H, W) uint8
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if not isinstance(org_img, np.ndarray):
        org_img = np.array(org_img)

    org_img = cv2.resize(org_img, (224, 224))
    overlay = cv2.addWeighted(org_img, 0.6, heatmap_color, 0.4, 0)

    return overlay


def predict_single(model, classes, image_path, device, use_cam=True):

    inputs, orig_img = preprocess_image(image_path)
    inputs = inputs.to(device)

    outputs = model(inputs)
    probs = torch.softmax(outputs, 1)
    conf, pred_idx = torch.max(probs, 1)

    pred_class = classes[pred_idx.item()]
    conf_value = conf.item()

    cam_path = None

    if use_cam:
        cam = GradCAM(model, model.layer4[-1].conv3)
        activation_map = cam.generate_cam(inputs, pred_idx.item())

        overlay = apply_colormap_on_image(orig_img, activation_map)

        cam_path = os.path.splitext(image_path)[0] + "_cam.jpg"
        cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return pred_class, conf_value, cam_path
