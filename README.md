# Flowers102 Image Classifier (ResNet18 + Grad-CAM)

## Summary
A course project: image classifier for Oxford Flowers-102 using transfer learning (ResNet18). Includes Streamlit demo with Grad-CAM visualization.

## Repo structure
- model/train.py        : training script (ImageFolder expects data/train, data/val)
- model/gradcam.py      : Grad-CAM utilities
- app/streamlit_app.py  : Streamlit demo
- requirements.txt
- agent_log.md

## Quick start
1. Prepare dataset with `data/train/<class>/*` and `data/val/<class>/*`
2. Train:
