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
2. Train:python model/train.py --data_dir data --epochs 10 --batch_size 32 --output model.pt
3. Run demo:
  pip install -r requirements.txt
  streamlit run app/streamlit_app.py
4. Deploy: push repo to GitHub → connect to Streamlit Cloud → set `model.pt` as file in repo (or use external storage)

## Notes
- If dataset is large, train on a GPU machine (Colab / local GPU).
- Use small epochs and lr scheduling for faster convergence.


