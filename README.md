# 🌸 Flowers102 Image Classification  
**ResNet18 + Transfer Learning + Grad-CAM | Streamlit Web App**

This project implements an image classification system for the Oxford **Flowers-102** dataset using **transfer learning (ResNet18)**. A **Streamlit web demo** is provided for real-time predictions and Grad-CAM heatmap visualization.  
Designed as a complete AIGC course project, the repository includes:  
- 📝 300-word English abstract  
- 🤖 Agent development log  
- 🧠 ResNet18 training script  
- 🔥 Grad-CAM visualization  
- 🌐 Streamlit demo (local & cloud deployable)

---

## 📂 Project Structure

```
Image_classfier/
│
├── model/
│   ├── train.py
│   ├── gradcam.py
│   └── evaluate.py
│
├── app/
│   └── streamlit_app.py
│
├── scripts/
│   └── prepare_data.py
│
├── model.pt                # (Optional) Pretrained weights
├── requirements.txt
├── agent_log.md
├── README.md
└── .gitignore
```

---

# 🔧 Installation

```bash
git clone https://github.com/CJohnLin/Image_classfier.git
cd Image_classfier
pip install -r requirements.txt
```

---

# 🌼 Dataset — Flowers102

Official dataset:  
🔗 https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

### ▶ (Recommended) Auto-download & prepare dataset

```bash
python scripts/prepare_data.py
```

This script will:
- Download Oxford Flowers-102  
- Extract all images  
- Create `train/`, `val/`, `test/` splits  
- Convert into **ImageFolder** format (PyTorch-ready)

After execution:

```
data/
 ├── train/ class_x/...
 ├── val/   class_x/...
 └── test/  class_x/...
```

---

# 🏋️ Training the Model

```bash
python model/train.py --data_dir data --epochs 10 --batch_size 32 --output model.pt
```

The best model checkpoint will be saved as **model.pt**.

---

# 📊 Evaluation (Accuracy + Confusion Matrix)

```bash
python model/evaluate.py --model model.pt --data_dir data
```

Generates:
- Overall Accuracy  
- Top-k accuracy  
- `confusion_matrix.png`

---

# 🌐 Streamlit Web Demo

Run locally:

```bash
streamlit run app/streamlit_app.py
```

Demo features:
- Upload any flower image  
- Top-3 predictions with probabilities  
- Grad-CAM heatmap overlay  
- Downloadable CAM result  

---

# ☁ Streamlit Cloud Deployment

1. Push repo to GitHub  
2. Go to: https://share.streamlit.io  
3. Create new app → select this repo  
4. Set main script:  
   ```
   app/streamlit_app.py
   ```  
5. Make sure `model.pt` is in repo root  
6. Deploy 🎉

---

# 📸 Example Demo Screenshots

(Add your own screenshots in `/images`)

| Prediction | Grad-CAM |
|-----------|----------|
| ![pred](images/demo_pred.jpg) | ![cam](images/demo_cam.jpg) |

---

# 📘 License

MIT License — free to modify and distribute.

---

# 🤖 Agent Development Log

A detailed record of AI-assisted development is included in:

```
agent_log.md
```

---

# 📝 Abstract (300 Words)

This project develops a practical and reproducible image classification system for the Oxford Flowers-102 dataset using transfer learning. The goal is to build an accurate and lightweight classifier that recognizes 102 flower species with a web demo for intuitive inspection. We adopt a ResNet-18 backbone pretrained on ImageNet and fine-tune the final layers to adapt to the domain-specific appearance of flowers. Data augmentation and class-balanced sampling are used to improve generalization across diverse viewpoints, backgrounds, and lighting. For interpretability, Grad-CAM visualization is integrated so users can see which regions influenced the prediction. The system is packaged as a Streamlit web application enabling users to upload an image, receive classification predictions with confidence scores, and view a Grad-CAM heatmap overlay. Training and inference scripts, a clear README, and an agent development log documenting AI-assisted coding and debugging steps are included in the repository to ensure reproducibility. Experimental results validate that transfer learning is an effective and practical approach for image classification tasks, and the completed GitHub repository and deployed demo ensure full accessibility.

---

# 🙌 Credits

Developed with assistance from ChatGPT.  
Dataset © Oxford VGG.

---
