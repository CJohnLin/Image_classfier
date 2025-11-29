[README.md](https://github.com/user-attachments/files/23835190/README.md)
# 🌸 Flower Image Classifier with Grad-CAM
A deep-learning application based on **ResNet-50**, trained on the **Oxford 102 Flowers** dataset.  
This project supports **local execution**, **cloud deployment**, and **Grad-CAM visualization** for model interpretability.

---

## ⭐ Features

- 🌼 **102-class Flower Classification**
- 🔥 **Grad-CAM visualization** (model attention heatmap)
- 🎨 **Heatmap overlay** on original images
- 🚀 **Streamlit interface**
- ☁️ **Streamlit Cloud ready**
- 📦 **Model packaged with class labels**

Model file format (`best_model.pt`):

```python
{
    "model": state_dict,      # model weights
    "classes": class_names    # list of 102 flower labels
}
```

---

## 📁 Project Structure

```
Image_classifier/
│
├─ streamlit_app.py           # Streamlit user interface
│
├─ model/
│   ├─ best_model.pt          # Trained model (weights + label names)
│   ├─ model_def.py           # ResNet50 architecture
│   ├─ predict.py             # Inference + Grad-CAM + visualization
│   ├─ gradcam.py             # Grad-CAM implementation
│   └─ evaluate.py            # (Optional) evaluation script
│
└─ requirements.txt           # Python dependencies
```

---

## 🖥️ Local Execution

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit application
```bash
streamlit run streamlit_app.py
```

### 3. Upload an image  
Supported formats:
```
jpg / jpeg / png
```

### 4. Toggle Grad-CAM  
Enable the Grad-CAM checkbox for visualization.

---

## ☁️ Streamlit Cloud Deployment

### 1. Upload the following files to GitHub:
```
streamlit_app.py
requirements.txt
model/model_def.py
model/predict.py
model/gradcam.py
model/best_model.pt   (if <100MB)
```

If your model is **larger than 100MB**, configure `streamlit_app.py` to download it from Google Drive.

### 2. Deploy on Streamlit Cloud  
Go to:
https://share.streamlit.io

Click **"Deploy an app"** → Select your repository → Set:

```
Main file: streamlit_app.py
```

### 3. Done!  
Your cloud app will appear at a URL like:
```
https://yourname-image-classifier.streamlit.app
```

---

## 🔥 Grad-CAM Example Interpretation

- ❤️ **Red** → Model heavily focuses  
- 💛 **Yellow** → Medium focus  
- 💙 **Blue** → Minimal attention  

Grad-CAM helps inspect what regions influenced the model decision.

---

## 📜 License
MIT License (modify as needed)

---

## ✨ Acknowledgements
- Dataset: Oxford 102 Flowers  
- Backbone: ResNet-50 (Torchvision)  
- UI Framework: Streamlit  
