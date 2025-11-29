# Agent Development Log (AI-Assisted)

## Day 1 — Problem Definition
Asked ChatGPT to propose suitable deep learning project topics for the Taica AIGC course.  
Chose **Flowers102 image classification** based on feasibility, dataset quality, and demo potential.

## Day 2 — Dataset & Model Selection
Discussed dataset structure and ImageFolder requirements.  
AI recommended using **ResNet18 transfer learning** for faster convergence and high accuracy.

## Day 3 — Training Pipeline
Used ChatGPT to write:
- train.py (augmentation, loaders, training loop)
- Grad-CAM implementation
- environment requirements

Handled errors:
- Shape mismatch in the final FC layer  
- DataLoader num_workers issue on Windows  

## Day 4 — Streamlit Demo
AI assisted in building:
- Image uploader  
- Top-3 predictions  
- Grad-CAM overlay  
- Downloadable CAM  

## Day 5 — Repo Structuring
ChatGPT helped:
- Design project folder structure  
- Write README  
- Add evaluation script & dataset preparation  
- Add .gitignore & MIT License  

## Day 6 — Finalization
Produced final report abstract.  
Packaged everything for GitHub + Streamlit deployment.
