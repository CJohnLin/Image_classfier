import torch
import torch.nn as nn
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import numpy as np
from model_def import create_model

def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    # 如果模型格式是 { "model": state_dict, "classes": [...] }
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        class_names = ckpt["classes"]
    else:
        raise ValueError("模型格式錯誤，沒有 'model' 與 'classes' 這兩個 key")

    # 建立相同架構
    model = create_model(num_classes=len(class_names))
    model = model.to(device)

    # 載入 state_dict（允許部分轉換）
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model, class_names

def get_dataloader(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(f"{data_dir}/test", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    return loader

def evaluate(model, loader, device, classes):
    preds, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, pred = torch.max(outputs, 1)

            preds.extend(pred.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = np.mean(np.array(preds) == np.array(labels_all))
    cm = confusion_matrix(labels_all, preds)
    report = classification_report(labels_all, preds, target_names=classes)

    return acc, cm, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model, classes = load_model(args.model_path, device)
    loader = get_dataloader(args.data_dir)

    acc, cm, report = evaluate(model, loader, device, classes)

    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()
