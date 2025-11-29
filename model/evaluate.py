import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import numpy as np
from model_def import create_model

def load_model(model_path, device, num_classes=102):
    model = create_model(num_classes).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def get_dataloader(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader, test_dataset.classes

def evaluate(model, dataloader, device, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    return acc, cm, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)
    dataloader, classes = get_dataloader(args.data_dir)

    acc, cm, report = evaluate(model, dataloader, device, classes)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()
