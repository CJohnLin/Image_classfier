import torch
import torch.nn as nn
from torchvision import datasets, transforms
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
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
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification Report
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
    dataloader, class_names = get_dataloader(args.data_dir)

    acc, cm, report = evaluate(model, dataloader, device, class_names)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()
