import torch.nn as nn
from torchvision.models import resnet18

def create_model(num_classes=102):
    model = resnet18(weights=None)  # 不載入預訓練
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
