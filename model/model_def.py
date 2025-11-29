import torchvision.models as models
import torch.nn as nn

def create_model(num_classes=102):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    return model
