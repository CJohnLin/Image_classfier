import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out shape: (B, C, H, W)
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] shape: (B, C, H, W)
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, x, class_idx, size=(224, 224)):
        self.model.zero_grad()

        out = self.model(x)
        target = out[0, class_idx]
        target.backward()

        # (C, H, W)
        gradients = self.gradients[0]
        activations = self.activations[0]

        # GAP
        weights = gradients.mean(dim=(1, 2), keepdim=True)

        # Weighted sum
        cam = (weights * activations).sum(dim=0)

        cam = torch.relu(cam).cpu().numpy()

        # Normalize + resize
        cam = (cam - cam.min()) / (cam.max() + 1e-7)
        cam = cv2.resize(cam, size)

        return cam
