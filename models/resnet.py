import torch
import torch.nn as nn
from torchvision import models

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes):
        super(ResNetBaseline, self).__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        
        try:
            self.resnet = models.resnet34(pretrained=True)  
        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet pretrained model: {e}")
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze_layers(self, layer_names=None):
        for name, param in self.resnet.named_parameters():
            if layer_names is None or any(layer in name for layer in layer_names):
                param.requires_grad = True