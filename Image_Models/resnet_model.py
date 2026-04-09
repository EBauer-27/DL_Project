import torch
import torch.nn as nn
import numpy as np
import torchvision

"""
ResNet18 pretrained on ImageNet
Use all 18 Layers including the final AvgPooling
Hidden Dim: 512
Out Layer: Fully connected linear layer 512 -> 1
           No activation function, because of the BCEWithLogitsLoss as criterion

Input: Tensor (B, C, H, W)
Output: Tensor (B, P) - P: possibility for malignant case
"""

class ResNet18(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()

        # load ResNet18 with the weights trained on ImageNet
        self.resnet = torchvision.models.resnet18(weights='DEFAULT')

        # remove last (linear) layer 
        self.hidden = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        fc_dim = self.resnet.fc.in_features

        # Output Layer with Sigmoid activation  
        self.out = torch.nn.Sequential(nn.Flatten(),
                                       nn.Linear(fc_dim, out_dim))

    def forward(self, x):
        hidden = self.hidden(x)
        out = self.out(hidden)
        return out 

