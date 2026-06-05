import torch
import torch.nn as nn

from model.module.ConvolutionVGG import ConvBlockVGG

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlockVGG(3, 32),
            ConvBlockVGG(32, 64),
            ConvBlockVGG(64, 128),
            ConvBlockVGG(128, 256),
            ConvBlockVGG(256, 256),

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
