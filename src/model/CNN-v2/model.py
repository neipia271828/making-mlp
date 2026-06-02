import torch
import torch.nn as nn

from model.module.ConvolutionVGG import ConvBlockVGG

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlockVGG(1, 32),
            ConvBlockVGG(32, 64),

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
