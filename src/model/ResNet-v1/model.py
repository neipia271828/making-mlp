import torch
import torch.nn as nn

from model.module.ConvolutionVGG import ConvBlockVGG
from model.module.BasicBlock import BasicBlock

class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, block_count=3, stride=1)
        self.layer2 = self._make_layer(128, block_count=4, stride=2)
        self.layer3 = self._make_layer(256, block_count=6, stride=2)
        self.layer4 = self._make_layer(512, block_count=3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        block_count: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * BasicBlock.expansion

        for _ in range(1, block_count):
            blocks.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
