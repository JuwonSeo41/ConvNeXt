import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from typing import Any, Callable, List, Optional, Tuple
import torch.nn.functional as F
import warnings


class TwoPathInceptionV3(nn.Module):
    def __init__(self, num_classes=38):
        super(TwoPathInceptionV3, self).__init__()

        self.inceptionV3 = models.inception_v3(pretrained=False, aux_logits=False)

        # 20%L + 80%AB
        self.L_branch = nn.Sequential(
            BasicConv2d(1, 6, kernel_size=3, stride=2),
            BasicConv2d(6, 6, kernel_size=3),
            BasicConv2d(6, 13, kernel_size=3, padding=1),
            self.inceptionV3.maxpool1
        )

        self.AB_branch = nn.Sequential(
            BasicConv2d(2, 26, kernel_size=3, stride=2),
            BasicConv2d(26, 26, kernel_size=3),
            BasicConv2d(26, 51, kernel_size=3, padding=1),
            self.inceptionV3.maxpool1
        )

        self.shared_layers = nn.Sequential(
            self.inceptionV3.Conv2d_3b_1x1,
            self.inceptionV3.Conv2d_4a_3x3,
            self.inceptionV3.maxpool2,
            self.inceptionV3.Mixed_5b,
            self.inceptionV3.Mixed_5c,
            self.inceptionV3.Mixed_5d,
            self.inceptionV3.Mixed_6a,
            self.inceptionV3.Mixed_6b,
            self.inceptionV3.Mixed_6c,
            self.inceptionV3.Mixed_6d,
            self.inceptionV3.Mixed_6e
        )

        self.head = nn.Sequential(
            self.inceptionV3.avgpool,
            self.inceptionV3.dropout
        )

        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, y):        # x: L image, y: AB image
        x = self.L_branch(x)
        y = self.AB_branch(y)

        features = torch.cat([x, y], dim=1)

        features = self.shared_layers(features)

        features = self.head(features)

        features = torch.flatten(features, 1)

        out = self.fc(features)

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

