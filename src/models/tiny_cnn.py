"""Tiny CNN model in PyTorch.

4-layer convolutional network with BatchNorm2d, ReLU, Flatten, and Linear classification head.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels, eps=1e-5)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2, eps=1e-5)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_channels * 4, eps=1e-5)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(hidden_channels * 4, eps=1e-5)
        self.relu4 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(hidden_channels * 4, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
