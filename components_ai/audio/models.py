import numpy as np
import matplotlib.pyplot as plt # Para reproducir audio
import os
import torch.nn.functional as F
from torch.nn import init
import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        init.kaiming_normal_(self.block[0].weight, a=0.1)
        self.block[0].bias.data.zero_()

    def forward(self, x):
        return self.block(x)

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBlock(2, 8, kernel_size=5, stride=2, padding=2, dropout=dropout),
            ConvBlock(8, 16, kernel_size=3, stride=2, padding=1, dropout=dropout),
            ConvBlock(16, 32, kernel_size=3, stride=2, padding=1, dropout=dropout),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1, dropout=dropout)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    # def forward(self, x):
    #     x = self.conv(x)
    #     x = self.pool(x)
    #     x = self.flatten(x)
    #     x = self.fc(x)
    #     return x

    def forward(self, x):
      for i, block in enumerate(self.conv):
          x = block(x)
          if i == len(self.conv) - 1:
              self.feature_map = x.detach().cpu()
      x = self.pool(x)
      x = self.flatten(x)
      x = self.fc(x)
      return torch.sigmoid(x).squeeze(1)