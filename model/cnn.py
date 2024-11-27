import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class CNN(nn.Module):
    def __init__(self, in_channels: int, output_size: int, device: torch.device) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1).to(device)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1).to(device)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1).to(device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_size).to(device)
        
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x