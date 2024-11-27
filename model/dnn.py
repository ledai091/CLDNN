import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 200).to(device)
        self.bn1 = nn.BatchNorm1d(200).to(device)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(200, 100).to(device)
        self.bn2 = nn.BatchNorm1d(100).to(device)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(100, output_size).to(device)
        
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x