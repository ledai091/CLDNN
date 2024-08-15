import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class DNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super(DNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, 200).to(device)
        self.bn1 = nn.BatchNorm1d(200).to(device)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(200, 100).to(device)
        self.bn2 = nn.BatchNorm1d(100).to(device)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(100, output_size).to(device)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.normal_(param, mean=0, std=0.01)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, device: torch.device = None) -> None:
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
                
        self.weight_ih = nn.Parameter(torch.empty(4*hidden_size, input_size, device=device))
        self.weight_hh = nn.Parameter(torch.empty(4*hidden_size, hidden_size, device=device))
        
        self.bias = bias
        
        if self.bias:
            self.bias_ih = nn.Parameter(torch.empty(4*hidden_size, device=device))
            self.bias_hh = nn.Parameter(torch.empty(4*hidden_size, device=device))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.peephole_i = nn.Parameter(torch.empty(hidden_size, device=device))
        self.peephole_f = nn.Parameter(torch.empty(hidden_size, device=device))
        self.peephole_o = nn.Parameter(torch.empty(hidden_size, device=device))
        
        self._init_weights()

    def _init_weights(self) -> None:
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.01)
        nn.init.normal_(self.peephole_i, mean=0, std=0.01)
        nn.init.normal_(self.peephole_f, mean=0, std=0.01)
        nn.init.normal_(self.peephole_o, mean=0, std=0.01)
    
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor] = None) -> Tuple[Tensor, Tensor]:
        hx, cx = state
        gates = torch.mm(input, self.weight_ih.t()) + torch.mm(hx, self.weight_hh.t())
        if self.bias:
            gates += self.bias_ih + self.bias_hh
        
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        
        input_gate = torch.sigmoid(input_gate + self.peephole_i * cx)
        forget_gate = torch.sigmoid(forget_gate + self.peephole_f * cx)
        cell_gate = torch.tanh(cell_gate)
        
        cy = (forget_gate * cx) + (input_gate * cell_gate)
        
        output_gate = torch.sigmoid(output_gate + self.peephole_o * cy)
        
        hy = output_gate * torch.tanh(cy)
        return hy, cy
    
class LSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_cells: int, device: torch.device) -> None:
        super(LSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.device = device
        
        self.lstm_cells = nn.ModuleList([LSTMCell(self.input_size, hidden_size, device=device)
                                         if i == 0
                                         else LSTMCell(self.hidden_size, self.hidden_size, device=device)
                                         for i in range(self.num_cells)])
        
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]=None)->Tuple[Tensor, Tensor]:
        batch_size = input.size(0)
        
        if state is None:
            zeros = torch.zeros(batch_size, self.hidden_size, device=self.device)
            state = (zeros, zeros)
            
        hx, cx = state
        
        outputs = []
        
        seq_len = input.size(1)
        for t in range(seq_len):
            x = input[:, t, :]
            for i, lstm_cell in enumerate(self.lstm_cells):
                hx, cx = lstm_cell(x, (hx, cx))
                x = hx
            outputs.append(hx)
            
        outputs = torch.stack(outputs, dim=1)
        return outputs, (hx, cx)
    
class MultiBlockLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_blocks: int, num_cells_per_block: int, device: torch.device) -> None:
        super(MultiBlockLSTM, self).__init__()
        self.device = device
        self.blocks = nn.ModuleList([
            LSTMBlock(input_size, hidden_size, num_cells_per_block, device)
            if i == 0
            else LSTMBlock(hidden_size, hidden_size, num_cells_per_block, device)
            for i in range(num_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x, _ = block(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, in_channels: int, output_size: int, device: torch.device) -> None:
        super(CNN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=3, stride=1, padding=1).to(device)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2, stride=1, padding=0).to(device)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=2, stride=1, padding=0).to(device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(30, output_size).to(device)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
    
class CLDNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 cnn_out_channels: int,
                 lstm_input_size: int,
                 lstm_hidden_size: int,
                 lstm_num_blocks: int,
                 lstm_num_cells_per_block: int,
                 dnn_output_size: int,
                 device: torch.device) -> None:
        super(CLDNN, self).__init__()
        self.device = device
        self.cnn = CNN(in_channels, cnn_out_channels, device)
        self.lstm = MultiBlockLSTM(lstm_input_size, lstm_hidden_size, lstm_num_blocks, lstm_num_cells_per_block, device)
        self.dnn = DNN(lstm_hidden_size, dnn_output_size, device)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, channels, height, width = x.size()

        cnn_out = []
        for t in range(seq_len):
            img = x[:, t, :, :, :]
            cnn_out.append(self.cnn(img))
            
        cnn_out = torch.stack(cnn_out, dim=1)
        
        lstm_out = self.lstm(cnn_out)
        
        last_output = lstm_out[:, -1, :]
        
        out = self.dnn(last_output)
        out = torch.sigmoid(out)
        return out