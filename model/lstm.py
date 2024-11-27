import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
from .attention import SelfAttention
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
        nn.init.orthogonal_(self.weight_hh)
        nn.init.xavier_normal_(self.weight_ih)
        nn.init.constant_(self.bias_hh, 0)
        nn.init.constant_(self.bias_ih, 0)
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
        self.dropout = nn.Dropout(0.5)
        self.lstm_cells = nn.ModuleList([LSTMCell(self.input_size, hidden_size, device=device)
                                         if i == 0
                                         else LSTMCell(self.hidden_size, self.hidden_size, device=device)
                                         for i in range(self.num_cells)])
        self.attention = SelfAttention(hidden_size)

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
                x = self.dropout(hx)
            outputs.append(hx)

        outputs = torch.stack(outputs, dim=1)
        context, _ = self.attention(outputs)
        # outputs = torch.cat([context, hx], dim=1)
        return context, (hx, cx)
        # return outputs, (hx, cx)

class BiLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_cells: int, device: torch.device) -> None:
        super(BiLSTMBlock, self).__init__()
        self.forward_lstm = LSTMBlock(input_size, hidden_size // 2, num_cells, device)
        self.backward_lstm = LSTMBlock(input_size, hidden_size // 2, num_cells, device)

    def forward(self, input: Tensor) -> Tensor:
        forward_output, _ = self.forward_lstm(input)
        backward_output, _ = self.backward_lstm(torch.flip(input, [1]))
        backward_output = torch.flip(backward_output, [1])
        return torch.cat((forward_output, backward_output), dim=2)

class MultiBiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_blocks: int, num_cells_per_block: int, device: torch.device) -> None:
        super(MultiBiLSTM, self).__init__()
        self.blocks = nn.ModuleList([
            BiLSTMBlock(input_size, hidden_size, num_cells_per_block, device)
            if i == 0
            else BiLSTMBlock(hidden_size, hidden_size, num_cells_per_block, device)
            for i in range(num_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x