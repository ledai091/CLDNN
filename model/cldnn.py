import torch
import torch.nn as nn
from torch import Tensor
from .dnn import DNN
from .lstm import MultiBiLSTM
from .cnn import CNN

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

        self.cnn = CNN(in_channels=in_channels, output_size=cnn_out_channels, device=device)
        
        self.lstm = MultiBiLSTM(input_size=lstm_input_size, 
                                hidden_size=lstm_hidden_size, 
                                num_blocks=lstm_num_blocks, 
                                num_cells_per_block=lstm_num_cells_per_block, 
                                device=device)
        
        self.dnn = DNN(input_size=lstm_hidden_size, output_size=dnn_output_size, device=device)
    
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
    
def main(args):
    model = CLDNN(
        in_channels=args.in_channels,
        cnn_out_channels=args.cnn_out_channels,
        lstm_input_size=args.lstm_input_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_blocks=args.lstm_num_blocks,
        lstm_num_cells_per_block=args.lstm_num_cells_per_block,
        dnn_output_size=args.dnn_output_size,
        device=args.device
    )
    return model