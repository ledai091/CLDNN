import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, encoder_outputs):
        attention_weights = self.attention(encoder_outputs)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)
        return context.squeeze(1), attention_weights