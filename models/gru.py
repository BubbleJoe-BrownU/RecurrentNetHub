import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUBase(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc_ir = nn.Linear(input_size, hidden_size)
        self.fc_hr = nn.Linear(hidden_size, hidden_size)
        self.fc_iz = nn.Linear(input_size, hidden_size)
        self.fc_hz = nn.Linear(hidden_size, hidden_size)
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_hn = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, inputs, h_0=None):
        if h_0 is None:
            h_t = torch.zeros([1, self.hidden_size], device=inputs.device)
        else:
            h_t = h_0
        
        output = []
        inputs = inputs.transpose(0, 1)
        for input_timestep in inputs:
            r_t = self.sigmoid(self.fc_ir(input_timestep) + self.fc_hr(h_t))
            z_t = self.sigmoid(self.fc_iz(input_timestep) + self.fc_hz(h_t))
            n_t = self.tanh(self.fc_in(input_timestep) + r_t*self.fc_hn(h_t))
            h_t = (1-z_t)*n_t + z_t*h_t
            output.append(h_t)
        output = torch.stack(output)
        return output, h_t
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([GRUBase(input_size, hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, h_0=None):
        for layer in self.layers:
            output, state = layer(inputs, h_0)
            input = output
            input = self.dropout(input)
        return output, state