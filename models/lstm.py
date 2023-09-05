import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBase(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc_ii = nn.Linear(input_size, hidden_size)
        self.fc_hi = nn.Linear(hidden_size, hidden_size)
        self.fc_if = nn.Linear(input_size, hidden_size)
        self.fc_hf = nn.Linear(hidden_size, hidden_size)
        self.fc_ig = nn.Linear(input_size, hidden_size)
        self.fc_hg = nn.Linear(hidden_size, hidden_size)
        self.fc_io = nn.Linear(input_size, hidden_size)
        self.fc_ho = nn.Linear(hidden_size, hidden_size)

        # we can also create a big chunk of weights first and split it into smaller chunks
        # but this way the time cost is actually greater (refer to the why_not_split notebook)
        # not recommended
        # self.weights_i = nn.Parameter(torch.zeros([input_size, 4*hidden_size]), requires_grad=True)
        # self.bias_i = nn.Parameter(torch.zeros([4*hidden_size]), requires_grad=True)
        # self.weights_h = nn.Parameter(torch.zeros([hidden_size, 4*hidden_size]), requires_grad=True)
        # self.bias_h = nn.Parameter(torch.zeros([4*hidden_size]), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Parameter):
            if module.data.dim() > 1:
                torch.nn.init.uniform_(module.data, -1*self.hidden_size**-0.5, self.hidden_size**-0.5)
        elif isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight, -1*self.hidden_size**-0.5, 1*self.hidden_size**-0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


        
    def forward(self, inputs, h_0=None, c_0=None):
        if h_0 is None:
            h_t = torch.zeros([inputs.shape[0], self.hidden_size], device=inputs.device)
        else: 
            h_t = h_0
        if c_0 is None:
            c_t = torch.zeros([inputs.shape[0], self.hidden_size], device=inputs.device)
        else:
            c_t = c_0

        # w_ii, w_if, w_ig, w_io = torch.split(self.weights_i, self.hidden_size, -1)
        # b_ii, b_if, b_ig, b_io = torch.split(self.bias_i, self.hidden_size, -1)
        # w_hi, w_hf, w_hg, w_ho = torch.split(self.weights_h, self.hidden_size, -1)
        # b_hi, b_hf, b_hg, b_ho = torch.split(self.bias_h, self.hidden_size, -1)
        
        output = []
        inputs = torch.transpose(inputs, 0, 1)
        for input_timestep in inputs:
            i_t = self.sigmoid(self.fc_ii(input_timestep) + self.fc_hi(h_t))
            f_t = self.sigmoid(self.fc_if(input_timestep) + self.fc_hf(h_t))
            g_t = self.tanh(self.fc_ig(input_timestep) + self.fc_hg(h_t))
            o_t = self.sigmoid(self.fc_io(input_timestep) + self.fc_ho(h_t))
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)

            output.append(o_t)
        output = torch.stack(output).transpose(0, 1)
        return output, h_t



class LSTM(nn.Module):
    """
    Multi-layer LSTM
    """
    def __init__(self, input_size, hidden_size, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([LSTMBase(input_size, hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, h_0=None, c_0=None):
        
        for layer in self.layers:
            output, state = layer(inputs, h_0, c_0)
            inputs = output
        
        return output, state
    

            

