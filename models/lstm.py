import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBase(nn.Module):
    def __init__(self, input_size, hidden_size):
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

        self.weights_i = nn.Parameter(torch.zeros([input_size, 4*hidden_size]), requires_grad=True)
        self.bias_i = nn.Parameter(torch.zeros([4*hidden_size]), requires_grad=True)
        self.weights_h = nn.Parameter(torch.zeros([hidden_size, 4*hidden_size]), requires_grad=True)
        self.bias_h = nn.Parameter(torch.zeros([4*hidden_size]), requires_grad=True)


    def _init_weights(self, module):
        if module.dat.dim() > 1:
            torch.nn.init.uniform_(module.data, -1*self.hidden_size**-0.5, self.hidden_size**-0.5)
            

        
    def forward(self, inputs, split=True, h_0=None, c_0=None):
        if h_0 is None:
            h_t = torch.zeros([inputs.shape[0], self.hidden_size], device=inputs.device)
        else: 
            h_t = h_0
        if c_0 is None:
            c_t = torch.zeros([inputs.shape[0], self.hidden_size], device=inputs.device)
        else:
            c_t = c_0

        w_ii, w_if, w_ig, w_io = torch.split(self.weights_i, 4, -1)
        b_ii, b_if, b_ig, b_io = torch.split(self.bias_i, 4, -1)
        w_hi, w_hf, w_hg, w_ho = torch.split(self.weights_h, 4, -1)
        b_hi, b_hf, b_hg, b_ho = torch.split(self.bias_h, 4, -1)
        
        output = []
        inputs = torch.transpose(inputs, 0, 1)
        for input_timestep in inputs:
            if split:
                i_t = self.sigmoid(input_timestep@w_ii + b_ii + h_t@w_hi + b_hi)
                f_t = self.sigmoid(input_timestep@w_if + b_if + h_t@w_hf + b_hf)
                g_t = self.tanh(input_timestep@w_ig + b_ig + h_t@w_hg + b_hg)
                o_t = self.sigmoid(input_timestep@w_io + b_io + h_t@w_ho + b_ho)
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * self.tanh(c_t)
            else:
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
    def __init__(self, units):
        self.units = units
        self.fc_s = nn.Linear(in_features=2*units, 
                              out_features=units)
        self.relu = nn.ReLU()
        self.fc_o = nn.Linear(in_features=units, out_features=units)

    def forward(self, inputs, prev_states):
        embed = torch.cat([inputs, prev_states], dim=-1)
        state = self.relu(self.fc_s(embed))
        output = self.relu(self.fc_o(embed))

        return output, state
    
    def generate(self, seq):
        length = seq.shape[1]
        states = torch.zeros([1, 1, self.units])
        for i in range(length):
            inputs = seq[:, i, :]
            outputs, states = self(inputs, states)
            seq = torch.cat([seq, outputs], dim=1)
            

