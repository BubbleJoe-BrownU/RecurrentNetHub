import torch
import torch.nn as nn
import torch.nn.functional as F

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
            

