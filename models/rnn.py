import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class RNNBase(nn.Module):

    def __init__(self, input_size, hidden_size, nonlinearity: Union[nn.ReLU, nn.Tanh]):
        self.fc_i = nn.Linear(in_features=input_size, 
                              out_features= hidden_size)
        self.fc_h = nn.Linear(in_features=hidden_size, 
                              out_features=hidden_size)
        
        self.nonlinearity = nonlinearity
        
    def forward(self, inputs, state):
        hidden_state = self.nonlinearity(self.fc_i(inputs) + self.fc_h(state))
        
        

class RNN(nn.Module):
    """
    Parameters:
        input_size: the number of features in the inputs
        hidden_size: the number of features in the hidden states
        num_layers: the number of recurrent layers. For num_layers > 1, the network stacks
            num_layer RNNs together, with each RNN layer taking in outputs of previous RNN layers as the input, 
            except for the first RNN layer, which takes in the original inputs
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc_s = nn.Linear(in_features=2*input_size, 
                              out_features=input_size)
        self.relu = nn.ReLU()
        self.fc_o = nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, inputs, prev_states):
        embed = torch.cat([inputs, prev_states], dim=-1)
        state = self.relu(self.fc_s(embed))
        output = self.relu(self.fc_o(embed))

        return output, state
    
    def generate(self, seq):
        length = seq.shape[1]
        states = torch.zeros([1, 1, self.input_size])
        for i in range(length):
            inputs = seq[:, i, :]
            outputs, states = self(inputs, states)
            seq = torch.cat([seq, outputs], dim=1)

