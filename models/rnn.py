import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class RNNBase(nn.Module):
    """
    Expects the inputs to be of shape (N, L, H_in), where N is the batch size, L is the sequence length, and H_in is the input embedding size
    Returns output and hidden_state. Output is of shape (N, L, H_out) containing the hidden states of all elements in the sequence.
    hidden_state is of shape (N, H_out) containing the final hidden state for each element in the batch.
    """

    def __init__(self, input_size, hidden_size, nonlinearity: Union[nn.ReLU, nn.Tanh]):
        super().__init__()
        self.fc_i = nn.Linear(in_features=input_size, 
                              out_features= hidden_size)
        self.fc_h = nn.Linear(in_features=hidden_size, 
                              out_features=hidden_size)
        self.fc_o = nn.Linear(in_features=hidden_size, 
                              out_features=hidden_size)
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity()

        # initialize all weights
        self.apply(self._init_weights)
        
    def forward(self, inputs, state=None):
        if state is None:
            state = torch.zeros([inputs.shape[0], self.hidden_size], device=inputs.device)
        outputs = torch.empty([inputs.shape[0], inputs.shape[1], self.hidden_size], device=inputs.device)
        length = inputs.shape[1]
        for i in range(length):
            state = self.nonlinearity(self.fc_i(inputs[:, i, :]) + self.fc_h(state))
            outputs[:, i, :] = state
        
        return outputs, state
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight, -1*self.hidden_size**-0.5, 1*self.hidden_size**-0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


        
        

class RNN(nn.Module):
    """
    Multi-layer RNN
    Expects the inputs to be of shape (N, L, H_in), where N is the batch size, L is the sequence length, and H_in is the input embedding size
    Returns output and hidden_state. Output is of shape (N, L, H_out) containing the output features from the last layer of the RNN.
    hidden_state is of shape (N, H_out) containing the final hidden state for each element in the batch.
    Parameters:
        input_size: the number of features in the inputs
        hidden_size: the number of features in the hidden states
        num_layers: the number of recurrent layers. For num_layers > 1, the network stacks
            num_layer RNNs together, with each RNN layer taking in outputs of previous RNN layers as the input, 
            except for the first RNN layer, which takes in the original inputs
    """
    def __init__(self, input_size: int, hidden_size: int, nonlinearity: Union[nn.ReLU, nn.Tanh], num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([RNNBase(input_size, hidden_size, nonlinearity) for _ in range(num_layers)])

    def forward(self, inputs, state=None):
        for layer in self.layers:
            output, state = layer(inputs, state)
            inputs = output

        return output, state


