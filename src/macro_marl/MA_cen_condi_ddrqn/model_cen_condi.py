import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):

    """
    Creat a linear layer.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    act_fn : str
        The activation function.
    init_weight_uniform : bool
        Whether uniformly sample initial weights.
    """
    gain = torch.nn.init.calculate_gain(act_fn)
    fc = torch.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc
    
class DDRQN(nn.Module):

    def __init__(self, input_dim, output_dim, mlp_layer_size=[32,32], rnn_layer_num=1, rnn_h_size=32, LSTM=False, **kwargs):
        super(DDRQN, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size[0])
        self.fc2 = Linear(mlp_layer_size[0], mlp_layer_size[0])
        if not LSTM:
            self.rnn = nn.GRU(mlp_layer_size[0], hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        else:
            self.rnn = nn.LSTM(mlp_layer_size[0], hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        self.fc3 = Linear(rnn_h_size, mlp_layer_size[1])
        self.fc4 = Linear(mlp_layer_size[1], output_dim, act_fn='linear')
       
    def forward(self, x, h=None):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x, h = self.rnn(x, h)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x, h

class BN_DDRQN(nn.Module):

    def __init__(self, input_dim, output_dim, rnn_layer_num=1, rnn_h_size=256, GRU=False, **kwargs):
        super(BN_DDRQN, self).__init__()

        self.fc1 = Linear(input_dim, 64)
        self.fc2 = Linear(64, rnn_h_size)
        if GRU:
            self.rnn = nn.GRU(rnn_h_size, hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        else:
            self.rnn = nn.LSTM(rnn_h_size, hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        self.fc3 = Linear(rnn_h_size, 64)
        self.fc4 = Linear(64, output_dim, act_fn='linear')

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(rnn_h_size)
        self.bn3 = nn.BatchNorm1d(rnn_h_size)
        self.bn4 = nn.BatchNorm1d(64)
       
    def forward(self, x, h=None):
        xx = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)
        mask = get_mask_from_input(xx)
        x = pad_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)

        x = F.leaky_relu(self.bn1(self.fc1(x).permute(0,2,1)).permute(0,2,1))
        x = F.leaky_relu(self.bn2(self.fc2(x).permute(0,2,1)).permute(0,2,1))

        x = pack_padded_sequence(x, mask.sum(1), batch_first=True, enforce_sorted=False)
        x, h = self.rnn(x, h)
        x = pad_packed_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)[0]

        x = F.leaky_relu(self.bn4(self.fc3(self.bn3(x.permute(0,2,1)).permute(0,2,1)).permute(0,2,1)).permute(0,2,1))
        x = self.fc4(x)
        x = x[mask]

        return x, h
