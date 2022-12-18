import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def get_mask_from_input(x):
    return ~torch.isnan(x).any(-1)

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

    def __init__(self, input_dim, output_dim, dec_mlp_layer_size=[32,32], rnn_layer_num=1, dec_rnn_h_size=32, **kwargs):
        super(DDRQN, self).__init__()

        self.fc1 = Linear(input_dim, dec_mlp_layer_size[0])
        self.fc2 = Linear(dec_mlp_layer_size[0], dec_mlp_layer_size[0])
        self.rnn = nn.GRU(dec_mlp_layer_size[0], hidden_size=dec_rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        self.fc3 = Linear(dec_rnn_h_size, dec_mlp_layer_size[1])
        self.fc4 = Linear(dec_mlp_layer_size[1], output_dim, act_fn='linear')
       
    def forward(self, x, h=None):
        # xx = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)
        # mask = get_mask_from_input(xx)
        # x = pad_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        # x = pack_padded_sequence(x, mask.sum(1), batch_first=True, enforce_sorted=False)
        x, h = self.rnn(x, h)
        # x = pad_packed_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)[0]

        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        # x = x[mask]

        return x, h

class Cen_DDRQN(nn.Module):

    def __init__(self, input_dim, output_dim, cen_mlp_layer_size=[32,32], rnn_layer_num=1, cen_rnn_h_size=64, **kwargs):
        super(Cen_DDRQN, self).__init__()

        self.fc1 = Linear(input_dim, cen_mlp_layer_size[0])
        self.fc2 = Linear(cen_mlp_layer_size[0], cen_mlp_layer_size[0])
        self.rnn = nn.GRU(cen_mlp_layer_size[0], hidden_size=cen_rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        self.fc3 = Linear(cen_rnn_h_size, cen_mlp_layer_size[1])
        self.fc4 = Linear(cen_mlp_layer_size[1], output_dim, act_fn='linear')
       
    def forward(self, x, h=None):
        # xx = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)
        # mask = get_mask_from_input(xx)
        # x = pad_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        # x = pack_padded_sequence(x, mask.sum(1), batch_first=True, enforce_sorted=False)
        x, h = self.rnn(x, h)
        # x = pad_packed_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)[0]

        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        # x = x[mask]

        return x, h
