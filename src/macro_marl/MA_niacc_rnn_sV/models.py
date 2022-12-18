import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    gain = torch.nn.init.calculate_gain(act_fn)
    fc = torch.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc

class Actor(nn.Module):

    def __init__(self, input_dim, output_dim, mlp_layer_size=32, rnn_layer_size=32):
        super(Actor, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size, act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size, mlp_layer_size, act_fn='leaky_relu')
        self.gru = nn.GRU(mlp_layer_size, hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(rnn_layer_size, mlp_layer_size, act_fn='leaky_relu')
        self.fc4 = Linear(mlp_layer_size, output_dim, act_fn='linear')

    def forward(self, x, h=None, eps=0.0, test_mode=False):

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x, h = self.gru(x, h)
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        action_logits = F.log_softmax(x, dim=-1)

        if not test_mode:
            logits_1 = action_logits + np.log(1-eps)
            logits_2 = torch.full_like(action_logits, np.log(eps)-np.log(action_logits.size(-1)))
            logits = torch.stack([logits_1, logits_2])
            action_logits = torch.logsumexp(logits,axis=0)

        return action_logits, h

class Critic(nn.Module):

    def __init__(self, input_dim, output_dim=1, mlp_layer_size=32, mid_layer_size=32):
        super(Critic, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size, act_fn='leaky_relu')
        self.fc2 = Linear(mlp_layer_size, mlp_layer_size, act_fn='leaky_relu')
        self.fc3 = Linear(mlp_layer_size, mid_layer_size, act_fn='leaky_relu')
        self.fc4 = Linear(mid_layer_size, mlp_layer_size, act_fn='leaky_relu')
        self.fc5 = Linear(mlp_layer_size, output_dim, act_fn='linear')

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        state_value = self.fc5(x)
        return state_value
