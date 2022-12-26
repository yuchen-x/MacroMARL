import torch 
import numpy as np

from torch.distributions import Categorical
from .envs_runner import EnvsRunner
from .models import Actor, Critic
from .utils import Agent, get_joint_avail_actions, get_conditional_action, get_conditional_logits

class MAC(object):
    
    def __init__(self, env, obs_last_action=False, 
                 a_mlp_layer_size=64, a_rnn_layer_size=64, 
                 c_mlp_layer_size=64,c_rnn_layer_size=64,
                 device='cpu'):

        self.env = env
        self.n_agent = env.n_agent
        self.obs_last_action = obs_last_action

        self.a_mlp_layer_size = a_mlp_layer_size
        self.a_rnn_layer_size = a_rnn_layer_size
        self.c_mlp_layer_size = c_mlp_layer_size
        self.c_rnn_layer_size = c_rnn_layer_size

        self.device = device

        self._build_agent()
        self._init_critic()

    def select_action(self, obses, last_actions, h_state, valids, avail_actions, eps=0.0, test_mode=False, using_tgt_net=False):
        actions = [] # List[Int]
        with torch.no_grad():
            if max(valids) == 1.0:
                joint_avail_actions = get_joint_avail_actions(avail_actions)
                if not using_tgt_net:
                    action_logits, new_h_state = self.agent.actor_net(torch.cat(obses, dim=1).view(1,1,-1), 
                                                                      h_state, 
                                                                      eps=eps, 
                                                                      test_mode=test_mode)
                else:
                    action_logits, new_h_state = self.agent.actor_tgt_net(torch.cat(obses, dim=1).view(1,1,-1), 
                                                                          h_state, 
                                                                          eps=eps, 
                                                                          test_mode=test_mode)

                action_logits = get_conditional_logits(action_logits,
                                                       get_conditional_action(torch.cat(last_actions, dim=1),
                                                                              torch.cat(valids, dim=1)),
                                                       joint_avail_actions,
                                                       self.env.n_action)
                #TODO check action_logitis shape
                action_prob = Categorical(logits=action_logits[0])
                action = action_prob.sample().item()
                actions = np.unravel_index(action, self.env.n_action)
            else:
                actions = last_actions
                new_h_state = h_state
        return actions, new_h_state

    def _build_agent(self):
        self.agent = Agent()
        self.agent.actor_net = Actor(self._get_input_dim(), 
                                     self._get_output_dim(), 
                                     self.a_mlp_layer_size, 
                                     self.a_rnn_layer_size).to(self.device)
        self.agent.actor_tgt_net = Actor(self._get_input_dim(), 
                                         self._get_output_dim(), 
                                         self.a_mlp_layer_size, 
                                         self.a_rnn_layer_size).to(self.device)
        self.agent.actor_tgt_net.load_state_dict(self.agent.actor_net.state_dict())

    def _init_critic(self):
        self.agent.critic_net = Critic(self._get_input_dim(), 
                                       1, 
                                       self.c_mlp_layer_size, 
                                       self.c_rnn_layer_size).to(self.device)
        self.agent.critic_tgt_net = Critic(self._get_input_dim(), 
                                           1, 
                                           self.c_mlp_layer_size, 
                                           self.c_rnn_layer_size).to(self.device)
        self.agent.critic_tgt_net.load_state_dict(self.agent.critic_net.state_dict())

    def _get_input_dim(self):
        if not self.obs_last_action:
            return sum(self.env.obs_size)
        else:
            return sum([o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])])

    def _get_output_dim(self):
        return np.prod(self.env.n_action)
