import torch 

from torch.distributions import Categorical

from macro_marl.MA_niacc_rnn_V.models import Actor, Critic
from .utils import Agent

class MAC(object):
    
    def __init__(self, env, obs_last_action=False, 
                 a_mlp_layer_size=32, a_rnn_layer_size=32, 
                 c_mlp_layer_size=32, c_rnn_layer_size=32,
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

    def select_action(self, obses, h_states, valids, avail_actions, eps=0.0, test_mode=False, using_tgt_net=False):
        actions = [] # List[Int]
        new_h_states = []
        with torch.no_grad():
            for idx, agent in enumerate(self.agents):
                if valids[idx]:
                    if not using_tgt_net:
                        action_logits, new_h_state = agent.actor_net(obses[idx].view(1,1,-1), 
                                                                     h_states[idx], 
                                                                     eps=eps, 
                                                                     test_mode=test_mode)
                    else:
                        action_logits, new_h_state = agent.actor_tgt_net(obses[idx].view(1,1,-1), 
                                                                         h_states[idx], 
                                                                         eps=eps, 
                                                                         test_mode=test_mode)
                    action_logits = self._get_masked_logits(action_logits[0], avail_actions[idx])
                    #TODO check action_logitis shape
                    action_prob = Categorical(logits=action_logits[0])
                    action = action_prob.sample().item()
                    actions.append(action)
                    new_h_states.append(new_h_state)
                else:
                    actions.append(-1)
                    new_h_states.append(h_states[idx])
        return actions, new_h_states

    def _get_masked_logits(self, action_logits, avail_action):
        masked_logits = action_logits.clone()
        return  masked_logits.masked_fill(avail_action==0.0, -float('inf'))

    def _build_agent(self):
        self.agents = []
        for idx in range(self.n_agent):
            agent = Agent()
            agent.idx = idx
            agent.actor_net = Actor(self._get_actor_input_shape(idx), self.env.n_action[idx], self.a_mlp_layer_size, self.a_rnn_layer_size).to(self.device)
            agent.actor_tgt_net = Actor(self._get_actor_input_shape(idx), self.env.n_action[idx], self.a_mlp_layer_size, self.a_rnn_layer_size).to(self.device)
            agent.actor_tgt_net.load_state_dict(agent.actor_net.state_dict())
            self.agents.append(agent)

    def _init_critic(self):
        for agent in self.agents:
            agent.critic_net = Critic(self._get_critic_input_shape(), 1, self.c_mlp_layer_size, self.c_rnn_layer_size).to(self.device)
            agent.critic_tgt_net = Critic(self._get_critic_input_shape(), 1, self.c_mlp_layer_size, self.c_rnn_layer_size).to(self.device)
            agent.critic_tgt_net.load_state_dict(agent.critic_net.state_dict())

    def _get_actor_input_shape(self, agent_idx):
        if not self.obs_last_action:
            return self.env.obs_size[agent_idx]
        else:
            return self.env.obs_size[agent_idx] + self.env.n_action[agent_idx]

    def _get_critic_input_shape(self):
        if not self.obs_last_action:
            return sum(self.env.obs_size)
        else:
            return sum([o_dim + a_dim for o_dim, a_dim in zip(*[self.env.obs_size, self.env.n_action])])
