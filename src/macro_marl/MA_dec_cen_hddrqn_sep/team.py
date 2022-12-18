import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import math
import copy
import random

from torch.optim import Adam, RMSprop
from itertools import chain

from .model import DDRQN, Cen_DDRQN
from .utils.Agent import Agent
from .utils.Cen_ctrl import Cen_Controller
from .utils.utils import Linear_Decay, get_conditional_argmax, get_conditional_action, get_masked_Q, get_joint_avail_actions
from .envs_runner import EnvsRunner

# parameters for e-greedy policy
EPS_START = 1.0
EPS_DECAY = 2000
EPS_DECAY_LINEAR_RATE = 0.9999

OPTIMIZERS = {'Adam': Adam,
              'RMSprop': RMSprop}

class Team:

    """Base class of a team of agents"""

    def __init__(self, 
                 envs,
                 memories,
                 n_agent, 
                 h_stable_at, 
                 dynamic_h=False, 
                 hysteretic=None, 
                 discount=0.99,
                 epsilon_end = 0.1,
                 epsilon_linear_decay=False, 
                 epsilon_linear_decay_steps=0,
                 eval_freq=100):

        """
        Parameters
        ----------
        envs : gym.env | Tuple(..)
            A tuple of domain environments for decentralized agents and centralized agents respectively.
        memories : ReplayBuffer | Tuple[..]
            A tuple of instances of the ReplayBuffer class for saving decentralized and centralized experiences respectively.
        n_agent : int
            The number of agent.
        h_stable_at : int
            The number of dacaying episodes/stpes for hysteretic learning rate.
        dynamic_h : bool
            Whether apply hysteratic learning rate decay.
        hysteretic : tuple
            A tuple of initialzed and ending hysteritic learning rates.
        discount : float
            Discount factor for learning.
        epsilon_end : float
            The ending value of epsilon.
        epsilon_linear_decay : bool
            Whether apply epsilon decay for explorating policy
        epsilon_linear_decay_steps : int
            The number of episodes/steps for epsilon decay
        eval_freq : int
            The frequency to perform evaluation.
        """

        self.env_cen = envs[0]
        self.env_dec = envs[1]

        self.n_agent = n_agent

        self.memory_cen = memories[0]
        self.memory_dec = memories[1]

        self.step_count = 0.0
        self.episode_count = 0.0
        self.episode_rewards = 0.0
 
        # hysteretic settings
        self.dynamic_h = dynamic_h
        (self.init_hysteretic, self.end_hysteretic) = hysteretic
        self.hysteretic = self.init_hysteretic
        self.discount = discount

        # epsilon for e-greedy
        self.epsilon = EPS_START
        self.epsilon_end = epsilon_end
        self.epsilon_linear_decay = epsilon_linear_decay
        self.eps_l_d = Linear_Decay(epsilon_linear_decay_steps, EPS_START, epsilon_end)
        
        self.eval_freq = eval_freq

        self.HYSTERESIS_STABLE_AT = h_stable_at

    def create_agents(self):
        raise NotImplementedError

    def create_cen_controller(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def get_next_actions(self):
        raise NotImplementedError

    def update_dec_target_net(self):
        for agent in self.agents:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    def update_cen_target_net(self):
        self.cen_controller.target_net.load_state_dict(self.cen_controller.policy_net.state_dict())

    def train(self):
        raise NotImplementedError

    def update_epsilon(self, step):
        # update epsilon:
        if self.epsilon_linear_decay:
            self.epsilon = self.eps_l_d._get_value(step)
        else:
            self.epsilon = self.epsilon_end + (EPS_START - self.epsilon_end) * math.exp(-1. * (step//8)  / EPS_DECAY)
    
    def update_hysteretic(self, step):
        if self.dynamic_h:
            self.hysteretic = min(self.end_hysteretic,
                                  ((self.end_hysteretic - self.init_hysteretic) / self.HYSTERESIS_STABLE_AT) * step + self.init_hysteretic)
        else:
            self.hysteretic = 1 - self.epsilon
    
    def get_init_inputs(self):
        raise NotImplementedError

    def sep_joint_exps(self, joint_exps):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def load_check_point(self, idx_run):
        for idx, parent in enumerate(self.envs_runner_dec.parents):
            PATH = "./performance/" + self.save_dir + "/check_point/" + str(idx_run) + "_dec_env_rand_state_" + str(idx) + "1.tar"
            rand_states = torch.load(PATH)
            parent.send(('load_rand_states', rand_states))
        for idx, parent in enumerate(self.envs_runner_cen.parents):
            PATH = "./performance/" + self.save_dir + "/check_point/" + str(idx_run) + "_cen_env_rand_state_" + str(idx) + "1.tar"
            rand_states = torch.load(PATH)
            parent.send(('load_rand_states', rand_states))

        PATH = "./performance/" + self.save_dir + "/check_point/" + str(idx_run) + "_trainInfo_" + "1.tar"
        ckpt = torch.load(PATH)
        self.step_count = ckpt['cur_step']
        self.episode_count = ckpt['n_epi']
        self.cen_controller.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
        self.cen_controller.target_net.load_state_dict(ckpt['target_net_state_dict'])
        self.cen_controller.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.hysteretic = ckpt['cur_hysteretic']
        self.epsilon = ckpt['cur_eps']
        self.TEST_PERFORM = ckpt['TEST_PERFORM']
        self.memory_dec.buf = ckpt['mem_dec_buf']
        self.memory_cen.buf = ckpt['mem_cen_buf']
        random.setstate(ckpt['random_state'])
        np.random.set_state(ckpt['np_random_state'])
        torch.set_rng_state(ckpt['torch_random_state'])

        for idx, agent in enumerate(self.agents):
            PATH = "./performance/" + self.save_dir + "/check_point/" + str(idx_run) + "_agent_" + str(idx) + "1.tar"
            ckpt = torch.load(PATH)
            agent.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
            agent.target_net.load_state_dict(ckpt['target_net_state_dict'])
            agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

class Team_RNN(Team):

    def __init__(self, 
                 envs, 
                 env_terminate_step,
                 n_env, 
                 memories, 
                 n_agent, 
                 training_method, 
                 h_stable_at, 
                 discount=0.99, 
                 sample_epi=False, 
                 dynamic_h=False, 
                 hysteretic=None, 
                 h_explore=False, 
                 soft_action_selection=False, 
                 epsilon_end=0.1, 
                 epsilon_linear_decay=False, 
                 epsilon_linear_decay_steps=0, 
                 epsilon_exp_decay=False, 
                 optimizer='Adam', 
                 learning_rate=0.001, 
                 device='cpu', 
                 eval_freq=100,
                 save_dir=None, 
                 nn_model_params={}, 
                 seed=None,
                 log=False,
                 **hyper_params):

        """
        Parameters
        ----------
        envs : gym.env | Tuple(..)
            A tuple of domain environments for decentralized agents and centralized agents respectively.
        n_env : int
            The number of envs running in parallel.
        memories : ReplayBuffer | Tuple[..]
            A tuple of instances of the ReplayBuffer class for saving decentralized and centralized experiences respectively.
        n_agent : int
            The number of agent.
        training_method : python function
            A algorithm for calculating loss and performing optimization.
        h_stable_at : int
            The number of dacaying episodes/stpes for hysteretic learning rate.
        discount : float
            Discount factor for learning.
        sample_epi : bool
            Whether simples entire episode in mini-batch learning.
        dynamic_h : bool
            Whether apply hysteratic learning rate decay.
        hysteretic : tuple
            A tuple of initialzed and ending hysteritic learning rates.
        h_explore : bool
            Whether uses history-based exploring policy.
        soft_action_selection : bool
            Whether apply soft action selection.
        epsilon_end : float
            The ending value of epsilon-based exlporation policy
        epsilon_linear_decay : bool
            Whether apply epsilon decay for explorating policy
        epsilon_linear_decay_steps : int
            The number of episodes/steps for epsilon decay
        epsilon_exp_decay : bool
            Whether apply exponentially decay for epsilon.
        optimizer : str
            Name of an optimizer.
        learning_rate : float
            Learning rate.
        device : str
            CPU/GPU for training.
        eval_freq : int
            The frequency to perform evaluation.
        save_dir : str
            Name of a directory to save results/ckpt.
        nn_model_params : dict[..]
            A dictionary of network parameters.
        hyper_params : dict[..] 
            A dictionary of some rest hyper-parameters.
        """

        super(Team_RNN, self).__init__(envs, memories, n_agent, h_stable_at, dynamic_h, hysteretic, discount,
                                       epsilon_end, epsilon_linear_decay, epsilon_linear_decay_steps, eval_freq=eval_freq)

        # create multiprocessor for multiple envs running parallel
        self.envs_runner_cen = EnvsRunner(self.env_cen, env_terminate_step, self.memory_cen, n_env, h_explore, self.get_next_action_cen, self.discount, seed, log=log)
        self.envs_runner_cen.reset()

        self.envs_runner_dec = EnvsRunner(self.env_dec, env_terminate_step, self.memory_dec, n_env, h_explore, self.get_next_action_dec, self.discount, seed)
        self.envs_runner_dec.reset()
 
        self.n_env = n_env
        self.env_terminate_step = env_terminate_step

        # sample the whole episode for training
        self.sample_epi = sample_epi
        
        # training method
        self.soft_action_selection = soft_action_selection
        self.training_method = training_method
        self.nn_model_params = nn_model_params
        self.hyper_params = hyper_params
        self.optimizer = optimizer
        self.lr = learning_rate
        
        # save model
        self.save_dir = save_dir
        self.device = device

        # statistic of training and testing
        self.TRAIN_PERFORM = []
        self.TEST_PERFORM = []

        # tensorboard log
        self.log = log
 
        # create agents
        self.create_agents()

        # create cen_controller
        self.create_cen_controller()

    def create_agents(self):
        self.agents=[]
        for i in range(self.n_agent):
            agent = Agent()
            agent.idx = i
            agent.policy_net = DDRQN(self.env_dec.obs_size[i], self.env_dec.n_action[i], **self.nn_model_params).to(self.device)
            agent.target_net = DDRQN(self.env_dec.obs_size[i], self.env_dec.n_action[i], **self.nn_model_params).to(self.device)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            agent.optimizer = OPTIMIZERS[self.optimizer](agent.policy_net.parameters(), lr=self.lr)
            self.agents.append(agent)

    def create_cen_controller(self):
        self.cen_controller = Cen_Controller()
        self.cen_controller.policy_net = Cen_DDRQN(np.sum(self.env_cen.obs_size),np.prod(self.env_cen.n_action), **self.nn_model_params).to(self.device)
        self.cen_controller.target_net = Cen_DDRQN(np.sum(self.env_cen.obs_size),np.prod(self.env_cen.n_action), **self.nn_model_params).to(self.device)
        self.cen_controller.target_net.load_state_dict(self.cen_controller.policy_net.state_dict())
        self.cen_controller.optimizer = OPTIMIZERS[self.optimizer](self.cen_controller.policy_net.parameters(), lr=self.lr)

    def step(self, idx_run):
        if self.step_count == 0:
            self.evaluate()
            with open("./performance/" + self.save_dir + "/test/test_perform" + str(idx_run) + ".pickle", 'wb') as handle:
                pickle.dump(self.TEST_PERFORM, handle)

        self.step_count += 1.0

        n_episode_done = self.envs_runner_dec.step()
        self.envs_runner_cen.step()

        self.episode_count += n_episode_done

        if n_episode_done > 0 and not self.episode_count % self.eval_freq:
            self.evaluate()
            if self.TEST_PERFORM[-1] == np.max(self.TEST_PERFORM):
                for agent in self.agents:
                     PATH = "./policy_nns/" + self.save_dir + "/" + str(idx_run) + "_agent_" + str(agent.idx) + ".pt"
                     torch.save(agent.policy_net, PATH)

            # PATH = "./policy_nns/" + self.save_dir + "/" + str(idx_run) + "_cen_controller.pt"
            # torch.save(self.cen_controller.policy_net, PATH)
            if not self.episode_count % (self.eval_freq*10):
                with open("./performance/" + self.save_dir + "/test/test_perform" + str(idx_run) + ".pickle", 'wb') as handle:
                    pickle.dump(self.TEST_PERFORM, handle)
                if self.log:
                    with open("./performance/" + self.save_dir + "/train/train_cen_perform" + str(idx_run) + ".pickle", 'wb') as handle:
                        pickle.dump(self.envs_runner_cen.train_returns, handle)

    def get_next_action_cen(self, 
                            joint_obs, 
                            dec_h_states, 
                            cen_h_state, 
                            cen_last_action, 
                            last_valid, 
                            avail_actions,
                            eval=False):

        """
        Parameters
        ----------
        joint_obs : ndarry | List[..]
            A list of each agent's observation.
        dec_h_states : ndarry | List[..]
            A list of hidden state of each agent's rnn-net
        cen_h_state : ndarry | Tuple(..)
            A tuple of hidden state of centralized rnn-net
        cen_last_action : int | List[..]
            A list of indice of agents' previous macro-actions selected by centralized policy.
        last_valid : binary (1/0) | List[..]
            A list of binary values indicates whether each agent has finished the previous macro-action or not.
        eval : bool
            Whether use evaluation mode or not.

        Returns
        -------
        cen_actions : int | List[..]
            A list of indice of agents' macro-actions selected by centralized policy.
        None
        cen_h_state : ndarry | tuple(..)
            A tuple of hidden states of centralized rnn net.
        """

        # explore using cen policy
        with torch.no_grad():
            if max(last_valid) == 1.0:
                joint_avail_actions = get_joint_avail_actions(avail_actions)
                self.cen_controller.policy_net.eval()
                Q, h = self.cen_controller.policy_net(torch.cat(joint_obs).view(1,1,np.sum(self.env_cen.obs_size)), cen_h_state)
                a = get_conditional_argmax(Q, 
                                           get_conditional_action(torch.cat(cen_last_action).view(1,-1), 
                                                                  torch.cat(last_valid).view(1,-1)), 
                                           joint_avail_actions, 
                                           self.env_cen.n_action).item()

                if self.soft_action_selection:
                    logits = torch.log_softmax((Q/self.epsilon), 1)
                    logits = logits.masked_fill(joint_avail_actions==0.0, -float('inf'))
                    a = torch.distributions.Categorical(logits=logits).sample().item()
                else:
                    if np.random.random() < self.epsilon:
                        prob = joint_avail_actions[0] / torch.sum(joint_avail_actions[0])
                        a = np.random.choice(range(Q.shape[-1]), 1, p=prob.numpy())[0]

                cen_actions = np.unravel_index(a, self.env_cen.n_action)
                cen_h_state = h
            else:
                cen_actions = [-1] * self.n_agent

        return cen_actions, None, cen_h_state

    def get_next_action_dec(self, 
                            joint_obs, 
                            dec_h_states, 
                            cen_h_state, 
                            cen_last_action, 
                            last_valid, 
                            avail_actions,
                            eval=False):

        """
        Parameters
        ----------
        joint_obs : ndarry | List[..]
            A list of each agent's observation.
        dec_h_states : ndarry | List[..]
            A list of hidden state of each agent's rnn-net
        cen_h_state : ndarry | Tuple(..)
            A tuple of hidden state of centralized rnn-net
        cen_last_action : int | List[..]
            A list of indice of agents' previous macro-actions selected by centralized policy.
        last_valid : binary (1/0) | List[..]
            A list of binary values indicates whether each agent has finished the previous macro-action or not.
        eval : bool
            Whether use evaluation mode or not.

        Returns
        -------
        dec_actions : int | List[..]
            A list of indice of agents' macro-actions selected by decentralized policy.
        new_dec_h_states : ndarry | List[tuple(..)]
            A list of hiddent states of each agent's rnn-net
        None
        """

        # explore using dec policies
        with torch.no_grad():
            dec_actions = []
            new_dec_h_states = []
            for i, agent in enumerate(self.agents):
                if last_valid[agent.idx]:
                    agent.policy_net.eval()
                    Q, h = agent.policy_net(joint_obs[agent.idx].view(1,1,self.env_dec.obs_size[agent.idx]), dec_h_states[agent.idx])
                    Q = get_masked_Q(Q, avail_actions[agent.idx])
                    a = Q.squeeze(1).max(1)[1].item()

                    if not eval:
                        if self.soft_action_selection:
                            logits = torch.log_softmax((Q/self.epsilon), 1)
                            a = torch.distributions.Categorical(logits=logits).sample().item()
                        else:
                            if np.random.random() < self.epsilon:
                                p = (avail_actions[agent.idx] / avail_actions[agent.idx].sum()).numpy().flatten()
                                a = np.random.choice(np.arange(self.env_dec.n_action[agent.idx]), p=p)

                    dec_actions.append(a)
                    new_dec_h_states.append(h)

                else:
                    dec_actions.append(-1)
                    new_dec_h_states.append(dec_h_states[agent.idx])

        return dec_actions, new_dec_h_states, None

    def train(self):
        if self.sample_epi:
            batch_cen, trace_len_cen = self.memory_cen.sample()
            batch_dec, trace_len_dec = self.memory_dec.sample()
            self.hyper_params['trace_lens'] = (trace_len_cen, trace_len_dec)
        else:
            batch_cen = self.memory_cen.sample()
            batch_dec = self.memory_dec.sample()
        cen_batch = self.cen_sep_joint_exps(batch_cen)
        dec_batch_cen = self.cen_sep_joint_exps(batch_dec)
        dec_batch_dec = self.dec_sep_joint_exps(batch_dec)

        self.training_method(self.env_dec, 
                             self.agents, 
                             self.cen_controller, 
                             (cen_batch, dec_batch_cen, dec_batch_dec), 
                             self.hysteretic, 
                             self.discount, 
                             **self.hyper_params)

    def get_init_inputs(self):
        return [torch.from_numpy(i).float() for i in self.env_dec.reset()], [None]*self.n_agent, None

    def dec_sep_joint_exps(self, joint_exps):

        """
        Parameters
        ----------
        joint_exps : List[List[tuple(..)]]
            A sampled batch of episodes/sequences, whose size equals to the number of episodes..

        Return
        ------
        exps : List[List[List(..)]]
            A separeted batch of episdoes/sequences for each agent, whose size equals to the number of agents. 
        """
 
        # seperate the joint experience for individual agents
        exps = [[] for _ in range(self.n_agent)]
        for o, avail_a, a, id_r, j_r, id_gamma, j_gamma, o_n, avail_a_n, t, id_v, j_v in chain(*joint_exps):
            for i in range(self.n_agent):
                exps[i].append([o[i], 
                                a[i], 
                                id_r[i], 
                                id_gamma[i],
                                o_n[i], 
                                t, 
                                id_v[i], 
                                j_v])
        return exps
    
    def cen_sep_joint_exps(self, joint_exps):

        """
        Parameters
        ----------
        joint_exps : List[List[tuple(..)]]
            A sampled batch of episodes/sequences, whose size equals to the number of episodes..

        Return
        ------
        exps : List[List(..)]
            A batch of episdoes/sequences of joint experiences.
        """

        # seperate the joint experience for individual agents
        exp = []
        for o, avail_a, a, id_r, j_r, id_gamma, j_gamma, o_n, avail_a_n, t, id_v, j_v in chain(*joint_exps):
            exp.append([torch.cat(o).view(1,-1), 
                        torch.cat(a).view(1,-1),
                        torch.tensor(np.ravel_multi_index(a, self.env_cen.n_action)).view(1,-1),
                        id_r, 
                        j_r, 
                        j_gamma,
                        torch.cat(o_n).view(1,-1),
                        get_joint_avail_actions(avail_a_n),
                        t,
                        torch.cat(id_v).view(-1),
                        j_v])
        return exp

    def evaluate(self, n_episode=1):

        R, L = 0, 0

        for _ in range(n_episode):
            t = 0
            step = 0

            last_obs, dec_h_states, cen_h_state = self.get_init_inputs()
            last_valid = [torch.tensor([[1]], dtype=torch.bool)] * self.n_agent
            cen_last_action = [torch.tensor([[-1]])] * self.n_agent
            avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in self.env_dec.get_avail_actions()]

            while not t and step < self.env_terminate_step:
                a, dec_h_states, cen_h_state = self.get_next_action_dec(last_obs, 
                                                                        dec_h_states, 
                                                                        cen_h_state, 
                                                                        cen_last_action, 
                                                                        last_valid, 
                                                                        avail_actions,
                                                                        eval=True)
                obs, r, t, info = self.env_dec.step(a)
                last_obs = [torch.from_numpy(o).float() for o in obs]
                cen_last_action = [torch.tensor(a_idx).view(1,1) for a_idx in info['cur_mac']]
                last_valid = [torch.tensor(_v, dtype=torch.bool).view(1,-1) for _v in info['mac_done']]
                avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in self.env_dec.get_avail_actions()]

                R += self.discount**step * sum(r)/self.env_dec.n_agent
                step += 1

        self.TEST_PERFORM.append(R/n_episode)
