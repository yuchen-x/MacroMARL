import numpy as np
import torch
import IPython

from IPython.core.debugger import set_trace
from collections import deque

class ReplayMemory:

    """Base class of a replay buffer"""

    def __init__(self, n_agent, obs_size, n_action, batch_size, size):

        """
        Parameters
        ----------
        n_agent : int
            The number of agents.
        obs_size : int | List[..]
            A list of agent's obsvation size.
        n_action : int | List[..]
            The number of actions of each agent.
        batch_size : int
            The number of episodes/sequences for batch sampling.
        size : int
            The number of episodes/sequences in replay buffer.
        """
        # obs_size is a list, coresponde to n_agents
        assert len(obs_size) == n_agent

        self.batch_size, self.obs_size = batch_size, obs_size

        # ZERO PADDINGs
        self.ZERO_JOINT_OBS = [torch.zeros(s) for s in obs_size]
        self.ZERO_JOINT_ACT = [torch.tensor(0).view(1,-1)] * n_agent
        self.ZERO_ID_REWARD = [torch.tensor(0.0).view(1,-1)] * n_agent
        self.ZERO_JOINT_REWARD = torch.tensor(0.0).view(1,-1)
        self.ZERO_JOINT_GAMMA = torch.tensor(0.0).view(1,-1)
        self.ZERO_JOINT_AVAIL_ACT = [torch.zeros(n).view(1,-1) for n in n_action]
        self.ZERO_ID_VALID = [torch.tensor(0, dtype=torch.bool).view(1,-1)] * n_agent
        self.ZERO_JOINT_VALID = torch.tensor(0, dtype=torch.bool).view(1,-1)

        self.ONE_ID_VALID = [torch.tensor(1, dtype=torch.bool).view(1,-1)] * n_agent
        self.ONE_JOINT_VALID = torch.tensor(1, dtype=torch.bool).view(1,-1)

        self.ZERO_PADDING = [(self.ZERO_JOINT_OBS, 
                              self.ZERO_JOINT_AVAIL_ACT,
                              self.ZERO_JOINT_ACT, 
                              self.ZERO_ID_REWARD,
                              self.ZERO_JOINT_REWARD, 
                              self.ZERO_JOINT_GAMMA,
                              self.ZERO_JOINT_OBS, 
                              self.ZERO_JOINT_AVAIL_ACT,
                              torch.tensor(0).float().view(1,-1), 
                              self.ZERO_ID_VALID, 
                              self.ZERO_JOINT_VALID)]

        self.ZEROS_ONE_PADDING = [(self.ZERO_JOINT_OBS,
                                   self.ZERO_JOINT_ACT, 
                                   self.ZERO_ID_REWARD,
                                   self.ZERO_JOINT_REWARD, 
                                   self.ZERO_JOINT_OBS, 
                                   torch.tensor(0).float().view(1,-1), 
                                   self.ONE_ID_VALID,
                                   self.ONE_JOINT_VALID)]

        self.buf = deque(maxlen=size)

    def append(self, transition):
        self.scenario_cache.append(transition)

    def scenario_cache_reset(self):
        raise NotImplementedError

    def flush_scenario_cache(self):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError

class ReplayMemory_rand(ReplayMemory):

    """A replay buffer collects sequences with certain length"""

    def __init__(self, n_agent, obs_size, n_action, trace_len, batch_size, size=100000):

        """
        Parameters
        ----------
        n_agent : int
            The number of agents.
        obs_size : int | List[..]
            A list of agent's obsvation size.
        trace_len : int 
            The length of sequential experiments for sampling.
        batch_size : int
            The number of episodes/sequences for batch sampling.
        size : int
            The number of episodes/sequences in replay buffer.
        """

        super(ReplayMemory_rand, self).__init__(n_agent, obs_size, n_action, batch_size, size)
        self.trace_len = trace_len
        self.scenario_cache_reset()

    def flush_scenario_cache(self):
        for i in range(len(self.scenario_cache)):
            trace = self.scenario_cache[i:i+self.trace_len]
            # end-of-episode padding
            trace = trace + self.ZERO_PADDING * (self.trace_len - len(trace))
            self.buf.append(trace)
        self.scenario_cache_reset()

    def scenario_cache_reset(self):
        self.scenario_cache = self.ZERO_PADDING * (self.trace_len - 1)

    def sample(self):
        indices = np.random.choice(len(self.buf), self.batch_size, replace=False)
        return [self.buf[i] for i in indices]

class ReplayMemory_epi(ReplayMemory):

    """A replay buffer collects episodes and samples entire episodes"""
    
    def __init__(self, n_agent, obs_size, n_action, batch_size, size=100000):
        super(ReplayMemory_epi, self).__init__(n_agent, obs_size, n_action, batch_size, size)
        self.scenario_cache_reset()

    def flush_scenario_cache(self):
        self.buf.append(self.scenario_cache)
        self.scenario_cache_reset()

    def scenario_cache_reset(self):
        self.scenario_cache = []

    def sample(self):
        indices = np.random.choice(len(self.buf), self.batch_size, replace=False)
        batch = [self.buf[i] for i in indices]
        return self.padding_batches(batch)
    
    def padding_batches(self, batch):
        """
        Parameters
        ----------
        batch :  List[List[tuple(..)]]
            A list of episodes, and each episode is a list of tuples.

        Returns
        -------
        batch : episode | List[List[tuple(..)]] 
            A new batch with zero paddings.
        max_len : int 
            The length of the longest episode before padding. 
        """
        max_len = max([len(epi) for epi in batch])
        batch = [epi + self.ZERO_PADDING * (max_len - len(epi)) for epi in batch]
        return batch, max_len
