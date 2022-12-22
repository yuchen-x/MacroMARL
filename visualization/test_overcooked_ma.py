import argparse
import numpy as np
import torch
import os
import sys
sys.path.append("..")
import time
import gym
from torch.distributions import Categorical

from macro_marl.MA_iaicc_rnn_V.utils import Agent


import argparse
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

def get_actions_and_h_states(env, agents, last_valid, joint_obs, joint_h_states):
    with torch.no_grad():
        actions = []
        new_h_states = []
        for idx, agent in enumerate(agents):
            action_logits, new_h_state = agent.policy_net(joint_obs[agent.idx].view(1,1,env.obs_size[agent.idx]), joint_h_states[agent.idx])
            action_prob = Categorical(logits=action_logits[0])
            action = action_prob.sample().item()
            actions.append(action)
            new_h_states.append(new_h_state)
    return actions, new_h_states


def get_init_inputs(env,n_agent):
    return [torch.from_numpy(i).float() for i in env.reset()], [None]*n_agent

def test(env_id, grid_dim, mapType, task, n_agent, p_id):

    TASKLIST = ["tomato salad", 
                "lettuce salad", 
                "onion salad", 
                "lettuce-tomato salad", 
                "onion-tomato salad", 
                "lettuce-onion salad", 
                "lettuce-onion-tomato salad"]
    rewardList = {"subtask finished": 10, 
                  "correct delivery": 200, 
                  "wrong delivery": -5, 
                  "step penalty": -0.1}
    env_params = {'grid_dim': grid_dim,
                  'task': TASKLIST[task],
                  'rewardList': rewardList,
                  'map_type': mapType,
                  'n_agent': n_agent,
                  'debug': True
                  }
    env = gym.make(env_id, **env_params)
    env = MacEnvWrapper(env)

    agents = []
    
    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        agent.policy_net = torch.load("./policy_nns/Overcooked/map" + mapType + "/agent_" + str(i) + ".pt")
        agent.policy_net.eval()
        agents.append(agent)

    R = 0
    discount=0.99
    step = 0.0
    n_episode = 1

    for e in range(n_episode):
        t = 0
        last_obs, h_states = get_init_inputs(env, n_agent)
        env.render()
        last_valid = [1.0] * n_agent
        while not t:
            a, h_states = get_actions_and_h_states(env, agents, last_valid, last_obs, h_states)
            last_obs, r, t, info = env.step(a)
            env.render()
            last_obs = [torch.from_numpy(o).float() for o in last_obs]
            last_valid = info['mac_done'] 
            R += discount**step*r[0]
            step += 1.0
        print(R)
        print("step", step)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='Overcooked-MA-v1')
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[7,7], choices=[[7, 7], [9, 9]])
    parser.add_argument('--mapType', action='store', type=str, default="C", choices=["A", "B", "C", "D", "E", "F"])
    parser.add_argument('--task', action='store', type=int, default=6, choices=[3, 6])
    parser.add_argument('--n_agent', action='store', type=int, default=2)
    parser.add_argument('--p_id',               action='store',        type=int,             default=0,                 help="The specific policy_id")

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()

