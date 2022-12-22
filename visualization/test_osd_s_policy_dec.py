import argparse
import numpy as np
import torch
import os
import sys
sys.path.append("..")
import time
import IPython
import logging
import gym
from torch.distributions import Categorical
import random

from macro_marl import my_env
from macro_marl.MA_iaicc_rnn_V.utils import Agent
from gym.wrappers import Monitor



def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def update_log(agent_idx, obs, action):
    if agent_idx == 2:
        fetch_logger.info(obs[0][0][0:3])
        fetch_logger.info(obs[0][0][3:])
        fetch_logger.info(action)
        fetch_logger.info("  ")
    elif agent_idx == 1:
        tb1_logger.info(obs[0][0][0:5])
        tb1_logger.info(obs[0][0][5:9])
        tb1_logger.info(obs[0][0][9:13])
        tb1_logger.info(obs[0][0][13:16])
        tb1_logger.info(obs[0][0][16:])
        tb1_logger.info(action)
        tb1_logger.info("  ")

    else:
        tb0_logger.info(obs[0][0][0:5])
        tb0_logger.info(obs[0][0][5:9])
        tb0_logger.info(obs[0][0][9:13])
        tb0_logger.info(obs[0][0][13:16])
        tb0_logger.info(obs[0][0][16:])
        tb0_logger.info(action)
        tb0_logger.info("  ")


def get_actions_and_h_states(env, agent, joint_obs, h_states_in, last_action, last_valid, log=False):
    with torch.no_grad():
        actions = []
        h_states_out = []
        for idx,agent in enumerate(agent):
            if last_valid[agent.idx]:
                jobs = joint_obs[agent.idx].view(1,1,env.obs_size[agent.idx])
                one_h_state = h_states_in[agent.idx]
                Q, h = agent.policy_net(jobs, one_h_state)
                a = Q.squeeze(1).max(1)[1].item()
                actions.append(a)
                h_states_out.append(h)
            else:
                actions.append(env.agents[agent.idx].cur_action.idx)
                h_states_out.append(h_states_in[agent.idx])

    return actions, h_states_out

def get_init_inputs(env,n_agent):
    return [torch.from_numpy(i).float() for i in env.reset()], [None]*n_agent

def test(env_id, scenario, env_terminate_step, grid_dim, n_agent, n_episode, p_id, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if env_id in ['OSD-D-v7']:
        human_speed = [[18,15,15,15], [48,18,15,15]] #pid 15 533
    elif env_id in ['OSD-T-v0']:
        human_speed = [[40]*4, [40]*4, [40]*4] # pid 9 801
    elif env_id in ['OSD-T-v1']:
        human_speed = [[38]*4, [38]*4, [27]*4] # pid 11 801
    else:
        human_speed = [[40]*4, [40]*4, [40]*4, [40]*4] # pid 19 1050
 
    env_params = {'grid_dim': grid_dim,
                  'n_agent': n_agent,
                  'penalty': -10,
                  'TB_move_speed': 0.8,
                  'delay_delivery_penalty': -20,
                  'human_speed_per_step': human_speed,
                  'fetch_pass_obj_tc': 4, 
                  'fetch_look_for_obj_tc': 6, 
                  'render': True
                  }

    env = gym.make(env_id, **env_params)
    env.reset()

    agents = []

    for i in range(n_agent):
        agent = Agent()
        agent.idx = i
        if "OSD-D" in env_id:
            agent.policy_net = torch.load("./policy_nns/OSD_D/" + scenario + "/agent_" + str(i) + ".pt")
        elif "OSD-T" in env_id:
            agent.policy_net = torch.load("./policy_nns/OSD_T/" + scenario + "/agent_" + str(i) + ".pt")
        else:
            agent.policy_net = torch.load("./policy_nns/OSD_F/" + scenario + "/agent_" + str(i) + ".pt")
        agent.policy_net.eval()
        agents.append(agent)

    R = 0
    discount = 1.0

    for e in range(n_episode):
        t = 0
        step=0
        last_obs, h_states = get_init_inputs(env, n_agent)
        last_valid = [torch.tensor([[1]]).byte()] * n_agent
        last_action = [torch.tensor([[-1]])] * n_agent
        while not t:
            a, h_states = get_actions_and_h_states(env, agents, last_obs, h_states, last_action, last_valid)
            time.sleep(0.2)

            last_obs, r, t, info = env.step(a)
            last_obs = [torch.from_numpy(o).float() for o in last_obs]
            last_action = [torch.tensor(a_idx).view(1,1) for a_idx in info['cur_mac']]
            last_valid = [torch.tensor(_v, dtype=torch.uint8).view(1,-1) for _v in info['mac_done']]
            R += discount**step*r[0]
            step += 1

        if t:
            print(R)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', action='store', type=str, default='OSD-D-v7')
    parser.add_argument('--scenario', action='store', type=str, default='18P20')
    parser.add_argument('--env_terminate_step', action='store', type=int, default=200)
    parser.add_argument('--grid_dim', action='store', type=int, nargs=2, default=[4,4])
    parser.add_argument('--n_agent', action='store', type=int, default=3)
    parser.add_argument('--seed', action='store', type=int, default=0)
    parser.add_argument('--n_episode', action='store', type=int, default=1)
    parser.add_argument('--p_id', action='store', type=int, default=0)

    test(**vars(parser.parse_args()))

if __name__ == '__main__':
    main()
