import pickle
import torch
import os
import random
import numpy as np

class Agent:

    def __init__(self):
        self.idx = None
        self.actor_net = None
        self.actor_optimizer = None
        self.actor_loss = None

        self.critic_net = None
        self.critic_tgt_net = None
        self.critic_optimizer = None
        self.critic_loss = None

def save_checkpoint(run_id, epi_count, eval_returns, controller, envs_runner, save_dir, max_save=2):

    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_genric_" + "{}.tar"

    for n in list(range(max_save-1, 0, -1)):
        os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    torch.save({
                'epi_count': epi_count,
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.random.get_rng_state(),
                'envs_runner_returns': envs_runner.train_returns,
                'eval_returns': eval_returns,
                }, PATH)

    for idx, parent in enumerate(envs_runner.parents):
        parent.send(('get_rand_states', None))
    for idx, parent in enumerate(envs_runner.parents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_env_rand_states_" + str(idx) + "{}.tar"
        rand_states = parent.recv()
        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)
        torch.save(rand_states, PATH)

    for idx, agent in enumerate(controller.agents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_agent_" + str(idx) + "{}.tar"

        for n in list(range(max_save-1, 0, -1)):
            os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
        PATH = PATH.format(1)

        torch.save({
                    'actor_net_state_dict': agent.actor_net.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic_net_state_dict': agent.critic_net.state_dict(),
                    'critic_tgt_net_state_dict': agent.critic_tgt_net.state_dict(),
                    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                    },PATH)

def load_checkpoint(run_id, save_dir, controller, envs_runner):

    # load generic stuff
    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_genric_" + "1.tar"
    ckpt = torch.load(PATH)
    epi_count = ckpt['epi_count']
    random.setstate(ckpt['random_state'])
    np.random.set_state(ckpt['np_random_state'])
    torch.set_rng_state(ckpt['torch_random_state'])
    envs_runner.train_returns = ckpt['envs_runner_returns']
    eval_returns = ckpt['eval_returns']

    # load random states in all workers
    for idx, parent in enumerate(envs_runner.parents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_env_rand_states_" + str(idx) + "1.tar"
        rand_states = torch.load(PATH)
        parent.send(('load_rand_states', rand_states))

    # load actor and ciritc models
    for idx, agent in enumerate(controller.agents):
        PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_agent_" + str(idx) + "1.tar"
        ckpt = torch.load(PATH)
        agent.actor_net.load_state_dict(ckpt['actor_net_state_dict'])
        agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        agent.critic_net.load_state_dict(ckpt['critic_net_state_dict'])
        agent.critic_tgt_net.load_state_dict(ckpt['critic_tgt_net_state_dict'])
        agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])

    return epi_count, eval_returns
