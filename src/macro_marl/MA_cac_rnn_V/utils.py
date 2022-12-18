import pickle
import torch
import os
import random
import numpy as np
import string

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

class Linear_Decay(object):

    def __init__ (self, total_steps, init_value, end_value):
        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def get_value(self, step):
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value-self.init_value)

def save_policy(run_id, agent, save_dir):
    PATH = "./policy_nns/" + save_dir + "/" + str(run_id) + "_agent_cen.pt"
    torch.save(agent.actor_net, PATH)

def save_train_data(run_id, data, save_dir):
    with open("./performance/" + save_dir + "/train/train_perform" + str(run_id) + ".pickle", 'wb') as handle:
        pickle.dump(data, handle)

def save_test_data(run_id, data, save_dir):
    with open("./performance/" + save_dir + "/test/test_perform" + str(run_id) + ".pickle", 'wb') as handle:
        pickle.dump(data, handle)

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

    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_agent_" + "{}.tar"

    for n in list(range(max_save-1, 0, -1)):
        os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    torch.save({
                'actor_net_state_dict': controller.agent.actor_net.state_dict(),
                'actor_optimizer_state_dict': controller.agent.actor_optimizer.state_dict(),
                'critic_net_state_dict': controller.agent.critic_net.state_dict(),
                'critic_tgt_net_state_dict': controller.agent.critic_tgt_net.state_dict(),
                'critic_optimizer_state_dict': controller.agent.critic_optimizer.state_dict(),
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
    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_id) + "_agent_" + "1.tar"
    ckpt = torch.load(PATH)
    controller.agent.actor_net.load_state_dict(ckpt['actor_net_state_dict'])
    controller.agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
    controller.agent.critic_net.load_state_dict(ckpt['critic_net_state_dict'])
    controller.agent.critic_tgt_net.load_state_dict(ckpt['critic_tgt_net_state_dict'])
    controller.agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])

    return epi_count, eval_returns

def get_joint_avail_actions(avail_actions):
    output = torch.ger(avail_actions[0].flatten(), avail_actions[1].flatten()).flatten()
    for i in range(2, len(avail_actions)):
        output = torch.ger(output, avail_actions[i].flatten()).flatten()
    return output.view(1,-1)

def get_conditional_logits(logits, action_idxes, joint_avail_actions, action_space):
    shape = logits.shape
    logits = logits.reshape(-1, shape[-1])
    size = []
    size.append(logits.shape[0])
    size += action_space
    _logits = logits.view(size)
    new_logits = []
    for l_i in range(_logits.shape[0]):
        N = _logits[l_i].shape
        avail_actions = joint_avail_actions[l_i].reshape(N)
        avail_logits = _logits[l_i].masked_fill(avail_actions==0.0, -float('inf'))
        masks = []
        for i, idx in enumerate(action_idxes[l_i]):
            if idx == -1:
                m = torch.ones(N[i]).byte()
            else:
                m = torch.zeros(N[i]).byte()
                m[idx] = 1
            masks.append(m)
        letters = string.ascii_letters[:avail_logits.ndimension()]
        rule = ','.join(letters) + '->' + letters
        mask = torch.einsum(rule, *masks)
        logits_masked = avail_logits.where(mask, torch.tensor(-float('inf')))
        new_logits.append(logits_masked.view(1,-1))
    return torch.cat(new_logits).reshape(shape)

def get_conditional_action(actions, v):
    condi_a = actions.clone()
    condi_a[v] = -1
    return condi_a 




