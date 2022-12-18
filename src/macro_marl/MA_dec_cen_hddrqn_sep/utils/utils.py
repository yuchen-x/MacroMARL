import torch
import os
import numpy as np
import string
import random

class Linear_Decay(object):

    """A class for achieving value linear decay"""

    def __init__ (self, total_steps, init_value, end_value):

        """
        Parameters
        ----------
        total_steps : int
            Total decay steps.
        init_value : float
            Initial value.
        end_value : float
            Ending value
        """

        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def _get_value(self, step):
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value-self.init_value)

def get_conditional_argmax(Q, action_idxes, joint_avail_actions, action_space):
    size = []
    size.append(Q.shape[0])
    size += action_space
    _Q = Q.view(size)
    a_idxes = []
    for q_i in range(_Q.shape[0]):
        N = _Q[q_i].shape
        avail_actions = joint_avail_actions[q_i].reshape(N)
        avail_Q = _Q[q_i].masked_fill(avail_actions==0.0, -float('inf'))
        masks = []
        for i, idx in enumerate(action_idxes[q_i]):
            if idx == -1:
                m = torch.ones(N[i]).byte()
            else:
                m = torch.zeros(N[i]).byte()
                m[idx] = 1
            masks.append(m)
        letters = string.ascii_letters[:avail_Q.ndimension()]
        rule = ','.join(letters) + '->' + letters
        mask = torch.einsum(rule, *masks)
        Qmasked = avail_Q.where(mask, torch.tensor(-float('inf')))
        a_idxes.append(Qmasked.argmax())
    return torch.tensor(a_idxes).view(-1,1)

def get_conditional_action(actions, v):
    condi_a = actions.clone()
    condi_a[v] = -1
    return condi_a 

def get_joint_avail_actions(avail_actions):
    output = torch.ger(avail_actions[0].flatten(), avail_actions[1].flatten()).flatten()
    for i in range(2, len(avail_actions)):
        output = torch.ger(output, avail_actions[i].flatten()).flatten()
    return output.view(1,-1)

def get_masked_Q(Q, avail_action):
    masked_Q = Q.clone()
    return  masked_Q.masked_fill(avail_action==0.0, -float('inf'))

def save_check_point(dec_env,
                     cen_env,
                     agents, 
                     cen_controller, 
                     cur_step, 
                     n_epi, 
                     cur_hysteretic, 
                     cur_eps, 
                     save_dir, 
                     mem_cen, 
                     mem_dec, 
                     run_id, 
                     test_perform):

    """
    Parameters
    ----------
    agents : Agent | List[..]
        A list of agent represented by an instance of Agent class.
    cen_controller : Cen_Controller
        An instance of Cen_Controller class.
    cur_step : int
        Current training step.
    n_epi : int
        Current training episode.
    cur_hysteretic : float
        Current hysteritic learning rate
    cur_eps : float
        Current epslion value.
    save_dir : str
        Directory name for saving ckpt
    mem_cen : RepalyMemory 
        Replay buffer for storing centralized experiences.
    mem_dec : RepalyMemory 
        Replay buffer for storing decentralized experiences.
    run_id : int
        Index of current run.
    test_perform : float | List[..]
        A list of testing discounted return
    max_save : int
        How many recent ckpt to be kept there.
    """

    for idx, parent in enumerate(dec_env.parents):
        parent.send(('get_rand_states', None))
    for idx, parent in enumerate(dec_env.parents):
        PATH = "./performance/" + save_dir + "/check_point/" + str(run_id) + "_dec_env_rand_state_" + str(idx) + "1.tar"
        rand_states = parent.recv()
        torch.save(rand_states, PATH)

    for idx, parent in enumerate(cen_env.parents):
        parent.send(('get_rand_states', None))
    for idx, parent in enumerate(cen_env.parents):
        PATH = "./performance/" + save_dir + "/check_point/" + str(run_id) + "_cen_env_rand_state_" + str(idx) + "1.tar"
        rand_states = parent.recv()
        torch.save(rand_states, PATH)


    PATH = "./performance/" + save_dir + "/check_point/" + str(run_id) + "_trainInfo_" + "1.tar"
    torch.save({
                'cur_step': cur_step,
                'n_epi': n_epi,
                'policy_net_state_dict':cen_controller.policy_net.state_dict(),
                'target_net_state_dict':cen_controller.target_net.state_dict(),
                'optimizer_state_dict':cen_controller.optimizer.state_dict(),
                'cur_hysteretic':cur_hysteretic,
                'cur_eps':cur_eps,
                'TEST_PERFORM': test_perform,
                'mem_dec_buf':mem_dec.buf,
                'mem_cen_buf':mem_cen.buf,
                'random_state':random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_random_state': torch.random.get_rng_state()
                }, PATH)

    for idx, agent in enumerate(agents):
        PATH = "./performance/" + save_dir + "/check_point/" + str(run_id) + "_agent_" + str(idx) + "1.tar"
        torch.save({
                    'policy_net_state_dict':agent.policy_net.state_dict(),
                    'target_net_state_dict':agent.target_net.state_dict(),
                    'optimizer_state_dict':agent.optimizer.state_dict(),
                    }, PATH)
