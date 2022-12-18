import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .utils.utils import get_masked_Q

n_iter=0

def QLearn_squ(agents, 
               batches, 
               hysteretic, 
               discount, 
               trace_len, 
               sub_trace_len, 
               batch_size, 
               sort_traj=False,
               huber_loss=False, 
               grad_clip=False, 
               grad_clip_value=None,
               grad_clip_norm=False, 
               grad_clip_max_norm=None, 
               rnn=True, 
               device='cpu', 
               writer=None,
               log=False,
               **kwargs):

    """
    Parameters
    ----------
    agents : Agent | List[..]
        A list of agent represented by an instance of Agent class.
    batches : List[List[tupe(..)]] 
        A list of each agent's episodic experiences, whoses size equals to the number of agents.
    discount : float
        Discount factor for learning.
    trace_len : int
        The length of the longest episode.
    sub_trace_len : int
        The length of the shortes episode for filtering.
    batch_size : int
        The number of episodes/sequences in a batch.
    sort_traj : bool
        Whether sort the sampled episodes/sequences according to each episode's valid length after squeezing operation. 
    huber_loss : bool
        Whether apply huber loss or not.
    grad_clip : bool
        Whether use gradient clip or not.
    grad_clip_value : float
        Absolute value limitation for gradient clip.
    grad_clip_norm : bool
        Whether use norm-based gradient clip
    grad_clip_max_norm : float
        Norm limitation for gradient clip.
    rnn : bool
        whether use rnn-agent or not.
    device : str
        Which device (CPU/GPU) to use.
    """

    # calculate loss for each agent
    for agent, batch in zip(agents, batches):

        agent.policy_net.train()
        policy_net = agent.policy_net
        target_net = agent.target_net

        # seperate elements in the batch
        state_b, avail_a_b, action_b, reward_b, gamma_b, state_next_b, avail_next_a_b, terminate_b, valid_b = zip(*batch)

        assert len(state_b) == trace_len * batch_size, "policy_net dim problem ..."
        assert len(state_next_b) == trace_len * batch_size, "target_net dim problem ..."

        s_b = torch.cat(state_b).view(batch_size, trace_len, -1).to(device)                    #dim: (batch_size, trace_len, policy.net.input_dim)
        avail_a_b = torch.cat(avail_a_b).view(batch_size, trace_len, -1).to(device)            #dim: (batch_size, trace_len, policy.net.input_dim)
        a_b = torch.cat(action_b).view(batch_size, trace_len, 1).to(device)                    #dim: (batch_size, trace_len, 1)
        r_b = torch.cat(reward_b).view(batch_size, trace_len, 1).to(device)                    #dim: (batch_size, trace_len, 1)
        gamma_b = torch.cat(gamma_b).view(batch_size, trace_len, 1).to(device)                 #dim: (batch_size, trace_len, 1)
        s_next_b = torch.cat(state_next_b).view(batch_size, trace_len, -1).to(device)          #dim: (batch_size, trace_len, policy.net.input_dim)
        avail_next_a_b = torch.cat(avail_next_a_b).view(batch_size, trace_len, -1).to(device)  #dim: (batch_size, trace_len, policy.net.input_dim)
        t_b = torch.cat(terminate_b).view(batch_size, trace_len, 1).to(device)                 #dim: (batch_size, trace_len, 1)
        v_b = torch.cat(valid_b).view(batch_size, trace_len).to(device)                        #dim: (batch_size, trace_len)


        selected_traj_mask = v_b.sum(1) >= sub_trace_len
        if torch.sum(selected_traj_mask).item() == 0:
            return
        selected_lengths = v_b.sum(1)[selected_traj_mask]
        s_b = torch.split_with_sizes(s_b[selected_traj_mask][v_b[selected_traj_mask]], list(selected_lengths))
        s_next_b = torch.split_with_sizes(s_next_b[selected_traj_mask][v_b[selected_traj_mask]], list(selected_lengths))

        avail_a_b = avail_a_b[selected_traj_mask][v_b[selected_traj_mask]]
        avail_next_a_b = avail_next_a_b[selected_traj_mask][v_b[selected_traj_mask]]
        a_b = a_b[selected_traj_mask][v_b[selected_traj_mask]]
        r_b = r_b[selected_traj_mask][v_b[selected_traj_mask]]
        gamma_b = gamma_b[selected_traj_mask][v_b[selected_traj_mask]]
        t_b = t_b[selected_traj_mask][v_b[selected_traj_mask]]

        # get mask and padd sequence with different lenghts
        mask_s_b = get_mask_from_input(s_b)
        mask_s_next_b = get_mask_from_input(s_next_b)
        padded_s_b = pad_sequence(s_b, padding_value=torch.tensor(0), batch_first=True) 
        padded_s_next_b = pad_sequence(s_next_b, padding_value=torch.tensor(0), batch_first=True) 
        padded_s_next_b = torch.cat([padded_s_b[:,0].unsqueeze(1),
                                     padded_s_next_b],
                                     dim=1)
        if rnn:
            Q_s = policy_net(padded_s_b)[0]
            Q_s_next = policy_net(padded_s_next_b)[0][:,1:,:]
        else:
            Q_s  = policy_net(padded_s_b)
            Q_s_next  = policy_net(padded_s_next_b)

        Q_s = Q_s[mask_s_b]
        Q_s_next = Q_s_next[mask_s_next_b]
        assert Q_s.size(0) == a_b.size(0), "number of Qs doesn't match with number of actions"
        Q = Q_s.gather(1, a_b)

        # apply double Q learning
        Q_s_next = get_masked_Q(Q_s_next, avail_next_a_b)
        a_next_b = Q_s_next.max(1)[1].view(-1, 1)

        if rnn:
            target_Q_s_next = target_net(padded_s_next_b)[0].detach()[:,1:,:]
        else:
            target_Q_s_next = target_net(padded_s_next_b).detach()
        target_Q_s_next = target_Q_s_next[mask_s_next_b]

        target_Q = target_Q_s_next.gather(1, a_next_b)
        assert not torch.any(gamma_b==0), "Gamma is 0 ..."
        target_Q = r_b + discount * gamma_b * target_Q * (-t_b + 1)
        
        td_err = (target_Q - Q)
        td_err = torch.max(hysteretic*td_err, td_err)

        if huber_loss:
            agent.loss = torch.mean(torch.min(td_err*td_err*0.5, torch.abs(td_err)-0.5))
        else:
            agent.loss = torch.mean(td_err*td_err)

            # log for tensorboard
            if log:
                global n_iter
                writer.add_scalar('Q_loss/agent_'+str(agent.idx), agent.loss, n_iter)
    if log:
        n_iter += 1
        
    # optimize params for each agent
    for agent in agents:
        if agent.loss is not None:
            agent.optimizer.zero_grad()
            agent.loss.backward()

            if grad_clip:
                assert grad_clip_value is not None, "no grad_clip_value is given"
                for param in agent.policy_net.parameters():
                    param.grad.data.clamp_(-grad_clip_value, grad_clip_value)
            elif grad_clip_norm:
                assert grad_clip_max_norm is not None, "no grad_clip_max_norm is given"
                clip_grad_norm_(agent.policy_net.parameters(), grad_clip_max_norm)
            agent.optimizer.step()
            agent.loss = None

def get_mask_from_input(x):
    padded_x = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True) 
    return ~torch.isnan(padded_x).any(-1)


