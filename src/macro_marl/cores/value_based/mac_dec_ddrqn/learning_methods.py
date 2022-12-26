import numpy as np
import time
import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .utils.utils import get_conditional_action, get_conditional_argmax

def QLearn_squ_dec_cen_0(env, 
                         agents, 
                         cen_controller, 
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
                         **kwargs):

    """
    Parameters
    ----------
    env : gym.env
        A macro-action-based gym envrionment.
    agents : Agent | List[..]
        A list of agent represented by an instance of Agent class.
    cen_controller : Cen_Controller
         An instance of Cen_Controller class.
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

    """y_i = r_i + max_a'Q_w-(h_t+1^i, argmax_a_t+1^i Q_cen(h_t+1, a_t+1 | a_/i_undone))"""

    cen_controller.policy_net.train()

    cen_batch, dec_batches = batches

    #==================train centralized Q==============================================
    policy_net = cen_controller.policy_net
    target_net = cen_controller.target_net

    # seperate elements in the batch
    (state_b, 
     id_action_b, 
     j_action_b, 
     id_reward_b, 
     j_reward_b, 
     j_gamma_b,
     state_next_b, 
     avail_next_j_a_b,
     terminate_b, 
     id_valid_b, 
     j_valid_b) = zip(*cen_batch)

    assert len(state_b) == trace_len * batch_size, "batch's data problem ..."
    assert len(state_next_b) == trace_len * batch_size, "batch's data problem ..."

    s_b = torch.cat(state_b).view(batch_size, trace_len, -1).to(device)             #dim: (batch_size, trace_len, policy.net.input_dim)
    id_a_b = torch.cat(id_action_b).view(batch_size, trace_len, -1).to(device)      #dim: (batch_size, trace_len, 1)
    j_a_b = torch.cat(j_action_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
    j_r_b = torch.cat(j_reward_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
    j_gamma_b = torch.cat(j_gamma_b).view(batch_size, trace_len, 1).to(device)                     #dim: (batch_size, trace_len, 1)
    s_next_b = torch.cat(state_next_b).view(batch_size, trace_len, -1).to(device)   #dim: (batch_size, trace_len, policy.net.input_dim)
    avail_next_j_a_b = torch.cat(avail_next_j_a_b).view(batch_size, trace_len, -1).to(device)
    t_b = torch.cat(terminate_b).view(batch_size, trace_len, 1).to(device)          #dim: (batch_size, trace_len, 1)
    id_v_b = torch.cat(id_valid_b).view(batch_size, trace_len, -1).to(device)
    j_v_b = torch.cat(j_valid_b).view(batch_size, trace_len).to(device)             #dim: (batch_size, trace_len)

    selected_traj_mask = j_v_b.sum(1) >= sub_trace_len
    if torch.sum(selected_traj_mask).item() == 0:
        return
    selected_lengths = j_v_b.sum(1)[selected_traj_mask]

    s_b = torch.split_with_sizes(s_b[selected_traj_mask][j_v_b[selected_traj_mask]], list(selected_lengths))
    s_next_b = torch.split_with_sizes(s_next_b[selected_traj_mask][j_v_b[selected_traj_mask]], list(selected_lengths))

    avail_next_j_a_b = avail_next_j_a_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    id_a_b = id_a_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    j_a_b = j_a_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    j_r_b = j_r_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    j_gamma_b = j_gamma_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    t_b = t_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    id_v_b = id_v_b[selected_traj_mask][j_v_b[selected_traj_mask]]

    # get mask and padd sequence with different lenghts
    mask_s_b = get_mask_from_input(s_b)
    mask_s_next_b = get_mask_from_input(s_next_b)
    padded_s_b = pad_sequence(s_b, padding_value=torch.tensor(0), batch_first=True) 
    padded_s_next_b = pad_sequence(s_next_b, padding_value=torch.tensor(0), batch_first=True) 
    padded_s_next_b = torch.cat([padded_s_b[:,0].unsqueeze(1),
                                 padded_s_next_b],
                                 dim=1)

    Q_s = policy_net(padded_s_b)[0]
    Q_s_next = policy_net(padded_s_next_b)[0][:,1:,:]
    Q_s = Q_s[mask_s_b]
    Q_s_next = Q_s_next[mask_s_next_b]
    assert Q_s.size(0) == j_a_b.size(0), "number of Qs doesn't match with number of actions"

    Q = Q_s.gather(1, j_a_b)

    # apply double Q learning
    # get conditional action indexes
    condi_a = get_conditional_action(id_a_b, id_v_b)
    j_a_next_b = get_conditional_argmax(Q_s_next, condi_a, avail_next_j_a_b, env.n_action)
    target_Q_s_next = target_net(padded_s_next_b)[0].detach()[:,1:,:]
    target_Q_s_next = target_Q_s_next[mask_s_next_b]

    target_Q = target_Q_s_next.gather(1, j_a_next_b)
    target_Q = j_r_b + discount * j_gamma_b * target_Q * (-t_b + 1)

    if huber_loss:
        cen_controller.loss = F.smooth_l1_loss(Q, target_Q)
    else:
        td_err = (target_Q - Q) 
        cen_controller.loss = torch.mean(td_err*td_err)

    if cen_controller.loss is not None:
        cen_controller.optimizer.zero_grad()
        cen_controller.loss.backward()

        if grad_clip:
            assert grad_clip_value is not None, "no grad_clip_value is given"
            for param in cen_controller.policy_net.parameters():
                param.grad.data.clamp_(-grad_clip_value, grad_clip_value)
        elif grad_clip_norm:
            assert grad_clip_max_norm is not None, "no grad_clip_max_norm is given"
            clip_grad_norm_(cen_controller.policy_net.parameters(), grad_clip_max_norm)

        cen_controller.optimizer.step()
        cen_controller.loss = None

    cen_Q_s_next = cen_controller.policy_net(padded_s_next_b)[0].detach()[:,1:,:]
    cen_Q_s_next = cen_Q_s_next[mask_s_next_b]
    condi_a = get_conditional_action(id_a_b, id_v_b)
    j_a_next_b = get_conditional_argmax(cen_Q_s_next, condi_a, avail_next_j_a_b, env.n_action)

    #==================train decentralized Q==============================================

    # calculate loss for each agent
    for agent, batch in zip(agents, dec_batches):

        agent.policy_net.train()

        policy_net = agent.policy_net
        target_net = agent.target_net

        # seperate elements in the batch
        (state_b, 
         action_b, 
         id_reward_b, 
         gamma_b,
         state_next_b, 
         terminate_b, 
         id_valid_b, 
         j_valid_b) = zip(*batch)

        assert len(state_b) == trace_len * batch_size, "policy_net dim problem ..."
        assert len(state_next_b) == trace_len * batch_size, "target_net dim problem ..."

        s_b = torch.cat(state_b).view(batch_size, trace_len, -1).to(device)            #dim: (batch_size, trace_len, policy.net.input_dim)
        a_b = torch.cat(action_b).view(batch_size, trace_len, 1).to(device)            #dim: (batch_size, trace_len, 1)
        r_b = torch.cat(id_reward_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
        gamma_b = torch.cat(gamma_b).view(batch_size, trace_len, 1).to(device)                 #dim: (batch_size, trace_len, 1)
        s_next_b = torch.cat(state_next_b).view(batch_size, trace_len, -1).to(device)  #dim: (batch_size, trace_len, policy.net.input_dim)
        t_b = torch.cat(terminate_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
        v_b = torch.cat(id_valid_b).view(batch_size, trace_len).to(device)             #dim: (batch_size, trace_len)

        selected_traj_mask = v_b.sum(1) >= sub_trace_len
        if torch.sum(selected_traj_mask).item() == 0:
            return
        selected_lengths = v_b.sum(1)[selected_traj_mask]
        s_b = torch.split_with_sizes(s_b[selected_traj_mask][v_b[selected_traj_mask]], list(selected_lengths))
        s_next_b = torch.split_with_sizes(s_next_b[selected_traj_mask][v_b[selected_traj_mask]], list(selected_lengths))

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

        Q_s, _ = policy_net(padded_s_b)
        Q_s = Q_s[mask_s_b]
        assert Q_s.size(0) == a_b.size(0), "number of Qs doesn't match with number of actions"
        Q = Q_s.gather(1, a_b)

        # apply double cen Q learning
        a_next_b = torch.tensor(np.unravel_index(j_a_next_b[id_v_b[:,agent.idx]], env.n_action)[agent.idx]).view(-1,1)
        target_Q_s_next = target_net(padded_s_next_b)[0].detach()[:,1:,:]
        target_Q_s_next = target_Q_s_next[mask_s_next_b]
        target_Q = target_Q_s_next.gather(1, a_next_b)
        target_Q = r_b + discount * gamma_b * target_Q * (-t_b + 1)

        td_err = (target_Q - Q) 
        td_err = torch.max(hysteretic*td_err, td_err)

        if huber_loss:
            agent.loss = torch.mean(torch.min(td_err*td_err*0.5, torch.abs(td_err)-0.5))
        else:
            agent.loss = torch.mean(td_err*td_err)

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


def QLearn_squ_dec_cen_1(env, 
                         agents, 
                         cen_controller, 
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
                         **kwargs):

    """y_i = r_i + max_a'Q_w-(h_t+1^i, argmax_a_t+1^i Q_w(h_t+1, a_t+1))"""

    cen_controller.policy_net.train()

    cen_batch, dec_batches = batches

    #==================train centralized Q==============================================
    policy_net = cen_controller.policy_net
    target_net = cen_controller.target_net

    # seperate elements in the batch
    (state_b, 
     id_action_b, 
     j_action_b, 
     id_reward_b, 
     j_reward_b, 
     j_gamma_b,
     state_next_b, 
     avail_next_j_a_b,
     terminate_b, 
     id_valid_b, 
     j_valid_b) = zip(*cen_batch)

    assert len(state_b) == trace_len * batch_size, "batch's data problem ..."
    assert len(state_next_b) == trace_len * batch_size, "batch's data problem ..."

    s_b = torch.cat(state_b).view(batch_size, trace_len, -1).to(device)             #dim: (batch_size, trace_len, policy.net.input_dim)
    id_a_b = torch.cat(id_action_b).view(batch_size, trace_len, -1).to(device)      #dim: (batch_size, trace_len, 1)
    j_a_b = torch.cat(j_action_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
    j_r_b = torch.cat(j_reward_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
    j_gamma_b = torch.cat(j_gamma_b).view(batch_size, trace_len, 1).to(device)                     #dim: (batch_size, trace_len, 1)
    s_next_b = torch.cat(state_next_b).view(batch_size, trace_len, -1).to(device)   #dim: (batch_size, trace_len, policy.net.input_dim)
    avail_next_j_a_b = torch.cat(avail_next_j_a_b).view(batch_size, trace_len, -1).to(device)
    t_b = torch.cat(terminate_b).view(batch_size, trace_len, 1).to(device)          #dim: (batch_size, trace_len, 1)
    id_v_b = torch.cat(id_valid_b).view(batch_size, trace_len, -1).to(device)
    j_v_b = torch.cat(j_valid_b).view(batch_size, trace_len).to(device)             #dim: (batch_size, trace_len)

    selected_traj_mask = j_v_b.sum(1) >= sub_trace_len
    if torch.sum(selected_traj_mask).item() == 0:
        return
    selected_lengths = j_v_b.sum(1)[selected_traj_mask]

    s_b = torch.split_with_sizes(s_b[selected_traj_mask][j_v_b[selected_traj_mask]], list(selected_lengths))
    s_next_b = torch.split_with_sizes(s_next_b[selected_traj_mask][j_v_b[selected_traj_mask]], list(selected_lengths))

    avail_next_j_a_b = avail_next_j_a_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    id_a_b = id_a_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    j_a_b = j_a_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    j_r_b = j_r_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    j_gamma_b = j_gamma_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    t_b = t_b[selected_traj_mask][j_v_b[selected_traj_mask]]
    id_v_b = id_v_b[selected_traj_mask][j_v_b[selected_traj_mask]]

    # get mask and padd sequence with different lenghts
    mask_s_b = get_mask_from_input(s_b)
    mask_s_next_b = get_mask_from_input(s_next_b)
    padded_s_b = pad_sequence(s_b, padding_value=torch.tensor(0), batch_first=True) 
    padded_s_next_b = pad_sequence(s_next_b, padding_value=torch.tensor(0), batch_first=True) 
    padded_s_next_b = torch.cat([padded_s_b[:,0].unsqueeze(1),
                                 padded_s_next_b],
                                 dim=1)

    Q_s = policy_net(padded_s_b)[0]
    Q_s_next = policy_net(padded_s_next_b)[0][:,1:,:]
    Q_s = Q_s[mask_s_b]
    Q_s_next = Q_s_next[mask_s_next_b]
    assert Q_s.size(0) == j_a_b.size(0), "number of Qs doesn't match with number of actions"

    Q = Q_s.gather(1, j_a_b)

    # apply double Q learning
    # get conditional action indexes
    condi_a = get_conditional_action(id_a_b, id_v_b)
    j_a_next_b = get_conditional_argmax(Q_s_next, condi_a, avail_next_j_a_b, env.n_action)
    target_Q_s_next = target_net(padded_s_next_b)[0].detach()[:,1:,:]
    target_Q_s_next = target_Q_s_next[mask_s_next_b]

    target_Q = target_Q_s_next.gather(1, j_a_next_b)
    target_Q = j_r_b + discount * j_gamma_b * target_Q * (-t_b + 1)

    if huber_loss:
        cen_controller.loss = F.smooth_l1_loss(Q, target_Q)
    else:
        td_err = (target_Q - Q) 
        cen_controller.loss = torch.mean(td_err*td_err)

    if cen_controller.loss is not None:
        cen_controller.optimizer.zero_grad()
        cen_controller.loss.backward()

        if grad_clip:
            assert grad_clip_value is not None, "no grad_clip_value is given"
            for param in cen_controller.policy_net.parameters():
                param.grad.data.clamp_(-grad_clip_value, grad_clip_value)
        elif grad_clip_norm:
            assert grad_clip_max_norm is not None, "no grad_clip_max_norm is given"
            clip_grad_norm_(cen_controller.policy_net.parameters(), grad_clip_max_norm)

        cen_controller.optimizer.step()
        cen_controller.loss = None

    #==================train decentralized Q==============================================

    # calculate loss for each agent
    for agent, batch in zip(agents, dec_batches):

        agent.policy_net.train()

        policy_net = agent.policy_net
        target_net = agent.target_net

        # seperate elements in the batch
        (state_b, 
         action_b, 
         id_reward_b, 
         gamma_b,
         state_next_b, 
         terminate_b, 
         id_valid_b, 
         j_valid_b) = zip(*batch)

        assert len(state_b) == trace_len * batch_size, "policy_net dim problem ..."
        assert len(state_next_b) == trace_len * batch_size, "target_net dim problem ..."

        s_b = torch.cat(state_b).view(batch_size, trace_len, -1).to(device)            #dim: (batch_size, trace_len, policy.net.input_dim)
        a_b = torch.cat(action_b).view(batch_size, trace_len, 1).to(device)            #dim: (batch_size, trace_len, 1)
        r_b = torch.cat(id_reward_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
        gamma_b = torch.cat(gamma_b).view(batch_size, trace_len, 1).to(device)                 #dim: (batch_size, trace_len, 1)
        s_next_b = torch.cat(state_next_b).view(batch_size, trace_len, -1).to(device)  #dim: (batch_size, trace_len, policy.net.input_dim)
        t_b = torch.cat(terminate_b).view(batch_size, trace_len, 1).to(device)         #dim: (batch_size, trace_len, 1)
        v_b = torch.cat(id_valid_b).view(batch_size, trace_len).to(device)             #dim: (batch_size, trace_len)

        selected_traj_mask = v_b.sum(1) >= sub_trace_len
        if torch.sum(selected_traj_mask).item() == 0:
            return
        selected_lengths = v_b.sum(1)[selected_traj_mask]
        s_b = torch.split_with_sizes(s_b[selected_traj_mask][v_b[selected_traj_mask]], list(selected_lengths))
        s_next_b = torch.split_with_sizes(s_next_b[selected_traj_mask][v_b[selected_traj_mask]], list(selected_lengths))

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

        Q_s, _ = policy_net(padded_s_b)
        Q_s = Q_s[mask_s_b]
        assert Q_s.size(0) == a_b.size(0), "number of Qs doesn't match with number of actions"
        Q = Q_s.gather(1, a_b)

        # apply double Q learning
        Q_s_next = policy_net(padded_s_next_b)[0].detach()[:,1:,:]
        Q_s_next = Q_s_next[mask_s_next_b]
        a_next_b = Q_s_next.max(1)[1].view(-1, 1)
        target_Q_s_next = target_net(padded_s_next_b)[0].detach()[:,1:,:]
        target_Q_s_next = target_Q_s_next[mask_s_next_b]
        target_Q = target_Q_s_next.gather(1, a_next_b)

        target_Q = r_b + discount * gamma_b * target_Q * (-t_b + 1)

        td_err = (target_Q - Q) 
        td_err = torch.max(hysteretic*td_err, td_err)

        if huber_loss:
            agent.loss = torch.mean(torch.min(td_err*td_err*0.5, torch.abs(td_err)-0.5))
        else:
            agent.loss = torch.mean(td_err*td_err)

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
