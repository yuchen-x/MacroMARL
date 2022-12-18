import torch
import copy
import numpy as np

from torch.optim import Adam
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from itertools import chain
from .models import Critic
from .utils import Linear_Decay

class Learner(object):
    
    def __init__(self, env, controller, memory, gamma, a_lr=1e-2, c_lr=1e-2, 
                 c_train_iteration=1, c_target_update_freq=50, tau=0.01,
                 grad_clip_value=None, grad_clip_norm=None,
                 n_step_TD=0, TD_lambda=0.0,
                 device='cpu'):

        self.env = env
        self.n_agent = env.n_agent
        self.controller = controller
        self.memory = memory
        self.gamma = gamma

        self.a_lr = a_lr
        self.c_lr = c_lr
        self.c_train_iteration = c_train_iteration
        self.c_target_update_freq = c_target_update_freq
        self.tau = tau
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.n_step_TD = n_step_TD
        self.TD_lambda = TD_lambda
        self.device = device

        self._set_optimizer()

    def train(self, eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys=False, adv_hys=False):

        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)

        ############################# train centralized critic ###################################
        cen_batch = self._cat_joint_exps(batch)
        cen_batch, trace_len, epi_len = self._squeeze_process(cen_batch, batch_size, trace_len)

        jobs, action, reward, n_jobs, terminate, discount, exp_valid = cen_batch

        if jobs.shape[1] == 0:
            return

        ##############################  calculate critic loss and optimize the critic_net ####################################
        for _ in range(self.c_train_iteration):
            if not self.TD_lambda:
                # NOTE WE SHOULD NOT BACKPROPAGATE CRITIC_NET BY N_STATE
                Gt = self._get_bootstrap_return(reward, 
                                                torch.cat([jobs[:,0].unsqueeze(1),
                                                           n_jobs],
                                                           dim=1),
                                                discount,
                                                terminate, 
                                                epi_len, 
                                                self.controller.agent.critic_tgt_net)
            else:
                Gt = self._get_td_lambda_return(batch_size, 
                                                trace_len, 
                                                epi_len, 
                                                reward, 
                                                torch.cat([jobs[:,0].unsqueeze(1),
                                                           n_jobs],
                                                           dim=1),
                                                terminate, 
                                                self.controller.agent.critic_tgt_net)

            TD = Gt - self.controller.agent.critic_net(jobs)[0]
            if critic_hys:
                TD = torch.max(TD*c_hys_value, TD)
            self.controller.agent.critic_loss = torch.sum(exp_valid * TD * TD) / exp_valid.sum()
            self.controller.agent.critic_optimizer.zero_grad()
            self.controller.agent.critic_loss.backward()
            if self.grad_clip_value:
                clip_grad_value_(self.controller.agent.critic_net.parameters(), self.grad_clip_value)
            if self.grad_clip_norm:
                clip_grad_norm_(self.controller.agent.critic_net.parameters(), self.grad_clip_norm)
            self.controller.agent.critic_optimizer.step()

        ##############################  calculate actor loss using the updated critic ####################################

        V_value = self.controller.agent.critic_net(jobs)[0].detach()

        # advantage value
        adv_value = Gt - V_value
        if adv_hys:
            adv_value = torch.max(adv_value*adv_hys_value, adv_value)
        action_logits = self.controller.agent.actor_net(jobs, eps=eps)[0]
        # log_pi(a|s) 
        log_pi_a = action_logits.gather(-1, action)
        # H(pi(.|s)) used as exploration bonus
        pi_entropy = torch.distributions.Categorical(logits=action_logits).entropy().view(batch_size, trace_len, 1)
        # actor loss
        actor_loss = torch.sum(exp_valid * discount * (log_pi_a * adv_value + etrpy_w * pi_entropy), dim=1)
        self.controller.agent.actor_loss = -1 * torch.sum(actor_loss) / exp_valid.sum()

        ############################# optimize each actor-net ########################################

        self.controller.agent.actor_optimizer.zero_grad()
        self.controller.agent.actor_loss.backward()
        if self.grad_clip_value:
            clip_grad_value_(self.controller.agent.actor_net.parameters(), self.grad_clip_value)
        if self.grad_clip_norm:
            clip_grad_norm_(self.controller.agent.actor_net.parameters(), self.grad_clip_norm)
        self.controller.agent.actor_optimizer.step()

    def update_critic_target_net(self, soft=False):
        if not soft:
            self.controller.agent.critic_tgt_net.load_state_dict(self.controller.agent.critic_net.state_dict())
        else:
            with torch.no_grad():
                for q, q_targ in zip(self.controller.agent.critic_net.parameters(), self.controller.agent.critic_tgt_net.parameters()):
                    q_targ.data.mul_(1 - self.tau)
                    q_targ.data.add_(self.tau * q.data)

    def update_actor_target_net(self, soft=False):
        if not soft:
            self.controller.agent.actor_tgt_net.load_state_dict(self.controller.agent.actor_net.state_dict())
        else:
            with torch.no_grad():
                for q, q_targ in zip(self.controller.agent.actor_net.parameters(), self.controller.agent.actor_tgt_net.parameters()):
                    q_targ.data.mul_(1 - self.tau)
                    q_targ.data.add_(self.tau * q.data)

    def _set_optimizer(self):
        self.controller.agent.actor_optimizer = Adam(self.controller.agent.actor_net.parameters(), lr=self.a_lr)
        self.controller.agent.critic_optimizer = Adam(self.controller.agent.critic_net.parameters(), lr=self.c_lr)

    def _cat_joint_exps(self, joint_exps):

        """
        seperate the joint experience for individual agents
        """

        exps = []
        for o, avail_a, a, r, j_r, n_o, n_avail_a, t, mac_v, j_mac_v, exp_v in chain(*joint_exps):
            exps.append([torch.cat(o, dim=1).view(1,-1), 
                         torch.tensor(np.ravel_multi_index(a, self.env.n_action)).view(1,-1), 
                         j_r, 
                         torch.cat(n_o, dim=1).view(1,-1), 
                         t, 
                         j_mac_v,
                         exp_v[0]])
        return exps

    def _squeeze_process(self, batch, batch_size, trace_len):

        """
        squeeze experience for each agent and re-padding
        """

        # seperate elements in the batch
        jobs_b, action_b, reward_b, next_jobs_b, terminate_b, mac_valid_b, exp_valid_b = zip(*batch)
        assert len(jobs_b) == trace_len * batch_size, "number of joint obsess mismatch ..."
        assert len(next_jobs_b) == trace_len * batch_size, "number of next joint obses mismatch ..."
        jo_b = torch.cat(jobs_b).view(batch_size, trace_len, -1)                    #dim: (batch_size, trace_len, policy.net.input_dim)
        a_b = torch.cat(action_b).view(batch_size, trace_len, -1)                   #dim: (batch_size, trace_len, 1)
        r_b = torch.cat(reward_b).view(batch_size, trace_len, -1)                   #dim: (batch_size, trace_len, 1)
        n_jo_b = torch.cat(next_jobs_b).view(batch_size, trace_len, -1)             #dim: (batch_size, trace_len, policy.net.input_dim)
        t_b = torch.cat(terminate_b).view(batch_size, trace_len, -1)                #dim: (batch_size, trace_len, 1)
        mac_v_b = torch.cat(mac_valid_b).view(batch_size, trace_len)                #dim: (batch_size, trace_len)
        exp_v_b = torch.cat(exp_valid_b).view(batch_size, trace_len, -1)            #dim: (batch_size, trace_len)
        discount_b = torch.pow(torch.ones(jo_b.shape[0],1)*self.gamma, torch.arange(jo_b.shape[1])).unsqueeze(-1) 

        # squeeze process
        squ_epi_len = mac_v_b.sum(1)
        squ_jo_b = torch.split_with_sizes(jo_b[mac_v_b], list(squ_epi_len))
        squ_a_b = torch.split_with_sizes(a_b[mac_v_b], list(squ_epi_len))
        squ_r_b = torch.split_with_sizes(r_b[mac_v_b], list(squ_epi_len))
        squ_n_jo_b = torch.split_with_sizes(n_jo_b[mac_v_b], list(squ_epi_len))
        squ_t_b = torch.split_with_sizes(t_b[mac_v_b], list(squ_epi_len))
        squ_exp_v_b = torch.split_with_sizes(exp_v_b[mac_v_b], list(squ_epi_len))
        squ_discount_b = torch.split_with_sizes(discount_b[mac_v_b], list(squ_epi_len))

        # re-padding
        squ_jo_b = pad_sequence(squ_jo_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_a_b = pad_sequence(squ_a_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_r_b = pad_sequence(squ_r_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_n_jo_b = pad_sequence(squ_n_jo_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_t_b = pad_sequence(squ_t_b, padding_value=torch.tensor(1.0), batch_first=True).to(self.device)
        squ_exp_v_b = pad_sequence(squ_exp_v_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)
        squ_discount_b = pad_sequence(squ_discount_b, padding_value=torch.tensor(0.0), batch_first=True).to(self.device)

        squ_cen_batch = (squ_jo_b,
                         squ_a_b,
                         squ_r_b,
                         squ_n_jo_b,
                         squ_t_b,
                         squ_discount_b,
                         squ_exp_v_b)

        return squ_cen_batch, squ_jo_b.shape[1], squ_epi_len

    def _get_discounted_return(self, reward, n_state, terminate, epi_len, critic_net):
        Gt = copy.deepcopy(reward)
        for epi_idx, epi_r in enumerate(Gt):
            end_step_idx = epi_len[epi_idx]-1
            if not terminate[epi_idx][end_step_idx]:
                # implement last step bootstrap
                epi_r[end_step_idx] += self.gamma * critic_net(n_state[epi_idx].unsqueeze(0))[0].detach()[:,1:,:][0][end_step_idx]
            for idx in range(end_step_idx-1, -1, -1):
                epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx+1]
        return Gt

    def _get_bootstrap_return(self, reward, n_state, discount, terminate, epi_len, critic_net):
        mac_discount = discount / torch.cat((self.gamma**-1*torch.ones((discount.shape[0],1,1)),
                                             discount[:,0:-1,:]),
                                             axis=1) 
        mask = mac_discount.isnan()
        mac_discount[mask] = 0.0
        if self.n_step_TD and self.n_step_TD != 1:
            # implement n-step bootstrap
            bootstrap = critic_net(n_state)[0].detach()[:,1:,:]
            Gt = copy.deepcopy(reward)
            for epi_idx, epi_r in enumerate(Gt):
                end_step_idx = epi_len[epi_idx]-1
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += mac_discount[epi_idx][end_step_idx] * bootstrap[epi_idx][end_step_idx]
                for idx in range(end_step_idx-1, -1, -1):
                    if idx > end_step_idx-self.n_step_TD:
                        epi_r[idx] = epi_r[idx] + mac_discount[epi_idx][idx] * epi_r[idx+1]
                    else:
                        if idx == 0:
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(reward[epi_idx][idx:idx+self.n_step_TD], 
                                                                                      bootstrap[epi_idx][idx+self.n_step_TD-1], 
                                                                                      discount[epi_idx][idx:idx+self.n_step_TD] / self.gamma**-1)
                        else: 
                            epi_r[idx] = self._get_n_step_discounted_bootstrap_return(reward[epi_idx][idx:idx+self.n_step_TD], 
                                                                                      bootstrap[epi_idx][idx+self.n_step_TD-1], 
                                                                                      discount[epi_idx][idx:idx+self.n_step_TD] / discount[epi_idx][idx-1])
        else:
            Gt = reward + mac_discount * critic_net(n_state)[0].detach()[:,1:,:] * (-terminate + 1)
        return Gt

    def _get_n_step_discounted_bootstrap_return(self, reward, bootstrap, discount):
        rewards = torch.cat((reward, bootstrap.reshape(-1,1)), axis=0)
        discounts = torch.cat((torch.ones((1,1)), discount), axis=0)
        Gt = torch.sum(discounts * rewards) 
        return Gt

    def _get_td_lambda_return(self, batch_size, trace_len, epi_len, reward, n_state, terminate, critic_net):
        # calculate MC returns
        Gt = self._get_discounted_return(reward, n_state, terminate, epi_len, critic_net)
        # calculate n-step bootstrap returns
        self.n_step_TD = 0
        n_step_part = self._get_bootstrap_return(reward, n_state, terminate, epi_len, critic_net)
        for n in range(2, trace_len):
            self.n_step_TD=n
            next_n_step_part = self._get_bootstrap_return(reward, n_state, terminate, epi_len, critic_net)
            n_step_part = torch.cat([n_step_part, next_n_step_part], dim=-1)
        # calculate the lmda for n-step bootstrap part
        lmdas = torch.pow(torch.ones(1,1)*self.TD_lambda, torch.arange(trace_len-1)).repeat(trace_len, 1).unsqueeze(0).repeat(batch_size,1,1)
        mask = (torch.arange(trace_len).view(-1,1) + torch.arange(trace_len-1).view(1,-1)).squeeze(0).repeat(batch_size,1,1)
        mask = mask >= epi_len.view(batch_size, -1, 1)-1
        lmdas[mask] = 0.0
        # calculate the lmda for MC part
        MC_lmdas = torch.zeros_like(Gt)
        for epi_id, length in enumerate(epi_len):
            last_step_lmda = torch.pow(torch.ones(1,1)*self.TD_lambda, torch.arange(length-1,-1,-1)).view(-1,1)
            MC_lmdas[epi_id][0:length] += last_step_lmda
        # TD LAMBDA RETURN
        Gt = (1 - self.TD_lambda) * torch.sum(lmdas * n_step_part, dim=-1, keepdim=True) +  MC_lmdas * Gt
        return Gt
