import time
import numpy as np

from macro_marl.cores.pg_based.mac_cac.memory import Memory_epi, Memory_rand
from macro_marl.cores.pg_based.mac_cac.controller import MAC
from macro_marl.cores.pg_based.mac_cac.envs_runner import EnvsRunner
from macro_marl.cores.pg_based.mac_cac.learner import Learner
from macro_marl.cores.pg_based.mac_cac.utils import Linear_Decay, save_train_data, save_test_data, save_checkpoint, load_checkpoint, save_policy

class MacCAC(object):

    def __init__(self,
            env,
            env_terminate_step, 
            n_env, 
            n_agent, 
            seed, 
            run_id, 
            save_dir, 
            resume, 
            device,
            total_epi, 
            gamma, 
            a_lr, 
            c_lr, 
            c_train_iteration, 
            eps_start, 
            eps_end, 
            eps_stable_at, 
            c_hys_start, 
            c_hys_end, 
            adv_hys_start, 
            adv_hys_end, 
            hys_stable_at, 
            critic_hys, 
            adv_hys, 
            etrpy_w_start, 
            etrpy_w_end, 
            etrpy_w_stable_at, 
            train_freq, 
            c_target_update_freq, 
            c_target_soft_update, 
            tau, 
            n_step_TD, 
            TD_lambda, 
            a_mlp_layer_size, 
            a_rnn_layer_size, 
            c_mlp_layer_size, 
            c_rnn_layer_size, 
            grad_clip_value, 
            grad_clip_norm, 
            obs_last_action, 
            eval_policy, 
            eval_freq, 
            eval_num_epi, 
            sample_epi, 
            trace_len, 
            *args, 
            **kwargs):

        self.total_epi = total_epi
        self.train_freq = train_freq
        self.eval_policy = eval_policy
        self.eval_freq = eval_freq
        self.eval_num_epi = eval_num_epi
        self.critic_hys = critic_hys
        self.adv_hys = adv_hys 
        self.sample_epi = sample_epi
        self.c_target_update_freq = c_target_update_freq
        self.c_target_soft_update = c_target_soft_update
        self.run_id = run_id
        self.save_dir = save_dir
        self.resume = resume

        # collect params
        actor_params = {'a_mlp_layer_size': a_mlp_layer_size,
                        'a_rnn_layer_size': a_rnn_layer_size}

        critic_params = {'c_mlp_layer_size': c_mlp_layer_size,
                         'c_rnn_layer_size': c_rnn_layer_size}

        hyper_params = {'a_lr': a_lr,
                        'c_lr': c_lr,
                        'c_train_iteration': c_train_iteration,
                        'c_target_update_freq': c_target_update_freq,
                        'tau': tau,
                        'grad_clip_value': grad_clip_value,
                        'grad_clip_norm': grad_clip_norm,
                        'n_step_TD': n_step_TD,
                        'TD_lambda': TD_lambda,
                        'device': device}

        self.env = env
        # create buffer
        if sample_epi:
            self.memory = Memory_epi(env.obs_size, env.n_action, obs_last_action, size=train_freq)
        else:
            self.memory = Memory_rand(trace_len, env.obs_size, env.n_action, obs_last_action, size=train_freq)
        # cretate controller
        self.controller = MAC(self.env, obs_last_action, **actor_params, **critic_params, device=device) 
        # create parallel envs runner
        self.envs_runner = EnvsRunner(self.env, n_env, self.controller, self.memory, env_terminate_step, gamma, seed, obs_last_action)
        # create learner
        self.learner = Learner(self.env, self.controller, self.memory, gamma, **hyper_params)
        # create epsilon calculator for implementing e-greedy exploration policy
        self.eps_call = Linear_Decay(eps_stable_at, eps_start, eps_end)
        # create hysteretic calculator for implementing hystgeritic value function updating
        self.c_hys_call = Linear_Decay(hys_stable_at, c_hys_start, c_hys_end)
        # create hysteretic calculator for implementing hystgeritic advantage esitimation
        self.adv_hys_call = Linear_Decay(hys_stable_at, adv_hys_start, adv_hys_end)
        # create entropy loss weight calculator
        self.etrpy_w_call = Linear_Decay(etrpy_w_stable_at, etrpy_w_start, etrpy_w_end)
        # record evaluation return
        self.eval_returns = []

    def learn(self):
        epi_count = 0
        if self.resume:
            epi_count, self.eval_returns = load_checkpoint(self.run_id, self.save_dir, self.controller, self.envs_runner)

        while epi_count < self.total_epi:

            if self.eval_policy and epi_count % (self.eval_freq - (self.eval_freq % self.train_freq)) == 0:
                self.envs_runner.run(n_epis=self.eval_num_epi, test_mode=True)
                assert len(self.envs_runner.eval_returns) >= self.eval_num_epi, "Not evaluate enough episodes ..."
                self.eval_returns.append(np.mean(self.envs_runner.eval_returns[-self.eval_num_epi:]))
                self.envs_runner.eval_returns = []
                print(f"{[self.run_id]} Finished: {epi_count}/{self.total_epi} Evaluate learned policies with averaged returns {self.eval_returns[-1]} ...", flush=True)
                if self.eval_returns[-1] == np.max(self.eval_returns):
                    save_policy(self.run_id, self.controller.agent, self.save_dir)

            # update eps
            eps = self.eps_call.get_value(epi_count)
            # update hys
            c_hys_value = self.c_hys_call.get_value(epi_count)
            adv_hys_value = self.adv_hys_call.get_value(epi_count)
            # update etrpy weight
            etrpy_w = self.etrpy_w_call.get_value(epi_count)
            # let envs run a certain number of episodes accourding to train_freq
            self.envs_runner.run(eps=eps, n_epis=self.train_freq)
            # perform hysteretic-ac update
            self.learner.train(eps, c_hys_value, adv_hys_value, etrpy_w, self.critic_hys, self.adv_hys)
            if not self.sample_epi:
                self.memory.buf.clear()

            epi_count += self.train_freq

            # update target net
            if self.c_target_soft_update:
                self.learner.update_critic_target_net(soft=True)
                self.learner.update_actor_target_net(soft=True)
            elif epi_count % self.c_target_update_freq == 0:
                self.learner.update_critic_target_net()
                self.learner.update_actor_target_net()

        ################################ saving in the end ###################################
        save_train_data(self.run_id, self.envs_runner.train_returns, self.save_dir)
        save_test_data(self.run_id, self.eval_returns, self.save_dir)
        save_checkpoint(self.run_id, epi_count, self.eval_returns, self.controller, self.envs_runner, self.save_dir)
        self.envs_runner.close()

        print(f"{[self.run_id]} Finish entire training ... ", flush=True)
