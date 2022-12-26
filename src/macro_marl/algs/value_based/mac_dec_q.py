import time
import pickle

from macro_marl.cores.value_based.mac_dec_q.team import Team_RNN
from macro_marl.cores.value_based.mac_dec_q.utils.memory import ReplayMemory_rand, ReplayMemory_epi
from macro_marl.cores.value_based.mac_dec_q.utils.utils import save_check_point
from macro_marl.cores.value_based.mac_dec_q.learning_methods import QLearn_squ


class MacDecQ(object):

    def __init__(self, 
            env, 
            env_terminate_step, 
            n_env, 
            n_agent, 
            total_epi, 
            replay_buffer_size, 
            sample_epi, 
            dynamic_h, 
            init_h, 
            end_h, 
            h_stable_at, 
            eps_l_d, 
            eps_l_d_steps, 
            eps_end, 
            eps_e_d, 
            softmax_explore, 
            h_explore, 
            db_step,
            optim, 
            l_rate, 
            discount, 
            huber_l, 
            g_clip, 
            g_clip_v, 
            g_clip_norm, 
            g_clip_max_norm,
            start_train, 
            train_freq, 
            target_update_freq, 
            trace_len, 
            sub_trace_len, 
            batch_size,
            sort_traj, 
            mlp_layer_size, 
            rnn_layer_num, 
            rnn_h_size, 
            lstm, 
            eval_freq, 
            run_id, 
            seed,
            resume, 
            save_ckpt, 
            save_dir, 
            device, 
            **kwargs):

        self.total_epi = total_epi
        self.start_train = start_train
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.db_step = db_step
        self.n_env = n_env
        self.resume = resume
        self.run_id = run_id
        self.save_dir = save_dir
        self.save_ckpt = save_ckpt

        # create replay buffer
        if sample_epi:
            memory = ReplayMemory_epi(env.n_agent, env.obs_size, env.n_action, batch_size, size=replay_buffer_size)
        else:
            memory = ReplayMemory_rand(env.n_agent, env.obs_size, env.n_action, trace_len, batch_size, size=replay_buffer_size)

        # collect hyper params:
        hyper_params = {'epsilon_linear_decay': eps_l_d,
                        'epsilon_linear_decay_steps': eps_l_d_steps,
                        'epsilon_end': eps_end,
                        'h_explore': h_explore,
                        'soft_action_selection': softmax_explore,
                        'epsilon_exp_decay': eps_e_d,
                        'dynamic_h': dynamic_h,
                        'hysteretic': (init_h, end_h),
                        'optimizer': optim,
                        'learning_rate': l_rate,
                        'discount': discount,
                        'huber_loss': huber_l,
                        'grad_clip': g_clip,
                        'grad_clip_value': g_clip_v,
                        'grad_clip_norm': g_clip_norm,
                        'grad_clip_max_norm': g_clip_max_norm,
                        'sample_epi': sample_epi,
                        'trace_len': trace_len,
                        'sub_trace_len': sub_trace_len,
                        'batch_size': batch_size,
                        'sort_traj': sort_traj,
                        'device': device}

        model_params = {'mlp_layer_size': mlp_layer_size,
                        'rnn_layer_num': rnn_layer_num,
                        'rnn_h_size': rnn_h_size,
                        'LSTM': lstm}

        # create team
        self.team = Team_RNN(env, env_terminate_step, n_env, memory, env.n_agent, QLearn_squ, h_stable_at,
                save_dir=save_dir, nn_model_params=model_params, eval_freq=eval_freq, seed=seed, **hyper_params)

    def learn(self):
        t = time.time()
        training_count=0
        target_updating_count = 0
        step = 0
        # continue training using the lastest check point
        if self.resume:
            self.team.load_check_point(self.run_id)
            step = self.team.step_count

        while self.team.episode_count < self.total_epi:
            step += 1

            self.team.step(self.run_id)
            if (not step % self.train_freq) and self.team.episode_count >= self.start_train:
                # update hysteretic learning rate
                if self.db_step:
                    self.team.update_hysteretic(step)
                else:
                    self.team.update_hysteretic(self.team.episode_count-self.start_train)

                for _ in range(self.n_env):
                    self.team.train()

                # update epsilon
                if self.db_step:
                    self.team.update_epsilon(step)
                else:
                    self.team.update_epsilon(self.team.episode_count-self.start_train)

                training_count += 1

            if not step % self.target_update_freq: 
                self.team.update_target_net() 
                target_updating_count += 1 

            if self.team.episode_count % self.team.eval_freq == 0 and self.team.envs_runner.step_count[0] == 0:
                print('[{}]run, [{:.1f}K] took {:.3f}hr to finish {} episodes {} trainning and {} target_net updating (eps={}) latest return ({})'.format(
                        self.run_id, step/1000, (time.time()-t)/3600, self.team.episode_count, training_count, target_updating_count, self.team.epsilon, self.team.TEST_PERFORM[-1]), flush=True)

        if self.save_ckpt:
            save_check_point(self.team.envs_runner, 
                             self.team.agents, 
                             step, 
                             self.team.episode_count, 
                             self.team.hysteretic, 
                             self.team.epsilon, 
                             self.save_dir, 
                             self.team.memory, 
                             self.run_id, 
                             self.team.TEST_PERFORM) 

        # save evaluation results
        with open("./performance/" + self.save_dir + "/test/test_perform" + str(self.run_id) + ".pickle", 'wb') as handle:
            pickle.dump(self.team.TEST_PERFORM, handle)
        print("Finish entire training ... ", flush=True)
        self.team.envs_runner.close()
