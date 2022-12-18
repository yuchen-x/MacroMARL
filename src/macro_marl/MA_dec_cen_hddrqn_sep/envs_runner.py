import numpy as np
import random
import torch

from multiprocessing import Process, Pipe

def worker(child, env, gamma, seed):
    """
    Worker function which interacts with the environment over remote
    """

    random.seed(seed)
    np.random.seed(seed)

    try:
        while True:
            # wait cmd sent by parent
            cmd, data = child.recv()
            if cmd == 'step':
                obs, reward, terminate, info = env.step(data)
                actions = info['cur_mac']
                valid = info['mac_done']

                for idx, v in enumerate(valid):
                    if last_id_valid[idx]:
                        accu_id_rewards[idx] = reward[idx]
                        mac_id_act_step[idx] = 1
                        mac_id_gamma[idx] = 1
                    else:
                        mac_id_act_step[idx] += 1
                        accu_id_rewards[idx] = accu_id_rewards[idx] + gamma**(mac_id_act_step[idx]-1) * reward[idx]
                        mac_id_gamma[idx] = gamma**(mac_id_act_step[idx]-1)

                # accumulate reward of joint-macro-action
                if last_joint_valid:
                    accu_joint_rewards = sum(reward)/env.n_agent 
                    mac_joint_act_step = 1
                    mac_joint_gamma = 1
                else:
                    mac_joint_act_step += 1
                    accu_joint_rewards += gamma**(mac_joint_act_step-1)*sum(reward)/env.n_agent 
                    mac_joint_gamma = gamma**(mac_joint_act_step-1)

                last_id_valid = valid
                last_joint_valid = max(valid)
                avail_actions = env.get_avail_actions()

                # sent experience back
                child.send((last_obs, 
                            actions, 
                            accu_id_rewards, 
                            accu_joint_rewards,
                            mac_id_gamma,
                            mac_joint_gamma,
                            obs, 
                            avail_actions,
                            terminate, 
                            valid,
                            max(valid)))

                last_obs = obs
                R += gamma**step * sum(reward) / env.n_agent
                step += 1

            elif cmd == 'get_return':
                child.send(R)

            elif cmd == 'reset':
                last_obs =  env.reset()
                dec_last_h = [None] * env.n_agent
                cen_last_h = None  # single network for cen control

                last_id_action = [-1] * env.n_agent
                last_id_valid = [1] * env.n_agent
                last_joint_valid = 1

                mac_id_gamma = [0.0] * env.n_agent
                mac_id_act_step = [0] * env.n_agent
                mac_joint_gamma = 0.0
                mac_joint_act_step = 0

                accu_id_rewards = [0.0] * env.n_agent
                accu_joint_rewards = 0.0
                avail_actions = env.get_avail_actions()

                step = 0
                R = 0.0

                child.send((last_obs, dec_last_h, cen_last_h, last_id_action, last_id_valid, avail_actions))
            elif cmd == 'close':
                child.close()
                break
            elif cmd == 'get_rand_states':
                rand_states = {'random_state': random.getstate(),
                               'np_random_state': np.random.get_state()}
                child.send(rand_states)
            elif cmd == 'load_rand_states':
                random.setstate(data['random_state'])
                np.random.set_state(data['np_random_state'])
            else:
                raise NotImplementerError
 
    except KeyboardInterrupt:
        print('EnvRunner worker: caught keyboard interrupt')
    except Exception as e:
        print('EnvRunner worker: uncaught worker exception')
        raise

class EnvsRunner(object):

    """
    Environment runner which runs multiple environemnts in parallel in subprocesses
    and communicates with them via pipe
    """

    def __init__(self, 
                 env, 
                 env_terminate_step,
                 memory, 
                 n_env, 
                 h_explore, 
                 get_actions,
                 gamma,
                 seed,
                 log=False):

        """
        Parameters
        ----------
        env : gym.env
            A macro-action-based gym envrionment.
        memory : ReplayBuffer
            An instance of RepalyBuffer class.
        n_env : int
            The number of envs running in parallel.
        h_explore : bool
            Whether use history-based policy for rollout.
        get_actions : python function
            A function for getting macro-actions 
        """
        
        self.n_env = n_env
        self.env_terminate_step = env_terminate_step
        # func for getting next action via current policy nn
        self.get_actions = get_actions
        # create connections via Pipe
        self.parents, self.children = [list(i) for i in zip(*[Pipe() for _ in range(n_env)])]
        # create multip processor with multiple envs
        self.envs = [Process(target=worker, args=(child, env, gamma, seed)) for child in self.children]
        # replay buffer
        self.memory = memory

        self.dec_hidden_states = [None] * n_env
        self.cen_hidden_states = [None] * n_env

        self.h_explore = h_explore
        self.episodes = [[]] * n_env

        # record train return
        self.train_returns = []

        # log
        self.log = log
 

        # trigger each processor
        for env in self.envs:
            env.daemon = True
            env.start()

        for child in self.children:
            child.close()

    def step(self):

        n_episode_done = 0

        for idx, parent in enumerate(self.parents):
            # get next action
            if self.h_explore:
                actions, self.dec_h_states[idx], self.cen_h_states[idx] = self.get_actions(self.last_obses[idx], 
                                                                                           self.dec_h_states[idx], 
                                                                                           self.cen_h_states[idx], 
                                                                                           self.actions[idx], 
                                                                                           self.last_valids[idx],
                                                                                           self.avail_actions[idx])
            else:
                actions, self.dec_hidden_states[idx], self.cen_hidden_states[idx] = self.get_actions(self.last_obses[idx], 
                                                                                                     self.dec_h_states[idx], 
                                                                                                     self.cen_h_states[idx], 
                                                                                                     self.actions[idx], 
                                                                                                     self.last_valids[idx],
                                                                                                     self.avail_actions[idx])

            # send cmd to trigger env step
            parent.send(("step", actions))
            self.step_count[idx] += 1 

        # collect envs' returns
        for idx, parent in enumerate(self.parents):
            # env_return is (last_obs, a, acc_r, obs, t, v)
            env_return = parent.recv()
            env_return = self.exp_to_tensor(idx, env_return)
            self.episodes[idx].append(env_return)

            self.last_obses[idx] = env_return[7]
            self.avail_actions[idx] = env_return[8]
            self.actions[idx] = env_return[2]
            self.last_valids[idx] = env_return[10]

            # if episode is done, add it to memory buffer
            if env_return[-3] or self.step_count[idx] == self.env_terminate_step:
                n_episode_done += 1
                self.memory.scenario_cache += self.episodes[idx]
                self.memory.flush_scenario_cache()
                if self.log:
                    parent.send(("get_return", None))
                    R = parent.recv()
                    self.train_returns.append(R)

                # when episode is done, immediately start a new one
                parent.send(("reset", None))
                self.last_obses[idx], self.dec_h_states[idx], self.cen_h_states[idx], self.actions[idx], self.last_valids[idx], self.avail_actions[idx] = parent.recv()
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                self.actions[idx] = self.action_to_tensor(self.actions[idx])
                self.last_valids[idx] = self.valid_to_tensor(self.last_valids[idx])
                self.avail_actions[idx] = self.avail_action_to_tensor(self.avail_actions[idx])
                self.episodes[idx] = []
                self.step_count[idx] = 0

        return n_episode_done

    def reset(self):
        # send cmd to reset envs
        for parent in self.parents:
            parent.send(("reset", None))

        self.last_obses, self.dec_h_states, self.cen_h_states, self.actions, self.last_valids, self.avail_actions = [list(i) for i in zip(*[parent.recv() for parent in self.parents])]
        self.last_obses = [self.obs_to_tensor(obs) for obs in self.last_obses]
        self.actions = [self.action_to_tensor(a) for a in self.actions]
        self.last_valids = [self.valid_to_tensor(id_v) for id_v in self.last_valids]
        self.avail_actions = [self.avail_action_to_tensor(avail_action) for avail_action in self.avail_actions]
        self.step_count = [0] * self.n_env

    def close(self):
        [parent.send(('close', None)) for parent in self.parents]
        [parent.close() for parent in self.parents]
        [env.terminate() for env in self.envs]
        [env.join() for env in self.envs]

    def obs_to_tensor(self, obs):
        return [torch.from_numpy(o).float() for o in obs]

    def valid_to_tensor(self,valid):
        return [torch.tensor(v, dtype=torch.bool).view(1,-1) for v in valid]

    def action_to_tensor(self,action):
        return [torch.tensor(a).view(1,1) for a in action]

    def avail_action_to_tensor(self, avail_action):
        return [torch.FloatTensor(a).view(1,-1) for a in avail_action]

    def exp_to_tensor(self, env_idx, exp):
        last_obs = [torch.from_numpy(o).float() for o in exp[0]]
        last_avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in self.avail_actions[env_idx]]
        a = [torch.tensor(a).view(1,-1) for a in exp[1]]
        acc_id_r = [torch.tensor(r).float().view(1,-1) for r in exp[2]]
        acc_joint_r = torch.tensor(exp[3]).float().view(1,-1)
        mac_id_gamma = [torch.tensor(g).float().view(1,-1) for g in exp[4]]
        mac_joint_gamma = torch.tensor(exp[5]).float().view(1,-1)
        obs = [torch.from_numpy(o).float() for o in exp[6]]
        avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in exp[7]]
        t = torch.tensor(exp[8]).float().view(1,-1)
        id_v = [torch.tensor(v, dtype=torch.bool).view(1,-1) for v in exp[9]]
        joint_v = torch.tensor(exp[10], dtype=torch.bool).view(1,-1)
        return (last_obs, 
                last_avail_actions, 
                a, 
                acc_id_r, 
                acc_joint_r, 
                mac_id_gamma, 
                mac_joint_gamma, 
                obs, 
                avail_actions, 
                t, 
                id_v, 
                joint_v)
