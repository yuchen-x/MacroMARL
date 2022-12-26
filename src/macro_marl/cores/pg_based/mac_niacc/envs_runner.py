import numpy as np
import random
import torch
import torch.nn.functional as F

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
                action = info['cur_mac']
                valid = info['mac_done']

                # accumulate reward of individual macro-action
                for idx, v in enumerate(valid):
                    if last_valid[idx]:
                        accu_rewards[idx] = reward[idx]
                        mac_act_step[idx] = 1
                        last_mac_start[idx] = 1
                    else:
                        mac_act_step[idx] += 1
                        accu_rewards[idx] = accu_rewards[idx] + gamma**(mac_act_step[idx]-1)*reward[idx]

                # accumulate reward of joint-macro-action
                if last_joint_valid:
                    accu_joint_reward = sum(reward)/env.n_agent 
                    mac_joint_act_step = 1
                else:
                    mac_joint_act_step += 1
                    accu_joint_reward += gamma**(mac_joint_act_step-1)*sum(reward)/env.n_agent 

                last_valid = valid
                last_joint_valid = max(valid)
                avail_actions = env.get_avail_actions()

                # sent experience back
                child.send((last_obs, 
                            last_mac_start,
                            action, 
                            accu_rewards, 
                            accu_joint_reward,
                            obs, 
                            avail_actions, 
                            terminate, 
                            valid, 
                            max(valid)))

                last_mac_start = [0] * env.n_agent
                last_obs = obs
                R += gamma**step * sum(reward) / env.n_agent
                step += 1
            
            elif cmd == 'get_return':
                child.send(R)

            elif cmd == 'reset':
                last_obs =  env.reset() # List[array]
                h_state = [None] * env.n_agent
                last_action = [-1] * env.n_agent
                last_valid = [1.0] * env.n_agent
                # record the moment when a new macro-action start
                last_mac_start = [0] * env.n_agent
                last_joint_valid = 1
                accu_rewards = [0.0] * env.n_agent
                accu_joint_reward = 0.0
                mac_act_step = [0] * env.n_agent
                mac_joint_act_step = 0
                avail_actions = env.get_avail_actions()
                step = 0
                R = 0.0

                child.send((last_obs, h_state, last_action, last_valid, avail_actions))
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
    Environment runner which runs mulitpl environemnts in parallel in subprocesses
    and communicates with them via pipe
    """

    def __init__(self, env, n_envs, controller, memory, env_terminate_step, gamma, seed, obs_last_action=False):
        
        self.env = env
        self.max_epi_step = env_terminate_step
        self.n_envs = n_envs
        self.n_agent = env.n_agent
        # controllers for getting next action via current actor nn
        self.controller = controller
        # create connections via Pipe
        self.parents, self.children = [list(i) for i in zip(*[Pipe() for _ in range(n_envs)])]
        # create multip processor with multiple envs
        self.envs = [Process(target=worker, args=(child, env, gamma, seed+idx)) for idx, child in enumerate(self.children)]
        # replay buffer
        self.memory = memory
        # observe last actions
        self.obs_last_action = obs_last_action
        # record parallel episodes
        self.episodes = [[] for i in range(n_envs)]
        # record train return
        self.train_returns = []
        # record eval return
        self.eval_returns = []

        # trigger each processor
        for env in self.envs:
            env.daemon = True
            env.start()

        for child in self.children:
            child.close()

    def run(self, eps=0.0, n_epis=1, test_mode=False):

        self._reset()

        if test_mode:
            while len(self.eval_returns) < n_epis:
                self._step(eps=eps, test_mode=test_mode)
        else:
            while self.n_epi_count < n_epis:
                self._step(eps=eps, test_mode=test_mode)

    def close(self):
        [parent.send(('close', None)) for parent in self.parents]
        [parent.close() for parent in self.parents]
        [env.terminate() for env in self.envs]
        [env.join() for env in self.envs]

    def _step(self, eps=0.0, test_mode=False):

        for idx, parent in enumerate(self.parents):
            
            actions, self.h_states[idx] = self.controller.select_action(self.last_obses[idx], 
                                                                        self.h_states[idx], 
                                                                        self.last_valids[idx],
                                                                        self.avail_actions[idx],
                                                                        eps=eps,
                                                                        test_mode=test_mode)
            # send cmd to trigger env step
            parent.send(("step", actions))
            self.step_count[idx] += 1

        # collect envs' returns
        for idx, parent in enumerate(self.parents):
            env_return = parent.recv()
            env_return = self._exp_to_tensor(idx, env_return, eps)
            self.episodes[idx].append(env_return)

            self.last_obses[idx] = env_return[6]
            self.avail_actions[idx] = env_return[7]
            self.last_valids[idx] = env_return[-3]
            if self.obs_last_action and sum(self.last_valids[idx]) > 0:
                for nth in range(self.n_agent):
                    if self.last_valids[idx][nth]:
                        self.last_actions[idx][nth] = env_return[3][nth]

            # if episode is done, add it to memory buffer
            if env_return[-4][0] or self.step_count[idx] == self.max_epi_step:
                self.n_epi_count += 1
                # collect the return
                parent.send(("get_return", None))
                R = parent.recv()
                if not test_mode:
                    self.memory.scenario_cache += self.episodes[idx]
                    self.memory.flush_buf_cache()
                    self.train_returns.append(R)
                else:
                    self.eval_returns.append(R)

                # when episode is done, immediately start a new one
                parent.send(("reset", None))
                self.last_obses[idx], self.h_states[idx], self.last_actions[idx], self.last_valids[idx], self.avail_actions[idx] = parent.recv()
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                self.last_actions[idx] = self.action_to_tensor(self.last_actions[idx])
                if self.obs_last_action:
                    self.last_obses[idx] = self.rebuild_obs(self.env, self.last_obses[idx], self.last_actions[idx])
                self.last_valids[idx] = self.mac_done_to_tensor(self.last_valids[idx])
                self.avail_actions[idx] = self.avail_action_to_tensor(self.avail_actions[idx])
                self.episodes[idx] = []
                self.step_count[idx] = 0

    def _reset(self):
        # send cmd to reset envs
        for parent in self.parents:
            parent.send(("reset", None))

        self.last_obses, self.h_states, self.last_actions, self.last_valids, self.avail_actions = [list(i) for i in zip(*[parent.recv() for parent in self.parents])]
        self.last_obses = [self.obs_to_tensor(obs) for obs in self.last_obses] #List[List[tensor]]
        self.last_actions = [self.action_to_tensor(a) for a in self.last_actions]
        if self.obs_last_action:
            # reconstruct obs to observe actions
            self.last_obses = [self.rebuild_obs(self.env, obs, a) for obs, a in zip(*[self.last_obses, self.last_actions])]
        self.last_valids = [self.mac_done_to_tensor(mac_done) for mac_done in self.last_valids]
        self.avail_actions = [self.avail_action_to_tensor(avail_action) for avail_action in self.avail_actions]

        self.n_epi_count = 0
        self.step_count = [0] * self.n_envs
        self.episodes = [[] for i in range(self.n_envs)]

    def _exp_to_tensor(self, env_idx, exp, eps):
        # exp (last_obs, a, r, obs, t, discnt)
        last_obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[0]]
        last_mac_start = [torch.tensor(start, dtype=torch.bool).view(1,-1) for start in exp[1]]
        last_avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in self.avail_actions[env_idx]]
        a = [torch.tensor(a).view(1,-1) for a in exp[2]]
        r = [torch.tensor(r).float().view(1,-1) for r in exp[3]]
        j_r = torch.tensor(exp[4]).float().view(1,-1) 
        obs = [torch.from_numpy(o).float().view(1,-1) for o in exp[5]]
        avail_actions = [torch.FloatTensor(avail_action).view(1,-1) for avail_action in exp[6]]
        # re-construct obs if obs last action
        if self.obs_last_action:
            last_obs = self.rebuild_obs(self.env, last_obs, self.last_actions[env_idx])
            obs = self.rebuild_obs(self.env, obs, a)
        t = torch.tensor(exp[7]).float().view(1,-1)
        mac_v = [torch.tensor(v, dtype=torch.bool).view(1,-1) for v in exp[8]]
        j_mac_v = torch.tensor(exp[9], dtype=torch.bool).view(1,-1)
        exp_v = [torch.tensor([1.0]).view(1,-1)] * self.n_agent
        return (last_obs, last_mac_start, last_avail_actions, a, r, j_r, obs, avail_actions, t, mac_v, j_mac_v, exp_v)

    @staticmethod
    def obs_to_tensor(obs):
        return [torch.from_numpy(o).float().view(1,-1) for o in obs]

    @staticmethod
    def action_to_tensor(action):
        return [torch.tensor(a).view(1,-1) for a in action]

    @staticmethod
    def mac_done_to_tensor(mac_done):
        return [torch.tensor(done, dtype=torch.bool).view(1,-1) for done in mac_done]

    @staticmethod
    def rebuild_obs(env, obs, action):
        new_obs = []
        for o, a, a_dim in zip(*[obs, action, env.n_action]):
            if a == -1:
                one_hot_a = torch.zeros(a_dim).view(1,-1)
            else:
                one_hot_a = F.one_hot(a.view(-1), a_dim).float()
            new_obs.append(torch.cat([o, one_hot_a], dim=1))
        return new_obs

    @staticmethod
    def avail_action_to_tensor(avail_action):
        return [torch.FloatTensor(a).view(1,-1) for a in avail_action]
