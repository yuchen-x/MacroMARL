import argparse
import gym
import torch
import random
import numpy as np
import os
import pickle
import time

from macro_marl.MA_niacc_rnn_V.utils import Linear_Decay, save_train_data, save_test_data, save_policies
from macro_marl.MA_niacc_rnn_sV.memory import Memory_epi, Memory_rand
from macro_marl.MA_niacc_rnn_sV.envs_runner import EnvsRunner

from macro_marl.MA_iaicc_rnn_sV.controller import MAC
from macro_marl.MA_iaicc_rnn_sV import Learner
from macro_marl.MA_iaicc_rnn_sV.utils import save_checkpoint, load_checkpoint

from macro_marl import my_env
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

Learners = [Learner]

def train(env_id, env_terminate_step, n_env, n_agent, l_mode, seed, run_id, save_dir, save_rate, save_ckpt, save_ckpt_time, resume, device, 
          total_epi, gamma, a_lr, c_lr, c_train_iteration, 
          eps_start, eps_end, eps_stable_at, 
          c_hys_start, c_hys_end, adv_hys_start, adv_hys_end, hys_stable_at, critic_hys, adv_hys, 
          etrpy_w_start, etrpy_w_end, etrpy_w_stable_at, 
          train_freq, c_target_update_freq, c_target_soft_update, tau, 
          n_step_TD, TD_lambda, 
          a_mlp_layer_size, a_rnn_layer_size, c_mlp_layer_size, c_mid_layer_size,
          grad_clip_value, grad_clip_norm, obs_last_action, eval_policy, eval_freq, eval_num_epi, sample_epi, trace_len, 
          grid_dim, big_box_reward, small_box_reward, penalty, 
          h0_speed_ps, h1_speed_ps, tb_m_speed, tb_m_noisy, tb_m_cost, f_p_obj_tc, f_l_obj_tc, f_m_noisy, f_drop_obj_pen, d_pen,
          task, map_type, step_penalty,
          *args, **kwargs):

    # set seed
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # create the dirs to save results
    os.makedirs("./performance/" + save_dir + "/train", exist_ok=True)
    os.makedirs("./performance/" + save_dir + "/test", exist_ok=True)
    os.makedirs("./performance/" + save_dir + "/ckpt", exist_ok=True)
    os.makedirs("./policy_nns/"+save_dir, exist_ok=True)

    # collect params
    actor_params = {'a_mlp_layer_size': a_mlp_layer_size,
                    'a_rnn_layer_size': a_rnn_layer_size}

    critic_params = {'c_mlp_layer_size': c_mlp_layer_size,
                     'c_mid_layer_size': c_mid_layer_size}

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

    if env_id.startswith('OSD') or env_id.startswith('BP'):
        env_params = {'grid_dim': grid_dim,
                      'big_box_reward': big_box_reward,
                      'small_box_reward': small_box_reward,
                      'penalty': penalty,
                      'n_agent': n_agent,
                      'terminate_step': env_terminate_step,
                      'TB_move_speed': tb_m_speed,
                      'TB_move_noisy': tb_m_noisy,
                      'TB_move_cost': tb_m_cost,
                      'fetch_pass_obj_tc': f_p_obj_tc,
                      'fetch_look_for_obj_tc': f_l_obj_tc,
                      'fetch_manip_noisy': f_m_noisy,
                      'delay_delivery_penalty': d_pen
                      }
        if env_id.startswith('OSD-D'):
            env_params['human_speed_per_step'] = [h0_speed_ps, h1_speed_ps]
        else:
            env_params['human_speed_per_step'] = [h0_speed_ps]
        env = gym.make(env_id, **env_params)
    else:
        TASKLIST = [
                "tomato salad", 
                "lettuce salad", 
                "onion salad", 
                "lettuce-tomato salad", 
                "onion-tomato salad", 
                "lettuce-onion salad", 
                "lettuce-onion-tomato salad"
                ]
        rewardList = {
                "subtask finished": 10, 
                "correct delivery": 200, 
                "wrong delivery": -5, 
                "step penalty": step_penalty
                }
        env_params = {
                'grid_dim': grid_dim,
                'map_type': map_type, 
                'task': TASKLIST[task],
                'rewardList': rewardList,
                'debug': False
                }
        env = gym.make(env_id, **env_params)
        if env_id.find("MA") != -1:
            env = MacEnvWrapper(env)

    # create buffer
    if sample_epi:
        memory = Memory_epi(env.state_size, env.obs_size, env.n_action, obs_last_action, size=train_freq)
    else:
        memory = Memory_rand(trace_len, env.state_size, env.obs_size, env.n_action, obs_last_action, size=train_freq)
    # cretate controller
    controller = MAC(env, obs_last_action, **actor_params, **critic_params, device=device) 
    # create parallel envs runner
    envs_runner = EnvsRunner(env, n_env, controller, memory, env_terminate_step, gamma, seed, obs_last_action)
    # create learner
    learner = Learners[l_mode](env, controller, memory, gamma, obs_last_action, **hyper_params)
    # create epsilon calculator for implementing e-greedy exploration policy
    eps_call = Linear_Decay(eps_stable_at, eps_start, eps_end)
    # create hysteretic calculator for implementing hystgeritic value function updating
    c_hys_call = Linear_Decay(hys_stable_at, c_hys_start, c_hys_end)
    # create hysteretic calculator for implementing hystgeritic advantage esitimation
    adv_hys_call = Linear_Decay(hys_stable_at, adv_hys_start, adv_hys_end)
    # create entropy loss weight calculator
    etrpy_w_call = Linear_Decay(etrpy_w_stable_at, etrpy_w_start, etrpy_w_end)

    #################################### training loop ###################################
    eval_returns = []
    epi_count = 0
    t_ckpt = time.time()
    if resume:
        epi_count, eval_returns = load_checkpoint(run_id, save_dir, controller, envs_runner)

    while epi_count < total_epi:

        if eval_policy and epi_count % (eval_freq - (eval_freq % train_freq)) == 0:
            envs_runner.run(n_epis=eval_num_epi, test_mode=True)
            assert len(envs_runner.eval_returns) >= eval_num_epi, "Not evaluate enough episodes ..."
            eval_returns.append(np.mean(envs_runner.eval_returns[-eval_num_epi:]))
            envs_runner.eval_returns = []
            print(f"{[run_id]} Finished: {epi_count}/{total_epi} Evaluate learned policies with averaged returns {eval_returns[-1]} ...", flush=True)
            # save the best policy
            if eval_returns[-1] == np.max(eval_returns):
                save_policies(run_id, controller.agents, save_dir)

        # update eps
        eps = eps_call.get_value(epi_count)
        # update hys
        c_hys_value = c_hys_call.get_value(epi_count)
        adv_hys_value = adv_hys_call.get_value(epi_count)
        # update etrpy weight
        etrpy_w = etrpy_w_call.get_value(epi_count)
        # let envs run a certain number of episodes accourding to train_freq
        envs_runner.run(eps=eps, n_epis=train_freq)
        # perform hysteretic-ac update
        learner.train(eps, c_hys_value, adv_hys_value, etrpy_w, critic_hys, adv_hys)
        if not sample_epi:
            memory.buf.clear()

        epi_count += train_freq

        # update target net
        if c_target_soft_update:
            learner.update_critic_target_net(soft=True)
            learner.update_actor_target_net(soft=True)
        elif epi_count % c_target_update_freq == 0:
            learner.update_critic_target_net()
            learner.update_actor_target_net()

        # save training and testing performance
        if epi_count % save_rate == 0:
            save_train_data(run_id, envs_runner.train_returns, save_dir)
            save_test_data(run_id, eval_returns, save_dir)
            save_checkpoint(run_id, epi_count, eval_returns, controller, envs_runner, save_dir)

        # save ckpt
        if save_ckpt and (time.time() - t_ckpt) // 3600 >= save_ckpt_time:
            save_checkpoint(run_id, epi_count, eval_returns, controller, envs_runner, save_dir)
            t_ckpt = time.time()
            break

    ################################ saving in the end ###################################
    save_train_data(run_id, envs_runner.train_returns, save_dir)
    save_test_data(run_id, eval_returns, save_dir)
    save_checkpoint(run_id, epi_count, eval_returns, controller, envs_runner, save_dir)
    envs_runner.close()

    print(f"{[run_id]} Finish entire training ... ", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id',                 action='store',        type=str,             default='OSD-S-v4',    help='Domain name')
    parser.add_argument('--env_terminate_step',     action='store',        type=int,             default=150,           help='Maximal steps for termination')
    parser.add_argument('--n_env',                  action='store',        type=int,             default=1,             help='Number of envs running in parallel')
    parser.add_argument('--n_agent',                action='store',        type=int,             default=2,             help='Number of agents')
    parser.add_argument('--l_mode',                 action='store',        type=int,             default=0,             help='Index of learning algorithm')
    parser.add_argument('--seed',                   action='store',        type=int,             default=0,             help='Random seed of a run')
    parser.add_argument('--run_id',                 action='store',        type=int,             default=0,             help='Index of a run')
    parser.add_argument('--save_dir',               action='store',        type=str,             default="test",        help='Directory name for storing trainning results')
    parser.add_argument('--save_rate',              action='store',        type=int,             default=1000,          help='Save result frequence')
    parser.add_argument('--save_ckpt',              action='store_true',                                                help='Wheter save ckpt or not')
    parser.add_argument('--save_ckpt_time',         action='store',        type=int,             default=23,            help='Save ckpt frequence')
    parser.add_argument('--resume',                 action='store_true',                                                help='Wheter resume training or not')
    parser.add_argument('--device',                 action='store',        type=str,             default='cpu',         help='Using cpu or gpu')
    # Hyper-params
    parser.add_argument('--total_epi',              action='store',        type=int,             default=40*1000,       help='Number of training episodes')
    parser.add_argument('--gamma',                  action='store',        type=float,           default=0.95,          help='Discount factor')
    parser.add_argument('--a_lr',                   action='store',        type=float,           default=0.0001,        help='Actor learning rate')
    parser.add_argument('--c_lr',                   action='store',        type=float,           default=0.001,         help='Critic learning rate')
    parser.add_argument('--c_train_iteration',      action='store',        type=int,             default=1,             help='Iteration for training critic')
    parser.add_argument('--eps_start',              action='store',        type=float,           default=1.0,           help='Start value of epsilon')
    parser.add_argument('--eps_end',                action='store',        type=float,           default=0.01,          help='End value of epsilon')
    parser.add_argument('--eps_stable_at',          action='store',        type=int,             default=4000,          help='End value of epsilon')
    parser.add_argument('--c_hys_start',            action='store',        type=float,           default=1.0,           help='Start value of the critic hysterisis')
    parser.add_argument('--c_hys_end',              action='store',        type=float,           default=1.0,           help='End value of the critic hysterisis')
    parser.add_argument('--adv_hys_start',          action='store',        type=float,           default=1.0,           help='Start value of the advantage hysterisis')
    parser.add_argument('--adv_hys_end',            action='store',        type=float,           default=1.0,           help='End value of the advantage hysterisis')
    parser.add_argument('--hys_stable_at',          action='store',        type=int,             default=4000,          help='End value of epsilon')
    parser.add_argument('--critic_hys',             action='store_true',                                                help='Whether uses hysterisis to critic or not')
    parser.add_argument('--adv_hys',                action='store_true',                                                help='Whether uses hysterisis to advantage value or not')
    parser.add_argument('--etrpy_w_start',          action='store',        type=float,           default=0.0,           help='Start entropy weigtht')
    parser.add_argument('--etrpy_w_end',            action='store',        type=float,           default=0.0,           help='Start entropy weigtht')
    parser.add_argument('--etrpy_w_stable_at',      action='store',        type=int,             default=4000,          help='End value of epsilon')
    parser.add_argument('--train_freq',             action='store',        type=int,             default=2,             help='Training frequence (epi)')
    parser.add_argument('--c_target_update_freq',   action='store',        type=int,             default=16,            help='Critic target-net update frequence (epi)')
    parser.add_argument('--c_target_soft_update',   action='store_true',                                                help='Wheter soft update critic target-net or not')
    parser.add_argument('--tau',                    action='store',        type=float,           default=0.01,          help='Soft updating rate')
    parser.add_argument('--n_step_TD',              action='store',        type=int,             default=0,             help='N-step TD')
    parser.add_argument('--TD_lambda',              action='store',        type=float,           default=0.0,           help='TD lambda')
    parser.add_argument('--a_mlp_layer_size',       action='store',        type=int,             default=32,            help='Number of neurons in actor-net MLP layers')
    parser.add_argument('--a_rnn_layer_size',       action='store',        type=int,             default=32,            help='Number of neurons in actor-net RNN layers')
    parser.add_argument('--c_mlp_layer_size',       action='store',        type=int,             default=32,            help='Number of neurons in critic-net MLP layers')
    parser.add_argument('--c_mid_layer_size',       action='store',        type=int,             default=32,            help='Number of neurons in critic-net MLP layers')
    parser.add_argument('--grad_clip_value',        action='store',        type=float,           default=0.0,           help='Abs value limitation for gradient clip')
    parser.add_argument('--grad_clip_norm',         action='store',        type=float,           default=0.0,           help='Norm limitation for gradient clip')
    parser.add_argument('--obs_last_action',        action='store_true',                                                help='Whether observes last action or not')
    parser.add_argument('--eval_policy',            action='store_true',                                                help='Whether evaluate policy or not')
    parser.add_argument('--eval_freq',              action='store',        type=int,             default=100,           help='Evaluation frequence')
    parser.add_argument('--eval_num_epi',           action='store',        type=int,             default=10,            help='Number of episodes per evaluation')
    parser.add_argument('--sample_epi',             action='store_true',                                                help='Whether use full-episode-based replay buffer or not')
    parser.add_argument('--trace_len',              action='store',         type=int,            default=10,            help='The length of each sequence saved in replay buffer when not using full-episode-based replay buffer') 
    # BPMA
    parser.add_argument('--grid_dim',               action='store',        type=int,  nargs=2,   default=[6,6],         help='Grid world size')
    parser.add_argument('--big_box_reward',         action='store',        type=float,           default=300.0,         help='Reward for pushing big box to the goal')
    parser.add_argument('--small_box_reward',       action='store',        type=float,           default=20.0,          help='Reward for pushing small box to the goal')
    parser.add_argument('--penalty',                action='store',        type=float,           default=-10.0,         help='Penalty for hitting wall or pushing big box alone')
    # OSD params
    parser.add_argument('--h0_speed_ps',            action='store',        type=int,  nargs='+', default=[18,18,18,18], help='Time-step cost for each work step of human_0 task')
    parser.add_argument('--h1_speed_ps',            action='store',        type=int,  nargs='+', default=[18,18,18,18], help='Time-step cost for each work step of human_1 task')
    parser.add_argument('--tb_m_speed',             action='store',        type=float,           default=0.6,           help='Turtlebot move speed m/s')
    parser.add_argument('--tb_m_noisy',             action='store',        type=float,           default=0.0,           help='Turtlebot dynamics noise')
    parser.add_argument('--tb_m_cost',              action='store',        type=float,           default=0.0,           help='Extra cost for turtlebot moving')
    parser.add_argument('--f_p_obj_tc',             action='store',        type=int,             default=4,             help='Time-step cost for finishing macro-action Pass_obj by Fetch robot')
    parser.add_argument('--f_l_obj_tc',             action='store',        type=int,             default=6,             help='Time-step cost for finishing macro-action Look_for_obj by Fetch robot')
    parser.add_argument('--f_m_noisy',              action='store',        type=float,           default=0.0,           help='Fetch robot dynamics nois')
    parser.add_argument('--f_drop_obj_pen',         action='store',        type=float,           default=-10.0,         help='Penalty for droping any tool')
    parser.add_argument('--d_pen',                  action='store',        type=float,           default=0.0,           help='Whether apply penatly for delayed tool delivery')
    #overcooked
    parser.add_argument('--task',                   action='store',        type=int,             default=0,             help='The receipt agent cooks')
    parser.add_argument('--map_type',               action='store',        type=str,             default="A",           help='The type of map')
    parser.add_argument('--step_penalty',           action='store',        type=float,           default=-0.1,          help='Penalty for every time step')


    params = vars(parser.parse_args())

    train(**params)

if __name__ == '__main__':
    main()
