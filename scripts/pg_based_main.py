import argparse
import gym
import torch
import random
import numpy as np
import os
import pickle
import time

from macro_marl.algs import MacIAC, MacCAC, NaiveMacIACC, NaiveMacIASC, MacIAICC, MacIAISC
from macro_marl import my_env
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

algs = {
        'MacIAC': MacIAC,
        'MacCAC': MacCAC,
        'NaiveMacIACC': NaiveMacIACC,
        'NaiveMacIASC': NaiveMacIASC,
        'MacIAICC': MacIAICC,
        'MacIAISC': MacIAISC,
        }

def main(args):

    # set seed
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # create the dirs to save results
    os.makedirs("./performance/" + args.save_dir + "/train", exist_ok=True)
    os.makedirs("./performance/" + args.save_dir + "/test", exist_ok=True)
    os.makedirs("./performance/" + args.save_dir + "/ckpt", exist_ok=True)
    os.makedirs("./policy_nns/" + args.save_dir, exist_ok=True)

    if args.env_id.startswith('OSD') or args.env_id.startswith('BP'):
        env_params = {'grid_dim': args.grid_dim,
                      'big_box_reward': args.big_box_reward,
                      'small_box_reward': args.small_box_reward,
                      'penalty': args.penalty,
                      'n_agent': args.n_agent,
                      'terminate_step': args.env_terminate_step,
                      'TB_move_speed': args.tb_m_speed,
                      'TB_move_noisy': args.tb_m_noisy,
                      'TB_move_cost': args.tb_m_cost,
                      'fetch_pass_obj_tc': args.f_p_obj_tc,
                      'fetch_look_for_obj_tc': args.f_l_obj_tc,
                      'fetch_manip_noisy': args.f_m_noisy,
                      'delay_delivery_penalty': args.d_pen,
                      'obs_human_wait': args.obs_h_wait
                      }
        if args.env_id.startswith('OSD-F'):
            env_params['human_speed_per_step'] = [args.h0_speed_ps, args.h1_speed_ps, args.h2_speed_ps, args.h3_speed_ps]
        elif args.env_id.startswith('OSD-T'):
            env_params['human_speed_per_step'] = [args.h0_speed_ps, args.h1_speed_ps, args.h2_speed_ps]
        elif args.env_id.startswith('OSD-D'):
            env_params['human_speed_per_step'] = [args.h0_speed_ps, args.h1_speed_ps]
        else:
            env_params['human_speed_per_step'] = [args.h0_speed_ps]
        env = gym.make(args.env_id, **env_params)
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
                "step penalty": args.step_penalty
                }
        env_params = {
                'grid_dim': args.grid_dim,
                'map_type': args.map_type, 
                'task': TASKLIST[args.task],
                'rewardList': rewardList,
                'debug': False
                }
        env = gym.make(args.env_id, **env_params)
        if args.env_id.find("MA") != -1:
            env = MacEnvWrapper(env)

    model = algs[args.alg](env, **vars(args))
    model.learn()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg',                    action='store',        type=str,             default='MacDecQ',     help='Name of the algorithm')
    parser.add_argument('--env_id',                 action='store',        type=str,             default='OSD-S-v4',    help='Gym environment id')
    parser.add_argument('--env_terminate_step',     action='store',        type=int,             default=150,           help='Maximal steps for termination')
    parser.add_argument('--n_env',                  action='store',        type=int,             default=1,             help='Number of envs running in parallel')
    parser.add_argument('--n_agent',                action='store',        type=int,             default=2,             help='Number of agents')
    parser.add_argument('--l_mode',                 action='store',        type=int,             default=0,             help='Index of learning algorithm')
    parser.add_argument('--seed',                   action='store',        type=int,             default=0,             help='Random seed of a run')
    parser.add_argument('--run_id',                 action='store',        type=int,             default=0,             help='Index of a run')
    parser.add_argument('--save_dir',               action='store',        type=str,             default="test",        help='Directory name for storing trainning results')
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
    parser.add_argument('--eps_stable_at',          action='store',        type=int,             default=4000,          help='Epsilon decay period')
    parser.add_argument('--c_hys_start',            action='store',        type=float,           default=1.0,           help='Start value of the critic hysterisis')
    parser.add_argument('--c_hys_end',              action='store',        type=float,           default=1.0,           help='End value of the critic hysterisis')
    parser.add_argument('--adv_hys_start',          action='store',        type=float,           default=1.0,           help='Start value of the advantage hysterisis')
    parser.add_argument('--adv_hys_end',            action='store',        type=float,           default=1.0,           help='End value of the advantage hysterisis')
    parser.add_argument('--hys_stable_at',          action='store',        type=int,             default=4000,          help='Hysterisis changing period')
    parser.add_argument('--critic_hys',             action='store_true',                                                help='Whether uses hysterisis to critic or not')
    parser.add_argument('--adv_hys',                action='store_true',                                                help='Whether uses hysterisis to advantage value or not')
    parser.add_argument('--etrpy_w_start',          action='store',        type=float,           default=0.0,           help='Start entropy weigtht')
    parser.add_argument('--etrpy_w_end',            action='store',        type=float,           default=0.0,           help='End entropy weigtht')
    parser.add_argument('--etrpy_w_stable_at',      action='store',        type=int,             default=4000,          help='Entroy weight decay period')
    parser.add_argument('--train_freq',             action='store',        type=int,             default=2,             help='Training frequence (epi)')
    parser.add_argument('--c_target_update_freq',   action='store',        type=int,             default=16,            help='Critic target-net update frequence (epi)')
    parser.add_argument('--c_target_soft_update',   action='store_true',                                                help='Wheter soft update critic target-net or not')
    parser.add_argument('--tau',                    action='store',        type=float,           default=0.01,          help='Soft updating rate')
    parser.add_argument('--n_step_TD',              action='store',        type=int,             default=0,             help='N-step TD')
    parser.add_argument('--TD_lambda',              action='store',        type=float,           default=0.0,           help='TD lambda')
    parser.add_argument('--a_mlp_layer_size',       action='store',        type=int,  nargs='+', default=[32,32],       help='Number of neurons in actor-net MLP layers (before, after) the RNN layer')
    parser.add_argument('--a_rnn_layer_size',       action='store',        type=int,             default=32,            help='Number of neurons in actor-net RNN layers')
    parser.add_argument('--c_mlp_layer_size',       action='store',        type=int,  nargs='+', default=[32,32],       help='Number of neurons in critic-net MLP layers (before, after) the RNN layer')
    parser.add_argument('--c_mid_layer_size',       action='store',        type=int,             default=32,            help='Number of neurons in critic-net mid layers (replacing RNN in state-based critic)')
    parser.add_argument('--c_rnn_layer_size',       action='store',        type=int,             default=32,            help='Number of neurons in critic-net RNN layers')
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
    parser.add_argument('--h2_speed_ps',            action='store',        type=int,  nargs='+', default=[18,18,18,18], help='Time-step cost for each work step of human_2 task')
    parser.add_argument('--h3_speed_ps',            action='store',        type=int,  nargs='+', default=[18,18,18,18], help='Time-step cost for each work step of human_3 task')
    parser.add_argument('--tb_m_speed',             action='store',        type=float,           default=0.6,           help='Turtlebot move speed')
    parser.add_argument('--tb_m_noisy',             action='store',        type=float,           default=0.0,           help='Turtlebot dynamics noise')
    parser.add_argument('--tb_m_cost',              action='store',        type=float,           default=0.0,           help='Extra cost for turtlebot moving')
    parser.add_argument('--f_p_obj_tc',             action='store',        type=int,             default=4,             help='Time-step cost for finishing macro-action Pass_obj by Fetch robot')
    parser.add_argument('--f_l_obj_tc',             action='store',        type=int,             default=6,             help='Time-step cost for finishing macro-action Look_for_obj by Fetch robot')
    parser.add_argument('--f_m_noisy',              action='store',        type=float,           default=0.0,           help='Fetch robot dynamics nois')
    parser.add_argument('--f_drop_obj_pen',         action='store',        type=float,           default=-10.0,         help='Penalty for droping any tool')
    parser.add_argument('--d_pen',                  action='store',        type=float,           default=0.0,           help='Whether apply penatly for delayed tool delivery')
    parser.add_argument('--obs_h_wait',             action='store_true',                                                help='Whether observes human wait or not')
    #overcooked
    parser.add_argument('--task',                   action='store',        type=int,             default=0,             help='The receipt agent cooks')
    parser.add_argument('--map_type',               action='store',        type=str,             default="A",           help='The type of map')
    parser.add_argument('--step_penalty',           action='store',        type=float,           default=-0.1,          help='Penalty for every time step')

    args = parser.parse_args()

    main(args)
