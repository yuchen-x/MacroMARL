import argparse
import numpy as np
import torch
import os
import sys
import time
import random
import gym

from macro_marl.algs import MacDecQ, MacCenQ, MacDecDDRQN, ParallelMacDecDDRQN
from macro_marl import my_env
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

algs = {
        'MacDecQ': MacDecQ,
        'MacCenQ': MacCenQ,
        'MacDecDDRQN': MacDecDDRQN,
        'ParallelMacDecDDRQN': ParallelMacDecDDRQN
        }

def main(args):

    # define the name of the directory to be created
    os.makedirs("./performance/"+args.save_dir+"/train", exist_ok=True)
    os.makedirs("./performance/"+args.save_dir+"/test", exist_ok=True)
    os.makedirs("./performance/"+args.save_dir+"/check_point", exist_ok=True)
    os.makedirs("./policy_nns/"+args.save_dir, exist_ok=True)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # create env 
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
        if args.env_id.startswith('OSD-T'):
            env_params['human_speed_per_step'] = [args.h0_speed_ps, args.h1_speed_ps, args.h2_speed_ps]
        elif args.env_id.startswith('OSD-D'):
            env_params['human_speed_per_step'] = [args.h0_speed_ps, args.h1_speed_ps]
        elif args.env_id.startswith('OSD-F'):
            env_params['human_speed_per_step'] = [h0_speed_ps, h1_speed_ps, h2_speed_ps, h3_speed_ps]
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
    parser.add_argument('--alg',                action='store',        type=str,             default='MacDecQ',     help='Name of the algorithm')
    parser.add_argument('--env_id',             action='store',        type=str,             default='OSD-S-v4',    help='Gym environment id')
    parser.add_argument('--env_terminate_step', action='store',        type=int,             default=150,           help='Maximal steps for termination')
    parser.add_argument('--obs_one_hot',        action='store_true',                                                help='Whether represents observation as a one-hot vector')
    parser.add_argument('--target_flick_prob',  action='store',        type=float,           default=0.3,           help='Probability of not observing target in Capture Target domain')
    parser.add_argument('--agent_trans_noise',  action='store',        type=float,           default=0.1,           help='Agent dynamics noise in Capture Target domain')
    parser.add_argument('--target_rand_move',   action='store_true',                                                help='Whether apply target random movement')
    parser.add_argument('--n_env',              action='store',        type=int,             default=1,             help='Number of envs running in parallel')
    parser.add_argument('--n_agent',            action='store',        type=int,             default=2,             help='Number of agents')
    parser.add_argument('--cen_mode',           action='store',        type=int,             default=0,             help='Index of learning algorithm')

    #BPMA
    parser.add_argument('--grid_dim',           action='store',        type=int,  nargs=2,   default=[6,6],         help='Grid world size')
    parser.add_argument('--big_box_reward',     action='store',        type=float,           default=300.0,         help='Reward for pushing big box to the goal')
    parser.add_argument('--small_box_reward',   action='store',        type=float,           default=20.0,          help='Reward for pushing small box to the goal')
    parser.add_argument('--penalty',            action='store',        type=float,           default=-10.0,         help='Penalty for hitting wall or pushing big box alone')
    
    #OSD params
    parser.add_argument('--h0_speed_ps',        action='store',        type=int,  nargs='+', default=[18,18,18,18], help='Time-step cost for each work step of human_0 task')
    parser.add_argument('--h1_speed_ps',        action='store',        type=int,  nargs='+', default=[18,18,18,18], help='Time-step cost for each work step of human_1 task')
    parser.add_argument('--h2_speed_ps',        action='store',        type=int,  nargs='+', default=[18,18,18,18], help='Time-step cost for each work step of human_2 task')
    parser.add_argument('--h3_speed_ps',        action='store',        type=int,  nargs='+', default=[18,18,18,18], help='Time-step cost for each work step of human_3 task')
    parser.add_argument('--tb_m_speed',         action='store',        type=float,           default=0.6,           help='Turtlebot move speed m/s')
    parser.add_argument('--tb_m_noisy',         action='store',        type=float,           default=0.0,           help='Turtlebot dynamics noise')
    parser.add_argument('--tb_m_cost',          action='store',        type=float,           default=0.0,           help='Extra cost for turtlebot moving')
    parser.add_argument('--f_p_obj_tc',         action='store',        type=int,             default=4,             help='Time-step cost for finishing macro-action Pass_obj by Fetch robot')
    parser.add_argument('--f_l_obj_tc',         action='store',        type=int,             default=6,             help='Time-step cost for finishing macro-action Look_for_obj by Fetch robot')
    parser.add_argument('--f_m_noisy',          action='store',        type=float,           default=0.0,           help='Fetch robot dynamics nois')
    parser.add_argument('--f_drop_obj_pen',     action='store',        type=float,           default=-10.0,         help='Penalty for droping any tool')
    parser.add_argument('--d_pen',              action='store',        type=float,           default=0.0,           help='Whether apply penatly for delayed tool delivery')
    parser.add_argument('--obs_h_wait',         action='store_true',                                                help='Whether observes human wait or not')

    #overcooked
    parser.add_argument('--task',               action='store',        type=int,             default=0,             help='The receipt agent cooks')
    parser.add_argument('--map_type',           action='store',        type=str,             default="A",           help='The type of map')
    parser.add_argument('--step_penalty',       action='store',        type=float,           default=-0.1,          help='Penalty for every time step')

    parser.add_argument('--total_epi',          action='store',        type=int,             default=40*1000,       help='Number of training episodes')
    parser.add_argument('--replay_buffer_size', action='store',        type=int,             default=1000,          help='Number of episodes/sequences in replay buffer')
    parser.add_argument('--sample_epi',         action='store_true',                                                help='Whether use full-episode-based replay buffer or not')
    parser.add_argument('--dynamic_h',          action='store_true',                                                help='Whether apply hysteritic learning rate decay or not')
    parser.add_argument('--init_h',             action='store',        type=float,           default=0.2,           help='Initial value of hysteretic learning rate')
    parser.add_argument('--end_h',              action='store',        type=float,           default=0.4,           help='Ending value of hysteretic learning rate')
    parser.add_argument('--h_stable_at',        action='store',        type=int,             default=4*1000,        help='Decaying period according to episodes/steps')

    parser.add_argument('--eps_l_d',            action='store_true',                                                help='Whether use epsilon linear decay for exploartion or not')
    parser.add_argument('--eps_l_d_steps',      action='store',        type=int,             default=4*1000,        help='Decaying period according to episodes/steps')
    parser.add_argument('--eps_end',            action='store',        type=float,           default=0.1,           help='Ending value of epsilon')
    parser.add_argument('--eps_e_d',            action='store_true',                                                help='Whether use episode-based epsilon linear decay or not')
    parser.add_argument('--softmax_explore',    action='store_true',                                                help='Whether apply softmac for exploration')
    parser.add_argument('--h_explore',          action='store_true',                                                help='whether use history-based policy for exploration or not')
    parser.add_argument('--cen_explore',        action='store_true',                                                help='Whether use centralized explore or not')
    parser.add_argument('--cen_explore_end',    action='store',        type=int,             default=1000*1000,     help='Number of episodes for centralied explore')
    parser.add_argument('--explore_switch',     action='store_true',                                                help='Whether switch between centralized explore and decentralized explore')
    parser.add_argument('--db_step',            action='store_true',                                                help='Whether use step-based decaying manner or not')

    parser.add_argument('--l_mode',             action='store',         type=int,            default=0,             help='Index of centralized training manner')
    parser.add_argument('--optim',              action='store',         type=str,            default='Adam',        help='Optimizer')
    parser.add_argument('--l_rate',             action='store',         type=float,          default=0.0006,        help='Learning rate')
    parser.add_argument('--discount',           action='store',         type=float,          default=0.95,          help='Discount factor')
    parser.add_argument('--huber_l',            action='store_true',                                                help='Whether use huber loss or not')
    parser.add_argument('--g_clip',             action='store_true',                                                help='Whether use gradient clip or not')
    parser.add_argument('--g_clip_v',           action='store',         type=float,          default=0.0,           help='Absolute Value limitation for gradient clip')
    parser.add_argument('--g_clip_norm',        action='store_true',                                                help='Whether use norm-based gradient clip')
    parser.add_argument('--g_clip_max_norm',    action='store',         type=float,          default=0.0,           help='Norm limitation for gradient clip')

    parser.add_argument('--start_train',        action='store',         type=int,            default=2,             help='Training starts after a number of episodes')
    parser.add_argument('--train_freq',         action='store',         type=int,            default=30,            help='Updating performs every a number of steps')
    parser.add_argument('--target_update_freq', action='store',         type=int,            default=5000,          help='Updating target net every a number of steps')
    parser.add_argument('--trace_len',          action='store',         type=int,            default=10,            help='The length of each sequence saved in replay buffer when not using full-episode-based replay buffer') 
    parser.add_argument('--sub_trace_len',      action='store',         type=int,            default=1,             help='Minimal length of a sequence for traning data filtering')
    parser.add_argument('--sort_traj',          action='store_true',                                                help='Whether sort sequences based on its valid length after squeezing experiences or not, redundant param for pytorch version later than 1.1.0')
    parser.add_argument('--batch_size',         action='store',         type=int,            default=16,            help='Number of episodes/sequences in a batch')

    parser.add_argument('--mlp_layer_size',     action='store',         type=int, nargs='+', default=[32,32],       help='MLP layer dimension of decentralized policy-net (before, after) the RNN layer')
    parser.add_argument('--dec_mlp_layer_size', action='store',         type=int, nargs='+', default=[32,32],       help='MacDecDDRQN:MLP layer dimension of decentralized policy-net')
    parser.add_argument('--cen_mlp_layer_size', action='store',         type=int, nargs='+', default=[32,32],       help='MacDecDDRQN:MLP layer dimension of centralized policy-net')
    parser.add_argument('--rnn_layer_num',      action='store',         type=int,            default=1,             help='Number of RNN layers')
    parser.add_argument('--rnn_h_size',         action='store',         type=int,            default=32,            help='RNN hidden layer dimension of decentralized policy-net')
    parser.add_argument('--dec_rnn_h_size',     action='store',         type=int,            default=32,            help='MacDecDDRQN:RNN hidden layer dimension of decentralized policy-net')
    parser.add_argument('--cen_rnn_h_size',     action='store',         type=int,            default=64,            help='MacDecDDRQN:RNN hidden layer dimension of centralized policy-net')
    parser.add_argument('--lstm',               action='store_true',                                                help='Whether use lstm or not')

    parser.add_argument('--eval_freq',          action='store',         type=int,            default=100,           help='Pause training every 100 episodes for evaluation')
    parser.add_argument('--resume',             action='store_true',                                                help='Whether use saved ckpt to continue training or not')
    parser.add_argument('--save_ckpt',          action='store_true',                                                help='Whether save ckpt or not')
    parser.add_argument('--run_id',             action='store',         type=int,            default=0,             help='Index of a run')
    parser.add_argument('--seed',               action='store',         type=int,            default=None,          help='Random seed of a run')
    parser.add_argument('--save_dir',           action='store',         type=str,            default=None,          help='Directory name for storing trainning results')
    parser.add_argument('--device',             action='store',         type=str,            default='cpu',         help='Which device (CPU/GPU) to use.')

    args = parser.parse_args()

    main(args)
