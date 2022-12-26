# MacroMARL

MacroMARL is a framework for deep multi-agent reinforcement learning that allows agents to asynchronously learn and execution with Macro-Actions. This framework includes implementations of the following algorithms:
- Value-Based Approaches:
  - [**MacDec-Q** and **MacCen-Q**: Macro-Action-Based Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2004.08646)
  - [**MacDec-DDRQN** and **Parallel-MacDec-DDRQN**: Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net](https://arxiv.org/abs/1909.08776)
- Policy-Gradient-Based Approaches:
  - [**Mac-IAC**, **Mac-CAC**, **Naive-Mac-IACC**, **Mac-IAICC**: Asynchronous Actor-Critic for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2209.10113)
  
 <p align="left">
  <img src="https://github.com/yuchen-x/gifs/blob/master/osd.GIF" width="40%" hspace="90">
  <img src="https://github.com/yuchen-x/gifs/blob/master/two_h_same.GIF" width="40%" hspace="90">
</p>
  
## Installation
Install dependencies:
```
cd MacroMarl/anaconda_env
conda env create -f environment.yml
```
Install macro_marl package:
```
cd MacroMarl/
pip install -e .
```  

## Macro-Action-Based Multi-Agent Enviroments:

| Box Pushing | [Overcooked](https://github.com/WeihaoTan/gym-macro-overcooked)-MapA | [Overcooked](https://github.com/WeihaoTan/gym-macro-overcooked)-MapB |
|:---:|:---:|:---:|
| <img src="https://github.com/yuchen-x/gifs/blob/master/BP14x14.png" width=150></img> | <img src="https://github.com/yuchen-x/gifs/blob/master/3_agent_D.png" width=150></img> | <img src="https://github.com/yuchen-x/gifs/blob/master/3_agent_F.png" width=150></img> |
| Warehouse Tool Delivery A | Warehouse Tool Delivery B | Warehouse Tool Delivery C |
| <img src="https://github.com/yuchen-x/gifs/blob/master/wtd_a_small.png" width=210></img> | <img src="https://github.com/yuchen-x/gifs/blob/master/wtd_b_small.png" width=230></img> | <img src="https://github.com/yuchen-x/gifs/blob/master/wtd_c_small.png" width=300></img> 
| Warehouse Tool Delivery D || Warehouse Tool Delivery E |
| <img src="https://github.com/yuchen-x/gifs/blob/master/wtd_e_small.png" width=300></img> || <img src="https://github.com/yuchen-x/gifs/blob/master/wtd_d_small.png" width=300></img> | 

## Run Experiments
All the commands for running experiments can be found in the bash files under `experiments/`. Note that each bash file can launch all experiments in parallel.

- Example for running **MacDec-Q** under **Warehouse-A**
```
value_based_main.py  --save_dir='ma_hddrqn_warehouse_A' --alg='MacDecQ' --env_id='OSD-D-v7' --n_agent=3 --env_terminate_step=200 --batch_size=64 --train_freq=128 --total_epi=40_000 --replay_buffer_size=2000 --l_rate=0.0001 --h_stable_at=10_000 --eps_l_d_steps=10_000 --eps_end=0.05 --discount=1.0 --start_train=2000 --dynamic_h --eps_l_d --sample_epi --h0_speed_ps 27 20 20 20 --h1_speed_ps 27 20 20 20 --d_pen=-20.0 --tb_m_speed=0.8 --run_id=$i
```
- Example for running **Mac-IAICC** under **Warehouse-A**
```
pg_based_main.py --save_dir='ma_iaicc_warehouse_A'--alg='MacIAICC' --run_id=$i --env_id='OSD-D-v7' --n_agent=3 --l_mode=0 --env_terminate_step=200 --a_lr=0.0005 --c_lr=0.0005 --train_freq=4 --n_env=4 --c_target_update_freq=32 --n_step_TD=5 --grad_clip_norm=0 --eps_start=1.0 --eps_end=0.05 --eps_stable_at=10_000 --total_epi=40_000 --gamma=1.0 --a_rnn_layer_size=32 --c_rnn_layer_size=64 --h0_speed_ps 27 20 20 20 --h1_speed_ps 27 20 20 20 --d_pen=-20.0 --tb_m_speed=0.8 --sample_epi --eval_policy
```

## Behavior Visualization
In the `visualization/` directory, we provide examples for behavior visualization by running learned policies.
### Box Pushing
```
python test_bp_ma.py --grid_dim 14 14 --scenario='14x14'
```
### Overcooked
- Map-A
```
python test_overcooked_ma.py --env_id  Overcooked-MA-v1 --mapType A --n_agent 3 --task 6
```
- Map-B
```
python test_overcooked_ma.py --env_id  Overcooked-MA-v1 --mapType B --n_agent 3 --task 6
```
### Warehouse Tool Delivery
- Warehouse-C
```
python test_osd_s_policy_dec.py --env_id='OSD-T-v0' --scenario=40 --n_agent=3
```
- Warehouse-D
```
python test_osd_s_policy_dec.py --env_id='OSD-T-v1' --scenario=4R383827 --n_agent=4
```
- Warehouse-E
```
python test_osd_s_policy_dec.py --env_id='OSD-F-v0' --scenario=v040 --n_agent=4
```

## Code Structure
- `./scripts/` includes main files for value-based and pg-based algorithms
- `./experiments/` includes commands to run experiments
- `./visualization/` includes code for visualizing learned behaviors under each domain 
- `./src/macro_marl/algs/` includes the main learning code for each algorithm
- `./src/macro_marl/cores/` includes the core modules for each algorithm
- `./src/macro_marl/my_env` code for Box Pushing and Warehouse Tool Delivery domains

## Citations
If you are using MacroMARL in your research, please cite the corresponding papers listed below:
```
@InProceedings{xiao_corl_2019,
  author = "Xiao, Yuchen and Hoffman, Joshua and Amato, Christopher",
  title = "Macro-Action-Based Deep Multi-Agent Reinforcement Learning",
  booktitle = "3rd Annual Conference on Robot Learning",
  year = "2019"
}
```
```
@InProceedings{xiao_icra_2020,
  author = "Xiao, Yuchen and Hoffman, Joshua and Xia, Tian and Amato, Christopher",
  title = "Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net",
  booktitle = "Proceedings of the International Conference on Robotics and Automation",
  year = "2020"
}
```
```
@InProceedings{xiao_neurips_2022,
  author = "Xiao, Yuchen and Wei, Tan and Amato, Christopher",
  title = "Asynchronous Actor-Critic for Multi-Agent Reinforcement Learning",
  booktitle = "Proceedings of the Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS)",
  year = "2022"
}
```
