# MacroMARL

MacroMARL is a framework for deep multi-agent reinforcement learning that allows agents to asynchronously learn and execution with Macro-Actions. This framework includes implementations of the following algorithms:
- Value-Based Approaches:
  - [**MacDec-Q** and **MacCen-Q**: Macro-Action-Based Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2004.08646)
  - [**MacDec-DDRQN** and **Parallel-MacDec-DDRQN**: Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net](https://arxiv.org/abs/1909.08776)
- Policy-Gradient-Based Approaches:
  - [**Mac-IAC**, **Mac-CAC**, **Naive-Mac-IACC**, **Mac-IAICC**: Asynchronous Actor-Critic for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2209.10113)
  
MacroMARL is written in PyTorch with the following macro-action-based multi-agent enviroment:
  - Box Pushing
  - [Overcooked](https://github.com/WeihaoTan/gym-macro-overcooked)
  - Warehouse Took Delivery

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

## Commands for Experiments
All the commands for experiments can be found in the bash files under `experiments/`. Note that each bash file can launch all experiments in parallel. Do not directly run the bash file if there is not enough computing resource.

## Code Structure
- `./scripts/ma_hddrqn.py` the main files for all algorithms
- `./experiments/` commands to run experiments
- `./visualization/` code for visualizing learned behaviors under each domain 
- `./src/macro_marl/{alg_name}/` the source files for a specific algorithm
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
