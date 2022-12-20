# MacroMARL

MacroMARL is a framework for deep multi-agent reinforcement learning that allows agents to asynchronously learn and execution with Macro-Actions. This framework includes implementations of the following algorithms:
- Value-Based Approaches:
  - [**MacDec-Q** and **MacCen-Q**: Macro-Action-Based Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2004.08646)
  - [**MacDec-DDRQN** and **Parallel-MacDec-DDRQN**:Learning Multi-Robot Decentralized Macro-Action-Based Policies via a Centralized Q-Net](https://arxiv.org/abs/1909.08776)
- Policy-Gradient-Based Approaches:
  - [**Mac-IAC**, **Mac-CAC**, **Naive-Mac-IACC**, **Mac-IAICC**: Asynchronous Actor-Critic for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2209.10113)
  
MacroMARL is written in PyTorch with the following macro-action-based multi-agent enviroment:
  - Box Pushing
  - [Overcooked](https://github.com/WeihaoTan/gym-macro-overcooked)
  - Warehouse Took Delivery

## Installation
```
cd anaconda_env
conda env create -f environment.yml
cd ..
conda activate macro_marl
```
```
pip install -e .
```
