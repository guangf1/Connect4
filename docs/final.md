---
layout: default
title: Final Report
---

## Video


## Project Summary
Connect Four is a board game that was released by Howard Wexler in 1974. In a 7x6 board, a player wins when the player places tiles in any direction to form a line of length four. In RL, Connect4's special features include the fact that it is a two-player zero-sum game with a fully observable state. the Connect4 board is small and the rules are simple, but an AI may face opponents with different levels of skills and playing strategies. As a result, when training an AI, the results are highly convergent and difficult to be applied to other opponents.  

The environment we tested on is [Connect four environment by Lucas Bertola](https://github.com/lucasBertola/Connect-4-Gym-env-Reinforcement-learning/tree/main/exemples). This environment is adapted to python gymnasium library and can be applied directly with the stable_baselines3 library. Although this environment is not very powerful, for example, it does not adapt to Cnn and has an oversimplified reward mechanism. However, we chose this environment for two reasons: (1) it has trained AI opponents for us to train our own AI and test it; (2) it has a powerful built-in Elo evaluation function for evaluating the training results of the model (see evaluation section).  

In our project, we will try Proximal policy optimization(PPO), Deep Q-network(DQN), and MCTS to train a powerful AI and evaluate their performances, and compare different training strategies(Mlp,Cnn). We will compare the performance of different RL algorithms and training strategies, look for methods that can train a generic and robust Connect4 AI, and analyze the connections behind them.

## Approaches


## Evaluation


## References


## AI Tool Usage
