---
layout: default
title:  Home
---
![image](https://upload.wikimedia.org/wikipedia/commons/a/ad/Connect_Four.gif)

Connect Four is a board game that was released by Howard Wexler in 1974. In a 7x6 board, a player wins when the player places tiles in any direction to form a line of length four. Under the off the shelf [Connect four environment by Lucas Bertola](https://github.com/lucasBertola/Connect-4-Gym-env-Reinforcement-learning/tree/main/exemples), which provides different levels of opponent players for us to train a model, our project explores different RL algorithms and reward strategy to train an AI that plays connect four. In our project, we will try Proximal policy optimization(PPO), Deep Q-network(DQN), and Alphazero to train a powerful AI and evaluate their performances, and compare different stable baselines3 strategies(Mlp,Cnn). The AI performance will be evaluated by rew mean score, elo ranking(environment built-in system), visulization of the actual game plays, and the winning rate against trained models.

Source code: [https://github.com/guangf1/Connect4](https://github.com/guangf1/Connect4)

Reports:

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)
