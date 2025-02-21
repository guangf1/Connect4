---
layout: default
title: Status
---

## Project Summary
Connect Four is a board game that was released by Howard Wexler in 1974. In a 7x6 board, a player wins when the player places pieces in any direction to form a line of length four. Under the off the shelf [Connect four environment by Lucas Bertola](https://github.com/lucasBertola/Connect-4-Gym-env-Reinforcement-learning/tree/main/exemples), which provides different levels of opponent players for us to train a model, our project explores different RL algorithms and reward strategy to train an AI that plays connect four. In our project, we will try Proximal policy optimization(PPO), Deep Q-network(DQN), and Alphazero to train a powerful AI and evaluate their performances, and compare different stable baselines3 stratigies(Mlp,Cnn). The AI performance will be evaluated by rew mean score, elo ranking(environment built-in system), visulization of the actual game plays, and the winnng rate against trained models.

## Approach
**Core ideas**:  
1. Since connect four is a two-player competitive game, we will train our model incrementally from easy to difficult opponents to avoid not having good enough training results due to consistent repetition of the reward results  
2. So far, we stop at TeenagerSmarterPlayer level opponents. We would like to achieve the desired results in a short amount of training first before expanding into more difficult areas to avoid wasting time  
3. We mimic the approach used in the exercise that utilizing the stable_baselines3 library to run the DQN and PPO algorithms. For Alphazero self-learning, we temporarily try the approach recommended in the Connect-4-Gym-env source code readme file

In the current environment, we follow the default setup. state is a 6x7 board, where dropping a piece in any column will leave the piece on the bottom row of the empty space. There are 7 actions, each action represents 1 of the 7 columns of the board. reward is +1 for winning the game, -1 for losing the game or dropping a piece in a full column, and 0 for a draw or an action that does not end the game.

**Algorithms**:  
·Proximal policy optimization(PPO)  
Proximal policy optimization is one type of reinforcement learning algorithm used to train AI. The algorithm uses a policy gradient algorithm. It combines the ideas from A2C and TRPO algorithm. The PPO training approach we used from stable_baseline3 library based on the paper "Proximal Policy Optimization Algorithms" by Schulman, John, et al. The paper summrizes that the ppo function we used is a policy gradient method for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent.  
The following formula defines the Clipped Surrogate Objective(CSO) of the PPO, which is used to stabilize policy updates and avoid excessive policy changes:

![1740126669424(1)](https://github.com/user-attachments/assets/c23b450a-c448-47b3-9183-71380da894f9)

In our project, we use the ppo method from the stable_baseline3 library. Starting with the simplest BabyPlayer provided in the environment, we gradually increase the difficulty and train up to the TeenagerSmarterPlayer, with each training session consisting of 250,000 timesteps. For the environment, we have so far kept the default parameters and just adjusted the learning_rate to 1e-4, to expect a good results after larger timesteps.

·Deep Q-network(DQN)  
DQN (Deep Q-Network) is a reinforcement learning algorithm that combines deep learning and Q-Learning to solve complex reinforcement learning problems. We still used the DQN function defined in stable_baseline3 library to train our AI model. The DQN approach defined in stable_baseline3 based on the paper "Playing Atari with Deep Reinforcement Learning" by Mnih, Volodymyr, et al. The paper indicated that the model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.  
The following function from the paper defined how DQN continuously adapts its strategy and converges to the optimal solution during training:

![image](https://github.com/user-attachments/assets/129449e0-bae5-4805-9c0b-a3da36b41992)

Just as we did with PPO, we kept increasing the difficulty of the AI opponent to train our model, but this time we only used up to 100,000 timesteps per training because the training time for DQN is longer, and we wanted to quickly validate the training for optimization purposes, and increase the timesteps when we were sure that the training was working relatively perfectly


·Alphazero  



## Evaluation

## Remaining Goals and Challenges

## Resources Used


# Video Summary
