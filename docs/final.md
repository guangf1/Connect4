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
The state is a 6x7 board, where dropping a tile in any column will leave the tile on the bottom row of the empty space. There are 7 actions, each action represents 1 of the 7 columns of the board. reward is +1 for winning the game, -1 for losing the game or dropping a tile in a full column, and 0 for a draw or an action that does not end the game. Since connect four is a two-player competitive game, we will train our model incrementally from easy to difficult opponents to avoid not having good enough training results due to consistent repetition of the reward results.  
The environment built-in reward system of win +1 lose -1 draw 0 is not effective, since the reward was always 0 when the game was not over. under such a system most of the AI's timesteps were wasted and it was very difficult to learn effective attack/defense strategies. Therefore, we have adjusted the reward mechanism of the environment in the hope that the AI receives a valid reward for each move. The new reward mechanism is as follows:
![1742102978578](https://github.com/user-attachments/assets/f83c41f0-5293-44a5-b232-4c6c16ce6227)


**·Baseline Approach**
A basic use case was given in the documentation of the ConnectFour environment library. We use this example as our baseline approach. we simply create an instance of the ConnectFour environment and train a 100,000-step model using the PPO algorithm + Mlp training strategy and the default hyperparameters. An example code of the Baseline approach is as follows:
```python
env = ConnectFourEnv(opponent=BabyPlayer())
model = PPO("MlpPolicy", env, verbose=1,)
model.learn(total_timesteps=(100000))
```  

**·Proximal Policy Optimization(PPO)**
Proximal policy optimization is one type of reinforcement learning algorithm used to train AI. The algorithm uses a policy gradient algorithm. It combines the ideas from A2C and TRPO algorithm. The PPO training approach we used from stable_baseline3 library based on the paper ["Proximal Policy Optimization Algorithms" by Schulman, John, et al](https://arxiv.org/pdf/1707.06347). The paper summarizes that the ppo function we used is a policy gradient method for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent.  
The following formula defines the Clipped Surrogate Objective(CSO) of the PPO, which is used to stabilize policy updates and avoid excessive policy changes:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip} \left( r_t(\theta), 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right]$$


**·Deep Q-network(DQN)**
DQN (Deep Q-Network) is a reinforcement learning algorithm that combines deep learning and Q-Learning to solve complex reinforcement learning problems. We still used the DQN function defined in stable_baseline3 library to train our AI model. The DQN approach defined in stable_baseline3 based on the paper ["Playing Atari with Deep Reinforcement Learning" by Mnih, Volodymyr, et al](https://arxiv.org/pdf/1312.5602). The paper indicated that the model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.  
The following function from the paper defined how DQN continuously adapts its strategy and converges to the optimal solution during training:

$$
\nabla_{\theta} L_i (\theta_i) = \mathbb{E}_{s, a \sim \rho(.)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i) \right) \nabla_{\theta} Q(s, a; \theta_i) \right]
$$

## Evaluation


## References


## AI Tool Usage
