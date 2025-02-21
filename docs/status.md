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

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip} \left( r_t(\theta), 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right]$$

In our project, we use the ppo method from the stable_baseline3 library. Starting with the simplest BabyPlayer provided in the environment, we gradually increase the difficulty and train up to the TeenagerSmarterPlayer, with each training session consisting of 250,000 timesteps. For the environment, we have so far kept the default parameters and just adjusted the learning_rate to 1e-4, to expect a good results after larger timesteps.

·Deep Q-network(DQN)  
DQN (Deep Q-Network) is a reinforcement learning algorithm that combines deep learning and Q-Learning to solve complex reinforcement learning problems. We still used the DQN function defined in stable_baseline3 library to train our AI model. The DQN approach defined in stable_baseline3 based on the paper "Playing Atari with Deep Reinforcement Learning" by Mnih, Volodymyr, et al. The paper indicated that the model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.  
The following function from the paper defined how DQN continuously adapts its strategy and converges to the optimal solution during training:

$$
\nabla_{\theta} L_i (\theta_i) = \mathbb{E}_{s, a \sim \rho(.)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i) \right) \nabla_{\theta} Q(s, a; \theta_i) \right]
$$

Just as we did with PPO, we kept increasing the difficulty of the AI opponent to train our model, but this time we only used up to 100,000 timesteps per training because the training time for DQN is longer, and we wanted to quickly validate the training for optimization purposes, and increase the timesteps when we were sure that the training was working relatively perfectly


·Alphazero  
We have not yet explored the actual approach of Alphazero, but have created a “forgery” Alphazero method based on the environment built-ins. Lucas Bertola's Connect four environment provides the ModelPlayer class, which allows the environment to compete with itself by setting its opponents to be the same as the model itself. the ModelPlayer class can be constantly updated, so our AI agent is always playing against its own level of opponent. As of now we have not explored the full usage of Alphazero much, and in the future we will either use the actual approach of alphazero or try other algorithms

## Evaluation  
After actual code implementation and model training, and understanding the CONNECT FOUR environment, we have more perspectives for evaluating the performance of the project.  

·Elo Score  
The funcionality to calculate elo scores is provided in the environment set up. This is a very interesting aspect of the environment, the higher the elo score, the better the AI model performs. As for new, we didn't test on all the AI opponents provided in the environment. The elo score comparison table is as follows:

![1740129250109(1)](https://github.com/user-attachments/assets/7e704f96-b643-4734-ab71-d25b4cd1bfac)

After we run get_elo() on all three of the models we trained, we get results as follow:

![image](https://github.com/user-attachments/assets/452b58a3-ece1-4521-b4d6-8cab79d665ad)

The plot generated by python matplotlib indicate that their performances are similar. Their performance is much better than ChildPlayer, and closer to ChildSmarterPlayer. According to the result, we think that there is a bottleneck for our model to outperform Teenager opponent, and this is a problem for us to figure out in the following weeks

·Rew Score

![image](https://github.com/user-attachments/assets/52046b7c-ff28-4197-be45-7c6a89942602)

We collected the rew mean achieved by the model at the final timestep when trained with the last few difficulty opponents. by looking at this we can see that: (i) PPO loses its performance boost as soon as it enters training with ChildSmarterModel. (ii) DQN has a better performance in shorter timesteps. Even though TeenagerModel is still a bottleneck, it has a small boost. (iii) Self-competing(Alphazero) training scores well within even shorter timesteps so it is well worth following up with further exploration

·Visualization

![1740132207454](https://github.com/user-attachments/assets/a1b1b75c-f90c-4513-90b1-64316c24ca19)

The image above shows the gameplay viewed in real time after rendering with pygame. By observing this, we realize that the trained AI will in most cases prevent the opponent from winning, or make the right move when it is about to win. In other cases, the trained AI tends to land on top of its own pieces. We believe this is due to insufficiently rich reward mechanisms. Simple +1 for winning and -1 for losing may not be able to handle the strategy in the middle of the game.

## Remaining Goals and Challenges
🟥

🟦

🟩

## Resources Used


# Video Summary
