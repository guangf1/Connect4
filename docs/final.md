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
The state is a 6x7 board, where dropping a tile in any column will leave the tile on the bottom row of the empty space. There are 7 actions, each action represents 1 of the 7 columns of the board. The default reward is +1 for winning the game, -1 for losing the game or dropping a tile in a full column, and 0 for a draw or an action that does not end the game. Since connect four is a two-player competitive game, we will train our model incrementally from easy to difficult opponents to avoid not having good enough training results due to consistent repetition of the reward results.  
  
The environment built-in reward system of win +1 lose -1 draw 0 is not effective, since the reward was always 0 when the game was not over. Under such a system most of the AI's timesteps were wasted and it was very difficult to learn effective attack/defense strategies. Therefore, we modified the environment to receive reward every step and adjusted the reward mechanism of the environment in the hope that the AI receives a valid reward for each move. The new reward mechanism is as follows:  
  
![1742102978578](https://github.com/user-attachments/assets/f83c41f0-5293-44a5-b232-4c6c16ce6227)  
  
On top of that, for each action, the AI receives a -0.0005 REWARD, which is used to encourage the AI to complete a game in a shorter number of steps.  
  
In addition to adjusting the reward system, we made a very important change to the environment by allowing the opponent to also receive single-step reward for each action and subtract it from our side. The main purpose of this change is to allow the model to learn about changes in rewards from the behavior of both players, thus speeding up training and avoiding that when too many single-step rewards are accumulated, the model will give up on winning by creating more streaks to get bigger rewards(this is something that we have observed does happen, especially in the PPO algorithm)  
  
**·Baseline Approach**  
A basic use case was given in the documentation of the ConnectFour environment library. We use this example as our baseline approach. we simply create an instance of the ConnectFour environment and train a 100,000-step model using the PPO algorithm + Mlp training strategy and the default hyperparameters. An example code of the Baseline approach is as follows:
```python
env = ConnectFourEnv(opponent=BabyPlayer())
model = PPO("MlpPolicy", env, verbose=1,)
model.learn(total_timesteps=(100000))
```  
The above baseline approach is our first attempt in unlocking our understanding of the ConnectFour environment and knowing what we're doing. BabyPlayer acts almost entirely at random, so the AI quickly finds a winning strategy, even with only 100,000 timesteps. However, such training is completely impractical, and the AI learns strategies that are almost impossible to use against other levels of opponents  
  
**·Proximal Policy Optimization(PPO)**  
Proximal policy optimization is one type of reinforcement learning algorithm used to train AI. The algorithm uses a policy gradient algorithm. It combines the ideas from A2C and TRPO algorithm. The PPO training approach we used from stable_baseline3 library based on the paper ["Proximal Policy Optimization Algorithms" by Schulman, John, et al](https://arxiv.org/pdf/1707.06347). The paper summarizes that the ppo function we used is a policy gradient method for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent.  
   
The following formula defines the Clipped Surrogate Objective(CSO) of the PPO, which is used to stabilize policy updates and avoid excessive policy changes:  
  
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip} \left( r_t(\theta), 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right]$$  
  
In our project, we use the ppo method from the stable_baseline3 library. Starting with the simplest BabyPlayer provided in the environment, we gradually increase the difficulty and train up to the AdultSmarterPlayer, which is the best built-in AI player with complex gaming strategy. Starting with the simplest opponents, we gradually increase the timestep required for training, because the model can quickly learn the game strategies from opponents with lower levels of skill to find winning strategies, so we gradually increase the timestep so that we stop training when the result converges. Each time we use change_opponent() to increase the difficulty of the game, we record the current state of the mod and the Elo score. An example code of training a model using PPO algorithm is as follows:  
```python
env = ConnectFourEnv()
model = PPO("CnnPolicy/MlpPolicy", env, verbose=1)
opponents = [BabyPlayer(), BabySmarterPlayer(), ChildPlayer(), ChildSmarterPlayer(), TeenagerPlayer(), TeenagerSmarterPlayer(), AdultPlayer(), AdultSmarterPlayer()]
Elos = []
Models = []
Timesteps = [...]

for i in range(8):
    env.change_opponent(opponents[i])
    model.set_env(env)
    model.learn(total_timesteps=Timesteps[i])
    myModelPlayer = ModelPlayer(model,name="Your trained Model1")
    Elos.append(EloLeaderboard().get_elo(myModelPlayer, num_matches=200))
    Models.append(model)
```  
The stability of PPO is an important advantage. It is a strategy gradient method that aims to find the strategy that maximizes the reward, so for most RL scenarios PPO is a worthwhile method to try. In addition to this, we tried to train the model using two different training strategies, CNN and MLP, in combination with PPO.CNN is theoretically more suitable for tasks such as Connect4 because CNN, as the main strategy for processing the image inputs, should be more powerful as it perceives the spatial relationships of the pieces. However, the Connect4 board is only 6x7 in size, and the difference between the whole and the local is not that big, and it is only through the results of the experiments that we can know whether CNN is more suitable for the Connect4 scenario than Mlp.  
  
**·Deep Q-network(DQN)**  
DQN (Deep Q-Network) is a reinforcement learning algorithm that combines deep learning and Q-Learning to solve complex reinforcement learning problems. We still used the DQN function defined in stable_baseline3 library to train our AI model. The DQN approach defined in stable_baseline3 based on the paper ["Playing Atari with Deep Reinforcement Learning" by Mnih, Volodymyr, et al](https://arxiv.org/pdf/1312.5602). The paper indicated that the model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.  
The following function from the paper defined how DQN continuously adapts its strategy and converges to the optimal solution during training:  

$$
\nabla_{\theta} L_i (\theta_i) = \mathbb{E}_{s, a \sim \rho(.)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s, a; \theta_i) \right) \nabla_{\theta} Q(s, a; \theta_i) \right]
$$  
  
As we did when training the PPO model, we still take the approach of gradually increasing the difficulty of the opponents while gradually increasing the total timestep of training to prevent the model from failing to learn an effective strategy due to always winning or always losing. We also continue to use two different training strategies, Cnn and Mlp, and observe the difference between their training results and efficiency, and analyze the reasons based on the results. The code we wrote to train a model with DQN is almost identical to the code we train models with PPO. The difference is that due to the inefficient use of samples in PPO, we trained the DQN with fewer total timesteps per training session. The other differences are the hyperparameters adjusted to accommodate the differences between PPO and DQN. An example code of training a model using DQN algorithm is as follows:  
```python
env = ConnectFourEnv()
model = DQN("CnnPolicy/MlpPolicy", env, verbose=1, #Using Different Hyperparameters)
opponents = [BabyPlayer(), BabySmarterPlayer(), ChildPlayer(), ChildSmarterPlayer(), TeenagerPlayer(), TeenagerSmarterPlayer(), AdultPlayer(), AdultSmarterPlayer()]
Elos = []
Models = []
Timesteps = [... #Fewer than the steps we train with PPO]

for i in range(8):
    env.change_opponent(opponents[i])
    model.set_env(env)
    model.learn(total_timesteps=Timesteps[i])
    myModelPlayer = ModelPlayer(model,name="Your trained Model1")
    Elos.append(EloLeaderboard().get_elo(myModelPlayer, num_matches=200))
    Models.append(model)
```
Since our environment contains a discrete action space, we conjecture that the DQN algorithm would be well suited to solve the DQN problem in our adopted environment, since DQN would project the behavior that is most likely to result in the greatest gain based on past experiences in a discrete action space. We have also considered possible problems with DQN in the Connect4 environment. For zero-sum game scenarios similar to Connect4, especially in fast-paced games like Connect4, we conjecture that DQNs might have difficulty adapting to changes in the opponent's strategy, and thus be under-generalizable. Similarly, we will try to use two training strategies, Cnn and Mlp, and observe their effects on training performance and efficiency.  

**·Self_Play_Training**  
Whether training with DQN or PPO algorithms, the effect of training ends up converged. The model we trained is still a long way from perfect AI, so we wanted to find a way to improve the performance of the model again after the DQN or PPO training is complete. The approach we ultimately took was to let the model play with itself. An example code of self-play-training is as follows:  
```python
env = ConnectFourEnv()
model = PPO/DQN.load(...)
Elos = []

for i in range(...):
    opponent = ModelPlayer(model,name="yourself")
    env.change_opponent(opponent)
    model.set_env(env)
    model.learn(total_timesteps=...)
    myModelPlayer = ModelPlayer(model,name="Your trained Model1")
    Elos.append(EloLeaderboard().get_elo(myModelPlayer, num_matches=200))
```
To ensure the sustainability of the training results. We set up a for loop and through this loop the unmodel keeps updating the opponents to keep up with itself. Within each loop, we train the model with only a small number of timesteps.  

## Evaluation
The results based on the environment built-in Elo scoring system will be our primary method of evaluating our programs. In addition to using Elo scores to judge the performance of AI models, we will also compare the effectiveness and convergence of using different training policies under different RL algorithms and analyze the reasons behind. Finally, we will summarize the strategies/behaviors learned by the AI based on observations of visualizing the game processes. First, let's explain how the Elo scoring system works.  
  
When we use the get_elo() function on our already trained AI, we initialize our AI with a score of 1400, which is a score relative to the median score among built-in opponents. Similarly, the EloLeaderBoard class creates different built-in opponents as attributes and sets corresponding scores for them. Our AI will play multiple rounds against the two opponents with the closest scores. Depending on the winning percentage against two different opponents, our AI will get a new score and play multiple rounds again against the two opponents with closest scores. After num_matches rounds of changing opponents, the final score of the AI will be determined. For better understanding the process of determining Elo score, the following graph depitches a visualized procedure of Elo system:  
<img src="https://github.com/user-attachments/assets/91b9bdae-dfdb-414f-8c15-a61cf8b5d5b7" width="600">  

The eight different levels of built-in AI players in the Connect4 library have pre-set Elo scores. From worst to best: the worst is the BabyPlayer, which is almost drop tiles randomly; the best is the AdultSmarterPlayer, which has a complex attack/defence strategy; and AlphaFour, which is the perfect AI that wins 100% if it drops tile first, but it is not provided in the library. The table of AI players with corresponding Elo scores is as follows:  
![image](https://github.com/user-attachments/assets/92610dfd-2e30-4478-b6db-83e1f3362382)  

**·Result of Baseline Approach**  
Although as we expected, the results of the baseline approach were not satisfactory, it can still give us a lot of inspiration. With the default environment, the model ends up with an Elo score of just over 1200. In the case where the opponent's strategy is almost completely random and will only receive a valid reward at the end of the game, the model will only adopt meaningful behavior with a small probability when there is an action to end the game. After we changed the rewards and adjusted the environment, the model was able to achieve a score close to 1400 after training with the baseline approach. By visualizing the board and observing the actual gameplay, we found that by adding reward for a single move, the model at least learns to connect tiles in line and to block the opponent's tiles when there are already 3 tiles in a line.  
  
**·Result of PPO Approach**  
In this report, we show sample code for training PPO, i.e., constantly updating the difficulty of the opponent in a for loop and increasing the total timesteps used for training, for reasons of logical readability of the code. In our actual training of a model, instead of using a for loop, we manually change the opponents and increase the timesteps, which is a choice we made in order to monitor the results and correct errors in real time. With the same timestep and different hyperparameters chosen, the performance of the model using the Mlp and Cnn training policies grows in the following two-plots trend, respectively.

## References
**⬢Environment:** Connect four environment by Lucas Bertola | [Github Repository](https://github.com/lucasBertola/Connect-4-Gym-env-Reinforcement-learning/tree/main/exemples)  
·Defined, adapted game logic for the gymnasium library interface  
·Pretrained AI opponents for training our own models  
·Elo system for evaluating AI training results  

**⬢Python Libraries**  
stable_baselines3, matplotlib, numpy, os, random, time, pygame, gymnasium, torch

**⬢Papers**  
·Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017) [URL](https://arxiv.org/pdf/1707.06347)  
·Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013) [URL](https://arxiv.org/pdf/1312.5602)  

**⬢Tools**  
·LucidChart

## AI Tool Usage
**⬢AI Chatbot:** Chatgpt, Deepseek  
·Environment Suggestion: ([Connect four environment by Lucas Bertola](https://github.com/lucasBertola/Connect-4-Gym-env-Reinforcement-learning/tree/main/exemples))  
·Error Message Analysis  
·Markdown File Editing Guidance (How to embed a video etc.)
·RL Algorithms explanation  
·Writing helper functions: *CustomCNN()*, *count_streaks()*, *is_threatening_move()*
