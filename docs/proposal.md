---
layout: default
title: Proposal
---

## Summary of the Project
Connect Four is a connect-the-dots board game that was released by Howard Wexler in 1974. In a 7x5 game, a player wins when he or she connects pieces in any direction to form a line of length four.
Our project will train an AI agent to participate in Connect Four, with the aim of exploring the efficiency of agent training under different RL algorithms and the win rate against different levels of players.
The simulated game environment will take as input the state of the board in the form of a numpy array, and return the column in which the piece fell.

## AI/ML Algorithms 
We are going to try to train our agent using Deep Q-Learning (DQN), Monte Carlo tree search (MCTS), or Imitation Learning (IL) if the conditions permit in the subsequent research process, to explore the efficiency of training agent under different algorithms as well as the winning rate of the agent against different difficulty levels.

## Evaluation Plan
Our projects will be evaluated on the basis of a multifaceted analysis of a single piece of quantitative data. Winning rate is the most important metric for evaluating the performance of an agent, and we will validate the effectiveness of the project by the winning rate of an agent using different algorithms with different levels of training against different levels of opponents.
We wish to determine the validity of the project by comparing the performance of the agents under different variables, visualizing the results through python Matplotlib, and verifying that there is a significant difference in the performance of the agents. In addition, since Connect Four is a game in which the first move has an absolute advantage, we will also validate the win rate of the agent playing with a forehand or a queen's move, and expect the agent to achieve a 100% win rate against the most difficult opponents at the end, as long as it is a forehand.

## Meet the Instructor
We plan to meet with the INSTRUCTOR bi-weekly starting in week 4, week 5 at the latest, to make sure our PROJECT is on the right path. In addition, when technical bottlenecks are encountered, make an appointment with the INSTRUCTOR as early as possible!

## AI Tool Usage
We will look to see if there is a Connect Four game environment readily available on the internet, and if not, we will simulate the game environment ourselves using numpy arrays. In addition, we would like to use existing AIs (of different levels) to play against our agent to train our agent and observe the agent's performance
