---
layout: default
title: Proposal
---

## Summary of the Project
Connect Four is a board game that was released by Howard Wexler in 1974. In a 7x6 board, a player wins when the player places pieces in any direction to form a line of length four.
Our project will train an AI agent to play Connect Four against existing trained agents, with the aim of exploring the efficiency of agent training under different RL algorithms and the win rate against different levels of players.
The simulated game environment will take the board's state as input and return the column where the piece fell. The agent will receive a reward of +1 for a win, 0 for a tie, and -1 for a loss.

## AI/ML Algorithms 
We plan to try to train our agent using Deep Q-Learning (DQN), Monte Carlo tree search (MCTS), or Imitation Learning (IL) if the actual implementation permits. The project will analyze the efficiency of training under these algorithms and evaluate the agent's win rate against different difficulty levels.

## Evaluation Plan
Our projects will be evaluated on the basis of a multifaceted analysis of a single piece of quantitative data. Winning rate is the most important metric for evaluating the performance of our agent, and we will evaluate our agent's performance by the winning rate under different algorithms with different amounts of training against different levels of opponents.
We wish to determine the validity of the project by comparing the performance of the agents under different variables, visualizing the results through python Matplotlib, and verifying that there is a significant difference in the performance of the agents. In addition, since Connect Four favors the player who moves first, we will also evaluate the win rate for the agent by its order of move . Our moonshot case is that the agent achieves an almost 100% win rate against the toughest opponents when playing first.

## Meet the Instructor
We plan to meet with the instructor bi-weekly starting in week 4, week 5 at the latest, to make sure our project is on the right path. In addition, when technical bottlenecks are encountered, make a drop-in appointment with the instructor as early as possible.

## AI Tool Usage
We will look to see if there is a Connect Four game environment readily available on the internet, and if not, we will simulate the game environment ourselves using numpy arrays. In addition, we would like to use existing AIs (of different levels) to play against our agent to train our agent and observe the agent's performance.
