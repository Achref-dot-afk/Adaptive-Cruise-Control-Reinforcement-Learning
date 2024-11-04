# Adaptive Cruise Control using Reinforcement Learning

## Overview

This project implements an Adaptive Cruise Control (ACC) system using Reinforcement Learning (RL) techniques. The goal of the ACC system is to automatically adjust the speed of a vehicle to maintain a safe distance from the vehicle ahead, ensuring smooth driving while enhancing safety and comfort. The project utilizes Q-learning and Monte Carlo methods to solve the Markov Decision Process (MDP) underlying the ACC problem.

## Table of Contents

- [Introduction](#introduction)
- [Reinforcement Learning Concepts](#reinforcement-learning-concepts)
- [Implementation](#implementation)
- [Greedy Policy](#greedy-policy)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Introduction

Adaptive Cruise Control systems are essential for modern vehicles, improving safety and efficiency. By employing reinforcement learning, we can create a model that learns to adjust vehicle speed based on the behavior of other vehicles on the road. This project focuses on the following key components:

- **Q-learning**: A model-free reinforcement learning algorithm that learns the value of actions in a given state.
- **Monte Carlo Every Visit Method**: A method to estimate the value of states based on the returns from multiple episodes, allowing the agent to learn from experience over time.
- **Markov Decision Process (MDP)**: A mathematical framework for modeling decision-making where outcomes are partly random and partly under the control of a decision maker.

## Reinforcement Learning Concepts

### Markov Decision Process (MDP)

An MDP is defined by:

- **States (S)**: The different situations the agent can be in, such as ego vehicle speed.
- **Actions (A)**: The choices available to the agent, such as accelerating, decelerating, or maintaining speed.
- **Rewards (R)**: The feedback received after taking an action in a state, aimed at encouraging desired behavior (e.g., maintaining a safe distance).
- **Transition Model (T)**: The probabilities of moving from one state to another given an action.

### Q-learning

Q-learning is used to learn the optimal action-value function (Q) for each state-action pair. The Q-value is updated using the following formula:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( R + \gamma \max_a Q(s', a) - Q(s, a) \right)
$$

Where:
- \( \alpha \) is the learning rate.
- \( R \) is the reward received after taking action \( a \) in state \( s \).
- \( \gamma \) is the discount factor.
- \( s' \) is the new state after taking action \( a \).

### Monte Carlo Every Visit Method

The Monte Carlo method estimates the value of each state by averaging the returns from multiple episodes where the state is visited. It helps to improve the policy based on the observed outcomes over time.

## Greedy Policy

In this project, we implement a **greedy policy** for action selection, which is crucial for guiding the agent's learning process. The greedy policy operates as follows:

- The agent always selects the action that maximizes the expected Q-value in a given state. This means that the agent will choose the action that it believes will yield the highest reward based on its current knowledge.

- While the greedy policy is effective for exploitation, it may lead to suboptimal performance if the agent does not sufficiently explore the environment. To balance exploration and exploitation, we can incorporate a strategy like **ε-greedy**, where the agent occasionally selects random actions (with probability ε) instead of always choosing the greedy action. This allows the agent to discover new states and improve its understanding of the environment over time.

By using the greedy policy, the agent focuses on refining its strategy based on learned experiences, ultimately leading to an efficient ACC system.

## Implementation

The implementation includes the following components:

1. **Environment Setup**: A simulation environment representing the vehicle dynamics and traffic conditions.
2. **Agent Design**: The agent employs Q-learning and Monte Carlo methods to learn and optimize its actions.
3. **Training Loop**: The agent interacts with the environment, collects data, and updates its knowledge.

## Results

The performance of the ACC system is evaluated based on its ability to maintain a safe distance from the lead vehicle while minimizing acceleration and deceleration. Graphs and metrics showcasing the learning progress and efficiency of the control system are included in the project directory.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
