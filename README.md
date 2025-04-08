<h1 align="center"> Reinforcement Learning for Decision making in Stochastic Environment </h1>
<p align="center">
  Rawan Badarneh, Mohammed Abu Al Heja
  <p align="center">
    Technion
  </p>
</p>

### Description
This project implements two types of agents using reinforcement learning algorithms to solve decision-making in a stochastic environment:

1. **Optimal Agent**: Designed for smaller state spaces, this agent aims to act optimally to achieve the highest possible reward by maximizing efficiency in decision-making.

2. **Non-Optimal Agent**: Suited for larger state spaces, this agent utilizes Q-learning to accumulate the greatest number of points (rewards) possible. It is built to consistently achieve a positive overall reward, even in complex scenarios.

These implementations demonstrate the adaptability and scalability of reinforcement learning techniques across different environmental complexities and in limited runtime.

## 🌟 Key Features
- **Optimal Agent**:
  - **State Space Representation**: The state space for the Optimal Agent is explicitly defined to facilitate precise decision-making.
  - **Value Iteration**: Utilizes value iteration to determine the optimal policy, ensuring the agent acts in a way that maximizes the reward in smaller state spaces.

- **Non-Optimal Agent**:
  - **Feature-Based State Representation**: Instead of a full state representation, this agent uses a feature-based approach to efficiently manage larger state spaces.
  - **Q-Learning**: Employs Q-learning, a form of reinforcement learning that allows the agent to learn the value of an action in a particular state. This approach helps the agent learn to accumulate points effectively and consistently achieve positive rewards.

 

## 🚀 Getting Started

### Prerequisites
- Python 3.11 

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Rawanbad/Reinforcement_Learning.git
cd Reinforcement_Learning
```

2.Install used libraries in the project

### Running the Project

To evaluate the performance of both the Optimal and Non-Optimal Agents, you can use the included checker that works on the given inputs.
```bash
python check.py
```
