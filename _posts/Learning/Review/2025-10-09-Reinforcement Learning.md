![]()


# Background


# Reinforcement learning

## Basics

- course: https://github.com/huggingface/deep-rl-class

- loop: state0, action0, reward1, state1, action1, reward2 ...

- state: complete description of the world, no hidden information

- observation: partial description of the state of the world

- goal: maximize expected cumulative reward

- policy: tells what action to take given a state

- task:
  
  - episodic
  
  - continuing

- method:
  
  - policy-based (learn which action to take given a state)
    
    - deterministic 给定 state s 会选择固定某个 action a <img src="https://huggingface.co/blog/assets/63_deep_rl_intro/policy_3.jpg" title="" alt="Policy" width="112">
    
    - stochatistic:  output a probability distribution over actions
  
  - value-based (maps a state to the expected value of being at that state)
    
    - <img src="https://huggingface.co/blog/assets/63_deep_rl_intro/value_1.jpg" title="" alt="Value based RL" width="329"> 随时间 reward 有一个衰减， γ < 1
    - 这时 policy 是个简单的人为确定的策略

- value-based methods
  
  - Monte Carlo: update the value function from a complete episode, and so we use the actual accurate discounted return of this episode (获得实际的一个 episode 的 reward)
  
  - Temporal Difference: update the value function from a step, so we replace Gt that we don't have with an estimated return called TD target （获得一个 action的 reward 以及对于下个 state 的估计来更新现在 state，基于 bellman equation）

- Bellman equation:
  
  - ![](/Users/huangruiming/Library/Application%20Support/marktext/images/2022-12-01-16-55-42-image.png)

## Q-learning

- Trains *Q-Function* (an **action-value function**) which internally is a *Q-table* **that contains all the state-action pair values**

- 一般 q-table 都初始化成 0

- 