![](https://images.pexels.com/photos/1591370/pexels-photo-1591370.jpeg)

# Background

course: [deep-rl-class](https://github.com/huggingface/deep-rl-class)

# Basics

- Loop: state0, action0, reward1, state1, action1, reward2 ...
- State: complete description of the world, no hidden information
- Observation: partial description of the state of the world
- Goal: maximize expected cumulative reward
- Policy: tells what action to take given a state
- Task:

  - episodic
  - continuing
- method:

  - policy-based (learn which action to take given a state)

    - deterministic 给定 state s 会选择固定某个 action a:  $a=\pi(s)$
    - stochastic: output a probability distribution over actions
  - value-based (maps a state to the expected value of being at that state)

    - ![](https://huggingface.co/blog/assets/63_deep_rl_intro/value_1.jpg) 随时间 reward 有一个衰减， $\gamma < 1$
    - 这时 policy 是个简单的人为确定的策略
- value-based methods

  - Monte Carlo: update the value function from a complete episode, and so we use the actual accurate discounted return of this episode (获得实际的一个 episode 的 reward)

    - $$
      V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]
      $$
  - Temporal Difference: update the value function from a step, so we replace Gt that we don't have with an estimated return called TD target (获得一个 action 的 reward 以及对于下个 state 的估计来更新现在 state, 基于 bellman equation)

    - $$
      V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1})- V(S_t)]
      $$
- we have 2 types of value-based functions:

  - state-value function $V_{\pi}(s)=E_{\pi}[G_t | S_t = s]$ : value of a state
  - action-value function $Q_{\pi}(s, a)=E_{\pi}[G_t | S_t = s, A_t = a]$ : value of state-action pair
- Bellman equation:

  - $$
    V_{\pi}{(s)} = E_{\pi}[R_{t+1} + \gamma \times V_{\pi}(S_{t+1}) | S_t = s]
    $$
  - value of state s = expected value of immediate reward + the discounted value of next state

# Q-learning

- is off-policy value-based, uses a TD approach
- Trains *Q-Function* (an **action-value function**) which internally is a *Q-table* **that contains all the state-action pair values**

  - **Q**uality of the action at the state
- 算法伪代码：
- ![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-2.jpg)
- 更新公式：注意到 $S_{t+1}$ 的 best state-action pair value 是通过 greedy (updating policy)取到 max，而不是 $\epsilon$-greedy (acting policy)，所以是 off-policy

  - ![](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/Q-learning-8.jpg)

# Deep Q-learning

# Policy gradient
