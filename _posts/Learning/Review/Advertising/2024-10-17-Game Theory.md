# ![](https://img.shetu66.com/2023/03/27/1679907048365193.jpg)

# 背景

内容来自 Lectures Notes on Algorithmic Game Theory

# 学习

# lecture 1

- POA(price of anarchy): ratio between the system performance with strategic players and the best-possible system performance

- 设计规则让 POA 接近 1 最好

- Every bimatrix game has a Nash equilibrium

- if a bimatrix game is zero-sum, then a Nash equilibrium can be computed in polynomial time

# lecture 2

- In a second-price (Vickrey) auction, every truthtelling bidder is guaranteed non-negative utility

- Vickrey auction has 3 properties: strong incentive guarantees (dominant-strategy incentive-compatible DSIC), stirng performance guarantees(social surplus), computational efficiency 

# lecture 3

- Myerson's Lemma: Fix a single-parameter environment.
  
  - An allocation rule x is implementable if and only if it is monotone.
  
  - If x is monotone, then there is a unique payment rule such that the sealed-bid mechanism (x, p) is DSIC
  
  - The payment rule in (b) is given by an explicit formula
    
    - $$
      p_i(b_i, b_{-i}) =\Sigma_{j=1}^{l}{z_j \cdot \text{ jump in } x_i(\cdot , b_{-i}) \text{ at } z_j}
      $$

# lecture 4

- knapsack auction: seller 有一个总量 capacity $W$, 每个广告有一个 size $w_i$，$x_i=1$ indicates $i$ is a winning bidder. $\Sigma_{i=1}^{n} w_ix_i \le W $
- knapsack auction is not awsome (no polynomial-time implementation of the allocation rule), however we can achieve pseudopolynomial time using dynamic programming
- knapsack auction greedy algorithm: 1) sort the bidders by $\frac{b_i}{w_i}$ 2) pick winners in descending order until one doen't fit 3) return step-2 solution or highest bidder
- previous algorithm, assuming truthful bids, the surpluis of the greedy allocation rule is at least $1-\alpha$ of the maximum-possible surplus ($w_i \le \alpha W$ for every bidder $i$)
- weakening the DSIC constraint often allows you accomplish things that are provably imporssible for DSIC mechanisms. DSIC and non-DSIC, the former enjoy stronger incentive guarantees, the latter better performance guarantees.
- different types of equilibrium lead to different mechanism design.
- Stronger equilibrium concepts (e.g. domiant-strategy equilibria) requiring weaker behavioral assumptions but with narrower reach than weaker equilibrium concepts (e.g. Bayes-Nash equilibrium)

# lecture 5

- revenue-maximizing auctions: 之前都是 welfare-maximization （无所谓 input，有一个 DSIC 机制），如果是 revenue-maximization 对于不同 input，不同的策略表现更好。
- Bayesian analysis: 1) a single-parameter environment 2) the privatre valuation $v_i$ is assumed to be drawn from a distribution $F_i$, al the distributions are independent but not necessarily identical. 3) distributions $F_i$ are known in advance to mechanism designer (by bids in past auctions)
- CASE: one bidder and one item, the expected revenue of a posted price $r$ is $r\cdot (1 -F(r)$ , 因为给出价格更高，但是成功拍出的概率也就更小。the optimal posted price is called monopoly price
- expected revenue = expected virtual welfare
- A distributino $F$ is regular if the corresponding virtual valuation function $v -\frac{1-F(v)}{f(v)}$ is strictly increasing
- 对于 i.i.d bidders, 直接 vickrey auction 加上 reserve price $\varphi^{-1}(0)$

# lecture 6: simple near-optimal auctions

- Prophet inequality: for every sequence $G_1,..,G_n$ of independent distributions, there is strategy that guarantees expected reward $\frac{1}{2}E_{\pi}[max_i \pi_i]$. There is such a threshold strategy $t$, which accepts prize $i$ if and only if $\pi_i >= t$

- Bulow-Klemperer Theorem: Let $F$ be a regular distribution and $n$ a positive integer. Then: $E_{v_1,..v_{n+1}\sim F}[Rev(VA)(n+1\, bidders)]>=E_{v_1,..v_{n+1}\sim F}[Rev(OPT_F)(n\, bidders)]$ where $VA$ and $OPT_F$ denote the Vickrey auction and the optimal auction for $F$

- Above theorem indicates invest resources into getting more serious participants, rather than sharpening knowledgte of their preferences.

# lecture 7: multi-parameter mechanism design and the VCG mechanism

- 之前的情形都是 single-parameter，也就是一个 participant 有一个 private 信息（物品价值），可以扩展到 multi-paramter，也就是物品给 A 或 B 对于我来说价值不同，或者 物品 C 和 D 对我来说有偏好
- multi-paramter mechanism key ingredients:
  - $n$ strategic participants
  - a finite set $\Omega$ of outcomes
  - each participant $i$ has a private valuation $v_i(w)$ for each outcome $w\in\Omega$
- Vickrey-Clarke-Groves mechanism: in every genral mechanism design environment, there is a DSIC welfare-maximizing mechanism
- Myerson's lemma does not hold beyond single-paramter environments (not able to define monotonicity in more than 1 dimension)
- 分 2 步设计，第一步在假定都真实 bid 的情况下，allocation rule 为 $x(b) = \underset{w\in \Omega}{argmax} \Sigma_{i=1}^{n}b_i(w)$
- 第二步设计 payment rule，也就是有我们 $i$ 导致的外部损失: $p_i(b)=max_{w \in \Omega} \Sigma_{j\ne i}b_j(w) - \Sigma_{j\ne i}b_j(w^{\*})$，其中 $w^{\*}=x(b)$ ，注意到 $p_i(b) \ge 0 $
- Combinatorial Auctions: 定义是 n 个 bidders，拍卖物是 set $M$，有 m 个物品，各不相同，每个 bidder 可以获取某种物品组合，有 2 个 assumptions, $v_i(\Phi)=0$ 也就是没有得到东西时效用是 0，$v_i(S)\lt v_i(T) $ 如果 $ S\subseteq T$，也就是不要白不要
- 组合竞价有多个难点：
  - 首先，组合非常多，impractical 交流所有组合的 value，一般可以通过 indirect auction 解决，也就是日常的竞拍逻辑，一个物品记住当前的 price 和潜在 winner，然后逐渐抬价直到只剩一个人
  - 其次，福利最大化 intractable: impossible to check, bidders' valuations 是未知的
  - VCG 机制即使 DSIC 仍可能具有较差的收入和激励特性
  - 组合 auctions 经常是多轮次的，也给了策略操作空间，比如通过 bids 发送信号给别的 bidder

# lecture 8:  Combinatorial and Wireless Spectrum Auctions

- 组合拍卖有 2 类，一个是 substitues, 一个是 complements；前者更容易设计机制和现实应用，因为理论上分开卖就行
- 现实中大部分拍卖是 mixture of substitutes and complements，例如频谱拍卖
- 
