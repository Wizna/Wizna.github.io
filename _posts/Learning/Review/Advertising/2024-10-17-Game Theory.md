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

- 
