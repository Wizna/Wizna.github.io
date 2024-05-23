# ![](https://webneel.com/wallpaper/sites/default/files/images/10-2014/scenery-wallpaper-15.jpg)

### Background

Given different targets of our recommendation system, we have 1 predicted score for each of them. How to ensemble these scores of different physical meanings and scales into one that can represent all the goals of the system is not easy.

I would like to divide this problem into two parts: function and score. Function is used to ensemble different targets into 1 and normally a sort is right after calculation of ensemble function on different scores. Score is the predicted value of a target, the raw score can be used, but often we do some normalization on it.

### 1. Function

#### 1.1 multiplication

One commonly used method is to use multiplication.

$$
score=\prod_{s\in targets}{(\alpha * s + bias)}^{\beta}
$$

#### 1.2 addition

$$
score=\sum_{s\in targets}{\alpha * s}
$$

Normally, for targets we want, e.g. watch_time, like, comment, share, $\alpha$ is positive.

For targets we don't want, e.g. reduce_similar, report, $\alpha$ is negative.

### 2. Score

#### 2.1 Raw score

Use the predicted score directly. Pro is that we have keep the information of predicted score. The con is that we must handle different scale by normalization. Mean predicted effective view rate maybe is 0.4, but like rate is around 0.01. A isotonic function is applied on the score.

Cons:

- The physical meaning, order of magnitude and distribution of different targets can be significantly different. Directly ensemble of raw scores will cause some of targets to fail.

- Explicit feedback behaviors, e.g. like, report, varies greatly among different users. Thus a fixed weight $\alpha$ is hard to choose.

#### 2.2 Rank

Sort values of target s in descending order. Normalize rank 1 ~ N to value in [0, 1]

A possible candidate function is 

$$
-\log{(\alpha * rank + bias)}

$$

#### 2.3 Percentile

Percentile is also commonly used. It's sort of similar to rank, while rank is request-wise, and percentile can be target-wise. By logging of all the predicted values of $x$, for a new given value $x_i$ , we can easily compute the percentile of it among all the values. It can be viewed as a different version of rank (target-wise)

Normally, sparse actions (e.g. like, comment, share) should use percentile instead of raw score; while dense actions (e.g. effective view, click) should use raw score.

#### 2.4 Normalization

For a monotonically increasing function, we can transform a value to another value without messing up the order.

Commonly used functions are as follows:



# To be continued ...
