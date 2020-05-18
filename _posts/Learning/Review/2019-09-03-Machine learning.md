

![](https://photo.settour.com.tw/900x600/https%3A%2F%2Fs2.settour.com.tw%2Fss_img%2FGFG%2F0000%2F0002%2F55%2Fori_9681881.jpg)

* TOC
{:toc}

# Background

Just a review of machine learning for myself (really busy recently, so ...)

内容来自dive into deep learning, pattern recognition and machine learning, 网络。

# Basics
- Batch normalization:  subtracting the batch mean and dividing by the batch standard deviation (2 trainable parameters for mean and standard deviation, mean->0, variance->1) to counter covariance shift (i.e. the distribution of input of training and testing are different) [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift]( https://arxiv.org/pdf/1502.03167v3.pdf ) 
- MXNet's ndarray比numpy的要多2特点，1是有automatic differentiation，2是支持在GPU, and distributed cloud architectures上的asynchronous computation.
- broadcast就是复制来填充新的维度
- missing data比如NaN可以用imputation(填充一些数)或者deletion来处理
- 使用`x+=y`或者`z[:]=x`可以在老地方设置新ndarray，节约内存
- scalar, vector, matrix, tensor: 0-, 1-, 2-, n-dimension
- $L_{p}$ norm: <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200517120714786.png" alt="image-20200517120714786" style="zoom:80%;" />
- calculus微积分: integration, differentiation
- product rule: <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200517210613582.png" alt="image-20200517210613582" style="zoom:80%;" />

* quotient rule: <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200517210755705.png" alt="image-20200517210755705" style="zoom:80%;" />
* chain rule: ![image-20200517213323541](https://raw.githubusercontent.com/Wizna/play/master/image-20200517213323541.png)
* matrix calculus: <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200517213215946.png" alt="image-20200517213215946" style="zoom:80%;" />[matrix calculus wiki](https://en.wikipedia.org/wiki/Matrix_calculus)
* A gradient is a vector whose components are the partial derivatives of a multivariate function
  with respect to all its variables
* Bayes' Theorem: <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200517214155675.png" alt="image-20200517214155675" style="zoom:80%;" />
* <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200517222203691.png" alt="image-20200517222203691" style="zoom:80%;" />[推导](https://en.wikipedia.org/wiki/Variance)
* dot product: a scalar; cross product: a vector
* stochastic gradient descent: update in direction of negative gradient of minibatch<img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200518114459580.png" alt="image-20200518114459580" style="zoom:80%;" />
* likelihood: <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200518120658206.png" alt="image-20200518120658206" style="zoom:80%;" />
* 经常用negative log-likelihood来将maximize multiplication变成minimize sum
* minimizing squared error is equivalent to maximum likelihood estimation of a linear model under the assumption of additive Gaussian noise
* one-hot encoding: 1个1，其他补0
* entropy of a distribution p: <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200518173223125.png" alt="image-20200518173223125" style="zoom:80%;" />
* cross-entropy is *asymmetric*: $H(p,q)=-\sum\limits_{x\in X}{p(x)\log q(x)}$
* minimize cross-entropy == maximize likelihood
* Kullback-Leibler divergence (也叫relative entropy或者information gain) is the difference between cross-entropy and entropy: <img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200518174004941.png" alt="image-20200518174004941" style="zoom:80%;" />
* KL divergence is *asymmetric* and does not satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality)
* 

## Hyper parameters

### Grid search

### Random search



## Transfer learning

- Chop off classification layers and replace with ones cater to ones' needs. Freeze pretrained layers during training. Enable training on batch normalization layers as well may get better results.


## Curriculum learning


## Objective function

### Mean absolute error

### Mean squared error

### Cross-entropy loss

- $loss(x,class)=-\log(\frac{exp(x[class])}{\Sigma_{j}exp(x[j])})=-x[class]+\log(\Sigma_{j}exp(x[j]))$

## Regularization

### Dropout

### Label smoothing

- Use not hard target 1 and 0, but a smoothed distribution. Subtract $\epsilon$  from target class, and assign that to all the classes based on a distribution (i.e. sum to 1). So the new smoothed version is $q'(k|x)=(1-\epsilon)\delta_{k,y}+\epsilon u(k)$ (x is the sample, y is the target class, u is the class distribution) [Rethinking the Inception Architecture for Computer Vision]( https://arxiv.org/pdf/1512.00567.pdf )

## Learning rate

###  Pick learning rate

- Let the learning rate increase linearly (multiply same number) from small to higher over each mini-batch, calculate the loss for each rate, plot it (log scale on learning rate), pick the learning rate that gives the greatest decline (the going-down slope for loss) [Cyclical Learning Rates for Training Neural Networks]( https://arxiv.org/pdf/1506.01186.pdf )

### Differential learning rate

- Use different learning rate for different layers of the model, e.g. use smaller learning rate for transfer learning pretrained-layers, use a larger learning rate for the new classification layer

### Learning rate scheduling

- Start with a large learning rate, shrink after a number of iterations or after some conditions met (e.g. 3 epoch without improvement on loss)

## Initialization

### Xavier initialization

## Optimizer

- Gradient descent: go along the gradient, not applicable to extremely large model (memory, time)
- weight = weight - learning_rate * gradient
- Stochastic gradient descent: pick a sample or a subset of data, go
- 

### Momentum 

### Adagrad 

### Adam

## Ensembles

- Combine multiple models' predictions to produce a final result (can be a collection of different checkpoints of a single model or models of different structures)

##  Activations

### ReLU

### LeakyReLU

### tanh

###Sigmoid

* sigmoid是一类s型曲线
* 代表：logit function, logistic function(logit的inverse function)，hyperbolic tangent function
* 

### Softmax

* 计算<img src="https://raw.githubusercontent.com/Wizna/play/master/image-20200518141705932.png" alt="image-20200518141705932" style="zoom:80%;" />
* softmax保证了each logit >=0且和为1
* 给softmax搭配cross entropy避免了exponential带来的数值overflow或者underflow问题

* 

# Convolutional neural network



# Recurrent neural network

- May suffer vanishing or exploding gradient



## LSTM

- [Long short-term memory]( https://www.bioinf.jku.at/publications/older/2604.pdf )
- Can be bidirectional (just stack 2 lstm together, with input of opposite direction)

# Computer vision

## Data augmentation

### general

- Mixup: superimpose e.g. 2 images together with a weight respectively e.g. 0.3, 0.7, classification loss modified to mean of the 2 class (with true labels not as 1s, but as 0.3, 0.7) [mixup: Beyond Empirical Risk Minimization]( https://arxiv.org/pdf/1710.09412.pdf )

### for image

- Change to RGB, HSV, YUV, LAB color spaces
- Change the brightness, contrast, saturation and hue: grayscale
- Affine transformation: horizontal or vertical flip, rotation, rescale, shear, translate 
- Crop

### for text

- Back-translation for machine translation task, use a translator from opposite direction and generate (synthetic source data, monolingual target data) dataset

### for audio

- SoX effects
- Change to spectrograms, then apply time warping, frequency masking (randomly remove a set of frequencies), time masking [ SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition ]( https://arxiv.org/pdf/1904.08779.pdf )

## Pooling

- 

# Natural language processing

## Embeddings

### Word2vec

## N-grams



## BPE

- Byte pair encoding:  the most common pair of consecutive bytes of data is replaced with a byte that does not occur within that data, do this recursively [Neural Machine Translation of Rare Words with Subword Units]( https://arxiv.org/pdf/1508.07909.pdf )
- 

## Metrics

### BLEU

### TER



## Attention

- [Attention Is All You Need]( https://arxiv.org/pdf/1706.03762.pdf )
- 

# Reinforcement learning



# To be continued ...

