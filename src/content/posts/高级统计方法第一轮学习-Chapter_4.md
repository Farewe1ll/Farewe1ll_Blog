---
title: '高级统计方法 第一轮学习 Chapter 4'
published: 2025-11-22
description: '高级统计方法 第一轮学习 Chapter 4 分类'
image: ''
tags: ['Statistical Learning']
category: 'Study'
draft: false
series: 'Maths'
---

# 高级统计方法 第一轮学习 Chapter 4 分类

## 分类问题概述

### 1.1 分类问题

#### 分类问题

给定一个特征向量 $X$ 和一个定性的响应变量 $Y$（$Y$ 的取值在集合 $\mathcal{C}$ 中），分类任务是：以特征向量 $X$ 为输入，建立函数 $C(x)$ ，预测响应变量 $Y$ 的取值，即 $C(X) = \mathcal{C}$

## 线性回归不可用

### 2.1 为什么线性回归不可用

#### 为什么线性回归不可用

1. 分类任务的编码（123，abc）没有“数值大小”的概念

2. 线性回归的预测值可能**超出类别范围**

此时可以采用多类别逻辑斯蒂回归（Multiclass Logistic Regression）或判别分析（Discriminant Analysis）。

## 逻辑斯谛回归

### 3.1 逻辑斯谛回归

#### 逻辑斯蒂回归

逻辑斯谛函数（Logistic Function）：

$$
p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}
$$

对该公式进行整理并两边同时**取对数**，得到

$$
\log\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X
$$

变化过程如下：

$$
\begin{aligned}
p(X) &= \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} \\
p(X)(1 + e^{\beta_0 + \beta_1 X}) &= e^{\beta_0 + \beta_1 X} \\
p(X) + p(X)e^{\beta_0 + \beta_1 X} &= e^{\beta_0 + \beta_1 X} \\
p(X) &= e^{\beta_0 + \beta_1 X} - p(X)e^{\beta_0 + \beta_1 X} \\
p(X) &= e^{\beta_0 + \beta_1 X}(1 - p(X)) \\
\frac{p(X)}{1 - p(X)} &= e^{\beta_0 + \beta_1 X} \\
\log\left(\frac{p(X)}{1 - p(X)}\right) &= \beta_0 + \beta_1 X
\end{aligned}
$$

等式的左边称为对数发生比（log-odd）或分对数（logit）。逻辑斯谛回归模型可以视为**分对数变换**下关于 $X$ 的一个线性模型。

一般采用**极大似然方法**估计系数：

$$
\ell(\beta_0, \beta_1) = \prod_{i:y_i=1} p(x_i) \prod_{i:y_i=0} (1 - p(x_i))
$$

该函数实际上是所有基于 Bernoulli 条件分布的二分类模型的通用似然函数。

实际上，由于逻辑斯谛回归模型是对分数变换后的线性模型，因此实际上我们就知道**最小二乘估计是最大似然估计的特例**。

### 3.2 多元逻辑斯谛回归

#### 多元逻辑斯谛回归

模型形式为：

$$
\log{\left(\frac{p(X)}{1 - p(X)}\right)} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p
$$

$$
p(X) = \frac{e^{\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p}}
$$

#### 混淆现象

与线性回归类似，只用一个预测变量得到的结果可能与多个预测变量得到的结果完全不一样，尤其是当这些因素之间存在相关性时更是如此，这种现象称为混淆现象（confounding）。

模型中认为 $x \rightarrow y$，但是实际上可能存在 $z$，使得 $z \rightarrow x$ 且 $z \rightarrow y$。此时 ，$z$ 就是一个混淆变量。此处的 $\rightarrow$ **代指因果关系**。

- 注意到因果关系具有**单向、无统一表现形式**的特点。此处应该直接从理性理解。

解决方法：通常来说将混淆变量 $z$ 纳入模型中进行控制即可。

### 3.3 其他问题

#### 多类逻辑斯谛回归

多类多元逻辑斯谛回归模型如下：

$$
\Pr(Y = k | X) = \frac{e^{\beta_{0k} + \beta_{1k} X_1 + \beta_{2k} X_2 + ... + \beta_{pk} X_p}}{\sum_{l=1}^{K} e^{\beta_{0l} + \beta_{1l} X_1 + \beta_{2l} X_2 + ... + \beta_{pl} X_p}}
$$

此处多类指的是 $Y$ 有**多个类别**，而非二分类问题；多元指的是 $X$ 有多个预测变量，而非单个预测变量。

当 $K=2$ 时，上式退化为二分类多元逻辑斯谛回归模型。

化简过程如下：

$$
\begin{aligned}
\Pr(Y = k | X) &= \frac{e^{\beta_{0k} + \beta_{1k} X_1 + \beta_{2k} X_2 + ... + \beta_{pk} X_p}}{\sum_{l=1}^{K} e^{\beta_{0l} + \beta_{1l} X_1 + \beta_{2l} X_2 + ... + \beta_{pl} X_p}} \\
&= \frac{e^{\beta_{0k} + \beta_{1k} X_1 + \beta_{2k} X_2 + ... + \beta_{pk} X_p}}{e^{\beta_{0k} + \beta_{1k} X_1 + \beta_{2k} X_2 + ... + \beta_{pk} X_p} + \sum_{l \neq k} e^{\beta_{0l} + \beta_{1l} X_1 + \beta_{2l} X_2 + ... + \beta_{pl} X_p}} \\
&= \frac{1}{1 + \sum_{l \neq k} e^{(\beta_{0l} - \beta_{0k}) + (\beta_{1l} - \beta_{1k}) X_1 + (\beta_{2l} - \beta_{2k}) X_2 + ... + (\beta_{pl} - \beta_{pk}) X_p}} \\
\end{aligned}
$$

化简过程实际上是将分子和分母同时除以了 $e^{\beta_{0k} + \beta_{1k} X_1 + \beta_{2k} X_2 + ... + \beta_{pk} X_p}$。

模型中每一类都有一个线性函数与之对应，用以判断属于这一类的概率（或不属于这一类的概率）。

多类逻辑回归也称多项（式）逻辑回归（multinomial regression），或 softmax 回归，用于多分类任务。

## 线性判别分析

### 4.0 线性判别分析

#### 线性判别分析

考虑另一类间接估计概率 $Pr(Y = k | X = x)$ 的方法。在这类方法中，首先分别对每种响应分类（给定的Y）建立预测变量 $X$ 的分布模型 (“先验知识”)， 然后运用贝叶斯定理（Bayes theorem）反过来估计 $Pr(Y = k | X = x)$（“后验概率”），最后再进行分类。

### 4.1 贝叶斯定理

#### 贝叶斯定理

假设定性变量 $Y$ 有 $K$ 类，设 $\pi_k = Pr(Y = k)$ 为一个随机选择的 $Y$ 属于第 $k$ 类的先验（prior）概率（即 $Y$ 的分布）。设 $f_k(x)$ 为在 $Y$ 属于第 $k$ 类的条件下，预测变量 $X$ 的概率密度函数（probability density function，pdf）。则根据贝叶斯定理，有：

$$
Pr(Y = k | X = x) = \frac{f_k(x) \pi_k}{\sum_{l=1}^{K} f_l(x) \pi_l}
$$

上式即表示待分类变量 $x$ 属于第 $k$ 类的概率，是后验概率。

如果写成原始形式，即为：

$$
Pr(Y = k | X = x) = \frac{Pr(X = x | Y = k) Pr(Y = k)}{\sum_{l=1}^{K} Pr(X = x | Y = l) Pr(Y = l)}
$$

对比 Beyas 公式原始形式：

$$
Pr(A | B) = \frac{Pr(B | A) Pr(A)}{\sum_{i} Pr(B | A_i) Pr(A_i)}
$$

我们发现：

1. $\Pr(Y = k)$ 即 $\pi_k$ 一般比较容易估计，取 $Y$ 的随机抽样计算即可。

2. $\Pr(X = x | Y = k)$ 即 $f_k(x)$ 的估计则**比较复杂**，一般需要对在 $Y = k$ 条件下 $X$ 的分布做出假设。

### 4.2 $p = 1$ 的线性判别分析

#### $p = 1$ 的线性判别分析

我们估计的重点是 $f_k(x)$。假设在 $Y = k$ 条件下，预测变量 $X$ 服从均值为 $\mu_k$，方差为 $\sigma^2$ 的正态分布，即

$$
f_k(x) = \frac{1}{\sqrt{2 \pi} \sigma} \exp\left(-\frac{(x - \mu_k)^2}{2 \sigma^2}\right)
$$

将其带入贝叶斯公式，得到

$$
Pr(Y = k | X = x) = \frac{\pi_k \frac{1}{\sqrt{2 \pi} \sigma} \exp\left(-\frac{(x - \mu_k)^2}{2 \sigma^2}\right)}{\sum_{l=1}^{K} \pi_l \frac{1}{\sqrt{2 \pi} \sigma} \exp\left(-\frac{(x - \mu_l)^2}{2 \sigma^2}\right)}
$$

#### 线性判别分析（判别函数）

为了分类，我们需要比较不同类别的后验概率大小。由于分母对所有类别均相同，因此只需比较分子大小即可。为简化计算，我们**对分子取对数**，得到判别函数：

$$\delta_k(x) = x \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2 \sigma^2} + \log(\pi_k)$$

该函数关于 $x$ 是线性的，因此称为线性判别分析（Linear Discriminant Analysis，LDA）。分类时，将 $x$ 分配给使 $\delta_k(x)$ 最大的类别 $k$。

化简过程如下：

$$
\begin{aligned}
\log\left(\pi_k \frac{1}{\sqrt{2 \pi} \sigma} \exp\left(-\frac{(x - \mu_k)^2}{2 \sigma^2}\right)\right) &= \log(\pi_k) + \log\left(\frac{1}{\sqrt{2 \pi} \sigma}\right) + \log\left(\exp\left(-\frac{(x - \mu_k)^2}{2 \sigma^2}\right)\right) \\
&= \log(\pi_k) + \log\left(\frac{1}{\sqrt{2 \pi} \sigma}\right) - \frac{(x - \mu_k)^2}{2 \sigma^2} \\
&= \log(\pi_k) + \log\left(\frac{1}{\sqrt{2 \pi} \sigma}\right) - \frac{x^2 - 2 x \mu_k + \mu_k^2}{2 \sigma^2} \\
&= -\frac{x^2}{2 \sigma^2} + x \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2 \sigma^2} + \log(\pi_k) + \log\left(\frac{1}{\sqrt{2 \pi} \sigma}\right)
\end{aligned}
$$

由于 $-\frac{x^2}{2 \sigma^2}$ 和 $\log\left(\frac{1}{\sqrt{2 \pi} \sigma}\right)$ 对所有类别均相同，因此在比较大小时可以忽略，得到最终的判别函数：

$$
\delta_k(x) = x \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2 \sigma^2} + \log(\pi_k)
$$

注意到此处虽然写的是 $x$ 为变量，但实际上的 $x$ 是一个常数，即待分类的样本点的预测变量取值，**真正的变量是类别 $k$**。

#### 为什么进行判别分析？

- 当类别的区分度高的时候，逻辑斯谛回归模型的参数估计不够稳定，这点在线性判别分析中是不存在的。

- 如果样本量 $n$ 比较小，而且在每一类响应分类中预测变量 $X$ 近似服从正态分布，那么线性判别分类模型比逻辑斯谛回归模型更稳定。

- 响应分类多于两类时，线性判别分析应用更普遍。

- 假设这些分布是正态分布时，则模型在形式上与逻辑斯谛回归很相似。但这种方法是通用的，也可以用于其它分布。我们接下来将集中讨论正态分布的情况。

#### 线性判别分析（概率密度）

需要把握核心矛盾：本质上归类到哪个类别，比较的是 $f_k(x) \pi_k$ 的大小。即 $Pr(X = x | Y = k) Pr(Y = k)$。

#### 线性判别分析（与贝叶斯分类器的关系）

线性判别分析实际上是贝叶斯分类器（Bayes Classifier）的一种特例。贝叶斯分类器的核心思想是：将 $x$ 分配给使后验概率 $Pr(Y = k | X = x)$ 最大的类别 $k$。而线性判别分析则假设在 $Y = k$ 条件下，预测变量 $X$ 服从均值为 $\mu_k$，方差为 $\sigma^2$ 的正态分布，从而得出具体的判别函数 $\delta_k(x)$。

在一维情况下，LDA 的“协方差矩阵相同”假设的含义是：**所有预测变量的方差相同**。

在多维情况下，LDA（线性判别分析）中“协方差矩阵相同”的假设的含义是：对于所有类别 $k$ ，不仅每个预测变量的方差相同，而且**任意两个预测变量之间的协方差**也相同。

在 LDA 中，假设各类别的协方差矩阵相同，因此判别函数是线性的。而如果各类别的协方差矩阵不同，则得到的是二次判别分析（Quadratic Discriminant Analysis，QDA），其判别函数是二次的。

### 4.3 $p > 1$ 的线性判别分析

#### $p > 1$ 的线性判别分析

LDA 推广到多元：假设不同类的群体对应的 $X=(X_1, X_2, \ldots, X_p)$ 服从均值不同、协方差矩阵相同的多元高斯分布。

- 多元高斯分布假设每一个预测变量服从一个一维正态分布，且每两个有预测变量之间都存在一些相关性。

多元高斯分布密度函数：

$$
f_k(x) = \frac{1}{(2 \pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_k)^T \Sigma^{-1} (x - \mu_k)\right)
$$

在预测变量的维度 $p>1$ 的情况下，LDA 分类器假设第 $k$ 类观测的 $x$ 服从一个多元高斯分布 $N(\mu_k, \Sigma)$，其中 $\mu_k$ 是一个均值向量，$\Sigma$ 是所有 $k$ 类共同的协方差矩阵。

带入贝叶斯公式，得到

$$
Pr(Y = k | X = x) = \frac{\pi_k \frac{1}{(2 \pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_k)^T \Sigma^{-1} (x - \mu_k)\right)}{\sum_{l=1}^{K} \pi_l \frac{1}{(2 \pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_l)^T \Sigma^{-1} (x - \mu_l)\right)}
$$

同样只关心分子并取对数，得到判别函数：

$$
\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)
$$

进一步展开：

$$
\delta_k(x) = \sum_{j=1}^{p} x_j \left(\sum_{m=1}^{p} \sigma_{jm}^{-1} \mu_{km}\right) - \frac{1}{2} \sum_{j=1}^{p} \sum_{m=1}^{p} \mu_{kj} \sigma_{jm}^{-1} \mu_{km} + \log(\pi_k)
$$

依然为线性函数。

### 4.4 二次判别分析（QDA）

#### 二次判别分析（QDA）

如果假设在 $Y = k$ 条件下，预测变量 $X$ 服从均值为 $\mu_k$，协方差矩阵为 $\Sigma_k$ 的多元正态分布，即

$$
f_k(x) = \frac{1}{(2 \pi)^{p/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)
$$

带入贝叶斯公式，得到

$$
Pr(Y = k | X = x) = \frac{\pi_k \frac{1}{(2 \pi)^{p/2} |\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)}{\sum_{l=1}^{K} \pi_l \frac{1}{(2 \pi)^{p/2} |\Sigma_l|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu_l)^T \Sigma_l^{-1} (x - \mu_l)\right)}
$$

同样只关心分子并取对数，得到判别函数：

$$
\delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log(\pi_k)
$$

该函数关于 $x$ 是二次的，因此称为二次判别分析（Quadratic Discriminant Analysis，QDA）。

如何选择 LDA 还是 QDA？

有 $p$ 个预测变量，有 $K$ 个类别时，QDA 需要预测 $K$ 个协方差，要 $Kp(p+1)/2$ 个参数；然而，通过假设 $K$ 类的协方差矩阵相同，LDA 只需要 $Kp$ 个参数需要估计。

一般地，如果训练集非常大，倾向于使用 QDA，这时，分类器的方差不再是一个主要关心的问题（因为数据量大）。

如果训练数据量相对较少，LDA 比 QDA 更好决策（QDA 要求拟合的数据更多，用少量数据时的精度不够）。

LDA 没有 QDA 分类器光滑，但拥有更低的方差；如果 LDA 假设的“$K$ 类的协方差矩阵相同”是不对的，那么 LDA 就会有很大的偏差。

## 方法比较

### 5.1 逻辑斯谛回归与LDA比较

#### 逻辑斯谛回归与LDA比较

逻辑斯谛回归和 LDA 是紧密相连的。考虑类别数为 $2$，LDA 框架下，得到如下对数形式：

$$
\log\left(\frac{Pr(Y = 1 | X = x)}{Pr(Y = 2 | X = x)}\right) = \log\left(\frac{\pi_1}{\pi_2}\right) + x^T \Sigma^{-1} (\mu_1 - \mu_2)
$$

它与逻辑斯谛回归有着相同的形式。但是两者在**拟合（参数估计）**过程中有一些差异。

- 逻辑斯谛回归中参数是由极大似然估计出来的

- LDA 是基于后验概率的分类，概率值通过估计的正态分布的均值和方差计算出来的

- 两者得到的结果应该是接近的。但 LDA 假设观测服从每一类协方差都相同的高斯分布，当假设成立时，LDA 能提供更好的结果。

逻辑斯谛回归也可以用于拟合类似于 QDA 的二次的决策边界，方法是在模型中显示的增加二次项。
