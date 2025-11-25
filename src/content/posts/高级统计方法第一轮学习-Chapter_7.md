---
title: '高级统计方法 第一轮学习 Chapter 7'
published: 2025-11-24
description: '高级统计方法 第一轮学习 Chapter 7 非线性模型'
image: ''
tags: ['Statistical Learning']
category: 'Study'
draft: false
series: 'Maths'
---

# 高级统计方法 第一轮学习 Chapter 7 非线性模型

## 多项式回归

### 多项式回归

#### 多项式回归

多项式回归（polynomial regression）：以预测变量的幂作为新的预测变量以替代原始变量。例如，三次回归模型有三个预测变量 $x, x_2, x_3$，是一种简单实用的表达数据非线性关系的模型。

模型为：

$$
y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \ldots + \beta_d x_i^d + \epsilon_i
$$

对阶数 $d$ 的选择不宜过大，一般不大于 $3$ 或 $4$。

$\hat{f}(x_0)$ 的标准误差计算如下：

- 最小二乘法能到每个系数 $\hat{\beta}_j$ 的方差估计，以及每一对系数估计之间的协方差。

- 由此可以计算出 $\hat{f}(x_0)$ 的方差：

  $$
  \text{Var}(\hat{f}(x_0)) = \text{Var}\left(\sum_{j=0}^d \hat{\beta}_j x_0^j\right) = \sum_{j=0}^d \sum_{k=0}^d x_0^j x_0^k \text{Cov}(\hat{\beta}_j, \hat{\beta}_k)
  $$

  $\hat{f}(x_0)$ 的逐点标准误差（函数在某个已知点的标准误）就是其方差的平方根。

R 中可采用 `lm()` 函数拟合多项式回归模型。`poly()` 函数避免了输入带有 $age$ 幂项的繁琐公式，其返回的是一个矩阵，列表示的是正交多项式的基。

`poly()` 函数也可通过内加 `raw=TRUE` 参数，实现直接估计 $age$, $age^2$, $age^3$ 和 $age^4$。

`raw=TRUE` 表示直接使用 $age$ 的原始值；如果不设置此项，R 函数将对年龄数据进行变换（一般是正交变换即施密特正交化）后，再进行回归，所以二者得到的系数不同。

将多项式函数作为预测变量的逻辑斯谛回归就能用来分类。即拟合的为下列模型：

$$
\log\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X + \beta_2 X^2 + \ldots + \beta_d X^d
$$

## 阶梯函数

### 阶梯函数

#### 阶梯函数

阶梯函数（step function）：将预测变量的取值空间划分为若干区域，以此来生成新的预测变量，并在每个区间内拟合一个常数值。

首先在 $X$ 的范围内选择 $K - 1$ 个切点 $c_1, c_2, \ldots, c_{K-1}$，将 $X$ 的取值空间划分为 $K$ 个区间：

$$
\begin{aligned}
C_1 & = I(X | X < c_1) \\
C_2 & = I(X | c_1 \leq X < c_2) \\
& \ldots \\
C_K & = I(X | X \geq c_{K-1})
\end{aligned}
$$

其中 $I$ 是示性函数，表示当括号内条件成立时取值为 $1$，否则取值为 $0$。

$X$ 只能落在 $K$ 个区间中的某一个，于是对任意 $X$ 的取值：$C_1(X) + C_2(X) + \ldots + C_K(X) = 1$

以 $C_1, C_2, \ldots, C_K$ 作为新的预测变量，拟合下列线性模型：

$$
y_i = \beta_0 + \beta_1 C_1(X_i) + \beta_2 C_2(X_i) + \ldots + \beta_K C_K(X_i) + \epsilon_i
$$

- 对于 $X$ 的一个给定值，$C_1(X), C_2(X), \ldots, C_K(X)$ 中至多只有一项是非零的。

- 当 $X < C_i$ 时，式中每个 $C_j(X) = 0$，$\hat{y} = \beta_0$，所以 $\beta_0$ 是 $y$ 在区间 $(-\infty, c_1)$ 上的预测值。

当 $c_{i-1} \leq X < c_i$ 时，$\hat{y} = \beta_0 + \beta_i$，所以 $\beta_i$ 是区间 $[c_{i-1}, c_i)$ 上的相应变量的平均增量。

- 本质上，是基于 $x$ 取值设置的哑变量。

- 回归结果相当于对 $x$ 分段赋值以固定值。

拟合阶梯函数前，需对数据先用`cut()` 函数处理，可自动对年龄变量进行分割点的选择，`cut()` 函数返回的实际上是一个有序变量，之后`lm()` 据此生成一系列回归中的哑变量。

## 基函数

### 基函数

#### 基函数

多项式和阶梯函数回归模型实际上是特殊的基函数方法。基本原理是使用变量X的函数或变换 $b_1(X), b_2(X), \ldots, b_K(X)$ 作为新的预测变量，拟合线性模型：

$$
y_i = \beta_0 + \sum_{j=1}^K \beta_j b_j(X_i) + \epsilon_i
$$

- $b_j(X)$ 称为基函数（basis function）。

多项式回归：基函数为 $b_j(X) = X^j$。

阶梯函数回归：基函数为 $b_j(X) = I(X \in C_j)$。

## 回归样条

###