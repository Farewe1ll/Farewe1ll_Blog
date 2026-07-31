---
title: 从零开始的 Post Training 2：DPO
published: 2026-06-20
description: "Post Training 的入门学习笔记 DPO 篇"
image: ""
tags: ["Deep_Learning", "Post_Training", "RL", "RLHF", "DPO"]
category: "Research"
draft: false
lang: ""
series: "Post Training"
---

# 从零开始的 Post Training 2：DPO

## DPO 算法

DPO（Direct Preference Optimization）是一种离线偏好优化方法。它从带有 KL 正则化的强化学习目标出发，将奖励建模和策略优化重新参数化为一个二分类损失，因此不需要显式训练奖励模型，也不需要执行策略梯度或在线强化学习采样。DPO 使用参考策略（reference policy）进行正则化，并通过系数 $\beta$ 调节偏离参考策略的强度，从而降低训练成本和实现难度。

### Bradley-Terry 模型

Bradley-Terry 模型是一种用于建模成对偏好的概率模型。它假设每个选项都有一个潜在的正数“能力”值，两个选项之间的偏好概率由这些能力值的相对大小决定。

具体来说，假设有两个选项 A 和 B，它们的能力值分别为 $w_A$ 和 $w_B$，则选项 A 被偏好于选项 B 的概率可以表示为：

$$
P(A \succ B) = \frac{w_A}{w_A + w_B} = \frac{1}{1 + e^{-(\ln w_A - \ln w_B)}}
$$

一个简单的例子：假设 A 对 B 历史战绩 8 胜 4 负，A 对 C 历史战绩 3 胜 5 负，求 B 战胜 C 的概率。

现实中的情况往往有多个选项和复杂的偏好关系，我们可以取对数最大似然估计来求解能力值 $w_A$、$w_B$ 和 $w_C$，从而计算 B 战胜 C 的概率。

似然函数可以表示为：

$$
L(w_A, w_B, w_C) = P(A \succ B)^{8} P(B \succ A)^{4} P(A \succ C)^{3} P(C \succ A)^{5}
$$

代入 Bradley-Terry 模型的概率表达式，我们可以得到：

$$
L(w_A, w_B, w_C) = \left(\frac{w_A}{w_A + w_B}\right)^{8} \left(\frac{w_B}{w_A + w_B}\right)^{4} \left(\frac{w_A}{w_A + w_C}\right)^{3} \left(\frac{w_C}{w_A + w_C}\right)^{5}
$$

我们对其取对数得到对数似然函数：

$$
\begin{aligned}
\ln L(w_A, w_B, w_C) &= 8 \ln \frac{w_A}{w_A + w_B} + 4 \ln \frac{w_B}{w_A + w_B} + 3 \ln \frac{w_A}{w_A + w_C} + 5 \ln \frac{w_C}{w_A + w_C} \\
&= 8 \ln w_A - 8 \ln (w_A + w_B) + 4 \ln w_B - 4 \ln (w_A + w_B) + 3 \ln w_A - 3 \ln (w_A + w_C) + 5 \ln w_C - 5 \ln (w_A + w_C) \\
&= 11 \ln w_A + 4 \ln w_B + 5 \ln w_C - 12 \ln (w_A + w_B) - 8 \ln (w_A + w_C)
\end{aligned}
$$

要使其最大化，我们可以对 $w_A$、$w_B$ 和 $w_C$ 分别求偏导数，并设置为零。

最后得到

$$
\begin{aligned}
\frac{\partial \ln L}{\partial w_A} &= \frac{11}{w_A} - \frac{12}{w_A + w_B} - \frac{8}{w_A + w_C} = 0 \\
\frac{\partial \ln L}{\partial w_B} &= \frac{4}{w_B} - \frac{12}{w_A + w_B} = 0 \\
\frac{\partial \ln L}{\partial w_C} &= \frac{5}{w_C} - \frac{8}{w_A + w_C} = 0
\end{aligned}
$$

解得

$$
\begin{aligned}
w_A &= 2w_B \\
w_C &= (10/3)w_B
\end{aligned}
$$

Bradley-Terry 模型只能确定这些能力值的相对比例，不能单独确定它们的共同尺度。因此可以任选一个正数作为 $w_B$，而下面的偏好概率不会受到影响。

从而

$$
P(B \succ C) = \frac{w_B}{w_B + w_C} = \frac{1}{1 + (10/3)} = \frac{3}{13} \approx 0.23
$$

除了直接求偏导，也可以通过 Bradley-Terry 模型的损失函数来求解能力值：

$$
L = -E_{(A,B)\sim D}\left[\ln\frac{w_A}{w_A+w_B}\right]
$$

上式中，$D$ 是偏好数据的分布，并约定每一对 $(A,B)$ 都把被偏好的选项放在前面；$w_A$ 和 $w_B$ 分别是两个选项的能力值。通过最小化这个损失函数，我们可以得到能力值的估计，从而计算 B 战胜 C 的概率。

上式就是一个二分类的交叉熵损失函数，本质是负对数似然。

在 RLHF 的偏好建模中，奖励模型输出的 $r(x,y)$ 更适合看作潜在效用，或者说 Bradley-Terry 能力值的对数。真正对应正数能力值的是

$$
w(x,y)=\exp(r(x,y))>0.
$$

这样既允许奖励 $r(x,y)$ 取任意实数，又能满足 Bradley-Terry 模型要求能力值为正的条件。

此时偏好概率可以表示为：

$$
P(y_A \succ y_B) = \frac{e^{r(x, y_A)}}{e^{r(x, y_A)} + e^{r(x, y_B)}}
$$

上式表示在给定输入提示 $x$ 时，生成内容 $y_A$ 被偏好于 $y_B$ 的概率。

注意 sigmoid 函数的定义为 $\sigma(z)=\frac{1}{1+e^{-z}}$，结合上式可以得到损失函数：

$$
\begin{aligned}
L
&=-E_{(x,y_A,y_B)\sim D}
\left[
\ln\frac{e^{r(x,y_A)}}{e^{r(x,y_A)}+e^{r(x,y_B)}}
\right] \\
&=-E_{(x,y_A,y_B)\sim D}
\left[
\ln\sigma(r(x,y_A)-r(x,y_B))
\right]
\end{aligned}
$$

上式中，$D$ 是偏好数据分布，$(x,y_A,y_B)$ 是一组输入和两个候选回答，$r(x,y_A)$ 与 $r(x,y_B)$ 是奖励模型给出的潜在效用。通过最小化这个损失函数，可以让奖励差拟合偏好数据。DPO 随后通过重新参数化奖励函数，避免显式训练奖励模型，并直接优化策略模型。

### DPO 的训练目标

我们引入如下几个关键元素：

- 潜在奖励函数 $r(x, y)$：用于描述偏好背后的奖励。在理论推导中假设该函数存在；实际 DPO 训练不需要单独训练或显式使用奖励模型。

- 参考策略 $\pi_{\mathrm{ref}}(y\mid x)$：一个固定的语言模型，用于提供生成内容的概率分布。

- 当前策略 $\pi_\theta(y\mid x)$：我们希望通过训练来优化的模型。

则 DPO 的训练目标可以定义为：

$$
\max_{\pi_\theta}
E_{x\sim D}
\left[
E_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]
-\beta D_{KL}
\left(
\pi_\theta(\cdot\mid x)
\parallel
\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
\right]
$$

即我们希望最大化生成内容的奖励，同时最小化训练策略与参考策略之间的 KL 散度，以确保训练策略不会偏离参考策略太远。$\beta$ 较小时，模型更强调奖励，可能更容易偏离参考策略；$\beta$ 较大时，模型更新更保守。

记上式方括号中的最大化目标为 $J(\pi_\theta)$。将它乘以负常数 $-1/\beta$，不会改变最优策略，只会把最大化问题改写成等价的最小化问题。逐步展开：

$$
\begin{aligned}
\arg\max_{\pi_\theta}J(\pi_\theta)
&=
\arg\max_{\pi_\theta}
E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[
r(x,y)
-
\beta
\ln\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right] \\
&=
\arg\min_{\pi_\theta}
E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[
\ln\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-\frac{1}{\beta}r(x,y)
\right] \\
&=
\arg\min_{\pi_\theta}
E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[
\ln\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
-
\ln\exp\left(\frac{1}{\beta}r(x,y)\right)
\right] \\
&=
\arg\min_{\pi_\theta}
E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[
\ln
\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)\exp\left(r(x,y)/\beta\right)}
\right]
\end{aligned}
$$

在这里引入 $Z(x)$ 是为了把分母归一化成一个合法的概率分布。由于 $Z(x)/Z(x)=1$，把它乘进分母不会改变对数里的值，但可以把分母拆成“归一化后的分布”和常数两项：

$$
Z(x)=\sum_{y'}\pi_{\mathrm{ref}}(y'\mid x)\exp\left(\frac{r(x,y')}{\beta}\right)
$$

于是

$$
\begin{aligned}
& \min_{\pi_\theta}
E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[
\ln
\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)\exp\left(r(x,y)/\beta\right)\frac{1}{Z(x)}Z(x)}
\right] \\
=\;& \min_{\pi_\theta}
E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[
\ln
\frac{\pi_\theta(y\mid x)}
{\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(r(x,y)/\beta\right)}
-\ln Z(x)
\right]
\end{aligned}
$$

它用于将奖励加权后的参考策略归一化为概率分布。定义目标分布：

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right)
$$

其中，$Z(x)$ 是归一化常数，整个分式才构成概率分布。

从而训练目标可以看作是最小化当前策略 $\pi_\theta(\cdot\mid x)$ 和目标分布 $\pi^*(\cdot\mid x)$ 之间的 KL 散度：

$$
\begin{aligned}
&\min_{\pi_\theta}
E_{x\sim D,\,y\sim\pi_\theta(\cdot\mid x)}
\left[
\ln\frac{\pi_\theta(y\mid x)}{\pi^*(y\mid x)}-\ln Z(x)
\right] \\
=\;&\min_{\pi_\theta}
E_{x\sim D}
\left[
D_{KL}
\left(
\pi_\theta(\cdot\mid x)
\parallel
\pi^*(\cdot\mid x)
\right)
-\ln Z(x)
\right]
\end{aligned}
$$

这里的 $\ln Z(x)$ 不依赖于 $\pi_\theta$，因此在优化过程中可以忽略。最终的训练目标可以简化为：

$$
\min_{\pi_\theta}
E_{x\sim D}
\left[
D_{KL}
\left(
\pi_\theta(\cdot\mid x)
\parallel
\pi^*(\cdot\mid x)
\right)
\right]
$$

由于 KL 散度的非负性，在策略模型能够表示该分布的理想情况下，最优策略满足 $\pi_\theta(\cdot\mid x)=\pi^*(\cdot\mid x)$。实际训练中，$\pi_\theta$ 会尽可能逼近由参考策略和潜在奖励函数共同定义的目标分布 $\pi^*$。

根据目标分布 $\pi^*(y\mid x)$ 的定义，可以反解潜在奖励函数：

$$
\begin{aligned}
\pi^*(y\mid x)
&=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right) \\
\implies r(x,y)
&=\beta\ln\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\ln Z(x)
\end{aligned}
$$

根据 Bradley-Terry 模型，我们需要比较偏好回答 $y_w$ 与非偏好回答 $y_l$ 的奖励差。将最优策略替换为待训练的参数化策略 $\pi_\theta$ 后，$\beta \ln Z(x)$ 会在奖励差中抵消，最终得到 DPO 损失：

$$
\begin{aligned}
L_{\mathrm{DPO}}(\theta)
&=
-E_{(x,y_w,y_l)\sim D}
\left[
\ln\sigma\left(
r(x,y_w)-r(x,y_l)
\right)
\right] \\
&=
-E_{(x,y_w,y_l)\sim D}
\left[
\ln\sigma\left(
\beta\ln\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
+\beta\ln Z(x)
-
\beta\ln\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
-\beta\ln Z(x)
\right)
\right] \\
&=
-E_{(x,y_w,y_l)\sim D}
\left[
\ln\sigma\left(
\beta\ln\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\ln\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right]
\end{aligned}
$$

其中，$y_w$ 表示数据中被选择的偏好回答，$y_l$ 表示被拒绝的非偏好回答。

### DPO 的优缺点

#### 优点

- 训练成本低、难度低：不需要显式训练奖励模型或价值模型，训练成本和难度相比于 PPO 算法显著降低，也不需要在线采样循环。

- 易于实现：损失函数和训练流程相对简单。

#### 缺点

- 对高质量二元偏好数据的数量和覆盖范围有较强依赖，偏好数据的收集与清洗成本较高。

- DPO 的效果高度依赖偏好数据质量，面对长链推理、在线探索或奖励稀疏问题时，可能存在局限。

### 参考资料

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
