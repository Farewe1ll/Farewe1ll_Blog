---
title: 从零开始的 Post Training 2：DPO
published: 2026-06-20
description: 'Post Training 的入门学习笔记 DPO 篇'
image: ''
tags: ['Deep_Learning', 'Post_Training', 'RL', 'RLHF', 'DPO']
category: 'Research'
draft: false
lang: ''
series: 'Post Training'
---

# 从零开始的 Post Training 2

## DPO 算法

DPO（Direct Preference Optimization）是一种直接优化偏好（Preference）的强化学习算法，将从偏好数据中学习奖励，再用 RL 优化的策略转变为一个关于策略（policy）参数的最大似然/二分类损失，因此不需要显式训练一个 reward 模型，也不需要在训练过程中用强化学习采样/策略梯度，DPO 用一个参考模型（reference policy）来做 KL/正则化的角色，并通过一个可控系数 $\beta$ 调节“偏离参考模型”的强度，从而大大降低了训练的成本和难度。

### Bradley-Terry 模型

Bradley-Terry 模型是一种用于建模偏好数据的概率模型，常用于二分类问题中。该模型假设每个选项都有一个潜在的非负“能力”值，偏好关系由这些能力值决定。

具体来说，假设有两个选项 A 和 B，它们的能力值分别为 $w_A$ 和 $w_B$，则选项 A 被偏好于选项 B 的概率可以表示为：

$$
P(A \succ B) = \frac{w_A}{w_A + w_B} = \frac{1}{1 + e^{-(\ln w_A - \ln w_B)}}
$$

一个简单的例子：假设 A 对 B 历史战绩 8 胜 4 负，A 对 C 历史战绩 3 胜 5 负，求 B 战胜 C 的概率。

现实中的情况往往有多个选项和复杂的偏好关系，我们可以取对数最大似然估计来求解能力值 $w_A$、$w_B$ 和 $w_C$，从而计算 B 战胜 C 的概率。

似然函数可以表示为：

$$
L(w_A, w_B, w_C) = \prod_{i} P(A_i \succ B_i)^{8} P(B_i \succ A_i)^{4} P(A_i \succ C_i)^{3} P(C_i \succ A_i)^{5}
$$

带入 Bradley-Terry 模型的概率表达式，我们可以得到：

$$
L(w_A, w_B, w_C) = \prod_{i} \left(\frac{w_A}{w_A + w_B}\right)^{8} \left(\frac{w_B}{w_A + w_B}\right)^{4} \left(\frac{w_A}{w_A + w_C}\right)^{3} \left(\frac{w_C}{w_A + w_C}\right)^{5}
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

从而

$$
P(B \succ C) = \frac{w_B}{w_B + w_C} = \frac{1}{1 + (10/3)} = \frac{3}{13} \approx 0.23
$$

除去求偏导，我们也可以通过 Bradley-Terry 模型一般的 Loss 函数来求解能力值：

$$
L = - E_{(A, B) \sim D} \left[ \ln \frac{w_A}{w_A + w_B} \right]
$$

上式中，$D$ 是偏好数据的分布，$(A, B)$ 是从数据中采样的一对选项，$w_A$ 和 $w_B$ 分别是选项 A 和 B 的能力值。通过最小化这个损失函数，我们可以得到能力值的估计，从而计算 B 战胜 C 的概率。

上式就是一个二分类的交叉熵损失函数，本质是负对数似然。

在 RL 中，Reward 模型的评估结果可以看作是选项的能力值，而偏好数据则是选项之间的比较结果。通过最小化上述的损失函数，我们可以直接优化策略参数，使得模型生成的内容更符合偏好。

需要注意，我们通常会在 Reward 模型的评估结果上加上指数函数来确保能力值为正数。

此时偏序的概率可以表示为：

$$
P(y_A \succ y_B) = \frac{e^{r(x, y_A)}}{e^{r(x, y_A)} + e^{r(x, y_B)}}
$$

上式为我们给定输入 Prompt x 的情况下，生成内容 $y_A$ 被偏好于生成内容 $y_B$ 的概率。

注意到 sigmoid 函数的定义为 $\sigma(z) = \frac{1}{1 + e^{-z}}$，结合上式可以得到 Loss 函数为：

$$
\begin{aligned}
L &= - E_{(x, y_A, y_B) \sim D} \left[ \ln \frac{e^{r(x, y_A)}}{e^{r(x, y_A)} + e^{r(x, y_B)}} \right] \\
&= - E_{(x, y_A, y_B) \sim D} \left[ \ln \sigma(r(x, y_A) - r(x, y_B)) \right]
\end{aligned}
$$

上式中，$D$ 是偏好数据的分布，$(x, y_A, y_B)$ 是从数据中采样的一组输入和两个生成内容，$r(x, y_A)$ 和 $r(x, y_B)$ 分别是 Reward 模型对生成内容 $y_A$ 和 $y_B$ 的评估结果。通过最小化这个损失函数，我们可以直接优化策略参数，使得模型生成的内容更符合人类的偏好，从而实现 RLHF 的目标。

### DPO 的训练目标

我们引入如下几个关键元素：

- 奖励函数 $r(x, y)$：用于评估生成内容的质量，通常是一个神经网络模型。

- 基准模型 $\pi(y|x)$：一个预训练的语言模型，用于提供生成内容的概率分布。

- 训练模型 $\pi'(y|x)$：我们希望通过训练来优化的模型。

则 DPO 的训练目标可以定义为：

$$
\max_{\pi'} E_{x \sim D, y \sim \pi'} \left[ r(x, y) - \beta D_{KL}(\pi'(y|x) || \pi(y|x)) \right]
$$

即我们希望最大化生成内容的奖励，同时最小化生成内容与基准模型之间的 KL 散度，以确保生成内容不会偏离基准模型太远。在这里的 $\beta$ 是一个超参数，用于控制奖励和 KL 散度之间的权重。$\beta$ 值大时，模型会更努力地拟合偏好数据，但可能过度拟合或偏离参考策略太远；$\beta$ 值小时，模型更新更保守，会紧紧锚定在参考策略附近。

进一步推导，得到

$$
\begin{aligned}
& \max_{\pi'} E_{x \sim D, y \sim \pi'} \left[ r(x, y) - \beta D_{KL}(\pi'(y|x) || \pi(y|x)) \right]\\
=& \max_{\pi'} E_{x \sim D, y \sim \pi'} r(x, y) - E_{x \sim D, y \sim \pi'} \beta \ln \frac{\pi'(y|x)}{\pi(y|x)} \\
=& \max_{\pi'} E_{x \sim D, y \sim \pi'} \left[ r(x, y) - \beta \ln \frac{\pi'(y|x)}{\pi(y|x)} \right] \\
=& \min_{\pi'} E_{x \sim D, y \sim \pi'} \left[ - \frac1{\beta} r(x, y) + \ln \frac{\pi'(y|x)}{\pi(y|x)} \right] \\
=& \min_{\pi'} E_{x \sim D, y \sim \pi'} \left[ \ln \frac{\pi'(y|x)}{\pi(y|x)} - \ln \exp \left( \frac1{\beta} r(x, y) \right) \right] \\
=& \min_{\pi'} E_{x \sim D, y \sim \pi'} \left[ \ln \frac{\pi'(y|x)}{\pi(y|x) \exp \left( \frac1{\beta} r(x, y) \right)} \right] \\
=& \min_{\pi'} E_{x \sim D, y \sim \pi'} \left[ \ln \frac{\pi'(y|x)}{\pi(y|x) \exp \left( \frac1{\beta} r(x, y) \right) \frac1{Z(x)} Z(x)} \right] \\
=& \min_{\pi'} E_{x \sim D, y \sim \pi'} \left[ \ln \frac{\pi'(y|x)}{\frac1{Z(x)} \pi(y|x) \exp \left( \frac1{\beta} r(x, y) \right)} - \ln Z(x) \right] \\
\end{aligned}
$$

在这里的 $Z(x)$ 是一个归一化常数，定义为：

$$
Z(x) = \int \pi(y|x) \exp \left( \frac1{\beta} r(x, y) \right) dy
$$

目的是将分母同样转化为一个概率分布的形式，从而使得训练目标可以看作是一个 KL 散度的形式。

故对于此时的分母，展开得

$$
\begin{aligned}
& \frac1{Z(x)} \pi(y|x) \exp \left( \frac1{\beta} r(x, y) \right) \\
=& \frac{\pi(y|x) \exp \left( \frac1{\beta} r(x, y) \right)}{\int \pi(y|x) \exp \left( \frac1{\beta} r(x, y) \right) dy}
\end{aligned}
$$

对于分母而言，其分子是在给定输入 $x$ 的情况下，生成内容 $y$ 的概率分布乘以一个与奖励函数相关的权重。分母则是对所有可能的生成内容 $y$ 进行积分，从而整个分母可以看作是一个新的概率分布，表示在给定输入 $x$ 的情况下，生成内容 $y$ 的概率分布经过奖励函数调整后的分布。我们可将其记为 $\pi^*(y|x)$。

从而训练目标可以看作是最小化 $\pi'(y|x)$ 和 $\pi^*(y|x)$ 之间的 KL 散度：

$$
\begin{aligned}
& \min_{\pi'} E_{x \sim D, y \sim \pi'} \left[ \ln \frac{\pi'(y|x)}{\pi^*(y|x)} - \ln Z(x) \right] \\
=& \min_{\pi'} E_{x \sim D} \left[ D_{KL}(\pi'(y|x) || \pi^*(y|x)) - \ln Z(x) \right]
\end{aligned}
$$

这里的 $\ln Z(x)$ 由于不依赖于 $\pi'$，因此在优化过程中可以被忽略掉。最终的训练目标可以简化为：

$$
\min_{\pi'} E_{x \sim D} \left[ D_{KL}(\pi'(y|x) || \pi^*(y|x)) \right]
$$

由于 KL 散度的非负性，我们可以得出上述优化问题的最优情况为 KL 散度为零，即 $\pi'(y|x) = \pi^*(y|x)$。因此，DPO 的训练目标可以看作是让训练模型 $\pi'(y|x)$ 尽可能地接近一个由基准模型 $\pi(y|x)$ 和奖励函数 $r(x, y)$ 共同定义的目标分布 $\pi^*(y|x)$。

关于新的目标分布 $\pi^*(y|x)$，我们可以变换得到其 Reward 的评估结果：

$$
\begin{aligned}
& \pi'(y|x) = \frac1{Z(x)} \pi(y|x) \exp \left( \frac1{\beta} r(x, y) \right) \\
\implies & \exp \left(\frac1{\beta} r(x, y) \right) = \frac{\pi'(y|x)}{\pi(y|x)} Z(x) \\
\implies & r(x, y) = \beta \ln \frac{\pi'(y|x)}{\pi(y|x)} + \beta \ln Z(x)
\end{aligned}
$$

由上文的 Bradley-Terry 模型，我们期望将 $\pi^*(y|x)$ 与 $\pi(y|x)$ 进行比较，从而得到以下 LOSS 函数：

$$
\begin{aligned}
L(\theta) &= - \ln \sigma(r(x, y_A) - r(x, y_B)) \\
&= - \ln \sigma\left( \beta \ln \frac{\pi'(y_A|x)}{\pi(y_A|x)} + \beta \ln Z(x) - \beta \ln \frac{\pi'(y_B|x)}{\pi(y_B|x)} - \beta \ln Z(x) \right) \\
&= - \ln \sigma\left( \beta \ln \frac{\pi'(y_A|x)}{\pi(y_A|x)} - \beta \ln \frac{\pi'(y_B|x)}{\pi(y_B|x)} \right) \\
\end{aligned}
$$

### DPO 的优缺点

#### 优点

- 训练成本低、难度低: 不需要奖励模型、价值模型，训练成本和难度相比于 PPO 算法显著降低，没有复杂的数据采样过程。

- 易于实现：损失函数，训练流程都非常简单，实际实现起来，操作起来非常简单，训练相比 PPO 算法能够更快的收敛，因为它本质是一个二分类模型。

#### 缺点

- 需要大量高精二元偏好数据：因此收集偏好数据较难，不过现在借助开源 LLM 可以生成很高质量的偏好数据。

- 只适用于简单偏好任务：由于高度依赖偏好数据，无法适用在相对复杂一些的推理场景，即虽然相对PPO算法简单、训练成本低，但是没有 PPO 算法的能力强。
