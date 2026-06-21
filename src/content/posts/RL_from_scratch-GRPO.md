---
title: 从零开始的 RL 与 RLHF——GRPO
published: 2026-06-21
description: 'RL 与 RLHF 的入门学习笔记'
image: ''
tags: ['Deep Learning', 'RL', 'RLHF', 'GRPO']
category: 'Research'
draft: false
lang: ''
series: 'Reinforcement Learning'
---

# 从零开始的 RL 与 RLHF——GRPO

## GRPO 算法

### 从 PPO 到 GRPO

在 PPO 算法过程中，我们使用一个价值模型来估计状态的价值，从而计算优势函数 $A^\pi(s, a)$。然而，训练一个准确的价值模型可能会比较困难，并且有相当大的内存开销。GRPO（Group Relative Policy Optimization）算法认为，我们可以通过对于每个时间步的回报进行采样来直接估计优势函数，而不需要显式地训练一个价值模型。

回顾一下 PPO 算法的 LOSS 函数：

$$
L(\theta') = -\frac{1}{N} \sum_{i=1}^N \sum_t \min \left( \frac{\pi'(a_t|s_t)}{\pi(a_t|s_t)}, \text{clip}\left( \frac{\pi'(a_t|s_t)}{\pi(a_t|s_t)}, 1 - \epsilon, 1 + \epsilon \right) \right) A^{GAE}_\lambda(s_t, a_t)
$$

对于 GRPO 算法而言，我们不需要显式地训练一个价值模型来估计状态的价值，而是直接对于每个时间步的回报进行采样，从而得到优势函数的估计。

具体来说，GRPO 的 LOSS 函数为：

$$
\begin{aligned}
    \mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]}  \\
    & \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \right)  \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]\right\} ,
\end{aligned}
$$

以上 LOSS 函数为 Deepseek 论文中的 LOSS 函数。PPO 算法的 LOSS 函数中，我们先对 N 条轨迹进行采样，然后对于每条轨迹中的每个时间步累加优势函数的估计。而 GRPO 算法的 LOSS 函数中，有一个外层的期望，表示对于每个问题 $q$，我们从旧策略 $\pi_{\theta_{old}}$ 中采样 $G$ 条轨迹 $\{o_i\}_{i=1}^G$，然后对于每条轨迹中的每个时间步累加优势函数的估计。也就是说，我们一共存在 $N \times G$ 条轨迹，每条轨迹中存在 $|o_i|$ 个时间步。

如果将 GRPO 的 LOSS 函数修改为本文中 PPO 的 LOSS 函数的形式，则如下：

$$
\begin{aligned}
L(\theta) &= - \frac1N \sum_{j = 1}^N \left[ \frac{1}{G}\sum_{i=1}^G\frac{1}{\lvert s^i \rvert} \sum_t \left\{ \min \left[ \frac{\pi_\theta(a_t^i | s_t^i)}{\pi_{\theta_{old}}(a_t^i | s_t^i)} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(a_t^i | s_t^i)}{\pi_{\theta_{old}}(a_t^i | s_t^i)}, 1 - \epsilon, 1 + \epsilon \right)  \hat{A}_{i,t} \right] - \beta D_{KL}\left[\pi' || \pi \right]\right\} \right], \quad a_t^i \sim \pi(a_t^i|\theta)
\end{aligned}
$$

公式中的 $\lvert s^i \rvert$ 表示第 $i$ 条轨迹的长度，除去这一项可使得每条轨迹对于 LOSS 函数的贡献保持一致，避免因为轨迹长度不同而导致某些轨迹对于 LOSS 函数的贡献过大或过小。

严格来说，上述公式中的 $a_t^i$ 和 $s_t^i$ 应该分别替换为 $a_t^{i, j}$ 和 $s_t^{i, j}$，为了简化表达，我们省略了这些索引。

论文使用 Bregman 散度构造的无偏估计量来估算 $D_{KL}$：

$$
D_{KL}(\pi' || \pi) = \frac{\pi(a_t^i | s_t^i)}{\pi'(a_t^i | s_t^i)} - \ln \frac{\pi(a_t^i | s_t^i)}{\pi'(a_t^i | s_t^i)} - 1
$$

这个估计量是无偏的、恒非负，且方差较优。

### 关于 $\hat{A}_{i,t}$ 的计算

在 GRPO 算法中，我们直接对于每个时间步的回报进行采样来估计优势函数 $\hat{A}_{i,t}$，而不需要显式地训练一个价值模型来估计状态的价值。

具体的，对于每个问题 $q$，我们从旧策略 $\pi_{\theta_{old}}$ 中采样 $G$ 条轨迹 $\{a_i\}_{i=1}^G$，使用奖励模型来计算每条轨迹的回报 $R_i$，然后对于每个时间步 $t$，该时间步的优势函数 $\hat{A}_{i,t}$ 可以通过以下方式计算：

$$
\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}
$$

实际上，该 $\hat{A}_{i, t}$ 对于每个时间步 $t$ 来说都是相同的，因为我们直接使用了每条轨迹的回报 $R_i$ 来计算优势函数，而没有考虑时间步之间的差异。所以该项实际上可以提出求和符号之外（）（）（）

### 关于惩罚

注意到 PPO 算法中，每个时间步的回报函数其实有一项 KL 散度的惩罚项，即

$$
R_t = r_t - \beta D_{KL}(\pi' || \pi)
$$

这里的 $r_t$ 是奖励模型的输出，而 $D_{KL}(\pi' || \pi)$ 是新旧策略之间的 KL 散度。GRPO 算法中，我们直接将 KL 散度的惩罚项加入到 LOSS 函数中，而不是加入到回报函数中。

### 关于过程监督与结果监督

原论文中提到，GRPO 算法可以同时支持过程监督（Process Supervision）和结果监督（Outcome Supervision）。过程监督是指在训练过程中，我们对于每个时间步的回报进行监督，而结果监督是指在训练过程中，我们对于整个轨迹的回报进行监督。

显然我们上文中提到的 GRPO 的 LOSS 函数是结果监督的形式，因为我们直接使用了每条轨迹的回报 $R_i$ 来计算优势函数 $\hat{A}_{i,t}$。如果我们想要实现过程监督的形式，我们使用过程奖励模型（Process Reward Model, PRM）对每条回复的每个推理步骤进行打分，而不是对每个 token 打分。具体来说，对于第 $i$ 条回复中第 $j$ 个推理步骤（结束于 token index $index(j)$），PRM 给出该步骤的奖励 $r_i^{index(j)}$。随后将所有 $G$ 条回复的所有步骤奖励放在一起做全局归一化：

$$
\tilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}
$$

其中 $\mathbf{R}$ 是所有 $G$ 条回复中所有步骤奖励的集合。最终每个 token 的优势函数为之后所有步骤归一化奖励的累加：

$$
\hat{A}_{i,t} = \sum_{index(j) \geq t} \tilde{r}_i^{index(j)}
$$

### GRPO 的优缺点

#### 优点

- 不需要显式地训练一个价值模型来估计状态的价值，从而减少了内存开销和训练难度。

- 避免了由于价值模型估计不准确而导致的训练不稳定问题。

- 有更强的探索能力。

#### 缺点

- 计算复杂度高，有新的超参数。

- 高方差，训练不稳定。GRPO 在进行重要性采样计算时是在 token 级别上进行的，而每个 token 只是采样了一次，若某一个 token 过于偏离组内平均值或原始策略，那么仅通过一次采样很难做到校正分布，需要多次采样。因此仅仅通过一次 token 采样会在训练过程中引入高方差的噪声，进而可能造成模型的梯度估计不稳定