---
title: 从零开始的 Post Training 3：GRPO
published: 2026-06-21
description: "Post Training 的入门学习笔记 GRPO 篇"
image: ""
tags: ["Deep_Learning", "Post_Training", "RL", "RLHF", "GRPO"]
category: "Research"
draft: false
lang: ""
series: "Post Training"
---

# 从零开始的 Post Training 3：GRPO

## GRPO 算法

### 从 PPO 到 GRPO

PPO 通常训练一个价值模型（Critic）来估计状态价值，再据此计算优势。这个价值模型会增加显存和计算开销，而且语言模型往往只在回答结束时得到结果奖励，要准确估计每个 token 位置的价值并不容易。

GRPO（Group Relative Policy Optimization，组相对策略优化）保留 PPO 的概率比裁剪，但不再单独训练价值模型。它针对同一个问题采样一组回答，用组内平均奖励作为基线，判断每条回答相对同组回答表现得更好还是更差。需要注意，GRPO 省去的是价值模型，并没有省去奖励信号；奖励仍可来自规则验证器、奖励函数或奖励模型。

为统一符号，本文使用：

- 当前策略 $\pi_\theta$：正在更新的模型；
- 旧策略 $\pi_{\mathrm{old}}$：生成当前训练数据的模型；
- 参考策略 $\pi_{\mathrm{ref}}$：用于 KL 正则化的模型，通常在一个更新阶段内保持固定。

对问题 $q$，旧策略采样 $G$ 条回答 $\{o_i\}_{i=1}^G$。记第 $i$ 条回答的长度为 $T_i=\lvert o_i\rvert$，并令

$$
s_{i,t}=(q,o_{i,<t}),
\qquad
a_{i,t}=o_{i,t}.
$$

当前策略与旧策略在已采样 token 上的概率比为：

$$
\rho_{i,t}(\theta)
=
\frac{\pi_\theta(a_{i,t}\mid s_{i,t})}
{\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})}.
$$

GRPO 要最大化的目标可以写成：

$$
\begin{aligned}
J_{\mathrm{GRPO}}(\theta)
=
E_{\substack{
q\sim P(Q)\\
\{o_i\}_{i=1}^G\sim\pi_{\mathrm{old}}(\cdot\mid q)
}}
\Bigg[
\frac{1}{G}
\sum_{i=1}^G
\frac{1}{T_i}
\sum_{t=1}^{T_i}
\Big(&
\min\big(
\rho_{i,t}(\theta)\widehat A_{i,t},
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)\widehat A_{i,t}
\big)\\
&-\beta\widehat D_{\mathrm{KL},i,t}
\Big)
\Bigg].
\end{aligned}
$$

其中，$\widehat A_{i,t}$ 是组相对优势估计，$\epsilon$ 控制概率比裁剪范围，$\beta$ 控制当前策略偏离参考策略的惩罚强度。训练代码通常最小化相反数 $L_{\mathrm{GRPO}}=-J_{\mathrm{GRPO}}$。

如果一个小批量包含 $N$ 个问题，那么通常会得到 $N\times G$ 条回答。目标中的 $1/T_i$ 表示先对每条回答的有效 token 取平均，避免长回答仅仅因为 token 更多而占据更大的权重。和 PPO 一样，GRPO 仍是 PPO 风格的在策略方法：一批旧策略数据可以复用若干次，但策略更新后需要重新采样较新的回答。

### 组相对优势如何计算

在结果监督下，先为同一问题的每条完整回答计算奖励 $r_i$。奖励可以来自答案是否正确等规则，也可以来自奖励模型。记组内均值和标准差为：

$$
\bar r
=
\frac{1}{G}\sum_{i=1}^G r_i,
\qquad
s_r
=
\sqrt{\frac{1}{G}\sum_{i=1}^G(r_i-\bar r)^2}.
$$

优势估计为：

$$
\widehat A_{i,t}
=
\frac{r_i-\bar r}{s_r+\varepsilon}.
$$

$\varepsilon$ 是防止除以零的小常数。如果组内所有奖励完全相同，那么这些回答之间没有相对差异，实践中可以把优势置为 0 或跳过这一组。

在结果监督下，同一条回答中的所有 token 共用同一个 $\widehat A_{i,t}$。因此，GRPO 提供的是“同一问题下不同回答之间”的相对信号，并没有自动判断一条回答中究竟哪个 token 导致了成败。概率比裁剪和 KL 惩罚仍按 token 计算，但这不等同于获得了 token 级奖励归因。

### KL 惩罚

GRPO 中有两种不同的策略比较，不能混淆：

1. $\pi_\theta$ 与 $\pi_{\mathrm{old}}$ 的概率比用于 PPO 裁剪；
2. $\pi_\theta$ 与 $\pi_{\mathrm{ref}}$ 的 KL 项用于限制模型偏离参考策略。

DeepSeekMath 使用下面的单 token 估计量：

$$
\widehat D_{\mathrm{KL},i,t}
=
\frac{\pi_{\mathrm{ref}}(a_{i,t}\mid s_{i,t})}
{\pi_\theta(a_{i,t}\mid s_{i,t})}
-
\ln
\frac{\pi_{\mathrm{ref}}(a_{i,t}\mid s_{i,t})}
{\pi_\theta(a_{i,t}\mid s_{i,t})}
-1.
$$

令概率比为 $u>0$，则 $u-\ln u-1\geq 0$，所以这个单样本估计量恒非负。当动作确实由当前策略采样，即 $a_{i,t}\sim\pi_\theta(\cdot\mid s_{i,t})$ 时，对它取期望正好得到

$$
D_{KL}
\left(
\pi_\theta(\cdot\mid s_{i,t})
\parallel
\pi_{\mathrm{ref}}(\cdot\mid s_{i,t})
\right).
$$

GRPO 实际使用旧策略采样的回答。因此，在一轮更新开始、$\pi_\theta=\pi_{\mathrm{old}}$ 时，上述无偏关系成立；经过多次参数更新后，如果仍复用同一批回答，它就是一个依赖“当前策略与旧策略仍然接近”的实用近似，而不能无条件称为无偏估计。

### 结果监督与过程监督

结果监督（outcome supervision）只在整条回答结束后给出奖励，前文的组相对优势就属于这种情况。

过程监督（process supervision）则在若干推理步骤结束时给出奖励。过程奖励模型（Process Reward Model，PRM）评价的是一个推理步骤，而不是每个 token。把同一问题下各回答的步骤奖励汇总为集合 $\mathcal R$，可以先进行标准化：

$$
\widetilde r_{i,j}
=
\frac{r_{i,j}-\operatorname{mean}(\mathcal R)}
{\operatorname{std}(\mathcal R)+\varepsilon}.
$$

若第 $i$ 条回答的第 $j$ 个步骤结束于位置 $b_{i,j}$，那么位置 $t$ 的优势可以写成之后各步骤标准化奖励之和：

$$
\widehat A_{i,t}
=
\sum_{j:\,b_{i,j}\geq t}
\widetilde r_{i,j}.
$$

这样，较早的 token 会接收到后续多个步骤的反馈，比只使用最终结果更容易定位问题。不过，过程监督需要可靠的步骤划分和过程奖励，标注或训练成本通常也更高。

### GRPO 的优缺点

#### 优点

- 不需要单独训练价值模型，能够减少显存、计算和实现复杂度。
- 避免了价值模型估计误差这一潜在的不稳定来源。
- 同一问题内直接比较多条回答，适合可验证答案或成对比较较可靠的任务。

#### 缺点

- 每个问题需要生成 $G$ 条回答，采样成本可能较高，组大小 $G$ 也需要调节。
- 优势依赖有限的组内样本。组较小、奖励稀疏或组内方差很小时，信号可能噪声较大或直接消失。
- 结果监督会把同一个优势分配给整条回答，无法直接完成细粒度的 token 级归因。
- 组内相对奖励只说明一条回答比同组回答更好，不保证它在绝对意义上已经足够好，因此方法仍高度依赖奖励设计与采样质量。

### 参考资料

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
