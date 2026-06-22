---
title: 从零开始的 Post Training 4——OPD
published: 2026-06-22
description: Post Training 的入门学习笔记
image: ""
tags:
  - Deep_Learning
  - Post_Training
  - RL
  - RLHF
  - OPD
category: Research
draft: false
lang: ""
series: Post Training
---

# 从零开始的 Post Training 4

## OPD

OPD（On Policy Distillation）是一种面向强化学习的在线策略蒸馏方法。

### 关于 KL 散度

Post Training 中，我们常使用 KL 散度来衡量两个概率分布之间的差异。KL 散度也可以被视作衡量目标分布与当前模型分布之间的距离。通过最小化 KL 散度，我们可以使模型的输出分布更接近目标分布，从而提升模型的性能。

但是，KL 散度并不是一个对称的距离度量，这就引出了两种不同的 KL 散度：正向 KL 散度和反向 KL 散度。

#### 正向 KL 散度

正向 KL 散度（Forward KL Divergence）定义为：

$$
D_{KL}(P_T || P_S) = E_{x \sim P_T} \left[ \log \frac{P_T(x)}{P_S(x)} \right]
$$

上式遍历的是目标分布 $P_T$ 的样本空间，因此它更关注目标分布中的高概率区域。如果在目标分布中存在一些高概率区域，而学生模型在这些区域的概率较低，那么正向 KL 散度会给予较大的惩罚。反之，如果学生模型在目标分布的低概率区域有较高的概率，那么正向 KL 散度的惩罚较小。

直观感受，可以感性理解为当我们期望正向 KL 散度最小时，变化的主动权在目标分布 $P_T$ 上，因此正向 KL 散度更倾向于覆盖目标分布的所有模式。FKL 又被称为 Mode-covering。

实际上 SFT 中优化的就是正向 KL 散度，因此 SFT 更倾向于覆盖目标分布的所有模式。

#### 反向 KL 散度

反向 KL 散度（Reverse KL Divergence）定义为：

$$
D_{KL} = E_{x \sim P_S} \left[ \log \frac{P_S(x)}{P_T(x)} \right]
$$

上式遍历的是学生模型分布 $P_S$ 的样本空间，因此它更关注学生模型中的高概率区域。如果学生模型在某些区域的概率较高，而目标分布在这些区域的概率较低，那么反向 KL 散度会给予较大的惩罚。反之，如果学生模型在目标分布的高概率区域有较低的概率，那么反向 KL 散度的惩罚较小。

直观感受，可以感性理解为当我们期望反向 KL 散度最小时，变化的主动权在学生模型 $P_S$ 上，因此反向 KL 散度更倾向于寻找目标分布中的一个模式。RKL 又被称为 Mode-seeking。

实际上之前所讲的 RL 中优化的就是反向 KL 散度。

### RL V.S OPD

RL 的目标通常为：

$$
\pi^* = \arg\max_{\pi} E_{y \sim \pi(·|x)} [R(y, x)]
$$

即最大化策略 $\pi$ 在输入 $x$ 上生成输出 $y$ 的预期奖励 $R(y, x)$。

但是在实践中，有时候会出现策略熵崩塌的情况出现，即策略 $\pi$ 过于集中在某些输出上，导致模型的多样性和泛化能力下降。之前在 PPO 的介绍中我们提到过，我们通常会在优化目标中加入一个熵正则项，其实就是为了防止策略熵崩塌的情况出现。

OPD 从另一个视角看待这个问题，定义目标分布 $\pi*$ 为一个 reward-tiltered distribution，即：

$$
\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(\frac{R(y, x)}{\beta})
$$

上式中，$\pi_{ref}$ 是一个参考策略，通常是一个预训练模型；$R(y, x)$ 是奖励函数；$\beta$ 是一个温度参数，用于控制奖励的影响程度。

这里的 reward-tiltered distribution 本质上就是一个带非均匀分布的加权 softmax 分布。$\pi_{ref}(y|x)$ 作为一个先验分布，提供了生成输出的基本模式，而 $\exp(\frac{R(y, x)}{\beta})$ 则根据奖励函数对输出进行加权，使得高奖励的输出具有更高的概率。最后再进行归一化，使得 $\pi^*(y|x)$ 成为一个合法的概率分布。

但是这里存在一个问题，即我们无法直接计算 $\pi^*(y|x)$，因为在计算 $\pi^*(y|x)$ 时，需要计算一个难以计算的归一化常数 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(\frac{R(y, x)}{\beta})$，因为它涉及到对所有可能输出 $y$ 的求和。

所以我们常常参数化一个近似的策略 $\pi_\theta$ 来近似 $\pi^*$。我们选择 FKL 还是 RKL 来优化 $\pi_\theta$ 和 $\pi^*$ 之间的距离呢？倘若是 FKL，我们需要在 $\pi^*$ 上取期望，这就需要我们能够从 $\pi^*$ 中采样或者计算 $\pi^*(y|x)$ 的概率，就算如重要性采样等方法来近似计算，也会面临计算量过大的问题。相反，如果我们选择 RKL，我们需要在 $\pi_\theta$ 上取期望，这就比较容易实现，因为我们可以直接从 $\pi_\theta$ 中采样，并且计算 $\pi_\theta(y|x)$ 的概率。这时候的优化目标可以写作：

$$
\begin{aligned}
\theta^* &= \arg\min_{\theta} D_{KL}(\pi_\theta || \pi^*) \\
&= \arg\min_{\theta} E_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi^*(y|x)} \right] \\
&= \arg\min_{\theta} E_{y \sim \pi_\theta} \left[ \log \pi_\theta(y|x) - \log \pi_{ref}(y|x) - \frac{R(y, x)}{\beta} \right] \\
&= \arg\max_{\theta} E_{y \sim \pi_{\theta}}[R(y, x)] - \beta * D_{KL}(\pi_\theta || \pi_{ref})
\end{aligned}
$$

### 散度度量方向的轴线

将上文中从 RKL 推导的优化目标，与之前在 RL 中提到的优化目标进行对比，我们可以有如下发现：

$$
\begin{aligned}
\text{RL:} \quad \theta^* &= \arg\max_{\theta} E_{y \sim \pi_{\theta}}[R(y, x)] - \beta * D_{KL}(\pi_\theta || \pi_{ref}) \\
\text{OPD:} \quad \theta^* &= \arg\min_{\theta} D_{KL}(\pi_\theta || \pi^*) \\
&= \arg\max_{\theta} E_{y \sim \pi_{\theta}}[R(y, x)] - \beta * D_{KL}(\pi_\theta || \pi_{ref})
\end{aligned}
$$

即，本质上，从 RKL 出发的 OPD 优化目标与 RL 的优化目标是相同的，都是最大化预期奖励 $E_{y \sim \pi_{\theta}}[R(y, x)]$，同时加入一个 KL 散度正则项 $D_{KL}(\pi_\theta || \pi_{ref})$ 来防止策略过于集中。这个策略的问题上文已经探讨，即可能会导致策略熵崩塌的情况出现。但是在 OPD 的理论中，对于 KL，我们实际上是有选择的余地的——可以考虑在 FKL 和 RKL 之间的光谱上进行选择，甚至可以考虑其他的散度度量。实际上，在这里 RL 可以被容纳入 OPD 的框架中，成为 OPD 的一个特例。

上述的散度度量方面的考察是 OPD 变体中一个重要轴线，而另一条轴线则是 Teacher 本身的更新。