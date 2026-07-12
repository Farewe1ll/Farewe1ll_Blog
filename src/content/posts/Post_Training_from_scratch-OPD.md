---
title: 从零开始的 Post Training 4：OPD
published: 2026-06-22
description: 'Post Training 的入门学习笔记 OPD 篇'
image: ''
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
D_{KL}(\pi_T \parallel \pi_\theta) = E_{x \sim \pi_T} \left[ \log \frac{\pi_T(x)}{\pi_\theta(x)} \right]
$$

上式遍历的是目标分布 $\pi_T$ 的样本空间，因此它更关注目标分布中的高概率区域。如果在目标分布中存在一些高概率区域，而学生模型在这些区域的概率较低，那么正向 KL 散度会给予较大的惩罚。反之，如果学生模型在目标分布的低概率区域有较高的概率，那么正向 KL 散度的惩罚较小。

直观感受，可以感性理解为当我们期望正向 KL 散度最小时，变化的主动权在目标分布 $\pi_T$ 上，因此正向 KL 散度更倾向于覆盖目标分布的所有模式。FKL 又被称为 Mode-covering。

实际上 SFT、标准 KD 中优化的就是正向 KL 散度。

#### 反向 KL 散度

反向 KL 散度（Reverse KL Divergence）定义为：

$$
D_{KL} = E_{x \sim \pi_\theta} \left[ \log \frac{\pi_\theta(x)}{\pi_T(x)} \right]
$$

上式遍历的是学生模型分布 $\pi_\theta$ 的样本空间，因此它更关注学生模型中的高概率区域。如果学生模型在某些区域的概率较高，而目标分布在这些区域的概率较低，那么反向 KL 散度会给予较大的惩罚。反之，如果学生模型在目标分布的高概率区域有较低的概率，那么反向 KL 散度的惩罚较小。

直观感受，可以感性理解为当我们期望反向 KL 散度最小时，变化的主动权在学生模型 $\pi_\theta$ 上，因此反向 KL 散度更倾向于寻找目标分布中的一个模式。RKL 又被称为 Mode-seeking。

实际上之前所讲的 RL 中优化的就是反向 KL 散度。

### 从 KD 到 OPD

#### KD

后训练中，除了 SFT、RLHF 等方法外，还有一种方法叫做 KD（Knowledge Distillation），即知识蒸馏。KD 的目标是让学生模型尽可能地模仿教师模型的输出分布，从而提升学生模型的性能。

KD 一般具有以下特点：

- Off-policy：KD 的训练数据是固定的，通常来自于教师模型的输出分布。

- FKL： KD 的优化目标是最小化学生模型分布与教师模型分布之间的正向 KL 散度。即本质是在做 mode-covering。目前也有部分 KD 变体会使用 RKL，但大多数情况下，KD 使用的是 FKL。

- 信号密度：KD 的训练信号常常是稠密的，在每个 token 处， 学生模型都可以获得教师模型的输出分布，从而进行优化。

本质上，KD 的优化目标是最小化以下 Loss 函数：

$$
L(\theta) = E_{x \sim D} \left[ D_{KL}\left( \pi_T(x) \parallel \pi_\theta(x; \theta) \right) \right]
$$

#### OPD

我们注意到，上述的 KD 方法是 Off-policy 的，即学生模型所能学习到的信息是有限的，同时优化目标为 FKL，无法充分利用学生模型自身的探索能力。回忆之前的 RLHF 方法，尽管模型所得到的信号密度是稀疏的，但 RLHF 的优化目标为 RKL，能够充分利用学生模型自身的探索能力。于是我们可以考虑一种结合 KD 和 RLHF 优点的方法，即 OPD。

我们希望 OPD 具有如下特点：

- On-policy：OPD 的训练数据是动态的，来自于学生模型自身的输出分布。

- RKL：OPD 的优化目标是最小化学生模型分布与教师模型分布之间的反向 KL 散度。即 OPD 在做 mode-seeking，充分利用学生模型自身的探索能力。

- 信号密度：OPD 的训练信号是稠密的，在每个 token 处，学生模型都可以获得教师模型的输出分布，从而进行高效的训练。

即，对于每次训练，我们让学生模型在给定输入 $x$ 的情况下，生成输出 $y^S \sim \pi_\theta(\cdot|x)$，然后逐 token 计算学生模型分布与教师模型分布之间的反向 KL 散度，最后对所有 token 的 Loss 进行求和，得到最终的优化目标：

标准目标：最小化序列级反向 KL 散度

$$
\mathcal{L}_{OPD}(\theta)=E_{x \sim D_x}\left[D_{KL}\left(\pi_\theta(\cdot|x)\parallel\pi_T(\cdot|x)\right)\right]
$$

token 级损失目标：

$$
\mathcal{L}_{OPD}(\theta)=E_{x \sim D_x}\left[E_{y^S \sim \pi_\theta(\cdot|x)}\left[\frac{1}{|y^S|}\sum_t D_{KL}\left(\pi_T(\cdot|x, y_{<t}^S)\parallel \pi_\theta(\cdot|x, y_{<t}^S)\right)\right]\right]
$$

则该过程即为 OPD 的核心训练流程。

显然，OPD 对于 teacher / student 的要求是，两者的词表必须一致。

### PG OPD

对于每个位置，我们可以计算 token 级的反向 KL 散度：

$$
r_{n} = \log \pi_T(y_n|x, y_{<n}) - \log \pi_\theta(y_n|x, y_{<n})
$$

对 $r_{n}$ 做 stop-gradient 后，我们可以将其作为一个隐式奖励，利用 REINFORCE 的策略梯度方法进行优化：

$$
L(\theta) = E_{y^S \sim \pi_\theta(\cdot|x)}\left[\frac{1}{|y^S|}\sum_n r_n \nabla_\theta \log \pi_\theta(y_n|x, y_{<n})\right]
$$

### 三种常见的 OPD 实现

1. `Sampled-token OPD` 仅对学生采样的 1 个 token 计算损失：

$$
l_t^{sample} = \log p_t(\hat{y}_t) - \log q_t(\hat{y}_t)
$$

进行一次无偏估计 token 级反向 KL。

2. `Full-vocab OPD` 整张词表计算 KL，最准确但容易导致显存爆炸 $O(BTV)$：

$$
l_t^{full} = \sum_{\nu \in \mathcal{V}} p_t(\nu) \log \frac{p_t(\nu)}{q_t(\nu)}
$$

3. `Top-k OPD` 仅在学生 Top-k 高概率 token 集合 $S_t$ 上计算：

$$
L^{top-k}=\mathbb{E}\left[\sum_t D_{KL}\left(\bar{p}_t^{(S_t)}\parallel \bar{q}_t^{(S_t)}\right)\right]
$$

可以平衡效率与精度。

### 动态监测指标

以下三个指标可以用于监测 OPD 训练过程中的动态变化，帮助我们更好地理解学生模型与教师模型之间的对齐情况。

-   重叠率 Overlap Ratio

$$
\mathcal{M}_{overlap} = \mathbb{E}_t\left[\frac{ \lvert S_{\theta}^t\cap S_{T}^t \rvert}{k}\right]
$$

即在学生模型的 Top-k 集合与在教师模型的 Top-k 集合的交集占 k 的比例。越接近 1 对齐越好。

-   重叠 Token 优势 Overlap-Token Advantage

$$
\mathcal{M}_{adv}=\mathbb{E}_t\left[\frac{1}{\lvert \cap \rvert}\sum_{\nu\in\cap} \bar{\pi}_{\theta}^t(\nu)(\log\bar{\pi}_{T}^t(\nu)-\log\bar{\pi}_{\theta}^t(\nu))\right]
$$

即在学生模型的 Top-k 集合与在教师模型的 Top-k 集合的交集内，计算 RKL。越接近 0 对齐越好。

-   熵差距 Absolute Entropy Gap

$$
\Delta H_t=\lvert H(\pi_{\theta})-H(\pi_{T}) \rvert
$$

即衡量学生模型与教师模型的置信度与多样性差异。

### OPD 变体的两条设计轴

在 OPD 的基础上，研究者们提出了许多变体方法，这些变体主要集中在两个设计轴上：

1. 教师模型的选择：教师模型可以是固定的、可以是 EMA 的、可以是同一个模型加 prompt 偏置的、可以是黑盒 API、可以是 rubric。而教师模型的选择会直接影响到学生模型的学习效果和泛化能力。因此，如何选择合适的教师模型是 OPD 研究中的一个重要问题。

2. 散度的选择：散度可以是 reverse KL、forward KL、JS、token-level entropy 等。不同的散度选择会影响学生模型的学习策略和优化目标。因此，如何选择合适的散度是 OPD 研究中的另一个重要问题。