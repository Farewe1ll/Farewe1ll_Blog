---
title: 从零开始的 Post Training 5：OPSD
published: 2026-07-01
description: 'Post Training 的入门学习笔记 OPSD 篇'
image: ''
tags:
  - Deep_Learning
  - Post_Training
  - RL
  - RLHF
  - OPSD
category: Research
draft: false
lang: ""
series: Post Training
---

# 从零开始的 Post Training 5

## OPSD

上文中提及，对于 OPD 的变种，我们可以围绕两条设计轴进行探索：

- 散度轴：如何找到合适的散度函数来衡量学生模型与教师模型之间的差异。

- 教师模型轴：如何设计教师模型以更好地指导学生模型的学习。

本文介绍的 OPSD（On Policy Self-Distillation）就是在 OPD 的基础上进行的探索。

### 非对称 OPD

前文中介绍过，OPD 兼具了 On-policy 的探索能力和高信息密度的监督信号。标准的 OPD 设定为 「信息对称，能力不对称」，即学生模型和教师模型的输入信息与上下文完全相同，但教师模型的能力更强。但是这个假设带来了三个问题：

1. 强依赖外部的教师模型，存在训练成本和数据合规性问题。

2. 教师模型和学生模型的分布本身差异较大，容易把学生模型推向 OOD 区域，遗忘通用的能力。

3. 在持续学习、领域适配、长上下文注入等场景中，"更强的教师模型"本身就难以获得。

于是产生了 OPD 的一个变种「非对称 OPD」：我们可以考虑打破"信息对称"的约束，让教师模型在推理时额外看到一段学生模型看不到的上下文 c——可以是 ground-truth 答案、参考 demonstration（专家示例）、甚至可以是 in-context 或一段领域知识。这样教师模型即便和学生模型是同一个模型，也能凭借这段额外信息产生显著更优的分布，从而充当蒸馏信号源。由此我们有三个直接收益：

1. 教师模型分布天然靠近学生模型的分布，OOD 遗忘大幅缓解。

2. 可以走 self-distillation 路线，彻底摆脱外部强模型依赖。

3. 天然适配 continual learning 与 context internalization 等"把上下文蒸进权重"的任务形态。

### OPSD 中的教师模型

基于上述的非对称 OPD 思路，OPSD 提出了一个新的教师模型设计：

$$
\pi_T(\cdot|x) := \pi_\theta(\cdot|x, y^*)
$$

我们给教师模型提供了 ground-truth 答案 $y^*$，而学生模型则只看到问题 $x$。

### OPSD 中的散度

在论文中，OPSD 采用的 D 为 Jensen-Shannon Divergence (JSD)：

$$
JSD_{\beta}(\pi_T \parallel \pi_{\theta}) = \beta D_{KL}(\pi_T \parallel m) + (1-\beta) D_{KL}(\pi_{\theta} \parallel m), m = \beta \pi_T + (1-\beta) \pi_{\theta}
$$

上式中的 $m$ 是 $\pi_T$ 和 $\pi_\theta$ 的混合分布，$\beta$ 是一个超参数，用于控制两者的权重。通常来说我们可以取 $\beta = 0.5$，使得两者的权重相等。

注意到我们这里的散度可以在整个词表上进行计算。因为我们可以采用因果掩码，即可以将多次的前传计算 logits 合并在一起做一次矩阵前传，利用 GPU 加速。在此基础上结合 LoRA，可以在相对较小的模型上实现 `Full-vocab OPD` 的训练。

### OPSD 的算法

从而我们可以得到 OPSD 的损失函数：

$$
L(\theta) = \mathbb{E}_{(x, y^*) \sim D} \left[ \mathbb{E}_{\hat{y} \sim \pi_\theta(\cdot|x)} \frac{1}{\lvert \hat{y} \rvert} \sum_{n = 1}^{\lvert \hat{y} \rvert} \left[ JSD_{\beta}(\pi_T(\cdot|x) \parallel \pi_\theta(\cdot|x))(\hat{y}) \right] \right]
$$

从而 OPSD 的训练流程大致为：

1. 学生模型自采样：从 $\pi_\theta(\cdot|x)$ 中采样生成回答 $\hat{y}$。

2. 生成两份输出分布：将 $\hat{y}$ 同时送入学生模型和教师模型，得到两份输出分布 $\pi_\theta(\cdot|x, \hat{y}_{<n})$ 和 $\pi_T(\cdot|x, y^*, \hat{y}_{<n})$。

3. 逐 token 计算散度：对每个位置 $n$，计算两份分布的散度。

4. 梯度回传：只将梯度回传给学生模型，教师模型作为固定的目标分布。

梯度即为：

$$
\nabla L(\theta) = E_{(x, y^*) \sim D} \left[ E_{\hat{y} \sim \pi_\theta(\cdot|x)} \left[ \nabla \frac{1}{|\hat{y}|} \sum_{n=1}^{|\hat{y}|} D(p_T(\cdot|x, y^*, \hat{y}_{<n}) || p_\theta(\cdot|x, \hat{y}_{<n})) \right] \right]
$$

### Per-token Pointwise Divergence Clipping

注意到，在训练中，style token 在散度中占比严重不均衡——少数风格性 token（标点、空格、连接词）的散度远大于“数学有意义的 token”（数字、运算符、变量名）。即模型极易变成”在学 Teacher 的语气”而不是”在学 Teacher 的推理”。

我们的解法是，在每个 token 的散度上加一个 clipping 操作：

$$
D_{clip}(\pi_{T} \parallel \pi_{\theta}) = \frac{1}{\lvert y \rvert} \sum_{n} \sum_{v \in V} \min (l_{n,v}, \tau)
$$

这里 $l_{n,v} = \pi_{\theta}(v) \log \frac{\pi_{\theta}(v)}{\pi_{T}(v)}$，即每个 token 的散度。把这一项 clip 掉过大的离群贡献，再求和。