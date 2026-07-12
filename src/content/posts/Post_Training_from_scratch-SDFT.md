---
title: 从零开始的 Post Training 6：SDFT
published: 2026-07-12
description: 'Post Training 的入门学习笔记 OPD 篇'
image: ''
tags:
  - Deep_Learning
  - Post_Training
  - RL
  - RLHF
  - SDFT
category: Research
draft: false
lang: ""
series: Post Training
---

# 从零开始的 Post Training 6

## SDFT

基础模型在语言、视觉和机器人等领域展现了广泛的能力。然而，现有的 AI 系统在部署后通常保持静态。虽然可以通过检索增强生成（RAG）或提示工程（Prompting）在推理阶段调整模型行为，但模型参数并未更新，因此无法内化新技能或知识。

为了实现真正的智能，模型需要具备持续学习（Continual Learning）的能力：即在不断获取新技能和知识的同时，不降低（或仅轻微降低）已有的通用能力。

### SFT 的局限性

SFT 将演示数据视为静态的监督信号，其本质是离线策略（Off-Policy）学习。

SFT 面临的核心问题在于训练数据分布与模型推理时的诱导分布（Induced Distribution）不匹配。

-   训练时：模型学习的是专家轨迹上的行为 $(s_{expert}, a_{expert})$。
-   推理时：模型一旦产生微小的误差，状态分布就会偏离专家轨迹，进入 OOD 领域（Out-of-Distribution states）。由于缺乏纠正反馈，误差会迅速累积，导致复合误差（Compounding Errors）。

此外，在持续学习的场景下，SFT 倾向于过度拟合当前任务的数据分布，导致模型参数大幅偏离预训练状态，从而引发灾难性遗忘（Catastrophic Forgetting）——即模型在学习任务 B 后，在任务 A 上的性能急剧下降。

故而 SDFT 希望能够解决的痛点为：能在 On-Policy 的情况下，不依赖于某种显式的奖励函数，同时避免模型发生遗忘。

### 从 RL 到 IRL

#### RL

RL 的目标通常为：

$$
\pi^* = \arg\max_{\pi} E_{y \sim \pi(·|x)} [R(y, x)]
$$

即最大化策略 $\pi$ 在输入 $x$ 上生成输出 $y$ 的预期奖励 $R(y, x)$。

该优化问题的闭式解为 $\pi*$ 为一个 reward-tiltered distribution：

$$
\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(\frac{R(y, x)}{\beta})
$$

这里的 reward-tiltered distribution 本质上就是一个带非均匀分布的加权 softmax 分布。$\pi_{ref}(y|x)$ 作为一个先验分布，提供了生成输出的基本模式，而 $\exp(\frac{R(y, x)}{\beta})$ 则根据奖励函数对输出进行加权，使得高奖励的输出具有更高的概率。最后再进行归一化，使得 $\pi^*(y|x)$ 成为一个合法的概率分布。

但是这里存在一个问题，即我们无法直接计算 $\pi^*(y|x)$，因为在计算 $\pi^*(y|x)$ 时，需要计算一个难以计算的归一化常数 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(\frac{R(y, x)}{\beta})$，因为它涉及到对所有可能输出 $y$ 的求和。

所以 RL 常常参数化一个近似的策略 $\pi_\theta$ 来近似 $\pi^*$。我们选择 FKL 还是 RKL 来优化 $\pi_\theta$ 和 $\pi^*$ 之间的距离呢？倘若是 FKL，我们需要在 $\pi^*$ 上取期望，这就需要我们能够从 $\pi^*$ 中采样或者计算 $\pi^*(y|x)$ 的概率，就算如重要性采样等方法来近似计算，也会面临计算量过大的问题。相反，如果我们选择 RKL，我们需要在 $\pi_\theta$ 上取期望，这就比较容易实现。这正是之前一众 RL 方法的思想。

#### IRL

虽然 SDFT 在形式上表现为学生-教师蒸馏，通过推导可证明该过程在数学上等价于最大化一个隐式奖励函数的逆强化学习（Inverse RL, IRL）过程。

我们将上述的公式进行重排，可以反表示我们之前所纠结的奖励函数：

$$
R(y, x) = \beta \left[ \log \frac{\pi^*(y | x)}{\pi_{ref}(y | x)} \right] + C
$$

在此处的 $\pi^*$ 如上所言是未知的，而在 SDFT 中引入一个假设：上下文学习假设，即：

在给定专家演示 $c$，后，条件化后的模型 $\pi(\cdot | x, c)$ 可以近似认为是该任务的最佳策略 $\pi^*$。

在认为条件化的模型可以生成高质量响应和认为条件化的模型输出分布在散度意义上接近于基础模型、于信赖域中的条件下，该假设是合理的。

于是可以带入假设得到 SDFT 的隐式奖励：

$$
R(y, x, c) = \log \pi(y | x, c) - \log \pi_k(y | x)
$$

基于该隐式奖励，我们可以得到我们的目标优化梯度：

$$
\nabla_{\theta} J(\pi_{k}) = E_{y \sim \pi_{k}} \left[ \left( \log \frac{\pi(y|x, c)}{\pi_k(y|x)} \right) \nabla_\theta \log \pi_k(y|x) \right]
$$

另一方面，我们 RL/OPD 的优化目标如下

$$
\begin{aligned}
\theta^* &= \arg\min_{\theta} D_{KL}(\pi_\theta || \pi^*) \\
&= \arg\min_{\theta} E_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi^*(y|x)} \right]
\end{aligned}
$$

可以发现，最小化逆向 KL 的梯度与最大化上述隐式奖励的策略梯度在期望上是等价的。这确立了 SDFT 的理论基础：它是一个利用 ICL 构造隐式奖励模型的在线强化学习算法。

### 梯度分析器

我们讨论过通用的后训练的目标，

令：

$$
l_{\theta}(\hat{y}) = \frac{1}{\lvert \hat{y} \rvert} \sum_{n = 1}^{\lvert \hat{y} \rvert}{\left[ D(\pi_T(\cdot|x, c) \parallel \pi_\theta(\cdot|x) \right]}
$$

则我们的目标实际上为

$$
L(\theta) = E_{(x, c) \sim D} \left[ E_{\hat{y} \sim \pi_\theta(\cdot|x)} l_{\theta}(\hat{y}) \right]
$$

对于外层的期望，与 $\theta$ 无关，因此我们可以将梯度传递到内层的期望，即

$$
\nabla L(\theta) = E_{(x, c) \sim D} \left[ \nabla_\theta E_{\hat{y} \sim \pi_\theta(\cdot|x)} l_{\theta}(\hat{y}) \right]
$$

里面这个参数实际上分为两个部分：

$$
\nabla_\theta E_{\hat{y} \sim \pi_\theta(\cdot|x)} l_{\theta}(\hat{y}) = E_{\hat{y} \sim \pi_\theta(\cdot|x)} \left[ l_{\theta}(\hat{y}) \nabla_\theta \log \pi_\theta(\hat{y}|x) + \nabla_\theta l_{\theta}(\hat{y}) \right]
$$

前者即为轨迹梯度（Trajectory Gradient），后者即为蒸馏梯度（Distillation Gradient）。

这两个部分在实际的训练中常常有所取舍，用于控制训练的方向与效果：

- 轨迹梯度：在 REINFORCE 中作为优化目标，抛弃蒸馏梯度，即假设 $A_{t}$ 与 $\theta$ 无关。本质上是避免模型走到高 Loss 的状态，只压低或者抬高整条轨迹的概率，而不调整 next-token 的分布。虽然有探索能力，但是也容易收缩到少数的安全路径。

- 蒸馏梯度：在 OPD 中作为优化目标，抛弃轨迹梯度，本质上是直接调整 next-token 的分布，而不考虑整条轨迹的概率。

在上节 OPSD 中，我们的目标函数为：

$$
L(\theta) = \mathbb{E}_{(x, y^*) \sim D} \left[ \mathbb{E}_{\hat{y} \sim \pi_\theta(\cdot|x)} \frac{1}{\lvert \hat{y} \rvert} \sum_{n = 1}^{\lvert \hat{y} \rvert} \left[ JSD_{\beta}(\pi_T(\cdot|x) \parallel \pi_\theta(\cdot|x))(\hat{y}) \right] \right]
$$

这里优化的常常为蒸馏梯度。即在学生模型 rollout 轨迹的时候，不记录如何从 $\theta$ 中采样 $\hat{y}$，即：

$$
\nabla L(\theta) = \mathbb{E}_{(x, y^*) \sim D} \left[ \nabla_{\theta} \mathbb{E}_{\hat{y} \sim \pi_\theta(\cdot|x)} \frac{1}{\lvert \hat{y} \rvert} \sum_{n = 1}^{\lvert \hat{y} \rvert} \left[ JSD_{\beta}(\pi_T(\cdot|x) \parallel \pi_\theta(\cdot|x))(\hat{y}) \right] \right]
$$

具体来说，打掉轨迹梯度或者蒸馏梯度都需要进行 stop-gradiant。

这里还有梯度分析器和EMA没写完 累晕过去了起来再写吧

### OPSD 与 SDFT 对比

二者在相当程度上相似，但是也有诸多不同：

|      维度      |                OPSD                |               SDFT               |
|:--------------:|:----------------------------------:|:--------------------------------:|
|    额外信息    |           参考答案 $y^*$           |    专家示范（demonstration）     |
|    目标场景    |           单任务推理提升           |  持续学习（continual learning）  |
|    核心问题    | 怎么用参考答案做 self-distillation |    怎么学新技能时不遗忘旧技能    |
|    散度选择    |                JSD                 |            Reverse KL            |
|   loss 形式    |            纯蒸馏 loss             | 蒸馏 loss + 可选的 SFT loss 混合 |
|      裁剪      |    Per-token pointwise clipping    |                -                 |
| 教师模型的参数 |           固定为初始权重           |               EMA                |
|     全词表     |       full / top-k / sampled       |   analytic per-token estimator   |

但是有趣的是，SDFT 的代码实现中，给定的是 FKL（也不知道他们要怎么圆）