---
title: 从零开始的 Post Training 5：OPSD
published: 2026-07-01
description: "Post Training 的入门学习笔记 OPSD 篇"
image: ""
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

# 从零开始的 Post Training 5：OPSD

## OPSD

上文中提及，对于 OPD 的变种，我们可以围绕两条设计轴进行探索：

- 散度轴：如何找到合适的散度函数来衡量学生模型与教师模型之间的差异。

- 教师模型轴：如何设计教师模型以更好地指导学生模型的学习。

本文介绍的 OPSD（On-Policy Self-Distillation，在策略自蒸馏）就是沿着第二条轴展开的探索：不再依赖一个独立的外部强教师，而是让同一个基础模型在不同上下文下分别扮演学生和教师。

这里的“在策略（on-policy）”表示训练所使用的回答由当前学生模型生成；“自蒸馏（self-distillation）”表示教师和学生来自同一个基础模型。OPSD 的主要目标仍然是蒸馏，即让学生的输出分布接近教师，而不一定要显式设计奖励函数。

### 从信息对称到信息不对称

前文介绍的典型 OPD 通常采用“信息相同、能力不同”的设定：学生模型和教师模型看到相同的问题与生成前缀，但教师模型本身更强。这种方式能够提供稠密的逐 token 监督，不过也存在一些限制：

1. 训练期间需要额外运行教师模型，计算和部署成本较高。

2. 教师和学生的模型家族、推理方式或输出风格可能不兼容，蒸馏效果并不总是稳定。

3. 在持续学习、领域适配等场景中，我们未必能够获得一个同时熟悉目标知识、又适合当前学生模型的外部教师。

OPSD 改变的不是模型能力，而是模型能够看到的信息。教师分支额外获得一段学生看不到的特权信息（Privileged Information），例如标准答案、参考推理过程或环境反馈。这里的“特权”只表示这些信息在训练时可用、在实际推理时不可用。

例如在一道数学题上，学生只看到题目；教师看到题目和参考解答。学生先写出自己的回答，然后教师沿着这条回答逐步判断：在每个位置上，哪些候选 token 更合理。

在获得参考解答后，同一个模型通常更容易判断或补全合理的推理步骤，因此可以为没有参考解答的学生分支提供监督。由于二者来自同一个基础模型，OPSD 不再依赖一个独立的更大教师模型；不过训练时仍需额外运行教师分支的前向计算。这种设计也不保证教师分布一定与学生接近，更不意味着训练不再依赖高质量参考数据。

这种“训练时读取上下文、推理时不再读取上下文”的形式，也可以用于研究如何把上下文中的知识或行为内化到模型参数中。不过，是否能够成功内化仍然取决于任务、参考信息质量和模型自身能力，并不是 OPSD 自动保证的结果。

### OPSD 中的学生与教师

给定问题 $x$ 和参考解答 $y^*$，学生策略和教师策略可以写成：

$$
p_S(\cdot\mid x) := p_\theta(\cdot\mid x),
\qquad
p_T(\cdot\mid x,y^*) := p_{\bar\theta}(\cdot\mid x,y^*)
$$

这里的 $y^*$ 可以只是正确答案，也可以是包含完整推理过程的参考解答。$\bar\theta$ 表示教师分支使用的参数。教师和学生从同一个基础模型出发，但教师额外看到 $y^*$。

在概念上，教师可以由当前模型构造；论文的主要实验则采用了更稳定的实现：冻结训练开始时的模型作为教师，只使用 LoRA 更新学生。LoRA 是一种低秩参数高效微调方法，只训练少量新增参数，而不全面更新基础模型权重。这样教师目标不会随着学生的每次更新同时变化。

### OPSD 中的散度

学生先生成回答 $\hat y$。在回答的第 $n$ 个位置，教师和学生都读取相同的学生前缀 $\hat y_{<n}$，但只有教师能够额外看到参考解答：

$$
p_{S,n}(v)=p_\theta(v\mid x,\hat y_{<n}),
\qquad
p_{T,n}(v)=p_{\bar\theta}(v\mid x,y^*,\hat y_{<n}),
\quad v\in\mathcal V
$$

其中 $\mathcal V$ 表示模型的整个词表，$v$ 表示其中一个候选 token。OPSD 可以使用正向 KL（Forward KL）、反向 KL（Reverse KL）或 Jensen-Shannon Divergence（JSD）比较这两个分布。

论文 v3 的实验发现，正向 KL 的效果最好，因此主要实验采用：

$$
D_{KL}(p_{T,n}\parallel p_{S,n})
=
\sum_{v\in\mathcal V}
p_{T,n}(v)
\ln\frac{p_{T,n}(v)}{p_{S,n}(v)}
$$

正向 KL 会重点惩罚这样一种情况：教师认为某个 token 很有可能，但学生给它的概率很低。直观地说，它鼓励学生覆盖教师认为合理的下一步选择。

论文也比较了 JSD。它先构造教师和学生的混合分布 $m$：

$$
\operatorname{JSD}_\beta(p_T\parallel p_S)
=
\beta D_{KL}(p_T\parallel m)
+
(1-\beta)D_{KL}(p_S\parallel m),
\qquad
m=\beta p_T+(1-\beta)p_S
$$

这里 $\beta\in[0,1]$ 控制混合分布中教师和学生所占的权重。当 $\beta=0.5$ 时，两边权重相同，得到通常所说的对称 JSD。不过在论文的消融实验中，$\beta=0.5$ 的 JSD 不如正向 KL，因此不应将 JSD 理解为 OPSD 唯一或默认的选择。

### 全词表 logit 蒸馏

论文主要采用全词表 logit 蒸馏（full-vocabulary logit distillation），即在每个位置上比较整个词表的概率，而不是只评价学生实际采样出的那个 token。Logit 是模型在 Softmax 之前为每个候选 token 给出的原始分数。

对于已经采样完成的学生回答，可以使用教师强制（teacher forcing）：把整段回答一次送入模型，并利用因果掩码保证第 $n$ 个位置只能读取 $\hat y_{<n}$。这里的“教师强制”是指训练时把已有 token 作为后续位置的输入，而不是让模型重新逐步生成。这样学生分支和教师分支各进行一次前向计算，就能分别得到所有位置的 logit，而不需要逐 token 重复调用模型。

全词表蒸馏能提供更完整的监督，但需要处理约 $O(BT\lvert\mathcal V\rvert)$ 规模的 logit，其中 $B$ 是批量大小，$T$ 是序列长度。LoRA 可以减少可训练参数、梯度和优化器状态的开销，但不会消除完整词表 logit 本身的显存成本。

### OPSD 的算法

OPSD 的一般损失函数为：

$$
L(\theta)
=
E_{(x,y^*)\sim\mathcal S}
\left[
E_{\hat y\sim p_S(\cdot\mid x)}
\left[
\frac{1}{\lvert\hat y\rvert}
\sum_{n=1}^{\lvert\hat y\rvert}
D\left(
p_T(\cdot\mid x,y^*,\hat y_{<n})
\parallel
p_S(\cdot\mid x,\hat y_{<n})
\right)
\right]
\right]
$$

其中 $\mathcal S$ 是问题与参考解答组成的数据集，$D$ 是选定的散度函数。论文主要实验令 $D$ 为正向 KL。

梯度只回传给学生模型。记方括号中的蒸馏损失为 $\ell_\theta(\hat y)$，则完整目标为 $J(\theta)=E_{\hat y\sim p_S(\cdot\mid x)}[\ell_\theta(\hat y)]$。由于 $\theta$ 既决定采样出什么回答，也出现在损失内部，用对数求导技巧展开完整梯度：

$$
\begin{aligned}
\nabla_\theta L(\theta)
&=
E_{(x,y^*)\sim\mathcal S}
\left[
\nabla_\theta
E_{\hat y\sim p_S(\cdot\mid x)}
\left[
\ell_\theta(\hat y)
\right]
\right] \\
&=
E_{(x,y^*)\sim\mathcal S}
\left[
E_{\hat y\sim p_S(\cdot\mid x)}
\left[
\ell_\theta(\hat y)\nabla_\theta\ln p_S(\hat y\mid x)
+
\nabla_\theta\ell_\theta(\hat y)
\right]
\right]
\end{aligned}
$$

外层的 $E_{(x,y^*)\sim\mathcal S}$ 与 $\theta$ 无关，可以直接放入。完整梯度包含两部分：轨迹梯度（第一项，来自 $\hat y$ 对 $\theta$ 的采样依赖）与蒸馏梯度（第二项，来自散度本身对 $\theta$ 的依赖）。实际训练把采样得到的回答 $\hat y$ 视为固定数据，即对采样过程停止梯度，只保留蒸馏梯度：

$$
\nabla_\theta L(\theta)
=
E_{(x,y^*)\sim\mathcal S}
\left[
E_{\hat y\sim p_S(\cdot\mid x)}
\left[
\frac{1}{\lvert\hat y\rvert}
\sum_{n=1}^{\lvert\hat y\rvert}
\nabla_\theta
D\left(
p_T(\cdot\mid x,y^*,\hat y_{<n})
\parallel
p_S(\cdot\mid x,\hat y_{<n})
\right)
\right]
\right]
$$

训练流程大致为：

1. 学生自采样：学生只读取问题 $x$，生成回答 $\hat y$。

2. 构造两份上下文：学生看到 $x$ 和自己的回答前缀；教师还会额外看到参考解答 $y^*$。

3. 计算两份分布：教师并不需要重新生成一条回答，而是沿着学生已经生成的轨迹，计算每个位置的下一 token 分布。

4. 逐位置计算散度：在每个位置比较教师和学生的词表分布，再对整段回答取平均。

5. 更新学生：把采样得到的离散回答视为固定训练数据，只对学生 logit 回传梯度；教师分支和采样过程都停止梯度。

这里的“停止梯度”表示教师只负责提供目标概率，训练程序不会根据教师分支的误差修改教师参数。下一轮训练再由更新后的学生重新采样，从而保持训练数据来自当前学生策略。

### 逐项散度裁剪

论文发现，少数风格 token，例如 `wait`、`think`、`therefore` 等推理连接词或自我反思词，可能产生远高于数学相关 token 的散度贡献。这样一来，训练信号容易被输出风格主导，而没有充分用于学习任务相关的推理内容。

为缓解这个问题，论文使用了逐项散度裁剪（Per-token Pointwise Divergence Clipping）：对每个位置、每个词表候选项的散度贡献分别设置上限。以主要实验采用的正向 KL 为例，位置 $n$ 上候选 token $v$ 的贡献为：

$$
\ell_{n,v}
=
p_{T,n}(v)
\ln\frac{p_{T,n}(v)}{p_{S,n}(v)}
$$

裁剪后的训练目标为：

$$
L_{\mathrm{clip}}
=
\frac{1}{\lvert\hat y\rvert}
\sum_{n=1}^{\lvert\hat y\rvert}
\sum_{v\in\mathcal V}
\min(\ell_{n,v},\tau)
$$

其中 $\tau$ 是裁剪阈值。它限制少数异常大的正向贡献，避免这些项支配整个更新。需要注意，单个 KL 项 $\ell_{n,v}$ 可能为负，因此裁剪后的目标主要是一个用于稳定训练的损失函数，不再需要把它理解为数学上严格非负的散度。

### 实验观察

论文在 Qwen3-1.7B、4B 和 8B 的数学推理任务上进行了实验，得到了一些值得注意的观察：

1. 在论文比较的三种散度中，正向 KL 的效果最好，反向 KL 和 $\beta=0.5$ 的 JSD 提升较小或不稳定。

2. 逐项裁剪能够避免少数风格 token 主导训练，在实验中明显改善了稳定性。

3. 全词表蒸馏在论文的对比实验中优于只评价已采样 token 的方法，但显存成本也更高。

4. 更长的学生回答不一定带来更好的效果。较后的 token 往往更容易根据已有前缀预测，也可能包含重复或噪声。

5. 在论文的特定配置下，OPSD 每个问题只采样一条较短回答，就取得了较高的 token 使用效率。不过这是特定模型、数据集和训练配置下的实验结果，不能直接理解为 OPSD 在所有任务上都一定优于 RL 或 SFT。

### OPSD 的适用条件与局限

OPSD 的关键假设是：模型在看到参考解答后，能够理解其中的信息，并对自己原本的推理轨迹给出更有帮助的下一 token 分布。如果问题远远超过模型的理解能力，即使提供正确答案，教师分支也未必能够产生有效监督。

此外，OPSD 仍然需要高质量的参考答案或推理过程。它消除的是对独立外部教师模型的依赖，但仍需教师分支的额外前向计算，也没有消除对监督信息的需求。教师和学生虽然来自同一个基础模型，但不同上下文仍可能导致明显的输出风格和分布差异。

原论文的实验主要集中在 1.7B 到 8B 模型和数学推理任务上。后续研究也发现，标准 OPSD 在部分长思维链模型或依赖样本专属参考信息的任务中可能表现不佳。因此，更稳妥的理解是：OPSD 提供了一种利用特权信息构造稠密监督的方法，但它的效果取决于教师构造、散度选择、模型能力和任务类型。

### 参考资料

- [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)
- [OPSD 官方代码](https://github.com/siyan-zhao/OPSD)
- [The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes](https://arxiv.org/abs/2605.11182)
- [Purified OPSD: On-Policy Self-Distillation Without Losing How to Think](https://arxiv.org/abs/2607.02234)
