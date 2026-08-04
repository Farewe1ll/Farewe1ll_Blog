---
title: 了解 PEFT 方法：LoRA、DoRA、PiSSA 与 InfLoRA
published: 2026-08-01
description: PEFT 的学习笔记
image: ""
tags:
  - Deep_Learning
  - PEFT
  - LoRA
  - DoRA
  - PiSSA
  - InfLoRA
category: Research
draft: false
lang: ""
series: PEFT
---

# 了解 PEFT 方法：LoRA、DoRA、PiSSA 与 InfLoRA

PEFT，即 Parameter-Efficient Fine-Tuning，是一类只训练少量新增参数或选定参数、同时冻结大部分预训练参数的微调方法。它的目标是在尽量保持全参数微调效果的同时，降低训练与存储成本。

## LoRA（Low-Rank Adaptation）

LoRA 是一种通过引入低秩矩阵来减少可训练参数量的微调方法。它并不分解或直接更新原始权重矩阵，而是用两个较小矩阵的乘积表示权重更新 $\Delta W$。因此，LoRA 可以在冻结原始权重的同时，显著降低训练与存储成本。

我们可以思考：为什么微调语言模型的时候，我们常常不需要与模型参数量同等规模的训练数据，也可以让庞大的模型适应新任务并取得不错的效果呢？一种解释是，特定任务所需的参数更新具有较低的内在维度：虽然全参数空间很大，但有效的微调过程可能主要发生在一个低维子空间中。

基于以上观察，我们可以推测，在模型参数更新的时候，可以只学习一个低秩的更新量，而不是更新所有参数。LoRA 就是基于这个思想提出的，它通过低秩参数化来实现参数高效微调。

对于预训练权重矩阵 $W_0$，我们可以用一个低秩分解来表示其参数更新 $\Delta W$：

$$
\begin{aligned}
\Delta W &= \frac{\alpha}{r}BA, \\
W &= W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA.
\end{aligned}
$$

这里的 $\frac{\alpha}{r}$ 是缩放因子，$r$ 是低秩分支的秩。$\alpha$ 控制分支输出的整体幅度；在使用 Adam 且初始化方式不变时，调节 $\alpha$ 与调节学习率有一定的相似性，但两者并不完全等价。$r$ 还会改变低秩分支的表达能力和参数量，因此调节 $r$ 不能视为调节学习率。$\alpha$、$r$ 及二者比例也没有适用于所有模型与任务的固定最优值，需要结合具体实验选择。在实践中，常见的起始配置包括 $\alpha=r$ 或 $\alpha=2r$，即缩放因子取 $1$ 或 $2$；部分配置也会取到 $2$。这些取值属于经验选项，并不存在适用于所有模型与任务的固定最优比例。

若 $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$，则可以令 $A \in \mathbb{R}^{r \times d_{in}}$、$B \in \mathbb{R}^{d_{out} \times r}$。$A$ 可视作先把输入维度压到 $r$，$B$ 再把维度升到输出空间，从而以较小的参数量表达更新 $\Delta W$。

在训练过程中我们冻结原始权重 $W_0$，只训练低秩矩阵 $A$ 和 $B$。初始时 $A$ 随机初始化、$B$ 初始化为零矩阵，因此初始的 $\Delta W=0$。这样可以确保引入 LoRA 分支后，模型在训练起点与原始预训练模型具有相同的输出。

对于 $h = W_0 x$，前向传播过程变为：

$$
h = W_0 x + \Delta W x = W_0 x + (\frac{\alpha}{r}) B A x
$$

在实际应用中，LoRA 可以应用到 attention 的 $Q$、$K$、$V$、$O$ 等投影矩阵上。LoRA 原始论文在 GPT-3 的固定参数预算实验中发现，将参数预算分散到多个投影矩阵上，尤其是同时作用于 $Q$ 和 $V$，通常优于只作用于 $Q$ 或 $K$；但最佳目标模块仍取决于模型和任务。

秩 $r$ 的具体取值需要根据实际任务、模型和目标模块进行调整。$4$、$8$、$16$ 可以作为常见的起始候选值，但不是固定范围；在部分任务中很小的秩已经足够，而更复杂的任务可能需要更高的秩。

## DoRA（Weight-Decomposed Low-Rank Adaptation）

DoRA 是对 LoRA 的改进方法。核心思路是把权重分解成「方向」与「幅度」两部分，分别用不同的策略微调，从而在接近全参数微调的表达能力下保持参数高效。

在 LoRA 训练过程中，有时会出现随着 $r$ 的增大、模型性能并不继续提升，甚至反而下降的现象。DoRA 论文通过分析权重的幅度变化与方向变化发现，全参数微调中的两类变化通常呈现负相关趋势，而 LoRA 中则更容易呈现正相关趋势。结合梯度分析，作者据此认为 LoRA 同时学习幅度与方向时可能受到低秩参数化的限制，难以复现全参数微调的更新模式。DoRA 正是为了缩小这一差距而设计的。

一个自然的想法是：将参数分解为方向与幅度两个部分，然后分别微调。DoRA 将预训练权重分解为方向矩阵 $V$ 和幅度向量 $m$，对方向矩阵的更新做低秩分解，对幅度向量做全量微调，从而让幅度与方向都能独立、灵活地更新。具体而言：

$$
W_0 = m \odot \frac{V}{\|V\|_c}
$$

这里的 $\|\cdot\|_c$ 是对 $V$ 计算逐列范数，$m$ 是由各列幅度组成的向量，$\odot$ 表示逐列广播乘法。初始时，方向分量沿用 LoRA 的 $B=0$（保证 $\Delta V$ 从零开始），而 $m$ 初始化为 $\|W_0\|_c$，因此训练起点严格等于预训练权重。在训练时，则有：

$$
\begin{aligned}
W' &= m' \odot \frac{V'}{\|V'\|_c} \\
&= (m + \Delta m) \odot \frac{V + \Delta V}{\|V + \Delta V\|_c} \\
&= (m + \Delta m) \odot \frac{V + \frac{\alpha}{r}BA}{\|V + \frac{\alpha}{r}BA\|_c}.
\end{aligned}
$$

在训练中，$\|V + \Delta V\|_c$ 仍会在每次前向传播时动态计算，以保持权重归一化的精确形式。为了减少反向传播图带来的额外显存开销，DoRA 在反向传播时对这个列范数分支使用 stop gradient，不再通过分母计算梯度。这里并不是用 $\|V\|_c$ 近似 $\|V + \Delta V\|_c$，而是只截断归一化项的梯度。

相比 LoRA，DoRA 只额外引入一个与权重列数同量级的幅度向量，因此可训练参数量通常只会小幅增加。DoRA 论文的实验表明，它在多项任务上能够以较低的秩 $r$ 达到或超过更高秩 LoRA 的表现，并且对 $r$ 的选择相对更鲁棒；具体参数占比则取决于模型大小、目标模块和是否训练偏置等配置。

不过，DoRA 在训练期间需要动态计算列范数，因此会比标准 LoRA 增加一些计算和显存开销。它虽然不能直接按 $W_0+BA$ 的简单形式合并，但可以在训练完成后计算完整的 $W'$，再用它替换原始权重；合并后的推理不需要保留分解结构，也不会引入额外的推理时延。总体而言，DoRA 以很小的参数增量换取了更接近全参数微调的幅度与方向更新模式，其主要额外代价发生在训练阶段。

## PiSSA（Principal Singular Values and Singular Vectors Adaptation）

LoRA 假设任务所需的参数更新 $\Delta W$ 可以用低秩形式表达；PiSSA 则进一步关注预训练权重本身的谱结构。当较大的前 $r$ 个奇异值占据主要部分时，可以认为权重中的主要信息集中在对应的奇异方向上。PiSSA 的核心思想是通过奇异值分解（SVD）提取这些主要成分，并将其作为可训练的低秩分支进行微调。

具体而言，对于预训练权重矩阵 $W_0$，我们可以进行奇异值分解：

$$
W_0 = U \Sigma V^T
$$

这里，$U$ 和 $V$ 的列分别构成正交归一基，$\Sigma$ 是对角矩阵，其对角线上的元素为奇异值。PiSSA 关注前 $r$ 个较大奇异值及其对应的奇异向量，由此形成一个截断的低秩近似：

$$
W_0 \approx U_r \Sigma_r V_r^T
$$

然后将左奇异向量 $U_r$ 和右奇异向量 $V_r$ 分别与奇异值的平方根相结合，令 $B=U_r\Sigma_r^{1/2}$、$A=\Sigma_r^{1/2}V_r^T$，得到 PiSSA 的表达：

$$
W_0 = U_r \Sigma_r^{1/2} \cdot \Sigma_r^{1/2} V_r^T + W_{res} = BA + W_{res}
$$

在训练时，我们更新 $B$ 和 $A$，可训练分支 $BA$ 的秩始终不超过 $r$，而由其余较小奇异值构成的 $W_{res}$ 保持冻结。PiSSA 论文将这个残差视为相对次要、噪声更强的高秩部分；这是一种由实验支持的建模解释，而不是说 $W_{res}$ 在所有任务中都完全无关。

对比 LoRA，PiSSA 的优势在于它直接利用了预训练权重的结构信息，而不是从零开始学习低秩表示。这使得 PiSSA 在微调过程中能够更有效地利用已有的知识，从而在参数效率和性能之间取得更好的平衡。

对比 DoRA，PiSSA 不需要在每次前向传播时动态计算列范数，其一次性 SVD 初始化完成后，训练形式与标准 LoRA 基本一致。

## InfLoRA（Interference-Free Low-Rank Adaptation）

在面对持续学习的要求时，朴素的 LoRA 并不能直接解决模型的灾难性遗忘问题。InfLoRA 面向基于预训练视觉 Transformer 的无样本类增量学习场景，通过预先设计降维矩阵来减少新旧任务之间的干扰，从而在持续学习过程中兼顾稳定性与可塑性。

具体而言，对于依次到来的学习任务，我们可以为第 $t$ 个任务拓展一个独立的 LoRA 分支 $A_tB_t$，其中 $B_t$ 需要预先设计。在学习的时候，预训练权重 $W_0$ 和其他分支权重全部冻结，只微调 $A_t$。需要注意的是，本节沿用 InfLoRA 论文的记号：$B_t$ 是降维矩阵、$A_t$ 是升维矩阵，与前文采用 $BA$ 时的字母角色相反。在完成第 $t$ 个任务的训练后，有效权重为：

$$
W = W_0 + \sum_{i=1}^{t} A_i B_i
$$

在 $B_t$ 的行向量正交归一时，仅微调 $A_t$ 对有效权重产生的更新 $\Delta_{A_t}W_t$，等价于将全权重更新 $\Delta W_t$ 投影到由 $B_t$ 的行向量张成的低秩子空间，即：

$$
\Delta_{A_t}W_t = \Delta A_t B_t = \Delta W_t B_t^T B_t.
$$

其中 $B_t^T B_t$ 是投影矩阵。这意味着，由 $A_t$ 的更新所引起的有效权重变化，始终位于 $B_t$ 的行向量张成的低秩子空间中。

由此不难注意到，以上分析需要对 $B_t$ 进行设计。首先，$B_t$ 的行向量需要正交归一，以保证 $B_t^T B_t$ 是一个正交投影矩阵。

其次，$B_t$ 张成的子空间 $\mathcal{S}_t = \operatorname{span}(B_t)$ 还需要满足两个关键特性：

1. 稳定性，即 $\mathcal{S}_t \subseteq \mathcal{M}_t^{\perp}$，也就是新任务更新避开旧任务梯度所占据的子空间 $\mathcal{M}_t$。

2. 可塑性，即 $\mathcal{S}_t \subseteq \mathcal{N}_t$，也就是更新方向从新任务的梯度子空间 $\mathcal{N}_t$ 中选取，以保留对新任务有用的方向。

故有：

$$
\mathcal{S}_t \subseteq \mathcal{N}_t \cap \mathcal{M}_t^{\perp}.
$$

以下给出简要的构造 $B_t$ 的方法：

1. 近似 $\mathcal{N}_t$ 和 $\mathcal{M}_t$：

    - 新任务 $t$ 的梯度子空间 $\mathcal{N}_t$ 可以用输入矩阵 $H_t$（即输入激活 $h$ 的集合）的列空间 $\operatorname{span}(H_t)$ 近似。

    - 所有旧任务的梯度子空间 $\mathcal{M}_t$ 的正交补空间 $\mathcal{M}_t^{\perp}$ 可以采用 DualGPM 方法维护。

2. 计算候选分量 $\hat{H}_t$：将新任务输入矩阵 $H_t$ 投影到 $\mathcal{M}_t^{\perp}$ 上。若矩阵 $M_t^{\perp}$ 的列构成子空间 $\mathcal{M}_t^{\perp}$ 的正交基，则：

    $$
    \hat{H}_t = M_t^{\perp}(M_t^{\perp})^T H_t.
    $$

    $\hat{H}_t$ 保留了新任务输入中位于旧任务梯度子空间正交补内的分量，其列空间可用于近似构造满足要求的 $\mathcal{S}_t$。一般而言，这个投影空间不能直接等同于严格的集合交空间 $\mathcal{N}_t \cap \mathcal{M}_t^{\perp}$。

3. 确定 $B_t$：对 $\hat{H}_t^T$ 进行奇异值分解，并将右奇异向量矩阵记为 $U_t$：

    $$
    \hat{H}_t^T = V_t\Sigma_tU_t^T.
    $$

    其中，$U_t$ 的列是 $\hat{H}_t^T$ 的右奇异向量。取与前 $r$ 个最大奇异值对应的右奇异向量并转置，得到：

    $$
    B_t = \left(U_t[:,1:r]\right)^T.
    $$

    这样得到的 $B_t$ 以秩 $r$ 保留投影后输入中的主要方向，在近似避开旧任务梯度空间的同时，为当前任务保留较有代表性的更新方向。

同样地，InfLoRA 可以在每个任务训练完成后将 $A_tB_t$ 融合进当前权重，不必在推理时逐个保留和计算历史分支，因此不会因为分支数量随任务增加而持续增加推理开销。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](https://arxiv.org/abs/2404.02948)
- [InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning](https://arxiv.org/abs/2404.00228)
