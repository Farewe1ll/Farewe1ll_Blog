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

# 从零开始的 Post Training 4：OPD

## OPD

OPD（On-Policy Distillation，在策略蒸馏）是一种使用学生当前输出进行训练的知识蒸馏方法。这里的“在策略（on-policy）”是指：训练数据不是预先固定好的，而是由当前学生模型自己生成。学生先生成回答，教师再逐步评价学生的选择。这里的 token 是模型处理和生成文本的基本单位。

从数学上看，OPD 可以直接计算学生与教师输出分布之间的差异，也可以把教师的评价写成奖励，再用策略梯度进行训练。

### 关于 KL 散度

Post Training 中，我们常使用 KL 散度来衡量两个概率分布之间的差异。KL 散度不满足对称性和三角不等式，因此并不是严格意义上的距离。通过最小化 KL 散度，我们可以使模型的输出分布更接近目标分布；如果目标分布提供了更好的行为，这种分布对齐才有可能进一步转化为性能提升。

由于 KL 散度不对称，交换两个分布的位置会得到性质不同的正向 KL 散度和反向 KL 散度。

#### 正向 KL 散度

正向 KL 散度（Forward KL Divergence）定义为：

$$
D_{KL}(\pi_T \parallel \pi_\theta) = E_{z \sim \pi_T} \left[ \ln \frac{\pi_T(z)}{\pi_\theta(z)} \right]
$$

上式遍历的是目标分布 $\pi_T$ 的样本空间，因此它更关注目标分布中的高概率区域。如果在目标分布中存在一些高概率区域，而学生模型在这些区域的概率较低，那么正向 KL 散度会给予较大的惩罚。反之，如果学生模型在目标分布的低概率区域有较高的概率，那么正向 KL 散度的惩罚较小。

直观上，正向 KL 会强烈惩罚学生遗漏教师认为很有可能的答案。当教师分布中存在多种不同的合理答案，而学生能力有限、无法完整模仿教师时，学生通常会尝试兼顾这些答案，而不是只集中在其中一种。这种倾向称为模式覆盖（mode covering）。

SFT 使用的交叉熵，在训练数据固定时，等价于最小化从数据分布到模型分布的正向 KL，二者只相差一个不会影响模型参数优化的常数项。标准 KD 如果直接让学生模仿教师给出的概率分布，通常也采用从教师分布到学生分布的正向 KL。

#### 反向 KL 散度

反向 KL 散度（Reverse KL Divergence）定义为：

$$
D_{KL}(\pi_\theta \parallel \pi_T) = E_{z \sim \pi_\theta} \left[ \ln \frac{\pi_\theta(z)}{\pi_T(z)} \right]
$$

上式遍历的是学生模型分布 $\pi_\theta$ 的样本空间，因此它更关注学生模型中的高概率区域。如果学生模型在某些区域的概率较高，而目标分布在这些区域的概率较低，那么反向 KL 散度会给予较大的惩罚。反之，如果学生模型在目标分布的高概率区域有较低的概率，那么反向 KL 散度的惩罚较小。

直观上，反向 KL 会强烈惩罚学生把概率放在教师认为不太可能的答案上，但不会直接惩罚学生遗漏其他合理答案。当教师分布中存在多种不同的合理答案，而学生能力有限时，学生更可能集中模仿其中少数几种。这种倾向称为模式寻找（mode seeking）。

需要注意，RL 并不天然等同于优化反向 KL。只有在特定形式下，例如同时使用任务奖励和参考模型 KL 约束时，才能构造奖励倾斜目标策略（reward-tilted target policy）：根据奖励重新调整参考模型对不同回答的偏好，得到一个新的目标分布。此时，RL 目标可以等价地解释为让当前模型通过反向 KL 向这个目标分布靠近。

### 从 KD 到 OPD

#### KD

后训练中，除了 SFT、RLHF 等方法外，还有一种方法叫做 KD（Knowledge Distillation），即知识蒸馏。KD 的目标是让学生模型尽可能地模仿教师模型的输出分布，从而提升学生模型的性能。

为了便于与 OPD 对比，下面讨论典型的离线、白盒 logit KD。其中“离线”表示训练数据预先固定；“白盒”表示训练程序能够读取教师内部输出；logit 是模型在 Softmax 之前为每个候选 token 给出的原始分数。它一般具有以下特点：

- 固定数据，即离策略（off-policy）：训练使用预先准备好的文本前缀，例如教师生成的回答或已有的监督数据，而不是让当前学生模型在训练过程中不断生成新回答。

- 正向 KL（FKL）：典型 KD 会最小化从教师分布到学生分布的正向 KL，即 $D_{KL}(\pi_T\parallel\pi_\theta)$，使学生尽量覆盖教师认为可能的各种选择。也有一些 KD 变体使用反向 KL，但正向 KL 更常见。

- 稠密监督（dense supervision）：如果能够读取教师的输出分数，学生在每个位置都可以看到教师对所有候选 token 的概率，而不只是看到教师最终选中了哪个 token。后者类似普通 SFT，只提供一个确定答案，包含的信息更少。

对于这里讨论的这种 KD，其典型优化目标为：

$$
L_{\mathrm{KD}}(\theta)
=
E_{(x,y) \sim D}
\left[
\frac{1}{\lvert y\rvert}
\sum_t
D_{KL}
\left(
\pi_T(\cdot\mid x,y_{<t})
\parallel
\pi_\theta(\cdot\mid x,y_{<t})
\right)
\right]
$$

#### OPD

上述离线 KD 始终使用固定的文本前缀训练，但学生在实际生成时必须继续处理自己已经生成的内容。如果学生在前面生成了一个训练数据中很少出现的错误，后续内容就可能一步步偏离正确方向。这种“训练时主要看到标准文本，推理时却要面对自己的输出”的差异称为暴露偏差（exposure bias）。

OPD 的核心改动是：让当前学生模型自己生成回答，再让教师评价这些回答中的每一步。这样，学生不仅能学习教师在标准答案附近会怎么选择，也能学习教师会如何处理学生自己生成的错误或异常前缀。

典型 OPD 具有如下特点：

- 在策略：训练回答由当前学生模型生成，因此会随着学生模型的更新而变化。

- 反向 KL（RKL）：本文讨论的典型 OPD 最小化从学生分布到教师分布的反向 KL。换句话说，当学生倾向于生成某个 token、但教师认为这个 token 不合适时，学生会受到较强的惩罚。更广义的 OPD 也可以使用正向 KL、JSD 等其他分布差异指标。

- 稠密监督：教师会评价学生生成过程中的每一步。它可以只评价学生实际选中的 token，也可以同时比较若干候选 token；比较的范围越大，反馈信息越完整。

具体来说，学生模型在给定输入 $x$ 后生成回答 $y^S \sim \pi_\theta(\cdot\mid x)$。在回答的每个位置 $t$，我们把学生此前已经生成的文本前缀 $y_{<t}^S$ 同时交给学生和教师，再比较它们对下一个 token 的预测分布。

标准目标：最小化序列级反向 KL 散度

$$
L_{\mathrm{OPD}}(\theta)
=
E_{x \sim D_x}
\left[
D_{KL}
\left(
\pi_\theta(\cdot\mid x)
\parallel
\pi_T(\cdot\mid x)
\right)
\right]
$$

利用自回归分解，序列级 RKL 可以精确分解为 token 级 RKL 之和：

$$
L_{\mathrm{OPD}}(\theta)
=
E_{x \sim D_x}
\left[
E_{y^S \sim \pi_\theta(\cdot\mid x)}
\left[
\sum_t
D_{KL}
\left(
\pi_\theta(\cdot\mid x,y_{<t}^S)
\parallel
\pi_T(\cdot\mid x,y_{<t}^S)
\right)
\right]
\right]
$$

这里的求和包含 EOS，即回答结束标记对应的位置。实际训练常常再除以回答中的有效 token 数，避免较长回答仅仅因为 token 更多而产生过大的损失。不过，经过这种长度平均后，训练使用的是一个实用的代理目标（surrogate objective），不再严格等于上面的序列级 RKL。

上述逐 token 比较要求教师和学生采用相同的词表，并且以相同方式切分文本，否则双方的“第一个 token”可能并不对应。OPD 在理论上并不强制教师和学生使用相同词表，但如果词表不同，就需要先建立 token 之间的对应关系，或者改为直接比较整段文本，不能原样使用上面的公式。

### PG OPD

前面的目标直接比较学生和教师的概率分布。另一种理解方式是：把教师对学生所选 token 的评价看作奖励，再使用策略梯度更新学生。下面把这种写法称为 PG OPD。

对于学生在位置 $n$ 采样的 token $y_n$，定义：

$$
d_n
=
\ln \pi_\theta(y_n\mid x,y_{<n})
-
\ln \pi_T(y_n\mid x,y_{<n})
$$

当 $y_n \sim \pi_\theta(\cdot\mid x,y_{<n})$ 时，$d_n$ 可以用一个采样 token 估计当前位置的 RKL。这里的“无偏”是指：如果在同一个固定前缀上反复采样并取平均，最终会得到真实的 token 级 RKL。将它取负，可以得到奖励：

$$
r_n
=
-d_n
=
\ln \pi_T(y_n\mid x,y_{<n})
-
\ln \pi_\theta(y_n\mid x,y_{<n})
$$

直观上，如果教师比学生更倾向于刚刚生成的 token，那么 $r_n$ 为正，学生会提高该 token 的概率；反之，学生会降低它的概率。

如果暂时把学生已经生成的前缀视为固定输入，只优化当前位置的选择，那么最大化该奖励的 REINFORCE 梯度为：

$$
\nabla_\theta J_{token}
=
E
\left[
\sum_n
\operatorname{sg}(r_n)
\nabla_\theta
\ln \pi_\theta(y_n\mid x,y_{<n})
\right]
$$

其中 $\operatorname{sg}(\cdot)$ 表示停止梯度（stop-gradient）：$r_n$ 只作为更新强度的权重，不对 $r_n$ 本身继续求导。实现时，也可以最小化下面这个等价的训练损失：

$$
L_{\mathrm{PG}}
=
-E
\left[
\sum_n
\operatorname{sg}(r_n)
\ln \pi_\theta(y_n\mid x,y_{<n})
\right]
$$

上式只考虑当前位置的直接影响，没有继续追踪“当前 token 改变后，后续文本也会随之改变”所产生的间接影响。若要得到完整回答对应的精确策略梯度，需要使用未来回报（reward-to-go），即从当前位置一直累加到回答结束的奖励：

$$
G_n=\sum_{t=n}^{T}r_t.
$$

实际的逐 token OPD 通常只保留当前位置的直接梯度。这样虽然忽略了一部分长期影响，但梯度波动更小，训练也更稳定。

### 三种常见的 OPD 实现

三种实现的核心区别是：教师在每一步评价多少个候选 token。只评价学生实际选中的 token 最便宜，但估计的波动较大；比较整个词表的信息最完整，但计算成本最高；Top-k 则介于两者之间。

对于学生已经生成的文本前缀 $x,y_{<t}^S$，记学生和教师对下一个 token 的预测分布为：

$$
p_t(v)=\pi_\theta(v\mid x,y_{<t}^S),
\qquad
q_t(v)=\pi_T(v\mid x,y_{<t}^S),
\qquad
v\in\mathcal V
$$

1. `Sampled-token OPD`：教师只评价学生在当前位置实际生成的一个 token：

$$
\ell_t^{\mathrm{sample}} = \ln p_t(\hat y_t) - \ln q_t(\hat y_t)
$$

在同一个前缀下反复采样并对 $\ell_t^{\mathrm{sample}}$ 取平均，可以得到真实的 token 级反向 KL，因此它是一个无偏估计。不过，单次采样的结果可能波动较大。训练时需要使用上一节的 REINFORCE / 停止梯度写法：$\ell_t^{\mathrm{sample}}$ 用来决定更新方向和强度，而不是把采样 token 当作固定的标准答案直接训练。

2. `Full-vocab OPD`：在每个位置比较整个词表中所有 token 的概率，因此不需要依靠单个采样 token 来估计当前位置的 KL，反馈更完整、波动也更小。代价是需要处理 $O(BTV)$ 规模的分数或概率，其中 $B$、$T$、$V$ 分别表示批量大小、序列长度和词表大小。分块计算可以降低峰值显存，但仍然需要遍历整个词表：

$$
\ell_t^{\mathrm{full}} = \sum_{\nu \in \mathcal V} p_t(\nu) \ln \frac{p_t(\nu)}{q_t(\nu)}
$$

这里的“精确”只表示：在当前前缀下，公式完整比较了整个词表。它仍然没有追踪当前 token 改变后续文本所产生的间接影响；若要计算完整的序列级策略梯度，仍需考虑上一节所说的未来奖励。

3. `Top-k OPD`：只比较学生认为最有可能的 $k$ 个候选 token。记这些 token 的集合为

$$
S_t=\operatorname{TopK}(p_t,k)
$$

由于只保留了 $k$ 个候选，需要把学生和教师在这些候选上的概率重新缩放，使它们各自的概率之和恢复为 1：

$$
\bar p_t^{(S_t)}(v)
=
\frac{p_t(v)}{\sum_{u\in S_t}p_t(u)},
\qquad
\bar q_t^{(S_t)}(v)
=
\frac{q_t(v)}{\sum_{u\in S_t}q_t(u)},
\qquad v\in S_t
$$

对应目标为：

$$
L_{\mathrm{top}\text{-}k}=E\left[\sum_t D_{KL}\left(\bar p_t^{(S_t)}\parallel \bar q_t^{(S_t)}\right)\right]
$$

该方法忽略了集合 $S_t$ 之外的 token，因此并不等于完整词表上的 KL，而是一种近似。它用一部分精度换取了更低的计算成本。

### 动态监测指标

除了观察最终任务得分，我们还可以从三个角度检查学生是否正在接近教师：双方看好的候选 token 是否相同、对共同候选 token 给出的概率是否接近，以及两者的确定程度是否相似。

记学生和教师在位置 $t$ 对下一个 token 的预测分布分别为 $p_t$ 和 $q_t$，它们各自最看好的 $k$ 个 token 分别组成集合 $S_\theta^t$ 和 $S_T^t$。

- 重叠率（Overlap Ratio）

$$
\mathcal M_{\mathrm{overlap}} = E_t\left[\frac{ \lvert S_{\theta}^t\cap S_{T}^t \rvert}{k}\right]
$$

它表示学生和教师共同看好的候选 token 占多少。越接近 1，说明双方关注的候选词越相似；但它只检查“候选是否相同”，不检查具体概率，所以不能单独证明两个分布已经一致。

- 重叠 token 优势（Overlap-Token Advantage）

首先定义两者的 Top-k 交集：

$$
I_t = S_\theta^t \cap S_T^t
$$

这个指标进一步检查：对于学生和教师都看好的 token，双方给出的概率是否接近。令 $\bar{\pi}_\theta^t$ 和 $\bar{\pi}_T^t$ 表示只保留交集 $I_t$ 后重新缩放得到的概率分布，则：

$$
\mathcal M_{\mathrm{adv}}
=
E_t
\left[
\frac{1}{\lvert I_t\rvert}
\sum_{\nu \in I_t}
\bar{\pi}_\theta^t(\nu)
\left(
\ln \bar{\pi}_T^t(\nu)
-
\ln \bar{\pi}_\theta^t(\nu)
\right)
\right]
$$

求和部分等于交集上归一化分布之间的负 RKL，因此位置 $t$ 对该指标的贡献为

$$
-\frac{1}{\lvert I_t\rvert}
D_{KL}\left(\bar{\pi}_\theta^t\parallel\bar{\pi}_T^t\right).
$$

在 $I_t\neq\varnothing$ 且 KL 有限时，该指标恒不大于 0；越接近 0，表示双方对共同候选 token 给出的概率越一致。由于公式还除以共同候选 token 的数量 $\lvert I_t\rvert$，比较不同位置或不同训练阶段时，也要同时观察交集大小。如果双方没有共同候选 token，即 $I_t=\varnothing$，则应跳过该位置，避免除以 0。

- 绝对熵差（Absolute Entropy Gap）

$$
\Delta H_t=\left\lvert H(p_t)-H(q_t)\right\rvert
$$

熵可以理解为模型对下一步选择的犹豫程度：熵越低，概率越集中，模型越确定；熵越高，说明多个候选都可能被选中。熵差距衡量学生和教师的确定程度是否相近。需要注意，熵差接近 0 只表示双方同样确定或同样犹豫，并不意味着它们看好的是同一批 token。

### OPD 变体的两条设计轴

在 OPD 的基础上，研究者们提出了许多变体方法，这些变体主要集中在两个设计轴上：

1. 教师分布的构造（教师从哪里来）：教师可以是一个固定的外部模型，也可以由学生模型参数的指数移动平均（EMA）得到。还可以让同一个模型同时扮演学生和教师，但给教师提供学生看不到的额外信息，例如参考答案或提示。对于只能通过 API 调用、无法读取内部输出的黑盒教师，前文介绍的逐 token RKL 形式至少需要 API 返回指定 token 的概率或等价评分。如果 API 只能生成文本，仍可用教师采样来估计某些正向 KL 或交叉熵目标，但不能原样计算这里的 RKL。评分规则只能判断整段回答好不好，不能直接给出 token 概率；如果只用规则打分，训练方法会更接近 RL，或者是 OPD 与 RL 的混合方法。

2. 分布差异目标的选择（如何比较学生与教师）：除了反向 KL，还可以使用正向 KL、Jensen-Shannon 散度（JSD）等指标。与 KL 不同，JSD 是对称且有界的。逐 token 熵（token-level entropy）衡量的是一个模型自身有多确定，而不是学生与教师之间的距离，因此更适合作为额外约束或监测指标。选择不同的比较方式，会影响学生是尽量覆盖教师的多种选择，还是更集中地学习少数选择，也会影响训练稳定性和计算成本。

上述两条轴描述的是 OPD 的目标设计。三种实现之间还需要进一步区分：`Sampled-token` 是固定前缀下 RKL 的随机估计，`Full-vocab` 完整计算当前前缀上的词表分布差异，而 `Top-k` 会丢弃集合外的概率质量，因此是有偏的代理目标，并不与完整词表目标严格等价。

### 参考资料

- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016)
- [KL for a KL: On-Policy Distillation with Control Variate Baseline](https://arxiv.org/abs/2605.07865)
