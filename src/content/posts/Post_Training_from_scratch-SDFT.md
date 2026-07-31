---
title: 从零开始的 Post Training 6：SDFT
published: 2026-07-12
description: "Post Training 的入门学习笔记 SDFT 篇"
image: ""
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

# 从零开始的 Post Training 6：SDFT

## SDFT

基础模型可以通过提示、上下文学习或检索增强生成（Retrieval-Augmented Generation，RAG）在推理时临时使用新信息，但这些方法不会更新模型参数。若希望模型真正内化新技能或知识，就需要继续训练；而在持续学习（continual learning）场景中，我们还希望它学习新任务时尽量保留原有能力。

SDFT（Self-Distillation Fine-Tuning，自蒸馏微调）尝试解决的正是这个问题：只有专家示范、没有显式奖励函数时，怎样把示范转换成由当前模型自身状态产生的训练信号。它让同一个模型分别扮演学生和教师：学生只看到问题，教师额外看到专家示范，再沿着学生生成的回答提供逐 token 监督。

这里的 token 是模型处理文本的基本单位；“逐 token 监督”表示训练不只判断整段回答好不好，而是在回答的每个位置比较学生和教师对下一步的预测。

> **版本说明：** SDFT 的 arXiv v1 按反向 KL（RKL）描述算法，并给出逆强化学习解释；官方代码仓库后来说明，论文中全部实验结果实际上来自“由当前学生生成回答，即在策略采样（on-policy rollout）+ 逐 token 正向 KL（FKL）”。这里的 rollout 就是模型生成的一条回答或轨迹。这两种表述不能混为一谈，本文会分别介绍。

### 为什么标准监督微调（SFT）容易遇到问题

SFT 使用固定的专家回答训练模型。在模仿学习中，它可以被视为离策略（off-policy）训练：模型主要读取专家提供的文本前缀，而不是当前模型自己会生成的前缀。

对语言模型而言，一个“状态”可以简单理解为“问题与目前已经生成的文本前缀”。训练和推理之间可能出现如下差异：

- 训练时，模型主要学习专家轨迹上的下一步选择。
- 推理时，模型的早期错误会改变后续输入前缀，使状态分布逐渐偏离专家演示所覆盖的分布。由于标准 SFT 通常没有在这些由模型自身诱导的状态上提供纠正信号，后续误差可能不断累积，形成复合误差（compounding errors）。

这类训练与推理不一致也常被称为暴露偏差（exposure bias）。它并不意味着 SFT 一定失败，而是说明：模型一旦走到训练数据很少覆盖的前缀上，监督信号就可能不足。

持续学习还带来另一个问题。模型在较窄的新任务上进行 SFT 时，参数可能明显偏向当前数据，从而损害原有能力，这被称为灾难性遗忘（catastrophic forgetting）。SDFT 的目标是缓解而不是彻底消除遗忘：它让学生在自己真正会到达的状态上学习，同时利用来自原模型的教师分布约束更新方向。

### SDFT 中的学生与教师

设训练数据为问题与专家示范组成的集合：

$$
\mathcal D=\{(x,c)\}
$$

其中 $x$ 是当前问题，$c$ 是与该问题配对的示范。示范可以是参考回答、解题样例，也可以包含完成任务所需的知识文本。

学生参数记为 $\theta$，教师参数记为 $\phi$。学生先只根据问题生成回答：

$$
\hat y\sim\pi_\theta(\cdot\mid x)
$$

在回答的第 $n$ 个位置，两条分支都读取同一段学生前缀 $\hat y_{<n}$，但教师还会额外读取示范 $c$：

$$
p_{S,n}(v)=\pi_\theta(v\mid x,\hat y_{<n}),
\qquad
p_{T,n}(v)=\pi_\phi(v\mid x,c,\hat y_{<n}),
\quad v\in\mathcal V
$$

这里 $\mathcal V$ 是词表，$v$ 是一个候选 token。教师并不是重新生成另一条回答，而是沿着学生已经生成的前缀，判断下一步应该把概率放在哪里。

这种设计同时包含两层含义：

1. **自蒸馏（self-distillation）：** 教师和学生来自同一个基础模型，而不是依赖一个独立的更大模型。
2. **在策略（on-policy）：** 用来训练的文本前缀由当前学生生成，并会随着学生更新而变化。

### SDFT 的训练流程

一次训练迭代可以概括为：

1. 从数据集中取出问题 $x$ 和示范 $c$。
2. 当前学生只读取 $x$，采样一条回答 $\hat y$。
3. 将同一条学生回答分别送入学生上下文和教师上下文，计算每个位置的词表分布。
4. 在学生前缀上比较教师与学生的分布，并对各位置的损失取平均。
5. 只更新学生参数，再用学生的新参数更新 EMA 教师。

写成一般形式，可以令 $d$ 表示所选择的分布差异：

$$
L_D(\theta)
=
E_{(x,c)\sim\mathcal D}
\left[
E_{\hat y\sim\pi_\theta(\cdot\mid x)}
\left[
\frac{1}{\lvert\hat y\rvert}
\sum_{n=1}^{\lvert\hat y\rvert}
d\!\left(p_{S,n},p_{T,n}\right)
\right]
\right]
$$

需要特别区分两个彼此独立的问题：

- **在策略 / 离策略** 决定训练使用谁生成的前缀，也就是在哪些状态上学习。
- **FKL / RKL** 决定在一个已经固定的前缀上，怎样比较学生和教师的下一 token 分布。

因此，“使用正向 KL”并不会自动变成离策略；只要前缀来自当前学生，训练仍然可以是在策略的。

### 论文 v1 的 RKL / IRL 理论视角

论文 v1 使用反向 KL 描述 SDFT，并从 KL 约束强化学习给出一个逆强化学习（Inverse Reinforcement Learning，IRL）视角。需要先强调：本节只解释论文 v1 的 RKL 理论目标，不是已发布 FKL 实验所实际优化的目标。

#### 奖励倾斜目标分布

先考虑第 $k$ 轮的 KL 约束策略更新：

$$
\pi_{k+1}^{*}
=
\arg\max_\pi
\left\{
E_{y\sim\pi}[r(y,x)]
-
\beta D_{KL}
\left(
\pi(\cdot\mid x)
\parallel
\pi_k(\cdot\mid x)
\right)
\right\}
$$

第一项鼓励高奖励回答，第二项限制新策略不要离当前策略 $\pi_k$ 太远；$\beta$ 越大，这个限制越强。这个带 KL 正则的目标具有下面的奖励倾斜（reward-tilted）闭式解：

$$
\pi_{k+1}^{*}(y\mid x)
=
\frac{1}{Z(x)}
\pi_k(y\mid x)
\exp\!\left(\frac{r(y,x)}{\beta}\right)
$$

其中 $Z(x)$ 是归一化常数。“奖励倾斜”可以直观理解为：先保留当前策略的基本分布，再按奖励对不同回答重新加权。

不过这里存在一个问题：我们无法直接计算 $\pi_{k+1}^{*}(y\mid x)$，因为归一化常数

$$
Z(x) = \sum_{y'} \pi_k(y'\mid x) \exp\!\left(\frac{r(y',x)}{\beta}\right)
$$

需要对所有可能的回答求和，通常不可行。因此 RL 往往参数化一个近似的策略 $\pi_\theta$ 来逼近 $\pi_{k+1}^{*}$。那么应当用 FKL 还是 RKL 来度量 $\pi_\theta$ 与 $\pi_{k+1}^{*}$ 之间的距离？

- 若是 FKL，我们需要在 $\pi_{k+1}^{*}$ 上取期望，这就要求能够从 $\pi_{k+1}^{*}$ 中采样或计算其概率；即使借助重要性采样等近似方法，计算量仍然过大。
- 若是 RKL，我们只需要在 $\pi_\theta$ 上取期望，这是容易实现的。这正是此前诸多 RL 方法的核心思想。

反过来整理可得：

$$
r(y,x)
=
\beta
\left[
\ln\pi_{k+1}^{*}(y\mid x)
-
\ln\pi_k(y\mid x)
\right]
+C(x)
$$

#### 用上下文学习近似最优策略

传统 IRL 希望从示范中恢复奖励函数，但这通常很困难。SDFT 没有单独训练奖励模型，而是采用一个关键假设：模型看到专家示范后形成的条件分布，可以近似当前任务中理想的下一策略：

$$
\pi_{k+1}^{*}(y\mid x)
\approx
\pi_{\phi_k}(y\mid x,c)
$$

代入上式后，可以得到一个隐式奖励：

$$
r(y,x,c)
\approx
\beta
\left[
\ln\pi_{\phi_k}(y\mid x,c)
-
\ln\pi_k(y\mid x)
\right]
+C(x)
$$

在只关心更新方向时，可以忽略正的整体缩放和与回答 $y$ 无关的常数。于是，一个 token 的直观奖励就是“教师对它的对数概率减去学生对它的对数概率”：教师比学生更认可该选择时，奖励为正；反之则为负。

基于该隐式奖励，我们可以得到第 $k$ 轮的策略梯度：

$$
\nabla_\theta J(\pi_{k})
=
E_{y \sim \pi_{k}(\cdot\mid x)}
\left[
\left(
\ln \frac{\pi_{\phi_k}(y\mid x,c)}{\pi_k(y\mid x)}
\right)
\nabla_\theta \ln \pi_k(y\mid x)
\right]
$$

另一方面，RL/OPD 一类的优化目标是最小化当前策略与最优策略之间的反向 KL：

$$
\begin{aligned}
\theta^* &= \arg\min_{\theta} D_{KL}(\pi_\theta(\cdot\mid x) \parallel \pi_{k+1}^{*}(\cdot\mid x)) \\
&= \arg\min_{\theta} E_{y \sim \pi_\theta} \left[ \ln \frac{\pi_\theta(y\mid x)}{\pi_{k+1}^{*}(y\mid x)} \right]
\end{aligned}
$$

对反向 KL 求梯度并用对数求导技巧展开：$\nabla_\theta D_{KL}(\pi_\theta\parallel\pi_{k+1}^{*}) = E_{y\sim\pi_\theta}[\ln\frac{\pi_\theta(y\mid x)}{\pi_{k+1}^{*}(y\mid x)}\nabla_\theta\ln\pi_\theta(y\mid x)] + E_{y\sim\pi_\theta}[\nabla_\theta\ln\frac{\pi_\theta(y\mid x)}{\pi_{k+1}^{*}(y\mid x)}]$。第二项中 $E_{y\sim\pi_\theta}[\nabla_\theta\ln\pi_\theta(y\mid x)]=\sum_y\nabla_\theta\pi_\theta(y\mid x)=0$（因为 $\sum_y\pi_\theta(y\mid x)=1$），而 $\pi_{k+1}^{*}$ 与 $\theta$ 无关，故第二项整体为零。代入 $\pi_\theta=\pi_k$、$\pi_{k+1}^{*}\approx\pi_{\phi_k}$ 后可以发现：最小化反向 KL 的梯度与最大化上述隐式奖励的策略梯度在期望上是等价的。

在教师分布固定、并从当前策略处计算梯度的条件下，最大化这个隐式奖励的策略梯度与最小化下式的梯度方向一致：

$$
D_{KL}
\left(
\pi_k(\cdot\mid x)
\parallel
\pi_{\phi_k}(\cdot\mid x,c)
\right)
$$

这就是论文 v1 所说的 IRL 解释。不过它依赖“示范条件化教师近似理想策略”这一假设，也依赖反向 KL 目标；它不表示 SDFT 实际恢复了一个独立、可复用的奖励模型。

如果把训练目标换成实际使用的正向 KL，上面的最后一步就不再成立。在一个固定前缀上，记学生分布为 $p$、教师分布为 $q$，并对教师分布停止梯度。此时，RKL 的负梯度可以写成由学生分布采样、并以 $\ln(q/p)$ 加权的策略梯度；FKL 的负梯度则为 $E_{v\sim q}[\nabla_\theta\ln p(v)]$，本质上是让学生用交叉熵模仿教师。两者可以拥有相同的理想最优点 $p=q$，但梯度、训练路径和理论解释并不相同。因此，奖励倾斜关系仍可作为“教师分布可以对应某个隐式奖励”的数学动机，却不能据此把实际 FKL 训练称为等价的 IRL 或在策略强化学习。

### 逐 token 蒸馏与停止梯度

设一条学生回答上的蒸馏损失为 $\ell_\theta(\hat y)$，则完整目标可以简写为：

$$
J(\theta)
=
E_{\hat y\sim\pi_\theta(\cdot\mid x)}
[\ell_\theta(\hat y)]
$$

因为 $\theta$ 既决定采样出什么回答，也出现在回答内部的蒸馏损失中，所以形式上的完整梯度包含两部分：

$$
\nabla_\theta J
=
E
\left[
\ell_\theta(\hat y)
\nabla_\theta\ln\pi_\theta(\hat y\mid x)
+
\nabla_\theta\ell_\theta(\hat y)
\right]
$$

- 第一项是**轨迹梯度（trajectory gradient）**。它根据整条采样轨迹的损失，调整轨迹中已采样动作的概率。它确实会改变下一 token 分布，只是没有直接比较未被采样的所有候选 token。
- 第二项是**蒸馏梯度（distillation gradient）**。它把已采样前缀视为固定输入，直接比较该前缀下的学生与教师分布。

实际训练通常不会对离散采样过程直接反向传播。SDFT 先生成回答，再把这条回答视为固定数据，只保留逐 token 蒸馏梯度。停止梯度（stop-gradient）就是在计算图中切断某条求导路径：数值仍然参与损失计算，但梯度不会沿该路径继续传播。

需要注意，去掉轨迹梯度后得到的是一个实用的代理梯度（surrogate gradient），而不是原序列级目标的精确完整梯度。训练仍然具有在策略性质，是因为下一轮会由更新后的学生重新采样，而不是因为采样过程本身参与了反向传播。

#### 论文中的解析式逐 token RKL 估计器

在一个固定的学生前缀上，记学生和教师的词表分布为 $p(v)$ 与 $q(v)$。反向 KL 为：

$$
D_{KL}(p\parallel q)
=
\sum_{v\in\mathcal V}
p(v)\ln\frac{p(v)}{q(v)}
$$

如果教师在当前优化步中保持冻结（$\nabla_\theta \ln q(v) = 0$），我们可以展开反向 KL 的梯度。先对第一项求梯度，利用乘积法则：

$$
\begin{aligned}
\nabla_\theta \sum_{v \in \mathcal V} p(v) \ln p(v)
&= \sum_{v \in \mathcal V} \left[ \nabla_\theta p(v) \right] \ln p(v) + \sum_{v \in \mathcal V} p(v) \nabla_\theta \ln p(v) \\
&= \sum_{v \in \mathcal V} \left[ \nabla_\theta p(v) \right] \ln p(v) + \sum_{v \in \mathcal V} p(v) \frac{\nabla_\theta p(v)}{p(v)} \\
&= \sum_{v \in \mathcal V} \left[ \nabla_\theta p(v) \right] \ln p(v) + \sum_{v \in \mathcal V} \nabla_\theta p(v)
\end{aligned}
$$

由于 $p$ 是概率分布，$\sum_{v \in \mathcal V} p(v) = 1$，故第二项

$$
\sum_{v \in \mathcal V} \nabla_\theta p(v) = \nabla_\theta \sum_{v \in \mathcal V} p(v) = 0
$$

于是第一项化简为：

$$
\nabla_\theta \sum_{v \in \mathcal V} p(v) \ln p(v)
=
\sum_{v \in \mathcal V} \left[ \nabla_\theta p(v) \right] \ln p(v)
$$

再处理第二项。由于教师分布 $q$ 与 $\theta$ 无关，$\nabla_\theta \ln q(v) = 0$，故：

$$
\nabla_\theta \sum_{v \in \mathcal V} p(v) \ln q(v)
=
\sum_{v \in \mathcal V} \left[ \nabla_\theta p(v) \right] \ln q(v)
$$

合并两项，得到：

$$
\begin{aligned}
\nabla_\theta D_{KL}(p\parallel q)
&=
\sum_{v\in\mathcal V}
\left[ \nabla_\theta p(v) \right]
\left[ \ln p(v) - \ln q(v) \right] \\
&=
\sum_{v\in\mathcal V}
\ln \frac{p(v)}{q(v)}
\nabla_\theta p(v) \\
&=
\sum_{v\in\mathcal V}
p(v)
\ln\frac{p(v)}{q(v)}
\nabla_\theta\ln p(v)
\end{aligned}
$$

推导中多出的 $\sum_v\nabla p(v)$ 会因为 $\sum_v p(v)=1$ 而抵消。这个式子完整遍历词表，因此在**固定前缀**上是精确的 RKL 梯度，也比只采样一个 token 的估计波动更小。论文将其称为 Full Analytic Per-token Estimator，这里可译作“解析式逐 token 梯度估计器”。

但“固定前缀上精确”不等于“对整条序列精确”。如果训练对学生采样的前缀停止梯度，它仍然没有计算早期 token 改变后续前缀所产生的影响，因此相对于完整序列梯度仍然有偏。

### 论文 RKL 与实验 FKL 的差异

截至本文修订时，arXiv 仍只有 v1。该版本把 SDFT 写成序列级反向 KL，并据此建立前面的隐式奖励与 IRL 解释：

$$
L_{\mathrm{RKL}}
=
D_{KL}
\left(
\pi_\theta(\cdot\mid x)
\parallel
\pi_\phi(\cdot\mid x,c)
\right)
$$

不过，官方代码仓库在 2026 年 4 月 7 日补充说明：论文中所有实验结果实际使用的是由当前学生生成回答，再计算逐 token 正向 KL；仓库默认设置也采用 FKL。在固定的学生前缀上，其损失为：

$$
L_{\mathrm{FKL},n}
=
D_{KL}(p_{T,n}\parallel p_{S,n})
=
\sum_{v\in\mathcal V}
p_{T,n}(v)
\ln\frac{p_{T,n}(v)}{p_{S,n}(v)}
$$

教师分布停止梯度后，这个目标的可训练部分等价于让学生用交叉熵模仿教师的完整词表分布。直观上，教师认为可能性较高、学生却遗漏的 token 会受到较大惩罚。

因此，更严谨的结论是：

1. 论文报告的实验结果支持“示范条件化教师 + 学生在策略前缀 + 逐 token FKL”这一实际配置。
2. RKL、解析式 RKL 估计器和 IRL 等价关系属于论文 v1 的理论表述，不能用来解释 FKL 的实际梯度，也不能直接当作这些实验已经验证的机制。
3. RKL 与 FKL 都可以在学生生成的前缀上训练；KL 方向与是否在策略并不是同一条设计轴。

### EMA 教师

如果教师始终冻结在初始模型，它无法跟随学生已经学到的新能力；如果直接使用当前学生作为教师，学生和教师又会同时快速变化，训练可能不稳定。SDFT 使用指数移动平均（Exponential Moving Average，EMA）在二者之间折中。

初始化时令 $\phi_0=\theta_0$。第 $k+1$ 次学生更新后，教师参数按下面的方式更新：

$$
\phi_{k+1}
=
(1-\alpha)\phi_k
+
\alpha\theta_{k+1}
$$

其中 $\alpha$ 是教师吸收最新学生参数的比例。论文中的超参数搜索范围为：

$$
\alpha\in\{0.01,0.02,0.05\}
$$

$\alpha$ 较小时，教师变化更慢，目标更稳定；$\alpha$ 较大时，教师更快跟上学生。训练学生时，教师分支本身不接收反向传播梯度，而是在学生更新完成后通过上面的 EMA 规则单独更新。

### OPSD 与 SDFT 对比

SDFT 与上文的 OPSD 共享相同的基本骨架：学生生成回答，同源教师获得额外信息，再沿着学生前缀提供逐 token 监督。主要区别在教师如何构造、研究目标以及论文与实现采用的具体设置。

| 维度 | OPSD | SDFT |
| --- | --- | --- |
| 学生轨迹 | 当前学生在策略生成 | 当前学生在策略生成 |
| 教师额外信息 | 当前问题的参考答案或解题过程 $y^*$ | 与样本配对的专家示范或知识上下文 $c$ |
| 教师参数 | v3 主要实验冻结初始模型 | 使用学生参数的 EMA |
| 主要实验散度 | v3 主要采用 FKL | 已发布结果实际采用逐 token FKL；v1 文本描述 RKL |
| 稳定化方式 | 逐项散度裁剪，避免少数风格 token 主导 | EMA 平滑教师更新；论文还用起始 token 掩码缓解语言痕迹 |
| 主要研究目标 | 利用特权信息提升推理任务表现 | 学习新技能或知识，同时减轻遗忘 |
| 理论侧重点 | 在策略自蒸馏 | v1 额外给出基于 RKL 的隐式奖励 / IRL 解释 |

两种方法的名称和应用背景不同，但算法边界并不绝对。更合适的理解是：它们都是“同源模型 + 特权上下文 + 学生在策略轨迹”这一方法族中的不同实例。

### 实验观察

除特别说明外，SDFT 论文主要使用 Qwen2.5-7B-Instruct，在科学问答、工具调用、医疗推理和新知识注入等任务上比较 SFT、DFT、持续预训练等基线。这里的 DFT 是一种通过重要性加权，让固定数据近似当前策略数据的方法。由于官方更正，下面的 SDFT 结果应理解为来自逐 token FKL 实现。

1. **新任务表现与能力保留：** 在三个技能学习任务中，SDFT 通常取得更高的新任务准确率，同时比 SFT 更接近基础模型原有的综合能力。在连续依次学习三个技能时，SDFT 也比 SFT 更能保留先前任务。

2. **知识注入：** 在新知识问答实验中，SFT 的严格准确率为 80%，SDFT 为 89%；在需要间接使用新知识的 OOD 问题上，二者分别为 80% 和 98%。这里的 OOD 指问题形式超出训练问答的直接表达，而不是完全无关的新领域。

3. **模型规模很重要：** 3B 模型的上下文学习能力较弱，SDFT 在科学问答上反而落后于 SFT；到 7B 和 14B 时，SDFT 相对 SFT 的优势分别扩大到约 4 个和 7 个百分点。这说明方法依赖模型先具备从示范中理解任务的能力。

4. **只有最终答案时仍可能保留推理习惯：** 在 Olmo-3-7B-Think 的医疗任务中，答案式 SFT 将准确率从 31.2% 降至 23.5%，而 SDFT 达到 43.7%，并较好地保留了原模型较长的推理输出。不过这只是特定模型与任务上的结果，不能直接推广到所有推理模型。

5. **计算成本高于 SFT：** 论文报告 SDFT 约需要 SFT 的 2.5 倍 FLOPs，并花费约 4 倍实际训练时间。FLOPs 可以粗略理解为总浮点计算量；额外成本主要来自生成学生回答，以及运行教师分支计算分布。

### SDFT 的适用条件与局限

SDFT 更适合这样的场景：有高质量示范，希望模型内化新技能或知识，但难以为每个输出设计可靠的显式奖励。使用时还需要注意以下限制：

1. **依赖上下文学习能力。** 如果模型看到示范后仍不能正确理解任务，教师分支就无法提供比学生更好的监督。小模型或超出模型能力范围的任务尤其容易遇到这个问题。

2. **依赖示范质量与配对方式。** 教师看到的是哪一段示范、示范是否真的表达了目标行为，都会影响训练信号。错误、含糊或与问题不匹配的示范可能被模型内化。

3. **减轻遗忘不等于零遗忘。** 论文观察到 SDFT 比 SFT 更能保留旧能力，但仍承认部分能力会下降。在策略训练和 EMA 都不能提供绝对保证。

4. **可能学习到教师上下文的语言痕迹。** 教师有时会输出“根据以上文本”或“按照示例”等表述，学生即使没有看到这些上下文也可能模仿。论文通过屏蔽回答开头若干 token 的损失来缓解，但这仍是经验性修补。

5. **不擅长彻底改变基础行为。** SDFT 更适合在原有能力上增加知识或技能。若目标是把一个不会显式推理的模型直接变成长期思考模型，示范条件化教师本身可能也无法产生足够强的行为变化。

6. **实验范围仍有限。** 当前证据主要来自论文选定的模型、数据集和训练配置。它展示了一条有潜力的持续学习路径，但还不能证明 SDFT 在所有任务上都优于 SFT、OPSD 或奖励驱动的在策略强化学习。

### 参考资料

- [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)
- [SDFT 官方代码与 README 更正说明](https://github.com/idanshen/Self-Distillation#updates)
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)
- [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)
