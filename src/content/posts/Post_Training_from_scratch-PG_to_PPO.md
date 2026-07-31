---
title: 从零开始的 Post Training 1：从 Policy Gradient 到 PPO
published: 2026-06-18
description: 'Post Training 的入门学习笔记 从 Policy Gradient 到 PPO 篇'
image: ''
tags: ['Deep_Learning', 'Post_Training', 'RL', 'RLHF', 'Policy_Gradient', 'Actor-Critic', 'PPO']
category: 'Research'
draft: false
lang: ''
series: 'Post Training'
---

# 从零开始的 Post Training 1：从 Policy Gradient 到 PPO

## Policy Gradient

策略梯度（Policy Gradient）是一类直接调整策略参数、使期望回报增大的方法。设策略为 $\pi_\theta(a\mid s)$，其中 $\theta$ 是需要训练的参数。

一条轨迹 $\tau$ 的概率由初始状态、策略选择和环境转移共同决定：

$$
p_\theta(\tau)
=
\rho_0(s_0)
\prod_t
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

这里 $\rho_0$ 是初始状态分布，$P$ 是环境转移概率。策略优化的目标是最大化轨迹的期望回报：

$$
J(\theta)
=
E_{\tau\sim p_\theta(\tau)}[G(\tau)]
=
\sum_\tau p_\theta(\tau)G(\tau).
$$

对 $\theta$ 求梯度。由于梯度运算可以穿过求和号，而 $G(\tau)$ 与 $\theta$ 无关，因此先把梯度作用在轨迹概率上，再逐步改写成期望形式：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta \sum_{\tau} p_\theta(\tau) G(\tau) \\
&= \sum_{\tau} \nabla_\theta p_\theta(\tau) G(\tau) \\
&= \sum_{\tau} p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} G(\tau) \\
&= \sum_{\tau} p_\theta(\tau) \nabla_\theta \ln p_\theta(\tau) G(\tau) \\
&= \sum_{\tau} p_\theta(\tau) G(\tau) \nabla_\theta \ln p_\theta(\tau) \\
&\approx \frac{1}{N} \sum_{i=1}^N G(\tau_i) \nabla_\theta \ln p_\theta(\tau_i), \quad \tau_i \sim p_\theta(\tau)
\end{aligned}
$$

第 3 行到第 4 行使用的就是对数求导技巧（Log-derivative trick）：它只是利用 $\nabla p = p\nabla\ln p$，把难以直接计算的分布梯度改写成可以通过采样估计的期望形式。

由于环境转移概率不依赖 $\theta$，轨迹的对数梯度只保留策略部分：

$$
\nabla_\theta\ln p_\theta(\tau)
=
\sum_t\nabla_\theta\ln\pi_\theta(a_t\mid s_t).
$$

如果去掉梯度符号并加上负号，就得到以整条轨迹为单位的策略梯度损失：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N G(\tau_i) \ln p_\theta(\tau_i), \quad \tau_i \sim p_\theta(\tau)
$$

进一步可以把优化目标从整条轨迹拆到每个时间步：利用因果性，时间步 $t$ 的动作不会影响此前已经获得的奖励，因此在对应的梯度项中，可以把整条轨迹回报替换为从 $t$ 开始的未来回报。使用 $N$ 条当前策略生成的轨迹，可以构造 REINFORCE 损失：

$$
L_{\mathrm{REINFORCE}}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^N
\sum_t
G_{i,t}
\ln\pi_\theta(a_{i,t}\mid s_{i,t}).
$$

这里 $G_{i,t}$ 是第 $i$ 条轨迹从时间步 $t$ 开始的回报。训练数据来自当前策略，因此 REINFORCE 属于在策略（on-policy）方法。

## Actor-Critic 方法

Actor-Critic 方法同时训练两个角色：

- Actor 根据状态选择动作，也就是策略 $\pi_\theta$。
- Critic 估计状态的长期价值，用来降低策略梯度的方差。

在 RLHF 中，语言模型通常承担 Actor；价值模型承担 Critic。奖励模型负责给生成结果打分，它提供奖励信号，但并不等同于 Critic。

### 优势估计

继续之前的推导，当我们将目光聚集到单个时间步时，上述损失函数中的整条轨迹回报 $G(\tau_i)$ 可以被替换为从 $t$ 开始的回报 $G_{i,t}$，于是 Actor 损失可以写成：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N G_{i,t} \ln \pi_\theta(a_{i,t}\mid s_{i,t})
$$

直接使用回报 $G_t$ 的方差通常很大，因此可以减去一个只依赖状态的基线 $b(s_t)$：

$$
\widehat A_t=G_t-b(s_t).
$$

为了将基线替换为价值函数，我们梳理一下价值函数之间的关系。利用单步采样近似期望，可以得到：

$$
\begin{aligned}
V^\pi(s) &\approx r(s, a, s') + \gamma V^\pi(s') \\
Q^\pi(s, a) &= r(s, a, s') + \gamma V^\pi(s') \\
A^\pi(s, a) &= r(s, a, s') + \gamma V^\pi(s') - V^\pi(s)
\end{aligned}
$$

由 $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ 可见，优势函数相当于以状态价值 $V^\pi(s)$ 作为基线 $b(s_t)$。由于 $\sum_a \pi(a\mid s)\nabla_\theta\ln\pi(a\mid s)=\sum_a\nabla_\theta\pi(a\mid s)=0$，减去这样的基线不会改变策略梯度的期望。实际中通常令 $b(s_t)=V_\phi(s_t)$，其中 $V_\phi$ 是 Critic 给出的价值估计。于是 Actor 损失可以进一步用优势函数改写为：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N A^\pi(s_{i,t}, a_{i,t}) \ln \pi_\theta(a_{i,t}\mid s_{i,t})
$$

为了进一步化简，可以将一条轨迹的概率展开。忽略环境转移概率中与策略参数无关的部分后，轨迹概率可以看作每个时间步动作概率的乘积：

$$
\begin{aligned}
p_\theta(\tau) &\propto \prod_t \pi_\theta(a_t\mid s_t) \\
\ln p_\theta(\tau) &= \sum_t \ln \pi_\theta(a_t\mid s_t)
\end{aligned}
$$

因此对应到逐时间步的优化形式，并加入停止梯度，Actor 损失可以写成：

$$
L_{\mathrm{actor}}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^N
\sum_t
\operatorname{sg}(\widehat A_{i,t})
\ln\pi_\theta(a_{i,t}\mid s_{i,t}).
$$

$\operatorname{sg}(\cdot)$ 表示停止梯度：优势值只决定更新方向和强度，不通过 Actor 损失反向修改优势估计本身。

### GAE

广义优势估计（Generalized Advantage Estimation，GAE）用于在偏差和方差之间取得平衡。先定义单步时序差分误差（temporal-difference error，TD error）：

$$
\delta_t
=
r_{t+1}
+\gamma V_{\phi_{\mathrm{old}}}(s_{t+1})
-V_{\phi_{\mathrm{old}}}(s_t).
$$

这里使用更新前的价值函数 $V_{\phi_{\mathrm{old}}}$ 计算固定的训练目标。

先定义 $n$ 步优势估计：对未来 $n$ 步的回报进行采样，可以得到递推展开：

$$
\begin{aligned}
A_1(s, a) &= r(s, a, s') + \gamma V^\pi(s') - V^\pi(s) \\
A_2(s, a) &= r(s, a, s') + \gamma \left[ r(s', a', s'') + \gamma V^\pi(s'') \right] - V^\pi(s) \\
A_3(s, a) &= r(s, a, s') + \gamma \left[ r(s', a', s'') + \gamma \left[ r(s'', a'', s''') + \gamma V^\pi(s''') \right] \right] - V^\pi(s) \\
&\vdots
\end{aligned}
$$

可以看到，采样步数越多，优势估计的偏差越小，但方差也越大。利用 $\delta_t$ 的定义展开，$n$ 步优势可以写成 TD error 的折扣和：

$$
\begin{aligned}
A_n(s, a) &= \sum_{k=0}^{n-1} \gamma^k r(s_{t+k}, a_{t+k}, s_{t+k+1}) + \gamma^n V^\pi(s_{t+n}) - V^\pi(s_t) \\
&= \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}
\end{aligned}
$$

为了平衡偏差和方差，我们可以引入一个权重 $\lambda$ 对多步优势估计做加权平均：

$$
A^{\mathrm{GAE}}_\lambda(s, a) = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} A_n(s, a)
$$

当 $\lambda = 0$ 时，GAE 退化为单步优势，即 TD error；当 $\lambda \to 1$ 时，GAE 退化为整条轨迹的回报，即 Monte Carlo 估计。代入 $A_n$ 的展开式并交换求和顺序，可以化简：

$$
\begin{aligned}
A^{\mathrm{GAE}}_\lambda(s, a) &= (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \sum_{k=0}^{n-1} \gamma^k \delta_{t+k} \\
&= (1 - \lambda) \sum_{k=0}^\infty \gamma^k \delta_{t+k} \sum_{n=k+1}^\infty \lambda^{n-1} \\
&= (1 - \lambda) \sum_{k=0}^\infty \gamma^k \delta_{t+k} \frac{\lambda^k}{1 - \lambda} \\
&= \sum_{k=0}^\infty (\gamma \lambda)^k \delta_{t+k}
\end{aligned}
$$

因此，GAE 最终可以写成较简洁的形式：

$$
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{k=0}^{\infty}
(\gamma\lambda)^k\delta_{t+k}.
$$

$\lambda$ 越小，估计越依赖短期 TD 目标，方差较小但更依赖 Critic 的准确性；$\lambda$ 越接近 1，估计越接近完整轨迹回报，偏差通常较小但方差更大。实际计算会在轨迹结束处截断。

## PPO 算法

PPO（Proximal Policy Optimization）仍然是一种在策略方法。它先用旧策略 $\pi_{\mathrm{old}}$ 采集一批轨迹（rollout），再在这批数据上进行若干轮小批量更新。完成这些更新后，需要重新采集较新的数据。这里的 rollout 就是策略与环境交互后得到的一条或一批轨迹。

### 1. 概率比与代理目标

PPO 复用了旧策略采集的数据，其理论依据是重要性采样（Importance Sampling）。对于一个期望，在 $p(x)$ 分布和 $q(x)$ 分布下的期望值有所不同，但可以通过乘上概率比来相互转换：

$$
\begin{aligned}
E_{p(x)}[f(x)] &= \int f(x) p(x) dx \\
&= \int f(x) \frac{p(x)}{q(x)} q(x) dx \\
&= E_{q(x)}\left[ f(x) \frac{p(x)}{q(x)} \right]
\end{aligned}
$$

如果将上式中的 $q(x)$ 分布用采样来近似，可以得到：

$$
E_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i) \frac{p(x_i)}{q(x_i)}, \quad x_i \sim q(x)
$$

把旧策略 $\pi_{\mathrm{old}}$ 当作 $q$、当前策略 $\pi_\theta$ 当作 $p$ 应用到 Actor-Critic 目标上。对于旧策略采样出的动作，定义当前策略与旧策略的概率比：

$$
\rho_{i,t}(\theta)
=
\frac{\pi_\theta(a_{i,t}\mid s_{i,t})}
{\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})}.
$$

当 $\rho_{i,t}(\theta)>1$ 时，当前策略比旧策略更倾向于该动作；当它小于 1 时，当前策略降低了该动作的概率。旧策略数据上的 REINFORCE 梯度需要用重要性比率 $\rho_{i,t}(\theta)$ 修正，推导如下：

$$
\begin{aligned}
& \frac1N \sum_{i=1}^N \sum_t \rho_{i,t}(\theta) \widehat A_{i,t} \nabla_{\theta} \ln \pi_\theta(a_{i,t}\mid s_{i,t}) \\
=& \frac1N \sum_{i=1}^N \sum_t \frac{\pi_\theta(a_{i,t}\mid s_{i,t})}{\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})} \widehat A_{i,t} \nabla_{\theta} \ln \pi_\theta(a_{i,t}\mid s_{i,t}) \\
=& \frac1N \sum_{i=1}^N \sum_t \widehat A_{i,t} \frac{\nabla_{\theta} \pi_\theta(a_{i,t}\mid s_{i,t})}{\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})}
\end{aligned}
$$

第一步中 $a_{i,t}\sim\pi_{\mathrm{old}}$ 与当前策略 $\pi_\theta$ 不一致，因此需要乘上概率比；第二步利用了 $\rho_{i,t}(\theta)\nabla_{\theta}\ln\pi_\theta=\nabla_{\theta}\pi_\theta/\pi_{\mathrm{old}}$ 化简。注意到 $\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})$ 是旧策略下的概率，而 $\pi_\theta(a_{i,t}\mid s_{i,t})$ 是当前策略下的概率。在 LLM 训练中，我们可以认为不存在环境的随机性，故用策略概率 $\pi_\theta$、$\pi_{\mathrm{old}}$ 替代轨迹概率中的动作概率部分。去掉梯度符号并加上负号，PPO 使用下面的代理目标复用旧策略数据：

$$
J_{\mathrm{sur}}(\theta)
=
E_{(s_{i,t},a_{i,t})\sim\pi_{\mathrm{old}}}
\left[
\rho_{i,t}(\theta)\widehat A_{i,t}
\right].
$$

这里的“代理目标”表示它在旧策略附近近似我们真正关心的策略改进目标。PPO 并不是可以长期使用任意旧数据的离策略算法；如果当前策略已经离采样策略太远，这种近似就会变差。

### 2. 裁剪（Clipping）

为了避免单次更新过大，PPO-Clip 使用：

$$
\begin{aligned}
J_{\mathrm{clip}}(\theta)
=
E
\Big[
\min\big(&\rho_{i,t}(\theta)\widehat A_{i,t},\\
&\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)\widehat A_{i,t}\big)
\Big].
\end{aligned}
$$

训练时最小化其相反数 $L_{\mathrm{clip}}=-J_{\mathrm{clip}}$。这里裁剪的是代理目标中的更新收益，而不是对模型概率施加硬约束；训练后的实际概率比仍然可能超出 $[1-\epsilon,1+\epsilon]$。

### 3. 两种不同的 KL 约束

需要区分两个容易混淆的策略：

1. $\pi_{\mathrm{old}}$：生成当前训练数据的旧策略，用于概率比和 PPO 更新限制。
2. $\pi_{\mathrm{ref}}$：固定参考策略。在 RLHF 中，它通常是 SFT 后的模型，用于防止策略为了追求奖励而偏离原模型太远。

上述的代理目标可能会导致策略更新过大，从而使训练不稳定。为了限制策略更新的幅度，PPO 引入了 KL 散度（Kullback-Leibler Divergence）来衡量当前策略与旧策略之间的差异。具体来说，KL 散度可以定义为：

$$
\begin{aligned}
D_{KL}(\pi_\theta \parallel \pi_{\mathrm{old}})
&= \sum_a \pi_\theta(a|s_t) \ln \frac{\pi_\theta(a|s_t)}{\pi_{\mathrm{old}}(a|s_t)} \\
&= E_{a \sim \pi_\theta(a|s_t)} \left[ \ln \frac{\pi_\theta(a|s_t)}{\pi_{\mathrm{old}}(a|s_t)} \right]
\end{aligned}
$$

KL 散度恒非负，当当前策略与旧策略完全相同时取零；且它不对称，即 $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$。将 KL 惩罚项加入损失函数：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_t \rho_{i,t}(\theta) \widehat A_{i,t} + \beta D_{KL}(\pi_\theta \parallel \pi_{\mathrm{old}})
$$

其中 $\beta$ 是控制 KL 惩罚权重的超参数。除了上述当前策略与旧策略之间的 KL 限制（横向 KL）之外，InstructGPT 中还引入了纵向的 KL 限制，即在每一步的 Reward 中加入一个当前策略与参考策略 $\pi_{\mathrm{ref}}$ 之间的 KL 惩罚项。

在此基础上，PPO 还引入了剪切（Clipping）机制来进一步限制策略更新的幅度。剪切函数定义为：

$$
\text{clip}(x, l, u) = \begin{cases}
l & \text{if } x < l \\
x & \text{if } l \leq x \leq u \\
u & \text{if } x > u
\end{cases}
$$

其实质上是对概率比 $\rho_{i,t}(\theta)$ 进行剪切，限制其在 $[1 - \epsilon, 1 + \epsilon]$ 的范围内。

PPO 可以监测或惩罚当前策略与旧策略之间的 KL 散度。RLHF 还常使用当前策略相对参考策略的约束：

$$
D_{KL}
\left(
\pi_\theta(\cdot\mid s)
\parallel
\pi_{\mathrm{ref}}(\cdot\mid s)
\right).
$$

实现中常用采样 token 的对数概率比近似这项惩罚，并把它加入每一步奖励。当 token 来自当前策略时，这个对数比率的期望等于相应的反向 KL；复用旧策略数据时，则依赖当前策略与旧策略仍然接近。旧策略和参考策略承担不同作用，不能混为同一个模型。

### 4. Critic 损失

令固定的回报目标为：

$$
\widehat R_t
=
\widehat A_t^{\mathrm{GAE}}
+V_{\phi_{\mathrm{old}}}(s_t).
$$

Critic 可以通过均方误差训练：

$$
L_{\mathrm{critic}}(\phi)
=
\frac{1}{N}
\sum_{i=1}^N
\sum_t
\left(
V_\phi(s_{i,t})-\widehat R_{i,t}
\right)^2.
$$

Critic 越准确，优势估计通常越稳定。不过 Critic 仍然只是价值函数的近似，也可能引入估计误差。

### 5. 可选的熵奖励

一些 PPO 实现会加入熵奖励，避免策略过早变得过于确定：

$$
H(\pi_\theta(\cdot\mid s))
=
-\sum_a
\pi_\theta(a\mid s)
\ln\pi_\theta(a\mid s).
$$

熵越高，策略分布越分散。熵奖励是常见的可选项，并不是 PPO 定义中不可缺少的部分。一个常见的组合损失形式为：

$$
L_{\mathrm{total}}
=
L_{\mathrm{clip}}
+c_VL_{\mathrm{critic}}
-c_HE_t\left[H(\pi_\theta(\cdot\mid s_t))\right],
$$

其中 $c_V$ 和 $c_H$ 根据任务设置。在 RLHF 中，还可以额外加入相对 $\pi_{\mathrm{ref}}$ 的 KL 惩罚。

### PPO 的优缺点

#### 优点

1. 通过代理目标和概率比裁剪提高了策略更新的稳定性。
2. 可以在一批 rollout 上进行多轮小批量更新，样本利用率高于只更新一次的策略梯度。
3. 实现相对简单，不需要复杂的二阶优化。

#### 缺点

1. 本质上仍需要持续采集较新的在策略数据，rollout 成本较高。
2. 需要训练价值函数，价值估计不准时会影响优势估计。
3. 概率比裁剪只是经验性的稳定机制，并不保证严格限制每次策略变化。
4. 学习率、裁剪范围、GAE 参数和 KL 系数等超参数仍需调节。

典型的 PPO-RLHF 系统可能包含策略、价值、奖励和参考四种模型角色，但这不是 PPO 算法本身必须包含四个独立模型；部分角色可以共享参数，普通 PPO 任务也可能直接由环境提供奖励。

## 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
