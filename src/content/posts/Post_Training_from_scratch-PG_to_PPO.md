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

# 从零开始的 Post Training 1

## Policy Gradient

策略梯度（Policy Gradient）方法是一类直接优化策略函数的强化学习算法。策略梯度方法通过直接调整策略参数来最大化预期回报。

也就是说，我们的目标是找到一个参数化的策略 $\pi_\theta(a|s)$，其中 $\theta$ 是策略的参数。我们希望通过调整 $\theta$ 来最大化预期回报 $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$。

我们可以展开预期回报的定义：

$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\pi_\theta}[G_t] \\
&= \sum_{\tau} P(\tau|\theta) G(\tau)
\end{aligned}
$$

这里我们对 $\theta$ 取梯度：

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \sum_{\tau} P(\tau|\theta) G(\tau) \\
&= \sum_{\tau} \nabla_\theta P(\tau|\theta) G(\tau) \\
&= \sum_{\tau} P(\tau|\theta) \frac{\nabla_\theta P(\tau|\theta)}{P(\tau|\theta)} G(\tau) \\
&= \sum_{\tau} P(\tau|\theta) \nabla_\theta \ln P(\tau|\theta) G(\tau) \\
&= \sum_{\tau} P(\tau|\theta) G(\tau) \nabla_\theta \ln P(\tau|\theta) \\
&\approx \frac{1}{N} \sum_{i=1}^N G(\tau_i) \nabla_\theta \ln P(\tau_i|\theta), \quad \tau_i \sim P(\tau|\theta)
\end{aligned}
$$

这里使用了 Log-derivative trick 来将梯度表达式转换为一个期望值的形式，并且通过采样来近似计算梯度。

如果我们将梯度去掉，加上负号就可以得到 Policy Gradient 的损失函数：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N G(\tau_i) \ln P(\tau_i|\theta), \quad \tau_i \sim P(\tau|\theta)
$$

即我们的目标是要最小化该损失函数。

当前是用整条轨迹的回报来估计策略梯度，这种方法被称为 REINFORCE 算法。进一步地，也可以把优化目标拆到每个时间步，并引入价值函数作为 Critic 来估计基线或优势函数，这就过渡到了 Actor-Critic 方法。

注意到我们的模型运行与数据采集是交替进行的，因此这是一个 On Policy 的方法。

## Actor-Critic 方法

Actor-Critic 方法是一种结合了策略优化（Actor）和价值函数估计（Critic）的强化学习算法。Actor 负责选择动作，而 Critic 负责评估 Actor 的动作选择。

在 RLHF 的语境下，语言模型通常可以看作 Actor，奖励模型或者价值模型则承担 Critic 的角色，用来评价模型生成内容的好坏。后续常见的 PPO 训练，本质上也是在这个框架下更新 Actor。

继续我们之前的推导，我们发现当我们将目光聚集到单个时间步时候，上述的 Loss 函数中的 $G(\tau_i)$ 可以被替换为 $G_t$，即当前时间步的回报。此时我们可以得到以下的 Actor 损失函数：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N G_t \ln P(a_t|\theta), \quad a_t \sim P(a_t|\theta)
$$

此时注意到 $G_t$ 的方差可能会比较大，因此我们可以引入一个基线（Baseline）来降低方差。这样我们就可以得到以下的 Actor 损失函数：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N (G_t - B^\pi(s_t)) \ln P(a_t|\theta), \quad a_t \sim P(a_t|\theta)
$$

### 价值函数改写

进一步的，我们可以用价值函数改写上述损失函数，引入之前得到的优势价值函数 $A^\pi(s, a)$，从而得到以下的 Actor 损失函数：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N A^\pi(s_t, a_t) \ln P(a_t|\theta), \quad a_t \sim P(a_t|\theta)
$$

为了进一步简化，我们可以将一条轨迹的概率展开。忽略环境转移概率中与策略参数无关的部分后，轨迹概率可以看作每个时间步动作概率的乘积：

$$
\begin{aligned}
P(\tau|\theta) &\propto \prod_t P(a_t|s_t) \\
\ln P(\tau|\theta) &= \sum_t \ln P(a_t|s_t)
\end{aligned}
$$

因此对应到逐时间步的优化形式，可以写为：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_t A^\pi(s_t, a_t) \ln P(a_t|s_t), \quad a_t \sim P(a_t|\theta)
$$

回到之前有关价值函数的推导，梳理价值函数之间的关系，我们有：

$$
\begin{aligned}
V^\pi(s) &\approx r(s, a, s') + \gamma V^\pi(s') \\
Q^\pi(s, a) &= r(s, a, s') + \gamma V^\pi(s') \\
A^\pi(s, a) &= r(s, a, s') + \gamma V^\pi(s') - V^\pi(s)
\end{aligned}
$$

此时对未来 $n$ 步的时间进行采样，我们可以得到以下的关系：

$$
\begin{aligned}
A_1(s, a) &= r(s, a, s') + \gamma V^\pi(s') - V^\pi(s) \\
A_2(s, a) &= r(s, a, s') + \gamma \left[ r(s', a', s'') + \gamma V^\pi(s'') \right] - V^\pi(s) \\
A_3(s, a) &= r(s, a, s') + \gamma \left[ r(s', a', s'') + \gamma \left[ r(s'', a'', s''') + \gamma V^\pi(s''') \right] \right] - V^\pi(s) \\
&\vdots
\end{aligned}
$$

采样的步数越多，优势价值函数的偏差就越小，但同时方差也会增加。因此在实际应用中，我们需要权衡采样步数和方差之间的关系，选择一个合适的采样步数来进行训练。

我们可以定义 $\delta_t = G_t - B^\pi(s_t)$，其中 $B^\pi(s_t)$ 是基线函数。实际应用中，基线函数常取当前的状态价值函数。展开即 $\delta_t = r(s_t, a_t, s_{t+1}) + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$。

则多步采样的优势价值函数可以定义为：

$$
\begin{aligned}
A_n(s, a) &= \sum_{k=0}^{n-1} \gamma^k r(s_{t+k}, a_{t+k}, s_{t+k+1}) + \gamma^n V^\pi(s_{t+n}) - V^\pi(s_t) \\
&= \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}
\end{aligned}
$$

为了平衡偏差和方差，我们可以引入一个权重 $\lambda$ 来控制多步采样的优势价值函数，即：

$$
A^{GAE}_\lambda(s, a) = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} A_n(s, a)
$$

该优势价值函数又称 Generalized Advantage Estimation (GAE)，它通过对多步采样的优势价值函数进行加权平均来平衡偏差和方差。

本质上，我们可以发现，当 $\lambda = 0$ 时，GAE 退化为单步采样的优势价值函数，即 TD Error；当 $\lambda \to 1$ 时，GAE 退化为全局采样的优势价值函数，即 Monte Carlo Error。

代入化简得

$$
\begin{aligned}
A^{GAE}_\lambda(s, a) &= (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} \sum_{k=0}^{n-1} \gamma^k \delta_{t+k} \\
&= (1 - \lambda) \sum_{k=0}^\infty \gamma^k \delta_{t+k} \sum_{n=k+1}^\infty \lambda^{n-1} \\
&= (1 - \lambda) \sum_{k=0}^\infty \gamma^k \delta_{t+k} \frac{\lambda^k}{1 - \lambda} \\
&= \sum_{k=0}^\infty (\gamma \lambda)^k \delta_{t+k}
\end{aligned}
$$

因此我们可以看到，$\lambda$ 的取值越大，优势价值函数的偏差越小，但同时方差也会增加。实际应用中，通常会选择一个适当的 $\lambda$ 来平衡偏差和方差，以获得更好的训练效果。

于是我们可以将 Actor-Critic 方法的 Actor 损失函数改写为：

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \ln P(a_t|s_t), \quad a_t \sim P(a_t|\theta)
$$

## PPO 算法

PPO（Proximal Policy Optimization）算法是一种基于策略梯度的强化学习算法，旨在通过限制策略更新的幅度来提高训练的稳定性和效率。

上述的 Actor-Critic 方法的局限性在于，方法本身是一个 On Policy 的方法，因此在每次更新策略时都需要重新采样数据，这可能会导致训练效率较低。PPO 算法引入了 Off Policy 的思想，通过使用旧策略的数据来更新新策略，从而提高了训练效率。

### 1. 重要性采样

对于一个期望，在 $p(x)$ 分布和 $q(x)$ 分布下的期望值有所不同。我们可以通过重要性采样（Importance Sampling）来将一个分布下的期望值转换为另一个分布下的期望值。具体来说，我们可以使用以下公式：

$$
\begin{aligned}
\mathbb{E}_{p(x)}[f(x)] &= \int f(x) p(x) dx \\
&= \int f(x) \frac{p(x)}{q(x)} q(x) dx \\
&= \mathbb{E}_{q(x)}\left[ f(x) \frac{p(x)}{q(x)} \right]
\end{aligned}
$$

如果将上述公式中的 $q(x)$ 分布使用采样的方式来近似，我们可以得到以下的近似公式：

$$
\mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i) \frac{p(x_i)}{q(x_i)}, \quad x_i \sim q(x)
$$

将上述公式应用到 Actor-Critic 方法中，我们可以将旧策略 $\pi_\theta$ 和新策略 $\pi_{\theta'}$ 进行比较，得到以下的 $J(\theta')$：

$$
\begin{aligned}
& \frac1N \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \nabla_{\theta'} \ln \pi'(a_t|s_t) \\
=& \frac1N \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \frac{\pi'(a_t|s_t)}{\pi(a_t|s_t)} \nabla_{\theta'} \ln \pi'(a_t|s_t) \\
=& \frac1N \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \frac{\nabla_{\theta'} \pi'(a_t|s_t)}{\pi(a_t|s_t)} \\
\end{aligned}
$$

注意到此处的 $\pi(a_t|s_t)$ 是旧策略下的概率，而 $\pi'(a_t|s_t)$ 是新策略下的概率。在 LLM 训练中，我们可以认为不存在环境的随机性，故上述损失函数中我们采用了 $\pi(a_t|s_t)$ 来替代 $P(a_t|s_t)$，并且 $\pi'(a_t|s_t)$ 来替代 $P'(a_t|s_t)$。我们是在新策略下进行优化的，因此需要使用新策略下的概率来调整旧策略下的优势价值函数。

故此时的 Actor 损失函数可以定义为：

$$
L(\theta') = -\frac{1}{N} \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \frac{\pi'(a_t|s_t)}{\pi(a_t|s_t)}, \quad a_t \sim \pi(a_t|\theta)
$$

这里我们实际上将旧策略下估计得到的 $A^{GAE}_\lambda(s_t, a_t)$ 视作一个常数。

### 2. Restrictions

上述的损失函数可能会导致策略更新过大，从而导致训练不稳定。为了限制策略更新的幅度，PPO 引入了 KL 散度（Kullback-Leibler Divergence）来衡量新旧策略之间的差异，并且在损失函数中加入一个限制项来限制策略更新的幅度。

具体来说，KL 散度可以定义为：

$$
\begin{aligned}
D_{KL}(\pi_\theta || \pi_{\theta'}) &= \sum_a \pi(a|s_t) \ln \frac{\pi(a|s_t)}{\pi'(a|s_t)} \\
&= E_{a \sim \pi(a|s_t)} \left[ \ln \frac{\pi(a|s_t)}{\pi'(a|s_t)} \right]
\end{aligned}
$$

注意到 KL 散度是一个非负的值，当新旧策略完全相同时，KL 散度为零；当新旧策略差异较大时，KL 散度会变得较大。

有性质 $D_{KL}(P || Q) \neq D_{KL}(Q || P)$。

则模型的损失函数可以加上惩罚项：

$$
L(\theta') = -\frac{1}{N} \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \frac{\pi'(a_t|s_t)}{\pi(a_t|s_t)} + \beta D_{KL}(\pi_\theta || \pi_{\theta'})
$$

其中 $\beta$ 是一个超参数，用于控制 KL 散度的权重。通过调整 $\beta$ 的值，我们可以控制策略更新的幅度，从而提高训练的稳定性。

除了横向的 KL 散度限制之外，InstructGPT 中还引入了一个纵向的 KL 散度限制，即在每一步的 Reward 中加入一个 KL 散度的惩罚项来限制新旧策略之间的差异。

在此基础上，PPO 还引入了一个剪切（Clipping）机制来进一步限制策略更新的幅度。具体来说，我们可以定义一个剪切函数：

$$
\text{clip}(x, l, u) = \begin{cases}
l & \text{if } x < l \\
x & \text{if } l \leq x \leq u \\
u & \text{if } x > u
\end{cases}
$$

其实质上是对 $P'(a_t|s_t) / P(a_t|s_t)$ 进行剪切，限制其在 $[1 - \epsilon, 1 + \epsilon]$ 的范围内。

这种情况下，PPO 的 Actor 损失函数可以定义为：

$$
L(\theta') = -\frac{1}{N} \sum_{i=1}^N \sum_t \min \left( \frac{\pi'(a_t|s_t)}{\pi(a_t|s_t)}, \text{clip}\left( \frac{\pi'(a_t|s_t)}{\pi(a_t|s_t)}, 1 - \epsilon, 1 + \epsilon \right) \right) A^{GAE}_\lambda(s_t, a_t)
$$

### 3. Critic Loss

Critic 的损失函数通常是一个回归问题，用于最小化预测的状态价值函数与实际回报之间的误差。常用的损失函数是均方误差（Mean Squared Error, MSE）：

$$
L_{critic}(\phi) = \frac{1}{N} \sum_{i=1}^N \sum_t \left( V_\phi(s_t) - R_t \right)^2
$$

其中 $\phi$ 是 Critic 的参数，$R_t = A^{GAE}_\lambda(s_t, a_t) + V_{\text{old}}(s_t)$ 为回报目标（$V_{\text{old}}$ 需 detach 以停止梯度）。

Critic 学得越准，$A^{GAE}_\lambda(s_t, a_t)$ 的估计就越可靠，Actor 的策略更新就越稳定。所以 critic 本质上就是一个目标逼近真实价值函数的监督学习回归任务。

### 4. 熵奖励

为了防止策略过早收敛到一个局部最优解，PPO 还引入了一个熵（Entropy）项来鼓励策略的探索。熵可以定义为：

$$
H(\pi) = -\sum_a \pi(a|s_t) \ln \pi(a|s_t)
$$

熵值越高, 表示策略随机性越强, 探索性越强。

最终的目标函数是以上内容的加权和：

$$
L(\theta', \phi) = L_{actor}(\theta') + c_1 L_{critic}(\phi) - c_2 H(\pi)
$$

常见的超参数设置是 $c_1 = 0.5$ 和 $c_2 = 0.01$，但具体的值需要根据实际情况进行调整。

### PPO 的优缺点

#### 优点

1. PPO 通过限制策略更新的幅度，提高了训练的稳定性和效率。

2. 易于实现，超参数少，不需要复杂的二阶优化方法。

#### 缺点

1. PPO 虽然通过重要性采样提高了样本利用率，但本质上仍是 on-policy（仍需从当前策略采集新鲜数据），采集-训练交替进行，效率提升有限。

2. PPO 本质需要四个模型参与，因此模型的复杂度较高，训练资源消耗较大。

3. clipping 机制虽然限制了策略更新的幅度，但也可能导致策略更新过于保守，从而影响训练效果。

4. 超参数的选择对 PPO 的性能有较大影响，需要进行调优。
