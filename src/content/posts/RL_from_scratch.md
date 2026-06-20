---
title: 从零开始的 RL 与 RLHF —— PPO、DPO 与 GRPO
published: 2026-06-18
description: 'RL 与 RLHF 的入门学习笔记'
image: ''
tags: ['Deep Learning', 'RL', 'RLHF']
category: 'Research'
draft: false
lang: ''
series: 'Reinforcement Learning'
---

# 从零开始的 RL 与 RLHF

## RL 基础：环境、状态、动作、奖励

### 1. 环境

环境是智能体（Agent）进行交互的外部世界，定义了智能体可以感知和影响的对象。

### 2. 状态

状态是环境在某一时刻的具体描述，包含了智能体所能感知到的所有信息。状态可以是完全可观测的（如棋盘游戏中的棋盘状态），也可以是部分可观测的（如机器人只能通过摄像头看到周围环境的一部分）。

### 3. 动作

动作是智能体在环境中可以执行的操作，动作的选择会影响环境的状态变化。动作可以是离散的（如向左、向右、跳跃）或连续的（如调整速度、改变方向）。

### 4. 奖励

奖励是智能体在环境中执行动作后获得的反馈信号，用于指导智能体学习。奖励可以是正的（表示好的行为）或负的（表示坏的行为）。智能体的目标是最大化累积奖励，即通过学习找到最优策略来选择动作，以获得最大的长期回报。

- 例子：在一个迷宫环境中，智能体的状态可能是它在迷宫中的位置，动作可能是向上、向下、向左、向右移动，而奖励可能是在到达目标位置时获得正奖励，在撞墙时获得负奖励。智能体通过不断尝试不同的动作来学习如何在迷宫中找到最短路径以最大化奖励。

## 马尔可夫决策过程（MDP）

马尔可夫决策过程（Markov Decision Process, MDP）是描述强化学习问题的数学框架，定义了智能体与环境交互的基本要素。其中需要注意以下几个概念：

### 1. 动作空间

动作空间是智能体在环境中可以选择的所有可能动作的集合。动作空间可以是离散的（如在棋盘游戏中，动作空间可能包括向上、向下、向左、向右四个动作）或连续的（如在机器人控制中，动作空间可能包括调整速度和方向的连续值）。智能体通过选择动作来与环境进行交互，并根据环境的反馈来学习最优策略。

- 如 $\{left, right, up, down\}$ 就是一个离散的动作空间，而调整速度和方向的连续值则构成了一个连续的动作空间。

### 2. 策略函数

策略函数（Policy Function）是智能体在给定状态下选择动作的规则或函数。输入状态，输出动作的概率分布。

- 如 $\pi(s) = a$ 表示在状态 $s$ 下选择动作 $a$ 的确定性策略，而 $\pi(a|s)$ 表示在状态 $s$ 下以概率 $\pi(a|s)$ 选择动作 $a$ 的随机策略。

- 注意到环境可能也具有随机性，此时我们可以将状态转移概率 $P(s'|s, a)$ 定义为在状态 $s$ 下执行动作 $a$ 后转移到状态 $s'$ 的概率。

### 3. 轨迹

轨迹（Trajectory）是智能体在环境中执行一系列动作后所经历的状态和奖励的序列。通常用 $\tau$ 表示。一个轨迹通常表示为一个时间序列，其中包含了智能体在每个时间步的状态、所选择的动作以及获得的奖励。通过分析轨迹，智能体可以学习如何调整策略以最大化累积奖励。

- 如一个轨迹可能表示为：$(s_0, a_0, r_1, s_1, a_1, r_2, s_2, ...)$，其中 $s_t$ 是时间步 $t$ 的状态，$a_t$ 是在状态 $s_t$ 下选择的动作，$r_{t+1}$ 是执行动作 $a_t$ 后获得的奖励。

### 4. 回报

回报（Return）是智能体在一个轨迹中获得的累积奖励的总和。

- 回报可以是所有奖励的总和，此时表示为 $R = \sum_{t=0}^T \gamma^t r_{t+1}$，其中 $\gamma$ 是折扣因子，取值范围在 $[0, 1]$ 之间，用于权衡当前奖励和未来奖励的重要性。注意折扣因子在整个学习过程中应该统一。

- 也可以是未来奖励的期望值，此时表示为 $G_{t'} = \mathbb{E}[\sum_{t=t'}^T \gamma^{t-t'} r_{t+1}]$，其中期望值是基于智能体在环境中执行动作的概率分布计算的。

此时我们可以注意到，回报的定义应该可以满足自洽性（Consistency），即在任何时间步 $t$，回报 $G_t$ 应该等于当前奖励 $r_{t+1}$ 加上未来回报的折扣值 $\gamma G_{t+1}$，即 $G_t = r_{t+1} + \gamma G_{t+1}$。

从而我们可以得到回报的递归定义 $G_t = r_{t+1} + \gamma G_{t+1}$，此时的 $G_{t + 1}$ 通常可以采取期望或者采样的方式获取。

## 在 MDP 上定义的价值函数

在 MDP 过程中，价值函数（Value Function）是用来评估在特定状态下智能体的表现的函数。它可以帮助智能体了解在当前状态下采取某个动作可能带来的长期回报，从而指导智能体选择最优策略。

### 1. 状态价值函数

状态价值函数（State Value Function）是指在给定状态下，智能体在遵循某一策略的前提下，所能获得的期望回报。通常用 $V^\pi(s)$ 表示，其中 $\pi$ 是策略，$s$ 是状态。

我们可以通过贝尔曼方程（Bellman Equation）来定义状态价值函数：

$$
\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi[G_t | S_t = s]
\\ &= \mathbb{E}_\pi[r_{t+1} + \gamma G_{t+1} | S_t = s]
\\ &= \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [r(s, a, s') + \gamma V^\pi(s')]
\end{aligned}
$$

公式的第一行为状态价值函数的定义，第二行为回报的递归定义，第三行为将期望值展开成动作和状态转移的概率分布，可以理解为在状态 $s$ 下，智能体选择动作 $a$ 的概率为 $\pi(a|s)$，执行动作 $a$ 后转移到状态 $s'$ 的概率为 $P(s'|s, a)$，在状态 $s$ 下执行动作 $a$ 后获得的奖励为 $r(s, a, s')$，同时未来回报的折扣值为 $\gamma V^\pi(s')$。

相似的，状态价值函数也满足自洽性（Consistency），即在任何状态 $s$，状态价值函数 $V^\pi(s)$ 应该等于当前奖励 $r(s, a, s')$ 加上未来回报的折扣值 $\gamma V^\pi(s')$ 的期望值，即 $V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [r(s, a, s') + \gamma V^\pi(s')]$。

此时如果对下一步的状态函数进行采样，我们可以得到状态价值函数的采样版本 $V^\pi(s) \approx r(s, a, s') + \gamma V^\pi(s')$，其中 $s'$ 是在状态 $s$ 下执行动作 $a$ 后转移到的状态。

### 2. 动作价值函数

动作价值函数（Action Value Function）是指在给定状态和动作的前提下，智能体在遵循某一策略的前提下，所能获得的期望回报。通常用 $Q^\pi(s, a)$ 表示，其中 $\pi$ 是策略，$s$ 是状态，$a$ 是动作。

我们可以通过贝尔曼方程来定义动作价值函数：

$$
\begin{aligned}
Q^\pi(s, a) &= \mathbb{E}_\pi[G_t | S_t = s, A_t = a] \\
&= \mathbb{E}_\pi[r_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
&= \sum_{s'} P(s'|s, a) \left[ r(s, a, s') + \gamma V^\pi(s') \right] \\
&= \sum_{s'} P(s'|s, a) [r(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]
\end{aligned}
$$

由上我们也可以发现 $V^\pi(s') = \sum_{a'} \pi(a'|s') Q^\pi(s', a')$。

公式的第一行为动作价值函数的定义，第二行为回报的递归定义，第三行为将期望值展开成状态转移的概率分布，可以理解为在状态 $s$ 下执行动作 $a$ 后转移到状态 $s'$ 的概率为 $P(s'|s, a)$，在状态 $s$ 下执行动作 $a$ 后获得的奖励为 $r(s, a, s')$，同时未来回报的折扣值为 $\gamma V^\pi(s')$。第四行进一步将 $V^\pi(s')$ 展开为 $\sum_{a'} \pi(a'|s') Q^\pi(s', a')$，表示到达状态 $s'$ 后继续按照策略 $\pi$ 选择动作所得到的期望动作价值。

### 3. 优势价值函数

优势价值函数（Advantage Value Function）是指在给定状态和动作的前提下，该动作价值相对于当前策略在该状态下平均水平的差值。通常用 $A^\pi(s, a)$ 表示，其中 $\pi$ 是策略，$s$ 是状态，$a$ 是动作。

我们可以通过以下公式来定义优势价值函数：

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

### 4. 各个价值函数之间的关系

我们可以通过以下公式来总结各个价值函数之间的关系：

$$
\begin{aligned}
V^\pi(s) &= \sum_{a} \pi(a|s) Q^\pi(s, a) \\
Q^\pi(s, a) &= \sum_{s'} P(s'|s, a) [r(s, a, s') + \gamma V^\pi(s')] \\
A^\pi(s, a) &= Q^\pi(s, a) - V^\pi(s)
\end{aligned}
$$

如果在忽略环境随机性并且采用采样的方式获取下一步状态函数，我们可以得到以下关系：

$$
\begin{aligned}
V^\pi(s) &\approx r(s, a, s') + \gamma V^\pi(s') \\
Q^\pi(s, a) &= r(s, a, s') + \gamma V^\pi(s') \\
A^\pi(s, a) &= r(s, a, s') + \gamma V^\pi(s') - V^\pi(s)
\end{aligned}
$$

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
& \frac1N \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \nabla_{\theta'} \ln P'(a_t|s_t) \\
=& \frac1N \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \frac{P'(a_t|s_t)}{P(a_t|s_t)} \nabla_{\theta'} \ln P'(a_t|s_t) \\
=& \frac1N \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \frac{\nabla_{\theta'} P'(a_t|s_t)}{P(a_t|s_t)} \\
\end{aligned}
$$

注意到此处的 $P(a_t|s_t)$ 是旧策略下的概率，而 $P'(a_t|s_t)$ 是新策略下的概率。我们是在新策略下进行优化的，因此需要使用新策略下的概率来调整旧策略下的优势价值函数。

故此时的 Actor 损失函数可以定义为：

$$
L(\theta') = -\frac{1}{N} \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \frac{P'(a_t|s_t)}{P(a_t|s_t)}, \quad a_t \sim P(a_t|\theta)
$$

这里我们实际上将旧策略下估计得到的 $A^{GAE}_\lambda(s_t, a_t)$ 视作一个常数。

### 2. Restrictions

上述的损失函数可能会导致策略更新过大，从而导致训练不稳定。为了限制策略更新的幅度，PPO 引入了 KL 散度（Kullback-Leibler Divergence）来衡量新旧策略之间的差异，并且在损失函数中加入一个限制项来限制策略更新的幅度。

具体来说，KL 散度可以定义为：

$$
\begin{aligned}
D_{KL}(\pi_\theta || \pi_{\theta'}) &= \sum_a P(a|s_t) \ln \frac{P(a|s_t)}{P'(a|s_t)} \\
&= E_{a \sim P(a|s_t)} \left[ \ln \frac{P(a|s_t)}{P'(a|s_t)} \right]
\end{aligned}
$$

注意到 KL 散度是一个非负的值，当新旧策略完全相同时，KL 散度为零；当新旧策略差异较大时，KL 散度会变得较大。

有性质 $D_{KL}(P || Q) \neq D_{KL}(Q || P)$。

则模型的损失函数可以加上惩罚项：

$$
L(\theta') = -\frac{1}{N} \sum_{i=1}^N \sum_t A^{GAE}_\lambda(s_t, a_t) \frac{P'(a_t|s_t)}{P(a_t|s_t)} + \beta D_{KL}(\pi_\theta || \pi_{\theta'})
$$

其中 $\beta$ 是一个超参数，用于控制 KL 散度的权重。通过调整 $\beta$ 的值，我们可以控制策略更新的幅度，从而提高训练的稳定性。

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
L(\theta') = -\frac{1}{N} \sum_{i=1}^N \sum_t \min \left( \frac{P'(a_t|s_t)}{P(a_t|s_t)}, \text{clip}\left( \frac{P'(a_t|s_t)}{P(a_t|s_t)}, 1 - \epsilon, 1 + \epsilon \right) \right) A^{GAE}_\lambda(s_t, a_t)
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
H(\pi) = -\sum_a P(a|s_t) \ln P(a|s_t)
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

## DPO

DPO（Direct Preference Optimization）是一种直接优化偏好（Preference）的强化学习算法，将从偏好数据中学习奖励，再用 RL 优化的策略转变为一个关于策略（policy）参数的最大似然/二分类损失，因此不需要显式训练一个 reward 模型，也不需要在训练过程中用强化学习采样/策略梯度，DPO 用一个参考模型（reference policy）来做 KL/正则化的角色，并通过一个可控系数 β 调节“偏离参考模型”的强度，从而大大降低了训练的成本和难度。

### Bradley-Terry 模型

Bradley-Terry 模型是一种用于建模偏好数据的概率模型，常用于二分类问题中。该模型假设每个选项都有一个潜在的非负“能力”值，偏好关系由这些能力值决定。

具体来说，假设有两个选项 A 和 B，它们的能力值分别为 $w_A$ 和 $w_B$，则选项 A 被偏好于选项 B 的概率可以表示为：

$$
P(A \succ B) = \frac{w_A}{w_A + w_B} = \frac{1}{1 + e^{-(w_A - w_B)}}
$$

一个简单的例子：假设 A 对 B 历史战绩 8 胜 4 负，A 对 C 历史战绩 3 胜 5 负，求 B 战胜 C 的概率。

现实中的情况往往有多个选项和复杂的偏好关系，我们可以取对数最大似然估计来求解能力值 $w_A$、$w_B$ 和 $w_C$，从而计算 B 战胜 C 的概率。

似然函数可以表示为：

$$
L(w_A, w_B, w_C) = \prod_{i} P(A_i \succ B_i)^{8} P(B_i \succ A_i)^{4} P(A_i \succ C_i)^{3} P(C_i \succ A_i)^{5}
$$

带入 Bradley-Terry 模型的概率表达式，我们可以得到：

$$
L(w_A, w_B, w_C) = \prod_{i} \left(\frac{w_A}{w_A + w_B}\right)^{8} \left(\frac{w_B}{w_A + w_B}\right)^{4} \left(\frac{w_A}{w_A + w_C}\right)^{3} \left(\frac{w_C}{w_A + w_C}\right)^{5}
$$

我们对其取对数得到对数似然函数：

$$
\begin{aligned}
\ln L(w_A, w_B, w_C) &= 8 \ln \frac{w_A}{w_A + w_B} + 4 \ln \frac{w_B}{w_A + w_B} + 3 \ln \frac{w_A}{w_A + w_C} + 5 \ln \frac{w_C}{w_A + w_C} \\
&= 8 \ln w_A - 8 \ln (w_A + w_B) + 4 \ln w_B - 4 \ln (w_A + w_B) + 3 \ln w_A - 3 \ln (w_A + w_C) + 5 \ln w_C - 5 \ln (w_A + w_C) \\
&= 11 \ln w_A + 4 \ln w_B + 5 \ln w_C - 12 \ln (w_A + w_B) - 8 \ln (w_A + w_C)
\end{aligned}
$$

要使其最大化，我们可以对 $w_A$、$w_B$ 和 $w_C$ 分别求偏导数，并设置为零。

最后得到

$$
\begin{aligned}
\frac{\partial \ln L}{\partial w_A} &= \frac{11}{w_A} - \frac{12}{w_A + w_B} - \frac{8}{w_A + w_C} = 0 \\
\frac{\partial \ln L}{\partial w_B} &= \frac{4}{w_B} - \frac{12}{w_A + w_B} = 0 \\
\frac{\partial \ln L}{\partial w_C} &= \frac{5}{w_C} - \frac{8}{w_A + w_C} = 0
\end{aligned}
$$

解得

$$
\begin{aligned}
w_A &= 2w_B \\
w_C &= (10/3)w_B
\end{aligned}
$$

从而

$$
P(B \succ C) = \frac{w_B}{w_B + w_C} = \frac{1}{1 + (10/3)} = \frac{3}{13} \approx 0.23
$$

除去求偏导，我们也可以通过 Bradley-Terry 模型一般的 Loss 函数来求解能力值：

$$
LOSS = - E_{(A, B) \sim D} \left[ \ln \frac{w_A}{w_A + w_B} \right]
$$

上式中，$D$ 是偏好数据的分布，$(A, B)$ 是从数据中采样的一对选项，$w_A$ 和 $w_B$ 分别是选项 A 和 B 的能力值。通过最小化这个损失函数，我们可以得到能力值的估计，从而计算 B 战胜 C 的概率。

上式就是一个二分类的交叉熵损失函数，本质是负对数似然。

在 RL 中，Reward 模型的评估结果可以看作是选项的能力值，而偏好数据则是选项之间的比较结果。通过最小化上述的损失函数，我们可以直接优化策略参数，使得模型生成的内容更符合偏好。

需要注意，我们通常会在 Reward 模型的评估结果上加上指数函数来确保能力值为正数。

此时偏序的概率可以表示为：

$$
P(y_A \succ y_B) = \frac{e^{r(x, y_A)}}{e^{r(x, y_A)} + e^{r(x, y_B)}}
$$

上式为我们给定输入 Prompt x 的情况下，生成内容 $y_A$ 被偏好于生成内容 $y_B$ 的概率。

注意到 sigmoid 函数的定义为 $\sigma(z) = \frac{1}{1 + e^{-z}}$，结合上式可以得到 Loss 函数为：

$$
\begin{aligned}
LOSS &= - E_{(x, y_A, y_B) \sim D} \left[ \ln \frac{e^{r(x, y_A)}}{e^{r(x, y_A)} + e^{r(x, y_B)}} \right] \\
&= - E_{(x, y_A, y_B) \sim D} \left[ \ln \sigma(r(x, y_A) - r(x, y_B)) \right]
\end{aligned}
$$

上式中，$D$ 是偏好数据的分布，$(x, y_A, y_B)$ 是从数据中采样的一组输入和两个生成内容，$r(x, y_A)$ 和 $r(x, y_B)$ 分别是 Reward 模型对生成内容 $y_A$ 和 $y_B$ 的评估结果。通过最小化这个损失函数，我们可以直接优化策略参数，使得模型生成的内容更符合人类的偏好，从而实现 RLHF 的目标。

### DPO 的训练目标

我们引入如下几个关键元素：

- 奖励函数 $r(x, y)$：用于评估生成内容的质量，通常是一个神经网络模型。

- 基准模型 $\pi_{ref}(y|x)$：一个预训练的语言模型，用于提供生成内容的概率分布。

- 训练模型 $\pi_\theta(y|x)$：我们希望通过训练来优化的模型。

则 DPO 的训练目标可以定义为：

$$
\max_{\pi} E_{x \sim D, y \sim \pi} \left[ r(x, y) - \beta D_{KL}(\pi(y|x) || \pi_{ref}(y|x)) \right]
$$

即我们希望最大化生成内容的奖励，同时最小化生成内容与基准模型之间的 KL 散度，以确保生成内容不会偏离基准模型太远。在这里的 $\beta$ 是一个超参数，用于控制奖励和 KL 散度之间的权重。

进一步推导，得到

$$
\begin{aligned}
& \max_{\pi} E_{x \sim D, y \sim \pi} \left[ r(x, y) - \beta D_{KL}(\pi(y|x) || \pi_{ref}(y|x)) \right]\\
=& \max_{\pi} E_{x \sim D, y \sim \pi} r(x, y) - E_{x \sim D, y \sim \pi} \beta \ln \frac{\pi(y|x)}{\pi_{ref}(y|x)} \\
=& \max_{\pi} E_{x \sim D, y \sim \pi} \left[ r(x, y) - \beta \ln \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right] \\
=& \min_{\pi} E_{x \sim D, y \sim \pi} \left[ - \frac1{\beta} r(x, y) + \ln \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right] \\
=& \min_{\pi} E_{x \sim D, y \sim \pi} \left[ \ln \frac{\pi(y|x)}{\pi_{ref}(y|x)} - \ln \exp \left( \frac1{\beta} r(x, y) \right) \right] \\
=& \min_{\pi} E_{x \sim D, y \sim \pi} \left[ \ln \frac{\pi(y|x)}{\pi_{ref}(y|x) \exp \left( \frac1{\beta} r(x, y) \right)} \right] \\
=& \min_{\pi} E_{x \sim D, y \sim \pi} \left[ \ln \frac{\pi(y|x)}{\pi_{ref}(y|x) \exp \left( \frac1{\beta} r(x, y) \right) \frac1{Z(x)} Z(x)} \right] \\
=& \min_{\pi} E_{x \sim D, y \sim \pi} \left[ \ln \frac{\pi(y|x)}{\frac1{Z(x)} \pi_{ref}(y|x) \exp \left( \frac1{\beta} r(x, y) \right)} - \ln Z(x) \right] \\
\end{aligned}
$$

在这里的 $Z(x)$ 是一个归一化常数，定义为：

$$
Z(x) = \int \pi_{ref}(y|x) \exp \left( \frac1{\beta} r(x, y) \right) dy
$$

目的是将分母同样转化为一个概率分布的形式，从而使得损失函数可以看作是一个 KL 散度的形式。

故对于此时的分母，展开得

$$
\begin{aligned}
& \frac1{Z(x)} \pi_{ref}(y|x) \exp \left( \frac1{\beta} r(x, y) \right) \\
=& \frac{\pi_{ref}(y|x) \exp \left( \frac1{\beta} r(x, y) \right)}{\int \pi_{ref}(y|x) \exp \left( \frac1{\beta} r(x, y) \right) dy}
\end{aligned}
$$

对于分母而言，其分子是在给定输入 $x$ 的情况下，生成内容 $y$ 的概率分布乘以一个与奖励函数相关的权重。分母则是对所有可能的生成内容 $y$ 进行积分，从而整个分母可以看作是一个新的概率分布，表示在给定输入 $x$ 的情况下，生成内容 $y$ 的概率分布经过奖励函数调整后的分布。我们可将其记为 $\pi^*(y|x)$。

从而训练目标可以看作是最小化 $\pi(y|x)$ 和 $\pi^*(y|x)$ 之间的 KL 散度：

$$
\begin{aligned}
& \min_{\pi} E_{x \sim D, y \sim \pi} \left[ \ln \frac{\pi(y|x)}{\pi^*(y|x)} - \ln Z(x) \right] \\
=& \min_{\pi} E_{x \sim D} \left[ D_{KL}(\pi(y|x) || \pi^*(y|x)) - \ln Z(x) \right]
\end{aligned}
$$

这里的 $\ln Z(x)$ 由于不依赖于 $\pi$，因此在优化过程中可以被忽略掉。最终的训练目标可以简化为：

$$
\min_{\pi} E_{x \sim D} \left[ D_{KL}(\pi(y|x) || \pi^*(y|x)) \right]
$$

由于 KL 散度的非负性，我们可以得出上述优化问题的最优情况为 KL 散度为零，即 $\pi(y|x) = \pi^*(y|x)$。因此，DPO 的训练目标可以看作是让训练模型 $\pi_\theta(y|x)$ 尽可能地接近一个由基准模型 $\pi_{ref}(y|x)$ 和奖励函数 $r(x, y)$ 共同定义的目标分布 $\pi^*(y|x)$。

关于新的目标分布 $\pi^*(y|x)$，我们可以变换得到其 Reward 的评估结果：

$$
\begin{aligned}
& \pi(y|x) = \frac1{Z(x)} \pi_{ref}(y|x) \exp \left( \frac1{\beta} r(x, y) \right) \\
\implies & \exp \left(\frac1{\beta} r(x, y) \right) = \frac{\pi(y|x)}{\pi_{ref}(y|x)} Z(x) \\
\implies & r(x, y) = \beta \ln \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \ln Z(x)
\end{aligned}
$$

由上文的 Bradley-Terry 模型，我们期望将 $\pi^*(y|x)$ 与 $\pi_{ref}(y|x)$ 进行比较，从而得到以下 LOSS 函数：

$$
\begin{aligned}
LOSS &= - \ln \sigma(r(x, y_A) - r(x, y_B)) \\
&= - \ln \sigma\left( \beta \ln \frac{\pi(y_A|x)}{\pi_{ref}(y_A|x)} + \beta \ln Z(x) - \beta \ln \frac{\pi(y_B|x)}{\pi_{ref}(y_B|x)} - \beta \ln Z(x) \right) \\
&= - \ln \sigma\left( \beta \ln \frac{\pi(y_A|x)}{\pi_{ref}(y_A|x)} - \beta \ln \frac{\pi(y_B|x)}{\pi_{ref}(y_B|x)} \right) \\
\end{aligned}
$$
