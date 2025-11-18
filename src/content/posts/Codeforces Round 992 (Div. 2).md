---
title: 'Codeforces Round 992 (Div. 2) A~E'
published: 2025-01-14
description: 'Codeforces Round 992 (Div. 2) VP 解题报告'
image: ''
tags: ['Codeforces', 'Div. 2', 'cpp', 'Virtual Participation']
category: 'Algorithm'
draft: false
series: 'Codeforces'
---

# [Codeforces Round 992 (Div. 2)](https://codeforces.com/contest/2040) (Virtual Participation)

差点倒闭在 $C$

感觉打的也挺烂的

赛后补题补了半天的 $E$，期望会不了一点，哎哎

## A. Game of Division

非常简单的暴力判断。

赛时读假题还 WA 了一发，肥肠若只。

时间复杂度 $\Theta (n^2)$

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void Farewe1ll() {
	int n, k; cin >> n >> k;
	vector<int> a(n + 10);
	rep (i, 1, n) cin >> a[i];

	int ans = -1;
	rep (i, 1, n) {
		bool flg = true;
		rep (j, 1, n) {
			if (i == j) continue;
			if (abs(a[i] - a[j]) % k == 0) flg = false;
		}
		if (flg) ans = i;
	}

	if (ans != -1) cout << "YES\n" << ans << char(10);
	else cout << "NO\n";
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) Farewe1ll();
	return 0;
}
```

## B. Paint a Strip

略微有一丢丢意思的题目。

首先，其实可以考虑构造最优情况，因为显然随着序列长度的增加，所需要的第一类操作个数也会增加。

那不妨直接考虑在有限个第一类操作的前提下，所能构造的序列的最长长度。

题目样例解释中已经给定部分解释：

当只有 $1$ 次第一类操作时，显然能构造出来的最长长度为 $1$，即 $1$。

当只有 $2$ 次第一类操作时，显然能构造出来的最长长度为 $4$，即 $0000 \rightarrow 1000 \rightarrow 1001 \rightarrow 1111$。

当只有 $3$ 次第一类操作时，显然能构造出来的最长长度为 $10$，即 $1111 \rightarrow 1111000001 \rightarrow 1111111111$

后续可发现，每当多一次第一类操作，可构造出来的最长长度应为前一次长度的两倍加 $2$，即：

$$
f(x) =
\begin{cases}
		1, & x = 1 \\\\
		2, & x = 2 \\\\
		2 f(x - 1) + 2, & x > 2 \\\\
\end{cases}
$$

显然 $f(x)$ 是指数增长的，故可以简单打表发现到 $10^5$ 级别时 $x$ 约为 $18$。

赛时代码（与上述描述略有出入）如下：

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

vector<int> table;

inline void init() {
	table[1] = 1; table[2] = 2; table[3] = 4;
	rep (i, 4, 18)
		table[i] = 2 * table[i - 1] + 2;
}

inline void Farewe1ll() {
	int n; cin >> n;
	rep (i, 1, 18) {
		if (table[i] >= n) {
			if (i == 1 || i == 2) cout << i << char(10);
			else cout << i - 1 << char(10);
			return;
		}
	}
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	table.assign(20, 0);
	init();

	int T; cin >> T;
	while (T--) Farewe1ll();
	return 0;
}
```

## C. Ordered Permutations

很显然是典型的思维题。

按位考虑贡献，不难发现 $1$ 的贡献区间长度必须为 $n$，$2$ 的贡献区间长度为 $n - 1$ 等等。

故而可以发现符合 $S(p)_{max}$ 的排列必然满足如下两种构造：

1. 从大到小考虑每个数，其只能加在现有序列的左侧或右侧。

2. 从小到大考虑每个数，其只能加在长度为 $n$ 的序列的左端或右端指针处。

两者应该是等价的。具体 $2$ 的表述可能需要参考代码。但是个人感觉 $2$ 直观上便于字典序计数。

故而符合 $S(p)_{max}$ 的排列应当有 $2^{(n - 1)}$ 种，从小到大考虑数字可以便于计数。

需要注意~~沟槽的~~ long long 精度问题及其特判。

~~md哪天我就秽土转生变成高贵的 py 选手了（恼~~

```cpp
#include <bits/stdc++.h>

using namespace std;

#define int long long
#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void Farewe1ll() {
	int n; int k; cin >> n >> k;
	if (n <= 63 && (1ll << (n - 1)) < k) return void(cout << -1 << char(10));

	vector<int> ans(n + 10);
	int p1 = 1, p2 = n;
	int t = 1;
	while (k > 0) {
		if (t == n) { ans[p1] = t; break; }
		if (n - t > 63) { ans[p1++] = t++; continue; }
		if (k > (1ll << ((n - t) - 1ll))) {
			k -= 1ll << ((n - t) - 1ll);
			ans[p2--] = t++;
		} else ans[p1++] = t++;
	}
	rep (i, 1, n) cout << ans[i] << " \n"[i == n];
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) Farewe1ll();
	return 0;
}
```

## D. Non Prime Tree

比较开放的构造题。

从中可以学习到的知识点有：

1. 构造非素数时可以直接考虑 **非2** 的偶数。

2. 树一定是二分图。

下面给出构造方案：

从根出发，维护标号 $cur$，如果子节点的标号可以是 $cur + 1$ 即钦定。否则 $cur$ 扩大到差值为偶数非 $2$ 的值。

可以证明，该种构造不会爆 $2n$。~~证明我也不会 guessforces全靠感觉（~~

如若利用 $2$，也可进行层次遍历然后标号。具体此处不展开。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void Farewe1ll() {
	int n; cin >> n;
	vector g(n + 10, vector<int>());

	rep (i, 1, n - 1) {
		int x, y;
		cin >> x >> y;
		g[x].push_back(y);
		g[y].push_back(x);
	}

	vector<int> ans(n + 10, 0);

	int cur = 1;
	auto dfs = [&](this auto &&dfs, int fa, int dep) -> void {
		for (auto it : g[dep]) {
			if (it == fa) continue;
			++cur;
			if (cur != ans[dep]) {
				if ((cur - ans[dep]) & 1) ++cur;
				if (cur - ans[dep] == 2) cur += 2;
			}
			ans[it] = cur;
			dfs(dep, it);
		}
	};
	ans[1] = 1;
	dfs(0, 1);
	rep (i, 1, n) cout << ans[i] << " \n"[i == n];
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) Farewe1ll();
	return 0;
}
```

## E. Control of Randomness

感觉题目只有差不多 1700~1800（?

但是我是 期望苦手 雾

首先这题的数据范围基本上暴力是完全足够的

[EternalEpic](https://codeforces.com/profile/EternalEpic) 和 [International_River](https://codeforces.com/profile/International_River) 老师讨论时指出其实完全可以离线之后做到极优的复杂度

首先可以预处理出各个点到 $1$ 的距离作为答案的基础贡献。对于走路径的过程而言，除了该基础贡献外，还有由于可能的游走所产生的额外贡献。一个直观的感觉是，儿子更多的节点更容易发生游走，故而自然且合理的一个处理就是每次直接处理出偶数步所对应的节点中前 $p$ 个节点，打上标记，之后的贡献计算中视为奇数步所对应的节点。

考虑到直接计算每个点单独的贡献会使得答案的计算与转移变得即为复杂，不妨考虑单独计算每个节点的额外贡献。

对于位于奇数步上或者打上标记的节点，显然其额外的贡献即为其转移而来的子节点的贡献，故而可直接列出其额外贡献的期望方程为

$$
dp(fa_u) = dp(u),\space{} u \in odd \vee tag(u)
$$

而对于偶数步上且未被打上标记的节点，其额外贡献的期望方程为

$$
dp(u) = \frac{deg(u) - 1}{deg(u)} * (dp(u) + 2) + \frac{1}{deg(u)} * dp(fa_u),\space{} u \in even \wedge \neg tag(u)
$$

化简可得

$$
dp(fa_u) = dp(u) + 2 * (deg(u) - 1),\space{} u \in even \wedge \neg tag(u)
$$

如果题目可以离线，似乎也可以将询问离线，然后 dfs 树上每个点时处理该点的所有查询，使用权值树状数组维护答案即可。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

using pii = pair<int, int>;

constexpr int P = 998244353;

void Farewe1ll() {
	int n, q; cin >> n >> q;
	vector g(n + 10, vector<int>());
	rep (i, 1, n - 1) {
		int x, y; cin >> x >> y;
		g[x].push_back(y);
		g[y].push_back(x);
	}

	vector<int> fa(n + 10, 0), dist(n + 10, 0);
	auto dfs = [&](this auto &&dfs, int rt, int dep) -> void {
		for (auto &it : g[dep]) {
			if (it == rt) continue;
			fa[it] = dep; dist[it] = dist[dep] + 1;
			dfs(dep, it);
		}
	};
	dfs(0, 1);

	while (q--) {
		int v, p; cin >> v >> p;

		vector<int> dp(n + 1, 0);
		vector<pii> son;
		vector<char> tag(n + 10, 0);
		son.reserve(n);
		for (int t = v, cnt = 1; fa[t]; t = fa[t], ++cnt)
			if (!(cnt & 1)) son.push_back({g[t].size(), t});
		sort(son.begin(), son.end(), greater<pii>());
		for (auto &[x, y] : son) {
			if (!p--) break;
			tag[y] = 1;
		}

		for (int t = v, cnt = 1; fa[t]; t = fa[t], ++cnt)
			if (!(cnt & 1) && !tag[t])
				dp[fa[t]] = (1ll * dp[t] + ((g[t].size() - 1) << 1) % P) % P;
			else dp[fa[t]] = dp[t];

		cout << (1ll * dist[v] + dp[1]) % P << char(10);
	}
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) Farewe1ll();
	return 0;
}
```