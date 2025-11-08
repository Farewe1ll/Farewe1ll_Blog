---
title: '2025 SSDUT 科技文化节程序设计竞赛 G'
published: 2025-04-26 #创建时间
updated: 2025-04-26 #更新时间
tags: [校赛, SSDUT, cpp, 笛卡尔树]
description: '2025 SSDUT 科技文化节程序设计竞赛 G 题解'
image: ''
category: 'Algorithm'
draft: false # 是否为草稿
series: 'SSDUT'
---

# [2025 SSDUT 科技文化节程序设计竞赛 G](https://hydro.ac/d/ssdut/p/O1043) 题解

本文提供一种*单调栈 + 笛卡尔树*解法。

## 起因 and 前置建树

由于当天看到题目某个杂鱼（）激动的觉得是 *36th CCF CSP E* 的类题，然后表示笛卡尔树可以秒了，结果补题时追求 $\Theta(n)$ 写成煞笔了。

感觉是很好的加深笛卡尔树结构与转移理解的题（如果是追求笛卡尔树解法）。

在此简要给出本人 **0idx** 建树代码。

```cpp
vector ls(n, n), rs(n, n); // n 表示无该子节点
vector<int> stk; stk.reserve(n);
FOR (i, 0, n) {
	int lst = n;
	while (!stk.empty() && a[i] > a[stk.back()]) {
		lst = stk.back(); stk.pop_back();
	}
	if (!stk.empty()) rs[stk.back()] = i;
	ls[i] = lst;
	stk.push_back(i);
}
int rt = max_element(a.begin(), a.end()) - a.begin();
```

## 36th CCF CSP E

首先讨论 [36th CCF CSP E](https://sim.csp.thusaac.com/contest/36/problem/4)，在同样使用笛卡尔树解法的前提下，个人觉得该题难度小于本题。

在使用笛卡尔树时，个人感觉大部分时候应该是 $\Theta(n)$，否则笛卡尔树一般可以被平替。我们通常可以在笛卡尔树每个节点上维护以该节点为根的子树的信息，该子树**一定对应原区间的一整段序列**，然后通过信息转移得到答案。一般使用**两次 dfs**，第一次预处理出**子树信息**，第二次转移得到**答案**。

则一般使用笛卡尔树需要思考：每个节点维护什么信息，如何转移得到答案。笛卡尔树上，**一个节点的父亲一定是当前节点左右单调栈中更弱的**，即大根堆是左右单调栈中较小的比自己大的元素，小根堆是左右单调栈中较大的比自己小的元素。故而梦魔这道题目节点信息为子树的 sumb，直接在父子节点之间转移即可。

```cpp
vector sum(n, 0ll);
auto dfs1 = [&](auto&& self, int u) -> i64 {
	sum[u] = b[u];
	for (auto v : {ls[u], rs[u]})
		if (v < n) sum[u] += self(self, v);
	return sum[u];
};
dfs1(dfs1, rt);

vector ans(n, 0);
auto dfs2 = [&](auto&& self, int u, int fa = -1) -> void {
	if (fa != -1) ans[u] = max(ans[fa], a[fa] - sum[u]);
	for (auto v : {ls[u], rs[u]})
		if (v < n) self(self, v, u);
};
dfs2(dfs2, rt);
```

## Solution

回到本题。每个节点维护的信息在本题并不难想，即为包含该节点的子树的点权和。但是在答案转移时，由于我们单次付出的**代价相同**（均为 1 颗糖果），故实际上转移需要注意为**当前节点左右单调栈中更强的**。虽然直接在父子之间转移的做法并不适用于本题。但是我们可以保证上述目标节点为当前节点的祖先，所以转移可行性是可以保证的。

我的解决思路是：预先通过两次单调栈预处理，找到每次转移的目标节点，直接转移即可。

```cpp
vector lm(n, n), rm(n, n), dst(n, n);; // 左侧、右侧比自己大的元素 idx 转移时的目标节点
stk.clear();
FOR (i, 0, n) {
	while (!stk.empty() && a[stk.back()] <= a[i]) stk.pop_back();
	if (!stk.empty()) lm[i] = stk.back();
	stk.push_back(i);
}
stk.clear();
ROF (i, n, 0) {
	while (!stk.empty() && a[stk.back()] <= a[i]) {
		stk.pop_back();
	}
	if (!stk.empty()) {
		rm[i] = stk.back();
	}
	stk.push_back(i);
}
FOR (i, 0, n) {
	if (lm[i] != n && rm[i] != n) {
		dst[i] = a[lm[i]] > a[rm[i]] ? lm[i] : rm[i];
	}
	if (lm[i] == n && rm[i] != n) {
		dst[i] = rm[i];
	}
	if (lm[i] != n && rm[i] == n) {
		dst[i] = lm[i];
	}
}
```

然后在 dfs2 时直接转移。

```cpp
vector ans(n, 0ll);
auto dfs2 = [&](auto&& self, int u) -> void {
	for (auto v : {ls[u], rs[u]})
		if (v < n) {
			ans[v] = ans[u];
			if (sum[v] < a[u]) ans[v] = ans[dst[v]] + 1;
			self(self, v);
		}
};
dfs2(dfs2, rt);
```

整体代码如下。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define FOR(i, a, b) for (auto i = (a); i < decltype(a)(b); ++i)
#define ROF(i, a, b) for (auto i = decltype(b)(a) - 1; i >= (b); --i)

using i64 = long long;

namespace IN {
	char buf[1 << 21], *p1 = buf, *p2 = buf;
	inline char getchar() { return p1 == p2 && (p2 = (p1 = buf) + cin.rdbuf()->sgetn(buf, 1 << 21), p1 == p2) ? EOF : *p1++; }
	inline void read(auto& p) {
		p = 0; char ch = getchar();
		for (; !isdigit(ch); ch = getchar());
		for (;  isdigit(ch); ch = getchar()) p = p * 10 + (ch ^ 48);
	}
	void read(auto& p, auto&... args) { read(p); read(args...); }
}
using IN::read;

namespace Farewe1ll {
	inline void solve() {
		int n; read(n);
		vector<int> a(n);
		FOR (i, 0, n) read(a[i]);

		// 构建笛卡尔树
		vector ls(n, n), rs(n, n);
		vector<int> stk; stk.reserve(n);
		FOR (i, 0, n) {
			int lst = n;
			while (!stk.empty() && a[i] > a[stk.back()]) {
				lst = stk.back();
				stk.pop_back();
			}
			if (!stk.empty()) rs[stk.back()] = i;
			ls[i] = lst;
			stk.push_back(i);
		}
		int rt = max_element(a.begin(), a.end()) - a.begin();

		// 第一次 dfs 预处理子树和
		vector sum(n, 0ll);
		auto dfs1 = [&](auto&& self, int u) -> i64 {
			sum[u] = a[u];
			for (auto v : {ls[u], rs[u]})
				if (v < n) sum[u] += self(self, v);
			return sum[u];
		};
		dfs1(dfs1, rt);

		// 找寻目标节点
		vector lm(n, n), rm(n, n), dst(n, n);
		stk.clear();
		FOR (i, 0, n) {
			while (!stk.empty() && a[stk.back()] <= a[i]) stk.pop_back();
			if (!stk.empty()) lm[i] = stk.back();
			stk.push_back(i);
		}
		stk.clear();
		ROF (i, n, 0) {
			while (!stk.empty() && a[stk.back()] <= a[i]) stk.pop_back();
			if (!stk.empty()) rm[i] = stk.back();
			stk.push_back(i);
		}
		FOR (i, 0, n) {
			if (lm[i] != n && rm[i] != n) dst[i] = a[lm[i]] > a[rm[i]] ? lm[i] : rm[i];
			if (lm[i] == n && rm[i] != n) dst[i] = rm[i];
			if (lm[i] != n && rm[i] == n) dst[i] = lm[i];
		}

		// 第二次 dfs 转移答案
		vector ans(n, 0ll);
		auto dfs2 = [&](auto&& self, int u) -> void {
			for (auto v : {ls[u], rs[u]})
				if (v < n) {
					ans[v] = ans[u];
					if (sum[v] < a[u]) ans[v] = ans[dst[v]] + 1;
					self(self, v);
				}
		};
		dfs2(dfs2, rt);

		for (auto it : ans) cout << it << ' ';
		cout << char(10);
	}
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T = 1; read(T);
	while (T--) Farewe1ll::solve();
	return 0;
}
```