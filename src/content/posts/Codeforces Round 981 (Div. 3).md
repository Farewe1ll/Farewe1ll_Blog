---
title: 'Codeforces Round 981 (Div. 3)'
published: 2024-10-25
description: 'Codeforces Round 981 (Div. 3) 解题报告'
image: ''
tags: ['Codeforces', 'Div. 3', 'C++', 'Competitive Programming']
category: 'Algorithm'
draft: false
series: 'Codeforces'
---

# [Codeforces Round 981 (Div. 3)](https://codeforces.com/contest/2033)

唉唉，这场搁楼道里打的，降温了忘添衣了，打得冻死了，咳嗽似乎还加重了。

水逆、衰。

## A. Sakurako and Kosuke

读题读完直觉就是判奇偶。何况依据题意双方路线固定。

复杂度 $\Theta(1)$。

代码如下：

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	int n; cin >> n;
	if (n & 1) cout << "Kosuke" << char(10);
	else cout << "Sakurako" << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) solve();
	return 0;
}
```

## B. Sakurako and Water

n范围很小。实际上就是求每一个斜行的最小的负数的总和。

复杂度 $\Theta(n)$。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)
#define per(i, a, b) for (decltype(b) i = (a); i >= b; --i)

inline void solve() {
	int n; cin >> n;
	vector< vector<int> > a(n + 10, vector<int>(n + 10));
	rep (i, 1, n) rep (j, 1, n)
        cin >> a[i][j];

	int ans = 0;
	rep (i, 1, n) {
		int mi = 0;
		for (int x = 1, y = i; x <= n && y <= n; ++x, ++y)
			if (a[x][y] < 0) mi = min(mi, a[x][y]);
		ans += -mi;
	}
	per (i, n, 2) {
		int mi = 0;
		for (int x = i, y = 1; x <= n && y <= n; ++x, ++y)
			if (a[x][y] < 0) mi = min(mi, a[x][y]);
		ans += -mi;
	}
	cout << ans << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) solve();
	return 0;
}
```

## C. Sakurako's Field Trip

一道莫名其妙通过率挺低的题。

实际上好像有dp解法，奈何我是dp苦手。

但是做题时贪心过的挺快的。

首先，注意到该题目对称性，故考虑 $i$ 与 $n - i + 1$ 的对称位置。

注意到对答案贡献的方式，不妨考虑相邻的对称位置，在确定思考方向是由中间向两边还是两边向中间的前提下，在已经确定一部分组的解时，考虑新的一组位置的贡献是无后效性的。

故考虑贪心。

实际上回头思考从两边向中间贪心和中间向两边贪心应该相似，因为考虑的第一组最外层或者最内层交换与否只是影响 $[L, mid - 1], [mid + 1, R]$ 是否需要对调。

实现的时候分奇偶了，实际上感觉或许不需要。

复杂度 $\Theta(n)$。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	int n; cin >> n;
	vector<int> a(n + 10);
	rep (i, 1, n) cin >> a[i];

	int mid = midpoint(1, n);
	auto check = [&](int i, int j) {
		return ((a[i] == a[i + 1]) + (a[j - 1] == a[j])) > ((a[j] == a[i + 1]) + (a[j - 1] == a[i]));
	};
	if (n & 1) { // odd
		for (int i = mid - 1, j = mid + 1; i >= 1 && j <= n; --i, ++j)
			if (check(i, j)) swap(a[i], a[j]);
	} else {
		for (int i = mid - 1, j = mid + 2; i >= 1 && j <= n; --i, ++j)
			if (check(i, j)) swap(a[i], a[j]);
	}

	int ans = 0;
	rep (i, 1, n - 1)
		if (a[i] == a[i + 1]) ++ans;
	cout << ans << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) solve();
	return 0;
}
```

## D. Kousuke's Assignment

做题时脑子抽了 insert(val) 写成 insert(a[i]) 错了就开始怀疑人生的一道题。。。

实际上思路没有太大毛病。

考虑如下转化：找到所有满足题设的 $[L, R]$，显然可以转化为给定区间，求最多多少区间覆盖问题。

则实际上我们只要发现出现相同的前缀和时就截断区间统计答案就好。如此可以尽可能让后方的区间有更多“空间”对答案进行贡献。

注意 Codeforces 需要 C++23 才能正常使用 __gnu_pbds。

复杂度 $\Theta(n)$。

```cpp
#include <bits/stdc++.h>
#include <bits/extc++.h>

using namespace std;
using namespace __gnu_pbds;

#define int long long
#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	int n; cin >> n;
	vector<int> a(n + 10);
	rep (i, 1, n) cin >> a[i];

	gp_hash_table<int, bool> hsh;
	int val = 0, ans = 0;
	hsh[0] = 1;
	rep (i, 1, n) {
		val += a[i];
		if (hsh.find(val) != hsh.end()) {
			hsh.clear(); hsh[0] = 1;
			++ans; val = 0;
		} else hsh[val] = 1;
	}

	cout << ans << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) solve();
	return 0;
}
```

## E. Sakurako, Kosuke, and the Permutation

额啊，是一道挺玄学的题。。。（仅仅对我个人而言）

听同学和jiangly群u说其实挺典的？（冒汗）

大致思路如下：

题意要求每个 $i$，都要满足自己构成自环或者二元环。故每次暴力的将点数大于二的环拆开即可。

以下是证明：拆环顺序与解的最优性无关 来自jiangly二群群友 @Arkweedy：

首先交换的两点必然是同一个点数大于二的环，在环内交换实际上是每次将一个环破成两个环，其中一个环点数为二。每次环点数个数减二，故分解次数和分解顺序或方式无关。

复杂度 $\Theta(n)$。

```cpp
#include <bits/stdc++.h>
#include <bits/extc++.h>

using namespace std;
using namespace __gnu_pbds;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	int n; cin >> n;
	vector<int> a(n + 10);
	gp_hash_table<int, int> hsh;
	rep (i, 1, n) { cin >> a[i]; hsh[a[i]] = i; }

	int ans = 0;
	rep (i, 1, n) {
		if (a[i] == i || a[a[i]] == i) continue;
		int x = a[a[i]];
		swap(a[a[i]], a[hsh[i]]);
		hsh[x] = hsh[i];
		hsh[i] = a[i];
		++ans;
	}
	cout << ans << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) solve();
	return 0;
}
```

## F. Kosuke's Sloth

实际上普遍认为这题比 E 简单。。。

显然在模 $k$ 意义下斐波那契数列仍然成立，不过将会退化成一个循环数列。

故每次直接将循环节暴力找出然后乘 $n$ 即可。

实际循环节长度不会很大，最大为 $6k$。

证明~~我也不会~~详见 [zky's blog](https://www.cnblogs.com/zkyJuruo/p/15714271.html)。

时间复杂度 $\Theta(k)$。

```cpp
#include <bits/stdc++.h>

using namespace std;
using i64 = long long;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

constexpr int P = 1e9 + 7;

inline void solve() {
	i64 n; int k;
	cin >> n >> k;
	vector<int> a(6 * k + 10);

	int len = 1;
	if (k == 1) goto lab;

	a[1] = 1 % k; a[2] = 1 % k;
	rep (i, 3, 6 * k) {
		a[i] = (1ll * a[i - 2] + a[i - 1]) % k;
		if (a[i] == 0) { len = i; break; }
	}

	lab:;
	n %= P;
	cout << (1ull * len * n) % P << endl;
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) solve();
	return 0;
}
```

## G. Sakurako and Chefir

啊啊啊鸽了没补

因为 LCA 和相关知识点还不很熟

等回头复习一遍图把这里补上 qwq

---

涩涩鹅说是求所有子树直径。先放这。