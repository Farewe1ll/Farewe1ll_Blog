---
title: "Codeforces Round 982 (Div. 2)"
published: 2024-10-27 #创建时间
updated: 2024-10-30 #更新时间
tags: [Codeforces, Div. 2, cpp]
description: "Codeforces Round 982 (Div. 2) 解题报告" #描述
image: ''
category: 'Algorithm'
draft: false # 是否为草稿
series: "Codeforces"
---

# [Codeforces Round 982 (Div. 2)](https://codeforces.com/contest/2027)

这场前半部分打得还行，可惜卡 D 了。

差不多算是想到做法了，但是调的有点糟。。。

哎 dp

## A. Rectangle Arrangement

看题没看懂在干啥，但是读样例解释看懂了x

考虑每一个可能的形状，通过平移它的边，其周长必然可以等价转换为一个矩形。

故我们实际上考虑的是这个矩形怎么维护。

很显然，矩形长是所有 stamp 的长的最大值，宽是所有 stamp 宽的最大值。

故直接维护即可。

复杂度 $\Theta(n)$

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	int n; cin >> n;
	int w = 0, h = 0;
	rep (i, 1, n) {
		int x, y; cin >> x >> y;
		w = max(w, x); h = max(y, h);
	}
	cout << 2 * (w + h) << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) solve();
	return 0;
}
```

## B. Stalin Sort

草 Cf better 加载出翻译的时候我人笑飞了

题意疑似有点地狱笑话

首先读完题意之后我是一头雾水基本没有任何思路的

但是可以思考样例解释，想为什么第一个样例删去的是 $3$ 和 $9$

如果我不要求是最优解，能否有其他删法呢

如果要保留 $9$，则会发现前面的 $3$、$6$、$4$ 都要删掉。

所以不难发现，题意就是要找出原数组的一个子数组，使得所有的元素都小于等于第一个元素，并且这个子数组的长度最长。

在这样的前提下，我们很容易就想到关键在于第一个元素是谁。

所以不妨倒序遍历原数组，每个数提供一个可能的答案，求答案的最小值。

每个数提供的答案的值就应该是从其位置往后所有的大于这个数的元素的个数加上这个数前面的元素的个数。

实际上赛时写了一个简易 vector 平衡树草过去了，实际上我也觉得它肘不动我。

upd: 草 以为数据范围是 $10^5$，没想到只有 $2000$。。。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)
#define per(i, a, b) for (decltype(b) i = (a); i >= b; --i)

inline void solve() {
	int n; cin >> n;
	vector<int> a(n + 10);

	rep (i, 1, n) cin >> a[i];
	vector<int> s;
	int ans = INT_MAX;
	per (i, n, 1) {
		int val = i - 1;
		val += s.end() - upper_bound(s.begin(), s.end(), a[i]);
		ans = min(ans, val);
		s.insert(lower_bound(s.begin(), s.end(), a[i]), a[i]);
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

## C. Add Zeros

我们的目标是数组长度的最大值，故而对于某种情形，我们最关心的应该是当前数组的长度。

故而考虑每次操作实际上是在 $\lvert a \rvert = a_i + i - 1$ 时，可以选择将当前数组长度增加 $i - 1$。

因为是往数组末尾添加 $0$，故不用考虑某次操作后可选择的操作增多。考虑到达某个数组长度时，我们并不关心是如何到达的，而可以直接以当前状态拓展其余状态，可以直接记忆化。

注意 **long long**。

时间复杂度 $\Theta(guess)$

```cpp
#include <bits/stdc++.h>
#include <bits/extc++.h>

using namespace std;
using namespace __gnu_pbds;

#define int long long
#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};

inline void solve() {
	int n; cin >> n;
	vector<int> a(n + 10);
	rep (i, 1, n) cin >> a[i];

	// 贡献 i - 1, 所需长度 a_i + i - 1
	gp_hash_table<int, vector<int>, custom_hash> mp;
	rep (i, 2, n)
		mp[a[i] + i - 1].push_back(i - 1);

	gp_hash_table<int, bool, custom_hash> len;
	int ans = n;

	auto dfs = [&](auto&& self, int dep) -> void {
		ans = max(ans, dep);
		if (mp.find(dep) != mp.end()) {
			for (auto &it : mp[dep]) {
				if (len.find(dep + it) == len.end()) {
					len[dep + it] = 1;
					self(self, dep + it);
				}
			}
		}
	};
	dfs(dfs, n);

	cout << ans << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; cin >> T;
	while (T--) solve();
	return 0;
}
```

## D1. The Endspeaker (Easy Version)

赛时看出来是二维 dp 加前缀和二分，但是调了一个小时没搞定。。。

寄

还没来得及补题，回头找机会补上罢。。。

咕咕咕