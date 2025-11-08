---
title: 'AtCoder Beginner Contest 380 A~E'
published: 2025-01-09 #创建时间
updated: 2025-01-10 #更新时间
tags: [Atcoder, AtCoder Beginner Contest, cpp, Virtual Participation]
description: 'AtCoder Beginner Contest 380 VP 解题报告' #描述
image: ''
category: 'Algorithm'
draft: false # 是否为草稿
series: 'Atcoder'
---

# [AtCoder Beginner Contest 380](https://atcoder.jp/contests/abc380) (Virtual Participation)

烂完了，读假又什么都不会呜呜呜。

要被自己菜哭了。

## A - 133233

很简单的签到。随便统计一下即可。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	int n; cin >> n;
	vector<int> v(10, 0);
	while (n) {
		int x = n % 10; n /= 10;
		if (x == 1) ++ v[1];
		if (x == 2) ++ v[2];
		if (x == 3) ++ v[3];
	}
	if (v[1] == 1 && v[2] == 2 && v[3] == 3) cout << "Yes" << char(10);
	else cout << "No" << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);
	int T; T = 1;
	while (T--) solve();
	return 0;
}
```

## B - Hurdle Parsing

签到 +1，扫一遍统计即可。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	string str; getline(cin, str);
	vector<int> ans;
	for (auto it : str) {
		if (it == '|') ans.push_back(0);
		if (it == '-') ++ans.back();
	}
	rep (i, 0, ans.size() - 2)
		cout << ans[i] << " \n"[i == int(ans.size()) - 2];
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);
	int T; T = 1;
	while (T--) solve();
	return 0;
}
```

## C - Move Segment

有点讨厌的模拟。主要是 $1$ 子段的截取需要一点判断。

将所有 $1$ 子段的区间截取出来，然后调整其中的 $K$ 到 $K - 1$ 的后面即可。

时间复杂度 $\Theta(n)$

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

using pii = pair<int, int>;

inline void solve() {
	int n, k; cin >> n >> k;
	string str; cin >> str;
	vector<pii> blk;
	[&]() {
		pii tmp; bool flg = false;
		rep (i, 0, n - 1) {
			if (i == 0) {
				if (str[0] == '1') { tmp.first = 1; flg = 1; }
				continue;
			}
			if (str[i] == '0') {
				if (str[i - 1] == '1') {
					tmp.second = i; flg = 0;
					blk.push_back(tmp); tmp = {0, 0};
				}
			}
			if (str[i] == '1') {
				if (str[i - 1] == '0') { tmp.first = i + 1; flg = 1; }
			}
		}
		if (flg == 1 && str.back() == '1') {
			tmp.second = n; blk.push_back(tmp);
		}
	}();
	int len = blk[k - 1].second - blk[k - 1].first;
	blk[k - 1].first = blk[k - 2].second + 1;
	blk[k - 1].second = blk[k - 1].first + len;
	fill(str.begin(), str.end(), '0');
	for (auto &it : blk)
		fill(str.begin() + it.first - 1, str.begin() + it.second, '1');
	cout << str << char(10);
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; T = 1;
	while (T--) solve();
	return 0;
}
```

## D - Strange Mirroring

注意到 $K_i$ 的范围很大，故而肯定考虑类似 $2$ 的次幂相关的化归。

不妨先观察最终串的组成，写出前几项

$$
STTSTSST\ldots
$$

不难发现，在 $1 \sim 2^n$ 的范围中，$1 \sim 2^{n - 1}$ 和 $2^{n - 1} + 1 \sim 2^n$ 应该是互反的。

所以可以考虑如下算法：

对于当前的位置 $p$ 所在串，通过不断减去 $2^{\lfloor \log_2{p} \rfloor}$，记录减去后的反转情况，可以最终规约到串 $S$ 以及是否需要翻转。

由于该题时限猜测一定不会太严，并且不高兴操心 $long\space long$ 溢出问题，故代码中存在 define int long long 之流的恶习，请不要随意模仿。

代码如下：

```cpp
#include <bits/stdc++.h>

using namespace std;

#define int long long
#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	string s; cin >> s;
	int q; cin >> q;
	while (q--) {
		int k; cin >> k;
		int val = ((k-- + s.size() - 1) / s.size());
		bool flg = false;
		while (val > 1) {
			int tmp = (1ll << (long long)(__lg(val - 1)));
			val -= tmp;
			k -= tmp * s.size();
			flg ^= 1;
		}
		if (!flg) cout << s[k] << ' ';
		else {
			if (s[k] >= 97) cout << char(s[k] - 97 + 65) << ' ';
			else cout << char(s[k] - 65 + 97) << ' ';
		}
	}
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; T = 1;
	while (T--) solve();
	return 0;
}
```

## E - 1D Bucket Tool

额啊

我是冰茶姬实现低手 QwQ

赛时想到冰茶姬了，但是一度写错加读错题意 (size 函数返回 int 的笨比)，所以小寄一波。

实际上这题个人觉得比较自然的想法是，使用类似珂朵莉树存放区间的方法，进行类似的合并和删除，实际上速度还挺快的（？

在这里放出两种不同的写法，感觉是加强了类似区间处理的一些想法。

Notice: 本段代码中 dsu 的 merge 函数非标准实现。珂朵莉树的节点定义中的 $operator<$ 重载需要加上两个 $const$，不然容易报神必错误。

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

struct Node {
	int l, r;
	mutable int v;
	Node(int l_, int r_ = -1, int v_ = 0) : l(l_), r(r_), v(v_) {};
	bool operator < (const Node &o) const { return l < o.l; }
};

inline void solve() {
	int n, q; cin >> n >> q;
	set<Node> s;
	rep (i, 1, n) { s.insert(Node(i, i, i)); }
	vector<int> cnt(n + 10, 1);
	while (q--) {
		int opt; cin >> opt;
		if (opt == 1) {
			int x, c; cin >> x >> c;
			auto it = s.lower_bound(Node(x, x, 0));
			if (it->l != x) --it;
			auto &[l, r, v] = *it;
			cnt[v] -= r - l + 1; v = c; cnt[c] += r - l + 1;
			bool flg1 = false, flg2 = false;
			if (it != s.begin() && prev(it)->v == c) flg1 = true;
			if (next(it) != s.end() && next(it)->v == c) flg2 = true;
			if (flg1 || flg2) {
				int nl = l, nr = r, nv = c;
				if (flg1) { nl = prev(it)->l; s.erase(prev(it)); }
				if (flg2) { nr = next(it)->r; s.erase(next(it)); }
				s.erase(it); s.insert(Node(nl, nr, nv));
			}
		}
		if (opt == 2) {
			int c; cin >> c;
			cout << cnt[c] << char(10);
		}
	}
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; T = 1;
	while (T--) solve();
	return 0;
}
```

```cpp
#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (auto i = (a); i <= decltype(i)(b); ++i)

inline void solve() {
	int n, q; cin >> n >> q;
	vector<int> fa(n + 10), siz(n + 10, 1);
	vector<int> cnt(n + 10, 1);
	vector<int> l(n + 10), r(n + 10), c(n + 10);
	iota(fa.begin(), fa.end(), 0);
	rep (i, 1, n) l[i] = r[i] = c[i] = i;

	auto find = [&](this auto&& find, int x) -> int {
		return fa[x] == x ? x : fa[x] = find(fa[x]);
	};
	auto merge = [&](int x, int y) {
		if (siz[x] < siz[y]) swap(x, y);
		siz[x] += siz[y];
		l[x] = min(l[x], l[y]);
		r[x] = max(r[x], r[y]);
		fa[y] = x;
	};

	while (q--) {
		int opt; cin >> opt;
		if (opt == 1) {
			int x, c_; cin >> x >> c_;
			int f = find(x);
			cnt[c[f]] -= r[f] - l[f] + 1;
			c[f] = c_;
			cnt[c_] += r[f] - l[f] + 1;
			if (l[f] != 1) {
				int f_ = find(l[f] - 1);
				if (c[f_] == c_) merge(f, f_);
			}
			if (r[f] != n) {
				int f_ = find(r[f] + 1);
				if (c[f_] == c_) merge(f, f_);
			}
		}
		if (opt == 2) {
			int c_; cin >> c_;
			cout << cnt[c_] << char(10);
		}
	}
}

signed main() {
	cin.tie(nullptr)->sync_with_stdio(false);

	int T; T = 1;
	while (T--) solve();
	return 0;
}
```

提交结束后两者的速度差距差不多在 $4$ 倍左右，尚且合理。

---
F、G: To be continued.