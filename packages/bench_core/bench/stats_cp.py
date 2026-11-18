"""Clopper–Pearson 区间（二分，只用标准库，与 g2/g45 stats 的做法相同）。"""
import math


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    return min(1.0, sum(math.exp(_log_comb(n, i) + i * math.log(p) + (n - i) * math.log(1 - p)) for i in range(k + 1)))


def _bisect(f, target, lo=0.0, hi=1.0, increasing=True):
    for _ in range(80):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp(k, n, alpha=0.05):
    low = 0.0 if k == 0 else _bisect(lambda p: 1 - _cdf(k - 1, n, p), alpha / 2)
    high = 1.0 if k == n else _bisect(lambda p: _cdf(k, n, p), alpha / 2, increasing=False)
    return low, high
