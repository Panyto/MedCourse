"""统计口径（照 G1-B、守则第五节）：按槽位（或住院）整簇 bootstrap 1,000 次，Clopper–Pearson，AUC，Kendall τ，带内判定。
不依赖 scipy。"""
import math
import random


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))

    def cdf(x, p):  # P(X <= x)
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0 if x < n else 1.0
        lp, lq = math.log(p), math.log1p(-p)
        terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
                 for i in range(0, x + 1)]
        m = max(terms)
        return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))

    lower = 0.0 if k == 0 else _cp_lower(k, n, alpha, cdf)
    upper = 1.0 if k == n else _cp_upper(k, n, alpha, cdf)
    return lower, upper


def _bisect(f, target, increasing):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        v = f(mid)
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _cp_lower(k, n, alpha, cdf):
    # 最小的 p 使 P(X >= k) = alpha/2；P(X>=k) 随 p 增大
    return _bisect(lambda p: 1 - cdf(k - 1, p), alpha / 2, True)


def _cp_upper(k, n, alpha, cdf):
    # 最大的 p 使 P(X <= k) = alpha/2；P(X<=k) 随 p 减小
    return _bisect(lambda p: cdf(k, p), alpha / 2, False)


def _ratio(units, keys):
    num = sum(units[k][0] for k in keys)
    den = sum(units[k][1] for k in keys)
    return num / den if den else float("nan")


def _pct(xs, q):
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q
    a, b = int(math.floor(i)), int(math.ceil(i))
    return xs[a] + (xs[b] - xs[a]) * (i - a)


def cluster_ratio(units, B=1000, seed=20261014):
    """units：{簇: (分子, 分母)}。返回 (点估计, 下限, 上限)。"""
    keys = sorted(units)
    est = _ratio(units, keys)
    rng = random.Random(seed)
    boots = [_ratio(units, [rng.choice(keys) for _ in keys]) for _ in range(B)]
    return est, _pct(boots, 0.025), _pct(boots, 0.975)


def paired_diff(units_a, units_b, B=1000, seed=20261014):
    """同一批槽位上 a − b 的差（各自分子之和/分母之和）。只用两边都有的槽位。返回 (差, 下限, 上限, 槽位数)。"""
    keys = sorted(set(units_a) & set(units_b))
    est = _ratio(units_a, keys) - _ratio(units_b, keys)
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        s = [rng.choice(keys) for _ in keys]
        boots.append(_ratio(units_a, s) - _ratio(units_b, s))
    return est, _pct(boots, 0.025), _pct(boots, 0.975), len(keys)


def paired_mean_diff(vals_a, vals_b, B=1000, seed=20261014):
    """连续量（复制率、事实数）：同槽位 a − b 的均值差。vals：{槽位: 值}，值为 None 的槽位不用。"""
    keys = sorted(k for k in set(vals_a) & set(vals_b) if vals_a[k] is not None and vals_b[k] is not None)
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0, float("nan")

    def m(ks, v):
        return sum(v[k] for k in ks) / len(ks)
    est = m(keys, vals_a) - m(keys, vals_b)
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        s = [rng.choice(keys) for _ in keys]
        boots.append(m(s, vals_a) - m(s, vals_b))
    return est, _pct(boots, 0.025), _pct(boots, 0.975), len(keys), m(keys, vals_b)


def band_verdict(lo, hi, ref_mean, rel=0.25, abs_min=0.0):
    """"落进真实带"：差的 95% 区间整个落在 ±δ 以内，δ = max(rel × 真实均值, abs_min)。
    区间整个在 +δ 以上记"带外偏高"，整个在 −δ 以下记"带外偏低"，其余"未定"。"""
    d = max(rel * abs(ref_mean), abs_min)
    if -d <= lo and hi <= d:
        return "带内"
    if lo > d:
        return "带外偏高"
    if hi < -d:
        return "带外偏低"
    return "未定"


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(相等)。pos：地板（应更高），neg：真实。"""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    rank_sum = 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2.0
        for t in range(i, j):
            if allv[t][1] == 1:
                rank_sum += r
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def kendall_tau(x, y):
    n = len(x)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    tot = n * (n - 1) / 2
    return (c - d) / tot if tot else float("nan")


def tau_ci(rows_a, rows_b, B=1000, seed=20261014):
    """rows_a、rows_b：{行名: {槽位: (分子, 分母)}}，两种测量（尺子、评委）在同一批行上的读数。
    按槽位重抽，算两种测量在各行比例上的 Kendall τ。返回 (τ, 下限, 上限)。"""
    rows = sorted(rows_a)
    keys = sorted(set.intersection(*[set(rows_a[r]) & set(rows_b[r]) for r in rows]))

    def tau_of(ks):
        xa = [_ratio(rows_a[r], ks) for r in rows]
        xb = [_ratio(rows_b[r], ks) for r in rows]
        return kendall_tau(xa, xb)
    est = tau_of(keys)
    rng = random.Random(seed)
    boots = [tau_of([rng.choice(keys) for _ in keys]) for _ in range(B)]
    return est, _pct(boots, 0.025), _pct(boots, 0.975)
