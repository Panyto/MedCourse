"""关6 长尾读数：H6-4"不典型表现"和 H6-5"罕见病上配额 = 抄唯一那个真病人"。

全部用 `entities.key_set`（没被否定的实体规范键）。真实一侧的集合只在内存里；这里的函数只返回计数和编号。

H6-4 一份病历"有不典型的东西" = 至少有一个实体，在**同诊断的其他真实病人**里很少见（份数 ≤ 门槛），
但在 fold 0 **别的真实病人**里见过（排除抽取噪声和编出来的词）。真实病历算自己时把自己去掉（留一），
合成病历对全部真实病历算。门槛 = max(1, floor(0.05 × 同诊断其他真实病人数))。
fold 0 里谁都没写过的实体记"孤词"，两边都单独报，不算不典型。

H6-5 一份合成病历"指向一个真人" = 它和某一个参考病人共有至少 3 个实体，且 fold 0 里没有第二个病人同时有这几个实体。
"""
from collections import Counter


def df_counter(sets):
    """[(住院号, 集合)] → Counter{实体: 份数}。"""
    c = Counter()
    for _, s in sets:
        c.update(s)
    return c


def rare_threshold(n_other, frac=0.05):
    return max(1, int(frac * n_other))


def atypical_items(keys, code_df, all_df, n_code_other, self_in_ref=False, frac=0.05):
    """keys：一份病历的实体集合；code_df：同诊断真实病历的份数表；all_df：fold 0 全部真实病历的份数表；
    n_code_other：同诊断"其他"真实病人数（真实病历是 N_c − 1，合成病历是 N_c）；
    self_in_ref：这份病历本身在两张份数表里（真实病历），算份数时减 1。
    返回 (不典型实体列表, 孤词列表)。"""
    thr = rare_threshold(n_code_other, frac)
    sub = 1 if self_in_ref else 0
    atyp, orphan = [], []
    for k in sorted(keys):
        other_all = all_df.get(k, 0) - sub
        if other_all <= 0:
            orphan.append(k)
            continue
        if code_df.get(k, 0) - sub <= thr:
            atyp.append(k)
    return atyp, orphan


def build_index(sets):
    """[(住院号, 集合)] → {实体: set(下标)}。"""
    idx = {}
    for i, (_, s) in enumerate(sets):
        for k in s:
            idx.setdefault(k, set()).add(i)
    return idx


def holders(items, index):
    """同时有 items 里每个实体的病人下标集合。"""
    out = None
    for k in sorted(items, key=lambda x: len(index.get(x, ()))):
        cur = index.get(k, set())
        out = set(cur) if out is None else out & cur
        if not out:
            return set()
    return out if out is not None else set()


def unique_combo(x_keys, ref_ids, all_sets, index, min_k=3):
    """x_keys：一份合成病历的实体集合；ref_ids：参考病人在 all_sets 里的下标；all_sets：fold 0 全部 [(住院号, 集合)]。
    对每个参考病人 a：共有 O = x ∩ E(a)；|O| ≥ min_k 且 fold 0 里同时有 O 的只有 a 一个人，算"指向 a"。
    返回 {"hit": 0/1, "ref": 下标或 None, "overlap": |O|, "holders": 同时有 O 的人数}：
    有命中时取命中里 |O| 最大的（并列取下标小的）；没有命中时取 |O| 最大的那个参考病人。"""
    hits, misses = [], []
    for a in sorted(ref_ids):
        o = x_keys & all_sets[a][1]
        h = holders(o, index) if len(o) >= min_k else None
        row = {"ref": a, "overlap": len(o), "holders": (len(h) if h is not None else None)}
        if h == {a}:
            hits.append(row)
        else:
            misses.append(row)
    if hits:
        best = sorted(hits, key=lambda r: (-r["overlap"], r["ref"]))[0]
        return {"hit": 1, **best}
    if misses:
        best = sorted(misses, key=lambda r: (-r["overlap"], r["ref"]))[0]
        return {"hit": 0, **best}
    return {"hit": 0, "ref": None, "overlap": 0, "holders": None}


def copy_share(x_keys, ref_sets):
    """max over 参考病人 |x ∩ E(a)| / |E(a)|：这份病历含了某一个真人实体清单的多少（参考病人实体为空的不算）。"""
    vals = [len(x_keys & s) / len(s) for s in ref_sets if s]
    return max(vals) if vals else 0.0


def real_real_copy_share(sets_by_code):
    """真实对真实的基线：同诊断里每个真人对其余真人的 copy_share（留一）。返回 [值]。"""
    out = []
    for code in sorted(sets_by_code):
        rows = sets_by_code[code]
        for i, (_, s) in enumerate(rows):
            others = [t for j, (_, t) in enumerate(rows) if j != i]
            if others:
                out.append(copy_share(s, others))
    return out
