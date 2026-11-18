"""从逐住院读数和行级读数算记分牌、负对照矩阵。只用标准库；区间是按住院整簇 bootstrap（1,000 次）。"""
import csv
import json
import math
import os
import random

from .registry import M, verdict

B = 1000
SEED = 20261101


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def clopper(k, n):
    if n == 0:
        return None, None
    from .stats_cp import cp
    return cp(k, n)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def boot(units, stat, seed):
    """units：每住院一个单元。按单元重抽。"""
    if not units or B == 0:
        return None, None
    rng = random.Random(seed)
    n = len(units)
    vals = [stat(rng.choices(units, k=n)) for _ in range(B)]
    vals = [v for v in vals if v is not None]
    return pct(vals, 0.025), pct(vals, 0.975)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _ratio(pairs):
    den = sum(b for _, b in pairs)
    return sum(a for a, _ in pairs) / den if den else None


def one_metric(m, recs, whole):
    """返回 dict：value、ci、分子、分母、n。recs：本行本读数的逐住院记录；whole：{metric: 行级记录}。"""
    if m["how"] == "row":
        r = whole.get(m["field"])
        if not r:
            return {"value": None, "n": None}
        return {"value": r.get("value"), "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"), "n": r.get("n")}
    seed = f"{SEED}|{m['id']}"
    if m["how"] == "mean":
        xs = [float(r[m["field"]]) for r in recs if r.get(m["field"]) is not None]
        lo, hi = boot(xs, _mean, seed)
        return {"value": _mean(xs), "ci_lo": lo, "ci_hi": hi, "n": len(xs)}
    if m["how"] == "pooled":
        ps = [(r[m["field"]], r[m["den"]]) for r in recs if r.get(m["den"]) is not None]
        lo, hi = boot(ps, _ratio, seed)
        return {"value": _ratio(ps), "ci_lo": lo, "ci_hi": hi, "n": len(ps), "numerator": sum(a for a, _ in ps),
                "denominator": sum(b for _, b in ps)}
    # rate
    use = [r for r in recs if (m["chk"] is None and r.get(m["field"]) is not None) or (m["chk"] and r.get(m["chk"]))]
    hits = [1 if r.get(m["field"]) else 0 for r in use]
    lo, hi = boot(hits, _mean, seed)
    cl, ch = clopper(sum(hits), len(hits))
    return {"value": _mean(hits), "ci_lo": lo, "ci_hi": hi, "n": len(hits), "numerator": sum(hits),
            "denominator": len(hits), "cp_lo": cl, "cp_hi": ch}


def scoreboard(raw_dir, row_kinds, efficiency=None, extra_row=None):
    """row_kinds：{行名: "admission"/"care7"}。返回记分牌的行列表。"""
    per, whole = {}, {}
    for reader in ("extra", "g1", "g2", "g3", "g3old", "g4", "g4old", "g6", "c2", "mauve"):
        for r in load_jsonl(os.path.join(raw_dir, f"all.{reader}.jsonl")):
            per.setdefault((reader, r["row"]), []).append(r)
        for r in load_jsonl(os.path.join(raw_dir, f"all.{reader}.row.jsonl")):
            whole.setdefault((reader, r["row"]), {})[r["metric"]] = r
    for row, d in (efficiency or {}).items():
        whole[("eff", row)] = {k: {"value": v, "n": d.get("admissions")} for k, v in d.items()}
    rows_out = []
    for row, kind in row_kinds.items():
        for m in M:
            base = {"metric": m["id"], "group": m["group"], "gate": m["gate"], "name": m["name"], "row": row,
                    "status": m["status"], "primary": m["primary"]}
            if m["src"] == "-":
                rows_out.append({**base, "value": None, "note": m["status"]})
                continue
            if kind not in m["need"]:
                rows_out.append({**base, "value": None, "note": "结构上不适用"})
                continue
            r = one_metric(m, per.get((m["src"], row), []), whole.get((m["src"], row), {}))
            rows_out.append({**base, **r, "floor_lo": (m["floor"] or (None, None))[0], "floor_hi": (m["floor"] or (None, None))[1],
                             "target_lo": (m["target"] or (None, None))[0], "target_hi": (m["target"] or (None, None))[1],
                             "floor_ok": verdict(r.get("value"), m["floor"]), "target_ok": verdict(r.get("value"), m["target"])})
    return rows_out


COLS = ["group", "gate", "metric", "name", "row", "status", "primary", "value", "ci_lo", "ci_hi", "numerator", "denominator",
        "cp_lo", "cp_hi", "n", "floor_lo", "floor_hi", "target_lo", "target_hi", "floor_ok", "target_ok", "note"]


def write_csv(path, rows, cols=COLS):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def not_worse(floor, nc_val, base_val):
    """负对照不差于基线。单侧门线看方向，相等也算；双侧门线看离带中点的距离。"""
    lo, hi = floor
    if lo is not None and hi is not None:
        mid = (lo + hi) / 2.0
        return abs(nc_val - mid) <= abs(base_val - mid)
    return nc_val <= base_val if hi is not None else nc_val >= base_val


def negcontrol_matrix(board, bases, nc_names, binding=None):
    """判据（PREREG_S0 第 4 节），只在真实基线 binding 上判，其他基线是参考：
    (a) 每个负对照至少被一个主打指标（A、B 组）拒绝；
    (b) 每个 B 组主打指标至少被一个负对照打破；
    (c) 负对照分数不差于基线、而且还在线内的指标，标"需配对量"。A 组的这一标只列表：闸门只管抓错，负对照也没错不算闸门失灵。
    拒绝 = 负对照落在底线外或读不出，而基线在线内。判别器和同病重合是"和真实行比"，真实行自己没有读数，按在线内算。
    返回 (矩阵行, 汇总行)。"""
    binding = binding or (bases[0] if bases else None)
    idx = {(r["row"], r["metric"]): r for r in board}
    primary = [m for m in M if m["primary"]]
    out, summ = [], []
    broken = {m["id"]: 0 for m in primary}
    for base in bases:
        for nc in nc_names:
            row = f"{nc}@{base}"
            rej = 0
            for m in primary:
                a, b = idx.get((row, m["id"])), idx.get((base, m["id"]))
                if a is None or b is None or a.get("note") == "结构上不适用":
                    continue
                nc_val, base_val = a.get("value"), b.get("value")
                if base_val is None and m["how"] != "row":
                    continue
                base_ok = b.get("floor_ok") is not False
                rejected = base_ok and (nc_val is None or a.get("floor_ok") is False)
                rej += rejected
                if base == binding:
                    broken[m["id"]] += rejected
                closer = False
                if m["floor"] and nc_val is not None and base_val is not None and a.get("floor_ok") is True:
                    closer = not_worse(m["floor"], nc_val, base_val)
                out.append({"negative_control": nc, "base_row": base, "binding": base == binding, "group": m["group"], "metric": m["id"], "nc_value": nc_val, "base_value": base_val,
                            "base_in_floor": b.get("floor_ok"), "nc_in_floor": a.get("floor_ok"), "rejected": rejected, "needs_paired_measure": closer})
            summ.append({"kind": "a_nc_rejected", "name": row, "binding": base == binding, "count": rej, "pass": rej >= 1})
    for m in primary:
        if m["group"] == "B":
            summ.append({"kind": "b_metric_broken", "name": m["id"], "binding": True, "count": broken[m["id"]], "pass": broken[m["id"]] >= 1})
    return out, summ


def paired_table(matrix):
    """判据 (c) 的点名表：真实基线上每个指标被几个负对照标了"需配对量"，分组列出。"""
    tally = {}
    for r in matrix:
        if r["binding"] and r["needs_paired_measure"]:
            k = (r["group"], r["metric"])
            tally.setdefault(k, []).append(r["negative_control"])
    return [{"group": g, "metric": m, "count": len(v), "negative_controls": "|".join(v)} for (g, m), v in sorted(tally.items())]
