"""关5：判别器。每行和 config["pairs"][行] 指定的真实行按槽位配对，逐段等长（同 Q45b 的 M1eq），另跑只看长度的对照。
空对照：真实行内部，同一三位码两两配对。需要 scikit-learn，只在 ys 环境跑。"""
from q45_kit import c2st, q45b
from q45_kit.realtext import care7_from_admission
from ruler_g1 import extract_facts, split_sections


def to_care7(x):
    if x["kind"] == "care7":
        return x["record"]
    rec, _ = care7_from_admission({"admission_id": x["id"], "docs": x["docs"]}, split_sections, extract_facts)
    return rec


def _result(row, metric, r, n):
    return {"row": row, "metric": metric, "value": r["auc_mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": n}


def run(rows, config):
    by = {}
    for x in rows:
        rec = to_care7(x)
        if rec is not None:
            by.setdefault(x["row"], {})[x["id"]] = rec
    out = []
    pairs = config.get("pairs") or {}
    for row, ref in sorted(pairs.items()):
        if row not in by or ref not in by:
            continue
        ids = sorted(set(by[row]) & set(by[ref]))
        if len(ids) < 20:
            out.append({"row": row, "metric": "auc_skipped_small", "value": None, "n": len(ids)})
            continue
        eq = {i: q45b.equal_pair(by[row][i], by[ref][i]) for i in ids}
        pos, neg = [(i, eq[i][0]) for i in ids], [(i, eq[i][1]) for i in ids]
        out.append(_result(row, "auc", c2st.c2st(pos, neg, "text"), len(ids)))
        ln = c2st.c2st([(i, q45b.equal_lengths(x)) for i, x in pos], [(i, q45b.equal_lengths(x)) for i, x in neg], "length")
        out.append(_result(row, "auc_length", ln, len(ids)))
    all_codes = config.get("codes") or {}
    for ref in sorted(set(pairs.values())):
        codes = all_codes.get(ref) or {}
        if ref in by and codes:
            items = [(i, by[ref][i]) for i in sorted(by[ref]) if i in codes]
            if len(items) >= 40:
                vals = q45b.null_split_equal(items, codes, reps=int(config.get("null_reps", 20)))
                out.append({"row": ref, "metric": "auc_null", "value": sum(vals) / len(vals), "n": len(items)})
    return [], out
