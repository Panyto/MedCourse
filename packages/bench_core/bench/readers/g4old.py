"""关4 历史口径：g4-rules-0.1，只用来复现 Q45/Q4_09 的锚点，和 g3old 一个道理。不进记分牌，新记分牌走 g4（0.2）。"""
from ruler_g4 import RULES_VERSION, analyze_admission

ROWS = ("R", "G-D", "G-E2")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run(rows, config):
    per = []
    for x in rows:
        if x["kind"] != "admission" or x["row"] not in ROWS:
            continue
        row = {"row": x["row"], "id": x["id"]}
        docs = analyze_admission({"admission_id": x["id"], "docs": x["docs"]})
        for role, tag in (("course", "course"), ("hr", "hr"), ("discharge", "ds"), ("fp", "fp")):
            part = [d for d in docs if d["role"] == role]
            row[f"{tag}_empty"] = _mean(d["empty_share"] for d in part)
            row[f"{tag}_iu_kchar"] = _mean(d["iu_per_kchar"] for d in part)
            row[f"{tag}_copy5"] = _mean(d["copy5_prev"] for d in part)
        per.append(row)
    return per, [{"row": "*", "metric": "rules_version_is_01", "value": RULES_VERSION.endswith("0.1")}]
