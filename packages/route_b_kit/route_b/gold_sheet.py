# -*- coding: utf-8 -*-
"""A 组金标审阅表：17 个病种 × 11 个字段 = 187 格。老板 11-11：先预判预填，存疑的排最前，老板重点审存疑、抽查其余。

预判只是帮老板排顺序，不是金标。判定标准（金标定义只写这一处，表头照抄）：
  对   = 8B 的答案回答了这个字段问的事，且和这份路径原文一致（不适用的格子：这个病种按路径确实不做手术或有创操作）
  错   = 答非所问、和原文矛盾，或原文有却答成"不适用"
  部分 = 列表型字段大部分对、个别项错或漏

机器预判规则（确定性，只用来排序）：
- 有正则 v0 逐字值的字段（住院日上限、麻醉、手术日、术后天数），8B 值和 v0 不一致 → 存疑
- 所引原句不含这个字段的提示词（如症状格的引句里没有"症状/表现"）→ 存疑
- 模型先验（没引句）、引文不实（答案不在引句里）→ 存疑
- 门控记"不适用"，但 v0 在原文里找到了麻醉或手术日 → 存疑
- 其余有出处的 → 建议对
人工复核（总经理逐格看后的改判）记在 overrides 里，理由写明。

python -m route_b.gold_sheet --v2 <kb_v2.1 目录> --v0 <kb_v0 目录> --config config/diseases_v0.json --out <输出 csv> [--overrides <json>]
"""
import argparse
import csv
import glob
import json
import os
import re

from .schema import GROUP_A

LABEL = {"los_max": "标准住院日（上限）", "chief_symptoms": "典型症状", "signs": "典型体征", "exams_required": "必需检查",
         "exams_optional": "可选检查", "procedure": "手术、操作或主要治疗", "anesthesia": "麻醉方式", "surgery_day_max": "手术日",
         "postop_days": "术后恢复天数", "antibiotic_rule": "抗菌药停药条件", "discharge_criteria": "出院标准"}
HINT = {"chief_symptoms": r"症状|表现|主诉", "signs": r"体征|查体|压痛|啰音|水肿|阳性|触及|听诊", "exams_required": r"必[需须]|检查项目|常规",
        "exams_optional": r"可选|根据.{0,4}病情|必要时|酌情", "procedure": r"行|术|治疗|介入|置|透析|化疗", "anesthesia": r"麻醉",
        "surgery_day_max": r"手术日|术前|入院第", "postop_days": r"术后", "antibiotic_rule": r"抗菌|抗生素|疗程|停用|预防用药",
        "discharge_criteria": r"出院|体温|愈合|症状|稳定|正常", "los_max": r"住院日"}
V0_EXACT = ("anesthesia", "surgery_day_max", "postop_days")
ORDER = {"存疑": 0, "建议错": 1, "建议部分": 2, "建议对": 3}


def _val_text(fld, v):
    if fld.startswith("exams") and isinstance(v, list):
        return "；".join(f"{e.get('name')}→{e.get('key')}" for e in v)
    if isinstance(v, dict) and "max" in v:
        return f"{v.get('min')}–{v.get('max')} 天（取上限 {v.get('max')}）"
    if isinstance(v, list):
        return "；".join(str(x) for x in v)
    return str(v)


def _v0_text(fld, kb0):
    fs = kb0["facts"].get(fld, []) if kb0 else []
    if not fs:
        return ""
    if fld in ("discharge_criteria",):
        return "；".join(str(f["value"]) for f in fs)
    v = fs[0]["value"]
    return "；".join(v) if isinstance(v, list) else str(v)


def _source(f):
    if f.get("source") == "not_applicable":
        return "不适用（门控）"
    if f.get("grounded"):
        return "有出处" + ("（指导原则）" if f.get("source") == "guideline" else "")
    if f.get("quote_failed"):
        n = f.get("items_in_quote")
        return "引文不实" + (f"（{n[0]}/{n[1]} 项在引句中）" if n else "")
    return "模型先验（没引句）"


def prejudge(fld, f, kb0):
    """返回 (预判, 理由)。"""
    v0 = _v0_text(fld, kb0)
    if f is None:
        return "存疑", "8B 没给出合格答案"
    if f.get("source") == "not_applicable":
        if fld in V0_EXACT and v0:
            return "存疑", f"门控判不做手术，但正则在原文找到了：{v0[:30]}"
        return ("建议对", "门控判不做手术，门控有出处") if f.get("gate_grounded") else ("存疑", "门控判不做手术，但门控没有出处")
    if not f.get("grounded"):
        return "存疑", _source(f)
    quote = f.get("quote", "")
    if fld == "los_max":
        vmax = (f.get("value") or {}).get("max") if isinstance(f.get("value"), dict) else f.get("value")
        if v0 and str(vmax) != v0:
            return "存疑", f"上限 {vmax} 和正则逐字抽到的 {v0} 不一致"
        return "建议对", "上限和正则一致" if v0 else "有出处"
    if fld in V0_EXACT and v0 and str(f.get("value")).strip() != v0.strip():
        return "存疑", f"和正则逐字值不一致：{v0[:30]}"
    if not re.search(HINT[fld], quote):
        return "存疑", "所引原句里没有这个字段的提示词，可能答非所问"
    if fld == "signs" and any("观察" in str(x) for x in f.get("value") or []):
        return "存疑", "体征里混有“观察……”，可能是术后观察"
    if fld == "antibiotic_rule" and f.get("source") == "guideline":
        return "建议对", "引自《抗菌药物临床应用指导原则》（通用条款，不是本病种路径）"
    return "建议对", "有出处，引句是这个字段的内容" + ("，和正则一致" if v0 and fld in V0_EXACT else "")


def build(v2_dir, v0_dir, config, overrides=None):
    cfg = json.load(open(config, encoding="utf-8"))["diseases"]
    rows = []
    for d in cfg:
        kb = json.load(open(os.path.join(v2_dir, f"kb_{d['id']}.json"), encoding="utf-8"))
        p0 = os.path.join(v0_dir, f"kb_{d['id']}.json")
        kb0 = json.load(open(p0, encoding="utf-8")) if os.path.exists(p0) else None
        gate = (kb["facts"].get("surgery_gate") or [{}])[0]
        for fld in GROUP_A:
            f = (kb["facts"].get(fld) or [None])[0]
            pj, why = prejudge(fld, f, kb0)
            key = f"{d['id']}|{fld}"
            if overrides and key in overrides:
                pj, why = overrides[key]["prejudge"], "总经理复核：" + overrides[key]["reason"]
            rows.append({"病种id": d["id"], "病种": d["name"], "路径版本": kb["meta"]["version"], "字段": fld, "字段含义": LABEL[fld],
                         "8B答案": _val_text(fld, f["value"]) if f else "", "来源": _source(f) if f else "没答出",
                         "所引原句": (f.get("quote") if f and f.get("source") != "not_applicable" else
                                  ("门控引句：" + gate.get("quote", "")) if f else "")[:400],
                         "正则v0值": _v0_text(fld, kb0)[:200], "预判": pj, "预判理由": why, "老板判定": "", "老板备注": ""})
    rows.sort(key=lambda r: (ORDER[r["预判"]], r["病种id"], GROUP_A.index(r["字段"])))
    for i, r in enumerate(rows, 1):
        r["序号"] = i
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v0", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overrides", default="")
    a = ap.parse_args()
    ov = json.load(open(a.overrides, encoding="utf-8")) if a.overrides else None
    rows = build(a.v2, a.v0, a.config, ov)
    cols = ["序号", "预判", "预判理由", "病种id", "病种", "路径版本", "字段", "字段含义", "8B答案", "来源", "所引原句", "正则v0值", "老板判定", "老板备注"]
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    c = {k: sum(r["预判"] == k for r in rows) for k in ORDER}
    print("gold_sheet_done", len(rows), c)


if __name__ == "__main__":
    main()
