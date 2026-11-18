"""H4-3：LLM-CARe 的修复环到底往病历里加了什么。逐段做字符级比对，取出新插进去或换掉的文字，
看里面有没有信息单元、碰到的是它 9 条完整性清单里的哪一条（按我们自己的关键词，只作描述）。"""
import difflib
import re

from ruler_g3.adapters import care_get
from ruler_g4.ruler import FIELDS, GENERAL, info_units, nchars

_ITEM = {
    "C02_起病时间": re.compile(r"\d+\s*(?:余|多)?\s*(?:个)?\s*(?:分钟|小时|天|日|周|月|年)|[一二两三四五六七八九十半数几]+(?:余|多)?(?:个)?(?:天|日|周|月|年)前?"),
    "C03_起病缓急": re.compile(r"急性|急起|突然|突发|骤然|逐渐|缓慢|隐匿|慢性|反复|间断|进行性|持续性|阵发性"),
    "C04_诱因": re.compile(r"诱因|受凉|着凉|劳累|饱餐|饮酒|外伤|情绪激动|活动后|感冒后|无明显原因"),
    "C06_一般情况": re.compile("|".join(w for ws in GENERAL.values() for w in ws)),
    "C07_检查": re.compile(r"检查|化验|CT|MRI|彩超|超声|心电图|血常规|胸片|造影|X线"),
    "C08_治疗": re.compile(r"治疗|手术|输液|吸氧|予|给予|口服|静滴|静脉"),
    "C09_剂量用法": re.compile(r"\d+(?:\.\d+)?\s*(?:mg|g|μg|ug|ml|mL|片|粒|支|单位|U)|每日|每天|bid|tid|qd|qn|q\d+h|次/日|一日"),
}


def inserted(before, after):
    """after 相对 before 新插进去或换掉的片段：[(start, end)]，是 after 里的位置。"""
    sm = difflib.SequenceMatcher(None, before or "", after or "", autojunk=False)
    return [(j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("insert", "replace") and j2 > j1]


def record_diff(before, after):
    """两版 7 字段病历的差别。返回每段：加了多少字、删了多少字、插入的片段（原文）、插入片段里的信息单元、
    插入片段碰到的清单条目；以及合计。"""
    out, tot_add, tot_del, items = {}, 0, 0, set()
    for k in FIELDS:
        b, a = care_get(before, k), care_get(after, k)
        if a == b:
            out[k] = {"changed": 0, "added_chars": 0, "removed_chars": 0, "inserts": [], "iu_in_inserts": [], "items": []}
            continue
        sm = difflib.SequenceMatcher(None, b, a, autojunk=False)
        removed = sum(nchars(b[i1:i2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("delete", "replace"))
        spans = inserted(b, a)
        segs = [a[s:e] for s, e in spans]
        ius = [u for u in info_units(a) if any(u["start"] < e and s < u["end"] for s, e in spans)]
        its = sorted({name for name, rx in _ITEM.items() for seg in segs if rx.search(seg)})
        added = sum(nchars(x) for x in segs)
        out[k] = {"changed": 1, "added_chars": added, "removed_chars": removed, "inserts": segs,
                  "iu_in_inserts": [{"type": u["type"], "key": u["key"]} for u in ius], "items": its}
        tot_add += added
        tot_del += removed
        items |= set(its)
    out["_total"] = {"changed_fields": sum(out[k]["changed"] for k in FIELDS), "added_chars": tot_add,
                     "removed_chars": tot_del, "iu_added": sum(len(out[k]["iu_in_inserts"]) for k in FIELDS),
                     "items": sorted(items)}
    return out
