"""地板和退化解。

- 错配地板：把成人的出院医嘱（7 字段病历里连同住院经过）整块换给一个 14 岁以下的宿主。
  宿主自己的年龄、性别、诊断不动。错误率必须明显高于宿主原样。
- 退化解：把每处药名连同后面的剂量、频次换成"药物治疗"。D 会降到 0，具体度也降到 0，两者同报。
"""
import copy
import random

from ruler_g3.adapters import from_care7, from_admission, care_get, doc_kind, CARE_KEYS
from ruler_g3.engine import mentions, WINDOW
from ruler_g3.lexicon import lexicon
from ruler_g3.parse import mention_context

CHILD = 14
ADULT = 18


def _k(rec, field, default):
    for k in CARE_KEYS[field]:
        if k in rec:
            return k
    return default


def swap_floor_care7(pool_name, records, seed="20261026"):
    """records：[(id, dict)]。返回 [(host_id, donor_id, 换过的 dict)]。"""
    rng = random.Random(f"{seed}|swap|{pool_name}")
    hosts, donors = [], []
    for rid, r in records:
        age = from_care7(r).age
        if age is None:
            continue
        if age < CHILD:
            hosts.append((rid, r))
        elif age >= ADULT:
            donors.append((rid, r))
    out = []
    if not donors:
        return out
    for rid, r in hosts:
        did, d = rng.choice(donors)
        x = copy.deepcopy(r)
        x[_k(x, "di", "Discharge Instructions")] = care_get(d, "di")
        x[_k(x, "hc", "Hospital Course")] = care_get(d, "hc")
        out.append((rid, did, x))
    return out


def swap_floor_admissions(pool_name, adms, seed="20261026"):
    """adms：[(id, adm)]。把成人住院的出院记录整篇换给 14 岁以下的住院。只在内存里用。"""
    rng = random.Random(f"{seed}|swap|{pool_name}")
    hosts, donors = [], []
    for rid, a in adms:
        rec = from_admission(a)
        if rec.age is None:
            continue
        ds = [d for d in a.get("docs") or [] if doc_kind(d) == "discharge"]
        if not ds:
            continue
        if rec.age < CHILD:
            hosts.append((rid, a))
        elif rec.age >= ADULT:
            donors.append((rid, ds[-1]))
    out = []
    if not donors:
        return out
    for rid, a in hosts:
        did, dd = rng.choice(donors)
        x = copy.deepcopy(a)
        for d in x["docs"]:
            if doc_kind(d) == "discharge":
                d["text"] = dd.get("text") or ""
        out.append((rid, did, x))
    return out


def degenerate_text(text):
    """把每处没有否定的药名连同其后剂量窗口换成"药物治疗"。"""
    lex = lexicon()
    found = list(lex.find(text or ""))
    pieces, last = [], 0
    for i, (s, e, alias, did) in enumerate(found):
        if s < last:
            continue
        ctx = mention_context(text, s, e, did)
        nxt = found[i + 1][0] if i + 1 < len(found) else len(text)
        wend = min(nxt, ctx["clause_end"], e + WINDOW)
        pieces.append(text[last:s])
        pieces.append("药物治疗")
        last = wend
    pieces.append(text[last:])
    return "".join(pieces)


def degenerate_care7(rec):
    x = copy.deepcopy(rec)
    for f in ("cc", "hpi", "hc", "di"):
        k = _k(x, f, None)
        if k:
            x[k] = degenerate_text(care_get(x, f))
    return x


def degenerate_admission(adm):
    x = copy.deepcopy(adm)
    for d in x.get("docs") or []:
        d["text"] = degenerate_text(d.get("text") or "")
    return x
