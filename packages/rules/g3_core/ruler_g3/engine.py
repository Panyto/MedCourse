"""判错引擎：对一个 Record 找药名、判 D 规则、找 N1 数值、算具体度。"""
import re

from .lexicon import lexicon, rules, weight_p50
from .parse import is_pregnant, mention_context, parse_dose
from .numbers import n1_hits

WINDOW = 40          # 药名后面最多看多少字找剂量
# 0.3：复方制剂的成分名（规则提案第 3 条）。成分名紧贴另一个药名或成分词尾，剂量多半是复方的总量，不判单成分的剂量规则
COMPOUND_BEFORE = re.compile(r"(?:沙坦酯|沙坦|普利|地平|他汀|双胍|列汀|列净|噻嗪)[/／]?$")
PED_AGE = 6          # 儿童剂量规则只管 6 岁以下
HARD_CATS = ("AGE", "PREG", "DOSE")


def _sev(ratio):
    if ratio >= 2.0:
        return "severe"
    if ratio > 1.5:
        return "mild"
    return None


def _cond_age(cond):
    m = re.match(r"age<(\d+(?:\.\d+)?)", cond)
    return float(m.group(1)) if m else None


def mentions(rec):
    """每处药名：位置、上下文判定、剂量。"""
    lex = lexicon()
    out = []
    for name, text, kind in rec.segs:
        found = list(lex.find(text or ""))
        for i, (s, e, alias, did) in enumerate(found):
            ctx = mention_context(text, s, e, did)
            if ctx["not_drug"] or ctx["stopped_here"]:
                continue
            nxt_s = found[i + 1][0] if i + 1 < len(found) else None
            prev_e = found[i - 1][1] if i > 0 else None
            compound = (nxt_s is not None and text[e:nxt_s] in ("", "/", "／")) \
                or (prev_e is not None and text[prev_e:s] in ("", "/", "／")) \
                or bool(COMPOUND_BEFORE.search(text[max(0, s - 4):s]))
            nxt = found[i + 1][0] if i + 1 < len(found) else len(text)
            wend = min(nxt, ctx["clause_end"], e + WINDOW)
            dose = parse_dose(text[e:wend], text[max(0, s - 8):s]) if not ctx["negated"] else None
            out.append({"seg": name, "kind": kind, "start": s, "end": e, "alias": alias, "drug_id": did,
                        "negated": ctx["negated"], "stop": ctx["stop"], "topical": ctx["topical"], "dose": dose,
                        "not_current": ctx["not_current"], "compound": compound})
    return out


def _hit(rule_id, cat, sev, m, rec, preg, **kw):
    h = {"rule_id": rule_id, "category": cat, "severity": sev,
         "hard": cat in HARD_CATS and sev in ("severe", "mild"),
         "drug_id": m["drug_id"], "alias": m["alias"], "seg": m["seg"], "start": m["start"], "end": m["end"],
         "age": rec.age, "age_hi": rec.age_hi, "sex": rec.sex, "pregnant": preg}
    d = m.get("dose") or {}
    h.update({"dose_text": d.get("text"), "single_mg": d.get("single_mg"), "daily_mg": d.get("daily_mg"), "freq": d.get("freq")})
    h.update(kw)
    return h


def evaluate(rec):
    lex, R = lexicon(), rules()
    end_scope = "\n".join(t for _, t, k in rec.segs if k in ("course", "discharge"))
    preg = is_pregnant(rec.dx_text, end_scope, rec.sex, rec.age)
    ms = mentions(rec)
    hits = []
    for m in ms:
        if m["negated"] or m["not_current"]:
            continue
        did = m["drug_id"]
        for r in R.rules:
            cat, target, cond = r["category"], r["target"], r["condition"]
            if cat not in ("AGE", "PREG", "SEX"):
                continue
            if r["systemic_only"] == "1" and m["topical"]:
                continue
            if not lex.in_target(did, target):
                continue
            if cat == "AGE":
                lim = _cond_age(cond)
                if rec.age is not None and lim is not None and rec.age < lim:
                    hits.append(_hit(r["rule_id"], cat, r["severity"], m, rec, preg, limit_age=lim))
            elif cat == "PREG":
                if preg and not m["stop"]:
                    hits.append(_hit(r["rule_id"], cat, r["severity"], m, rec, preg))
            elif cat == "SEX":
                want = cond.split("=")[1]
                if rec.sex == want:
                    hits.append(_hit(r["rule_id"], cat, r["severity"], m, rec, preg))
        d = R.dose.get(did)
        dose = m["dose"]
        if not d or not dose or m["topical"] or m["compound"]:
            continue
        amount = dose["daily_mg"] if dose["daily_mg"] is not None else dose["single_mg"]
        basis = "daily" if dose["daily_mg"] is not None else "single"
        if amount is None:
            continue
        if rec.age is not None and rec.age < PED_AGE:
            if d["ped_max_mgkg"]:
                w = weight_p50(rec.age_hi if rec.age_hi is not None else rec.age)   # 整岁按上限取体重
                lim = d["ped_max_mgkg"] * w
                if d["ped_cap"]:
                    lim = min(lim, d["ped_cap"])
                ratio = amount / lim
                sev = _sev(ratio)
                if sev:
                    hits.append(_hit("D-DOSE-PED", "DOSE", sev, m, rec, preg, limit_mg=round(lim, 1),
                                     ratio=round(ratio, 2), basis=basis, weight_kg=round(w, 1)))
            elif d["ped_adult_threshold"] and amount >= d["ped_adult_threshold"]:
                hits.append(_hit("D-PEDADULT", "PEDADULT", "warning", m, rec, preg,
                                 limit_mg=d["ped_adult_threshold"], ratio=round(amount / d["ped_adult_threshold"], 2), basis=basis))
        elif d["adult_max"]:
            ratio = amount / d["adult_max"]
            sev = _sev(ratio)
            if sev:
                hits.append(_hit("D-DOSE-ADULT", "DOSE", sev, m, rec, preg, limit_mg=d["adult_max"],
                                 ratio=round(ratio, 2), basis=basis))
    n1 = []
    for name, text, kind in rec.segs:
        for h in n1_hits(text):
            h["seg"] = name
            n1.append(h)
    systemic = [m for m in ms if not m["negated"] and not m["topical"] and not m["not_current"]]
    return {"rid": rec.rid, "age": rec.age, "sex": rec.sex, "pregnant": preg,
            "mentions": ms, "hits": hits, "n1": n1,
            "spec": {"mentions": len(systemic), "with_dose": sum(1 for m in systemic if m["dose"])}}


def summarize(res):
    hs = res["hits"]
    row = {"rid": res["rid"], "age": res["age"], "sex": res["sex"], "pregnant": res["pregnant"],
           "n_mentions": res["spec"]["mentions"], "n_with_dose": res["spec"]["with_dose"],
           "any_hard": any(h["hard"] for h in hs),
           "any_severe": any(h["hard"] and h["severity"] == "severe" for h in hs),
           "any_pedadult": any(h["category"] == "PEDADULT" for h in hs),
           "any_sex": any(h["category"] == "SEX" for h in hs),
           "any_warning": any(h["severity"] == "warning" for h in hs),
           "any_n1": bool(res["n1"])}
    for h in hs:
        row["n_" + h["rule_id"]] = row.get("n_" + h["rule_id"], 0) + 1
    return row
