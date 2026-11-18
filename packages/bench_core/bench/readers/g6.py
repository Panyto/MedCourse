"""关6：同病实体重合（和真实行配对截等长，同 Q6 表 H6-3a）、Self-BLEU、Vendi、不同年龄数、留出词表覆盖率。

Vendi 依赖 scikit-learn，缺时记 None。覆盖率口径同 Q6 H6-1：参考词表由 REF 建，HOLD 是 15 个真人；
合成行每个头部诊断取槽位号排序后的前 15 份，读数是 合成覆盖 − HOLD 覆盖，诊断间等权平均。
"""
import random

from q45_kit import demog
from q45_kit.realtext import care7_from_admission
from q6_kit import diversity, entities, kcov, split
from ruler_g1 import extract_facts, split_sections
from ruler_g3.adapters import care_get

N_AGE = 256


def to_care7(x):
    if x["kind"] == "care7":
        return x["record"]
    rec, _ = care7_from_admission({"admission_id": x["id"], "docs": x["docs"]}, split_sections, extract_facts)
    return rec


def _codes_of(ids, codes):
    by = {}
    for i in ids:
        c = codes.get(i)
        if c:
            by.setdefault(c, []).append(i)
    return by


def _vendi_by_code(records, by_code):
    vals, used = [], 0
    for c, ids in sorted(by_code.items()):
        if len(ids) < 2:
            continue
        texts = [diversity.record_text(records[i]) for i in ids]
        try:
            vals.append(diversity.vendi(texts)[0])
        except ImportError:
            return None, 0
        except ValueError:      # 负对照投影后全空，词表为空：这个诊断不计
            continue
        used += 1
    return (sum(vals) / len(vals) if vals else None), used


def _self_bleu(records, by_code):
    vals = []
    for c, ids in sorted(by_code.items()):
        if len(ids) >= 2:
            vals.append(diversity.self_bleu([(i, diversity.record_text(records[i])) for i in ids]))
    return sum(vals) / len(vals) if vals else None


def _kcov_rel(records, ref_records, codes):
    """头部诊断（够 30 份能分 HOLD、REF 的）上的 合成覆盖 − HOLD 覆盖。"""
    by_code = _codes_of(sorted(ref_records), codes)
    sp = split.split_hold_ref(by_code)
    vals = []
    for c, (hold, ref) in sorted(sp.items()):
        if not hold:
            continue
        mine = sorted(i for i in records if codes.get(i) == c and i not in set(hold))[:15]
        if len(mine) < 15:
            continue
        vocab = kcov.ref_vocab([ref_records[i] for i in ref])
        vals.append(kcov.kcov([records[i] for i in mine], vocab)["all"] - kcov.kcov([ref_records[i] for i in hold], vocab)["all"])
    return (sum(vals) / len(vals) if vals else None), len(vals)


def run(rows, config):
    by = {}
    for x in rows:
        rec = to_care7(x)
        if rec is not None:
            by.setdefault(x["row"], {})[x["id"]] = rec
    all_codes = config.get("codes") or {}
    pairs = config.get("pairs") or {}
    out = []
    for row, recs in sorted(by.items()):
        codes = all_codes.get(row) or {}
        rng = random.Random(f"20261101|age|{row}")
        ids = sorted(recs)
        sample = ids if len(ids) <= N_AGE else rng.sample(ids, N_AGE)
        ages = {demog.stated_age(care_get(recs[i], "age") or "") for i in sample} - {None}
        out.append({"row": row, "metric": "distinct_age_256", "value": len(ages), "n": len(sample)})
        by_code = _codes_of(ids, codes)
        v, used = _vendi_by_code(recs, by_code)
        out.append({"row": row, "metric": "vendi", "value": v, "n": used})
        out.append({"row": row, "metric": "self_bleu", "value": _self_bleu(recs, by_code), "n": len(by_code)})
        ref = pairs.get(row)
        if ref and ref in by:
            common = sorted(set(recs) & set(by[ref]))
            mine, theirs = {}, {}
            for i in common:
                a, b = diversity.truncate_pair(recs[i], by[ref][i])
                mine[i], theirs[i] = entities.key_set(a), entities.key_set(b)
            slots = _codes_of(common, codes)
            out.append({"row": row, "metric": "jaccard", "value": diversity.within_code_jaccard(mine, slots), "n": len(common)})
            out.append({"row": row, "metric": "jaccard_ref", "value": diversity.within_code_jaccard(theirs, slots), "n": len(common)})
            rel, n_codes = _kcov_rel(recs, by[ref], codes)
            out.append({"row": row, "metric": "kcov_rel", "value": rel, "n": n_codes})
    return [], out
