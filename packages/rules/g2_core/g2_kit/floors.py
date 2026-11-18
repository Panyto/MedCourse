"""地板（校7）和退化解（校10）。

地板：
  floor_easy     每篇文书换成另一位随机病人同一角色的文书（入院记录保留本人的）
  floor_hard     同上，但供体限同科室、同疾病章、住院天数相差 ≤2 天
  floor_shuffle  同一次住院，入院记录仍在最前、出院记录仍在最后，中间各篇的先后随机重排（日子跟着位置走）
退化解（都从一个干净住院构造）：
  degen_copy         每篇病程原样复制上一篇（首次病程之后的第二篇起都等于第一篇日常病程）
  degen_plan_only    删去所有含结果触发词的句子，只剩计划和描述
  degen_all_excluded 出院记录诊疗经过末尾写"鉴别诊断里的每一个都已排除"
  degen_template     同一科室的所有住院共用同一套病程（取该科室第一个住院的病程）
真实病历的地板版本只在内存里，不落盘。
"""
import re

from ruler_g2 import diagnoses as dxm


def _roles(adm):
    return [(d.idx, d.role or d.subtype) for d in adm.docs]


def _donor_doc(donor, role):
    for d in donor.docs:
        if (d.role or d.subtype) == role:
            return d
    return None


def _swap_docs(adm, donors, rng, source):
    new = adm.clone(source)
    swapped = 0
    for d in new.docs:
        if d.doc_type == "hospital_record":
            continue
        role = d.role or d.subtype
        cands = [x for x in donors if _donor_doc(x, role) is not None]
        if not cands:
            continue
        src = _donor_doc(rng.choice(cands), role)
        new.replace_text(d.idx, src.text)
        swapped += 1
    if swapped == 0:
        return None
    new.meta["swapped"] = swapped
    return new


def floor_easy(adm, pool, rng):
    donors = [x for x in pool if x.admission_id != adm.admission_id]
    return _swap_docs(adm, donors, rng, "floor:easy")


def floor_hard(adm, pool, meta, rng):
    """meta：{admission_id: {"dept":…, "chapter":…, "los":…}}"""
    m0 = meta.get(adm.admission_id)
    if not m0:
        return None
    donors = [x for x in pool if x.admission_id != adm.admission_id and x.admission_id in meta
              and meta[x.admission_id]["dept"] == m0["dept"] and meta[x.admission_id]["chapter"] == m0["chapter"]
              and abs(meta[x.admission_id]["los"] - m0["los"]) <= 2]
    if not donors:
        return None
    return _swap_docs(adm, donors, rng, "floor:hard")


def floor_shuffle(adm, rng, tries=10):
    mids = [d for d in adm.docs if d.doc_type == "progress_note"]
    if len(mids) < 2:
        return None
    for _ in range(tries):
        perm = mids[:]
        rng.shuffle(perm)
        if [d.idx for d in perm] != [d.idx for d in mids]:
            break
    else:
        return None
    new = adm.clone("floor:shuffle")
    slots = [(d.day, d.pos) for d in mids]          # 原来中间各篇占的日子和位置
    by_idx = {d.idx: d for d in new.docs}
    for (day, pos), d in zip(slots, perm):
        x = by_idx[d.idx]
        x.day, x.pos = day, pos
    new.docs.sort(key=lambda d: d.pos)
    return new


def degen_copy(adm):
    new = adm.clone("degen:copy")
    progs = [d for d in new.docs if d.doc_type == "progress_note" and d.subtype != "首次病程"]
    if len(progs) < 2:
        return None
    for d in progs[1:]:
        new.replace_text(d.idx, progs[0].text)
    return new


_RES = re.compile(r"示|提示|显示|回报|报告|结果|证实|可见")


def degen_plan_only(adm):
    new = adm.clone("degen:plan_only")
    for d in new.docs:
        out, cursor = [], 0
        text = d.text
        a = 0
        for i, ch in enumerate(text + "\n"):
            if ch in "。；;\n":
                if _RES.search(text[a:i]):
                    out.append(text[cursor:a])
                    cursor = min(i + 1, len(text))
                a = i + 1
        out.append(text[cursor:])
        new.replace_text(d.idx, "".join(out))
    return new


def degen_all_excluded(adm):
    fp, ds = adm.fp, adm.ds
    if fp is None or ds is None:
        return None
    diffs = [it.raw for it in dxm.differential_items(fp)]
    if not diffs:
        return None
    new = adm.clone("degen:all_excluded")
    t = new.doc(ds.idx)
    span = t.section("诊疗经过")
    piece = "经相关检查，" + "、".join(diffs) + "均已排除。"
    at = span[1] if span else len(t.text.rstrip())
    while at > 0 and t.text[at - 1] == "\n":
        at -= 1
    new.replace_text(t.idx, t.text[:at] + piece + t.text[at:])
    return new


def degen_template(pool, dept_of):
    """pool：[Admission]；dept_of：{admission_id: 科室}。返回 [Admission]（每个科室第一个住院原样保留）。"""
    first = {}
    out = []
    for adm in pool:
        dept = dept_of.get(adm.admission_id)
        if dept not in first:
            first[dept] = adm
            out.append(adm.clone("degen:template"))
            continue
        tpl = [d for d in first[dept].docs if d.doc_type == "progress_note"]
        new = adm.clone("degen:template")
        progs = [d for d in new.docs if d.doc_type == "progress_note"]
        for i, d in enumerate(progs):
            if tpl:
                new.replace_text(d.idx, tpl[min(i, len(tpl) - 1)].text)
        out.append(new)
    return out
