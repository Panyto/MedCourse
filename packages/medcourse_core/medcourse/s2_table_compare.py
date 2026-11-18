"""S2 准入四类事件建表 vs 现在的规则读数（0.1.13，方案 `方案_S2四类建表对比_v1.md`，老板 11-10 批准）。只出计数和比例。

python -B -m medcourse.s2_table_compare --path fold0.jsonl --dx-map dx_map.csv --events events_norm.jsonl --out tables/S2T_compare.csv

同一批住院两边各读一遍：
- 规则读数：和 stats.build_stats 同一套读数器（主诉、现病史的阳性症状词和部位补全，评测台结果口吻的检查，出院诊断清单，
  既往史公开词，入院体格检查的生命体征正则，病程写没写生命体征），逐住院记下；
- S2 读数：events_norm.jsonl 里准入四类（诊断、检查、症状、生命体征）的事件，按类型筛状态和文书、节，名称两种归一：
  甲（引文优先）在原文引文的位置上用同一套词表匹配，配不上用模型名称；乙（名称归一）把同一套词表用在模型名称上。

输出 metric,value 两列：不出病人、正文、引文、模型名称，只出公开词表里的固定词（既往史公开词、生命体征五项）做读数名。
分层的表照 k 压：层的住院数 < k 的不比，项 < k 的不进前 N 名。
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import statistics

from . import icd
from .rules import setup
from .safe import guarded, log
from .stats import (K, PAST_TERMS, QUAL_TAIL, SITE_TERMS, _NEG_BEFORE, _SYM_CLAUSE_END, _primary_name, _terms, dx_map_check, positive_symptoms,
                    site_regex)

VERSION = "s2t-compare-0.2"
TYPES = ("exam", "symptom", "diagnosis", "vital")
# 0.2（S2T v2，老板 11-10 批准）：评测台的检查词表一半是化验套餐（第十一次规划里检验类占 54.2%），S2 把它们归"化验"。
# 检查只比非检验类，两边同口径；化验单列（规则的检验类 vs S2 的化验事件），只描述
READ_TYPES = TYPES + ("lab",)
_NEG_WORD = re.compile(r"否认|无|未见|没有")    # 同 stats._NEG_BEFORE 的否定词
METHODS = ("jia", "yi")                 # 甲：引文优先；乙：名称归一
NOTE_SUBTYPES = ("日常查房", "主任医师查房", "主治医师查房", "上级医师查房", "其他病程")   # 同 stats 的 note_vitals
LINES = {   # 方案第 5 节（老板 11-10 批准）
    "exam": {"kinds_ratio": (0.8, 1.25), "top10_overlap": 0.7, "spearman20": 0.6},
    "symptom": {"chief_top5_overlap": 0.6, "chief_cov_drop": 0.05},
    "diagnosis": {"count_ratio": (0.8, 1.25), "comorb_top10_overlap": 0.7, "past_diff": 0.05},
    "vital": {"T": 0.2, "P": 3, "R": 1, "SBP": 5, "DBP": 5, "note_share_diff": 0.05},
}
TIE, COVER_GAP = 0.02, 0.10           # 方案第 2 节：一致度差 < 0.02 选甲；覆盖比另一种低 10 个百分点以上不选
_VITAL_ITEM = (("BP", re.compile(r"血压|BP", re.I)), ("T", re.compile(r"体温|T(?![a-zA-Z])")),
               ("P", re.compile(r"脉搏|心率|脉率|HR|P(?![a-zA-Z])")), ("R", re.compile(r"呼吸|R(?![a-zA-Z])")))
_DX_PIECE = re.compile(r"[；;，,、\n。]")


# ---------------- 名称归一 ----------------

class Vocab:
    def __init__(self):
        setup(("BENCH_G3", "BENCH_G2", "BENCH_G45"))
        from ruler_g2.events import EXAM_RE, EXAMS, NOT_PATH_AFTER, PRE_ADMISSION_SECTIONS, _ALIAS, _REGION, _region_before
        from ruler_g2.text import clause_span
        from ruler_g2 import diagnoses as dxm
        self.EXAM_RE, self.EXAMS, self.NOT_PATH, self.ALIAS, self.region_before, self.clause_span = (
            EXAM_RE, EXAMS, NOT_PATH_AFTER, _ALIAS, _region_before, clause_span)
        self.pre_adm = set(PRE_ADMISSION_SECTIONS)
        self.dxm = dxm
        symptoms = sorted(_terms(), key=len, reverse=True)
        self.sym_re = re.compile("|".join(re.escape(s) for s in symptoms))
        self.sites = site_regex(_REGION)

    def exam_keys(self, text, a=0, b=None):
        """text[a:b] 里的检查（键|部位），部位按评测台的写法在整段 text 上往前找（同 mentions_in）。"""
        b = len(text) if b is None else b
        out = []
        for m in self.EXAM_RE.finditer(text, a, b):
            key = self.ALIAS[m.group(0)]
            if key == "病理" and self.NOT_PATH.match(text, m.end()):
                continue
            ca, _ = self.clause_span(text, m.start(), m.end())
            region = self.region_before(text, m.start(), ca) if self.EXAMS[key]["region"] else None
            out.append(f"{key}|{region or ''}")
        return out

    def symptom_terms(self, text, a=0, b=None):
        """起点落在 text[a:b] 里的症状词。部位补全（往前 8 个字、不跨分句）和修饰词同 stats.positive_symptoms；
        否定不在这里判：S2 用事件状态，规则那边照 positive_symptoms。"""
        b = len(text) if b is None else b
        hits = [(m.start(), m.end(), m.group(0)) for m in self.sym_re.finditer(text, a, b)]
        hits += [(m.start(), m.end(), m.group(0)) for m in self.sites[1].finditer(text, a, b)]
        out = []
        for s0, e0, term in sorted(hits):
            start = 0
            for e in _SYM_CLAUSE_END.finditer(text, 0, s0):
                start = e.end()
            if term in SITE_TERMS:
                m = self.sites[0].search(text[max(start, s0 - 8):s0])
                term = (m.group(0) + term) if m else term
            q = next((x for x in QUAL_TAIL if text.startswith(x, e0)), None)
            if q and not term.endswith(q):
                term += q
            out.append(term)
        return out

    def is_lab(self, key):
        """"检查|部位"这个键在评测台分类里是不是检验（化验套餐）。"""
        return self.EXAMS.get((key or "").split("|", 1)[0], {}).get("cls") == "检验"

    @staticmethod
    def pick(cands, name_keys):
        """引文里配到多个时，先取模型名称也配到的那个，否则取第一个。"""
        for c in cands:
            if c in name_keys:
                return c
        return cands[0] if cands else None

    def dx_norm(self, text):
        return self.dxm.norm_dx(self.dxm.NUM_ITEM.sub("", text or "")).strip()

    def set_dx_vocab(self, names):
        """诊断的"词表"：规则这边全部出院诊断清单条目的归一名（评测台 norm_dx）。只在进程里用，不输出。
        几千个名字拼成一个正则太慢，改成逐个起点、从长到短查集合。"""
        self.dx_vocab = {x for x in names if len(x) >= 2}
        self.dx_maxlen = max((len(x) for x in self.dx_vocab), default=0)

    def _dx_hits(self, s):
        """s 里每个起点上最长的词表名。"""
        out = []
        for i in range(len(s) - 1):
            for j in range(min(len(s), i + self.dx_maxlen), i + 1, -1):
                if s[i:j] in self.dx_vocab:
                    out.append(s[i:j])
                    break
        return out

    def dx_from_evidence(self, ev, name):
        """甲：引文（按清单编号、标点切开后逐段归一）里配到的词表诊断名；先取和模型名称相同、互相包含的，否则取最长的。配不上返回 None。"""
        if not getattr(self, "dx_vocab", None):
            return None
        n = self.dx_norm(name)
        cands = []
        for piece in _DX_PIECE.split(self.dxm.NUM_ITEM.sub("\n", ev or "")):
            cands += self._dx_hits(self.dx_norm(piece))
        if not cands:
            return None
        same = [x for x in cands if x == n] or [x for x in cands if n and (n in x or x in n)]
        return (same or sorted(cands, key=len, reverse=True))[0]


def vital_item(text):
    for item, rx in _VITAL_ITEM:
        if rx.search(text or ""):
            return item
    return None


def vital_values(item, value, ev):
    """(项, 数) 列表；血压拆成收缩压、舒张压。"""
    s = str(value or "")
    if item == "BP":
        m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", s) or re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", ev or "")
        return [("SBP", float(m.group(1))), ("DBP", float(m.group(2)))] if m else []
    m = re.search(r"\d+(?:\.\d+)?", s)
    return [(item, float(m.group(0)))] if m and item else []


def normalize(vocab, e, text):
    """一条事件在甲、乙两种做法下归到的词（None 表示归不上）。text 是事件所在那篇文书的正文。"""
    t, name, ev = e["event_type"], e.get("name") or "", e.get("evidence") or ""
    a, b = e.get("ev_start"), e.get("ev_end")
    span_ok = isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(text)
    if t in ("exam", "lab"):
        nk = vocab.exam_keys(name)
        ek = vocab.exam_keys(text, a, b) if span_ok else vocab.exam_keys(ev)
        return {"jia": vocab.pick(ek, nk) or (nk[0] if nk else None), "yi": nk[0] if nk else None}
    if t == "symptom":
        nk = vocab.symptom_terms(name)
        ek = vocab.symptom_terms(text, a, b) if span_ok else vocab.symptom_terms(ev)
        return {"jia": vocab.pick(ek, nk) or (nk[0] if nk else None), "yi": nk[0] if nk else None}
    if t == "diagnosis":
        j = vocab.dx_from_evidence(ev, name)
        y = vocab.dx_norm(name)
        y = y if len(y) >= 2 else None
        return {"jia": j or y, "yi": y}
    if t == "vital":
        return {"jia": vital_item(ev) or vital_item(name), "yi": vital_item(name)}
    return {"jia": None, "yi": None}


# ---------------- 逐住院读数 ----------------

def neg_distance_bucket(t, pos):
    """既往史里规则判"有"的那次命中，同一句（到上一个句号、分号、换行为止）里前面最近的否定词离它多远：
    none 同一句前面没有否定词；le10 ≤ 10 字（中间隔着逗号，规则的否定窗口不跨逗号）；gt10 > 10 字（长清单，规则 10 字窗口够不着）。"""
    start = max(t.rfind(ch, 0, pos) for ch in "。；;\n") + 1
    negs = list(_NEG_WORD.finditer(t, start, pos))
    if not negs:
        return "none"
    return "le10" if pos - negs[-1].end() <= 10 else "gt10"


def rule_reading(vocab, adm, h):
    """和 stats.build_stats 同一套读数器，逐住院。h 是评测台模块（build_real 等）。"""
    out = {"syms": set(), "chief": set(), "past": set(), "dx": [], "exams": set(), "exam_tone": {}, "vitals": {}, "notes": {},
           "exams_nonlab": set(), "labs": set(), "past_bucket": {}, "sym_text": set()}
    hr, ds = adm.hr, adm.ds
    if hr is not None:
        for name in PAST_TERMS:
            sec = hr.section("既往史")
            if sec:
                t = hr.text[sec[0]:sec[1]]
                for f in re.finditer(name, t):
                    if not _NEG_BEFORE.search(t[max(0, f.start() - 12):f.start()]):
                        out["past"].add(name)
                        out["past_bucket"][name] = neg_distance_bucket(t, f.start())
                        break
        cc, hpi = hr.section("主诉"), hr.section("现病史")
        for sec in (cc, hpi):
            if sec:
                out["syms"] |= set(positive_symptoms(hr.text[sec[0]:sec[1]], vocab.sym_re, vocab.sites))
                # 0.2：不管否定、这两节里写到的全部症状词（给"S2 有、规则没有"归因：规则按否定去掉了，还是原文根本没有）
                out["sym_text"] |= set(vocab.symptom_terms(hr.text, sec[0], sec[1]))
        if cc:
            out["chief"] = set(positive_symptoms(hr.text[cc[0]:cc[1]], vocab.sym_re, vocab.sites))
        sec = hr.section("体格检查")
        if sec:
            t = hr.text[sec[0]:sec[1]][:200]
            x = h["T"].search(t)
            if x:
                out["vitals"]["T"] = float(x.group(1))
            for rx in h["rates"]:
                for x in rx.finditer(t):
                    key = {"脉搏": "P", "心率": "P", "呼吸": "R", "P": "P", "HR": "P", "R": "R"}[x.group(1)]
                    out["vitals"].setdefault(key, float(x.group(2)))
            x = h["BP"].search(t)
            if x:
                out["vitals"]["SBP"], out["vitals"]["DBP"] = float(x.group(1)), float(x.group(2))
    if ds is not None:
        out["dx"] = [it.norm for it in h["dxm"].list_items(ds) if it.section == "出院诊断"]
    ments = h["admission_mentions"](adm)
    for d in adm.docs:
        for x in ments[d.idx]:
            k = f"{x.key}|{x.region or ''}"
            out["exam_tone"].setdefault(k, set()).add(x.tone)
            if x.tone == "R":
                out["exams"].add(k)
                out["labs" if vocab.is_lab(k) else "exams_nonlab"].add(k)
        if d.doc_type == "progress_note" and d.subtype in NOTE_SUBTYPES:
            hits = bool(h["T"].search(d.text)) + bool(h["BP"].search(d.text)) + any(rx.search(d.text) for rx in h["rates"])
            out["notes"][d.idx] = hits >= 2
    return out


def s2_reading(vocab, events, adm):
    """一次住院的 S2 读数，两种归一各一份；另记三向计数要的原始量。"""
    docs = {d.idx: d for d in adm.docs}
    hr_idx = adm.hr.idx if adm.hr is not None else None
    ds_idx = adm.ds.idx if adm.ds is not None else None
    r = {m: {"syms": set(), "chief": set(), "past": set(), "dx": set(), "exams": set(), "exams_hrds": set(), "vitals": {}, "notes": {},
             "prog_syms": collections.Counter(), "exam_planned": {}, "exam_done_day": {},
             "course_exam_planned_day": {}, "course_exam_done_day": {}, "prog_sym_days": {}, "fp_day0_syms": set(),
             "labs": set(), "exam_status": {}, "sym_status": {}, "past_status": {}} for m in METHODS}
    raw = []
    for e in events:
        t = e.get("event_type")
        if t not in READ_TYPES:
            continue
        d = docs.get(e.get("doc_idx"))
        if d is None:
            continue
        sec, st = e.get("section") or "", e.get("assertion")
        nm = normalize(vocab, e, d.text)
        name_in_ev = re.sub(r"\s", "", e.get("name") or "") in re.sub(r"\s", "", e.get("evidence") or "")
        raw.append((t, st, d.doc_type, sec if d.doc_type != "progress_note" else "病程", nm, name_in_ev))
        for m in METHODS:
            key = nm[m]
            if key is None:
                continue
            x = r[m]
            pre = d.idx == hr_idx and sec in vocab.pre_adm      # 入院记录的入院前段落（现病史、既往史……），同评测台记 H
            if t == "symptom":
                if d.idx == hr_idx and sec in ("主诉", "现病史"):
                    x["sym_status"].setdefault(key, set()).add(st or "none")
                    if st == "done":
                        x["syms"].add(key)
                        if sec == "主诉":
                            x["chief"].add(key)
                if d.doc_type == "progress_note":
                    if st == "done":
                        x["prog_syms"][d.idx] += 1
                        if d.day == 0:
                            x["fp_day0_syms"].add(key)
                    if d.day is not None and d.day >= 1 and st in ("done", "negated"):
                        x["prog_sym_days"].setdefault(key, {"done": set(), "negated": set()})[st].add(int(d.day))
            elif t in ("exam", "lab"):
                # 0.2：两类事件都记状态（给规则那边的检查归因）；检查只留非检验类、去掉入院前段落；化验事件只留检验类
                x["exam_status"].setdefault(key, set()).add(("pre_" if pre else "") + (st or "none"))
                if pre:
                    continue
                if t == "lab":
                    if st == "done" and vocab.is_lab(key):
                        x["labs"].add(key)
                    continue
                if vocab.is_lab(key):
                    continue
                if st == "done":
                    x["exams"].add(key)
                    if d.idx in (hr_idx, ds_idx):
                        x["exams_hrds"].add(key)
                    if d.day is not None:
                        x["exam_done_day"][key] = min(x["exam_done_day"].get(key, d.day), d.day)
                        if d.idx != ds_idx:
                            x["course_exam_done_day"][key] = min(x["course_exam_done_day"].get(key, d.day), int(d.day))
                elif st == "planned" and d.day is not None:
                    x["exam_planned"][key] = min(x["exam_planned"].get(key, d.day), d.day)
                    if d.idx != ds_idx:
                        x["course_exam_planned_day"][key] = min(x["course_exam_planned_day"].get(key, d.day), int(d.day))
            elif t == "diagnosis":
                if d.idx == ds_idx and sec == "出院诊断" and st == "done":
                    x["dx"].add(key)
                if d.idx == hr_idx and sec == "既往史":
                    for p in PAST_TERMS:
                        if p in key:
                            x["past_status"].setdefault(p, set()).add(st or "none")
                    if st in ("done", "history"):
                        x["past"] |= {p for p in PAST_TERMS if p in key}
            elif t == "vital" and st == "done":
                vals = vital_values(key, e.get("value"), e.get("evidence"))
                if d.idx == hr_idx and sec == "体格检查":
                    for k, v in vals:
                        x["vitals"].setdefault(k, v)
                if d.doc_type == "progress_note" and d.subtype in NOTE_SUBTYPES:
                    x["notes"].setdefault(d.idx, set()).update(k for k, _ in vals)
    for m in METHODS:
        x = r[m]
        x["notes"] = {i: (("T" in s) + (("SBP" in s) or ("DBP" in s)) + (("P" in s) or ("R" in s))) >= 2 for i, s in x["notes"].items()}
    return r, raw


# ---------------- 汇总 ----------------

def jaccard(a, b):
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return "NA"
    i = (len(xs) - 1) * p
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 4)


def _ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for x in order[i:j + 1]:
            r[x] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def top(counter, n, k):
    return [x for x, c in sorted(counter.items(), key=lambda z: (-z[1], z[0])) if c >= k][:n]


def overlap(a, b):
    if not a or not b:
        return None
    return len(set(a) & set(b)) / min(len(a), len(b))


def strata_compare(rows, n, k, rank_n=None):
    """rows：[(层元组, 规则集合, S2 集合)]。每层（住院数 ≥ k）比前 n 名重合率、（给了 rank_n 时）前 rank_n 名的秩相关。
    前 n 名只取计数 ≥ k 的项（同统计表压格子）；一边没有够 k 的项时这一层不比，计进 skipped。"""
    by = {}
    for st, a, b in rows:
        for s in st:
            g = by.setdefault(s, [0, collections.Counter(), collections.Counter()])
            g[0] += 1
            g[1].update(a)
            g[2].update(b)
    ov, sp, skipped = [], [], 0
    for s, (cnt, ca, cb) in by.items():
        if cnt < k:
            continue
        ta, tb = top(ca, n, k), top(cb, n, k)
        o = overlap(ta, tb)
        if o is None:
            skipped += 1
            continue
        ov.append(o)
        if rank_n:
            keys = sorted(set(top(ca, rank_n, k)) | set(top(cb, rank_n, k)))
            r = spearman([ca[x] for x in keys], [cb[x] for x in keys])
            if r is not None:
                sp.append(r)
    return ov, sp, skipped


def layer_keys(st):
    """stats 的层元组里取三位码、节、章三层（不要全局层和诊断名层）。"""
    return tuple(s for s in st if s != "*" and not s.startswith("n:"))


def compare(lines, codes, events_by, k=K):
    vocab = Vocab()
    pre = rule_pass(lines, codes, vocab)
    vocab.set_dx_vocab({x for *_, rr in pre for x in rr["dx"]})
    c, per, three = collections.Counter(), [], collections.Counter()
    # 第二遍：S2 读数
    for rid, adm, st, pname, age, rr in pre:
        sr, raw = s2_reading(vocab, events_by.get(rid, ()), adm)
        for t, a, dt, sec, nm, inev in raw:
            g = "inev" if inev else "notinev"
            three[(t, a or "none", dt)] += 1
            c[f"sec_{t}_{dt}_{sec}"] += 1
            for m in METHODS:
                c[f"norm_{t}_{m}_{g}_n"] += 1
                # 诊断的"归上"是落在规则的诊断词表里；其余三类是词表配上了
                c[f"norm_{t}_{m}_{g}_mapped"] += (nm[m] in vocab.dx_vocab) if t == "diagnosis" else (nm[m] is not None)
            if nm["jia"] is not None and nm["yi"] is not None:
                c[f"norm_{t}_both_{g}"] += 1
                c[f"norm_{t}_conflict_{g}"] += nm["jia"] != nm["yi"]
        per.append((st, pname, age, rr, sr))
    return summarize(per, three, c, k)


def rule_pass(lines, codes, vocab):
    """第一遍：逐住院的规则读数和分层（同 stats.build_stats：编码表没有的三位码，过了 dx_map_check 才用公开词典补）。
    返回 [(住院号, Admission, 层元组, 主诊断名, 年龄, 读数)]。"""
    setup(("BENCH_G3", "BENCH_G2", "BENCH_G45"))
    from ruler_g2 import build_real
    from ruler_g2 import diagnoses as dxm
    from ruler_g2.events import admission_mentions
    from ruler_g3.adapters import from_admission
    from ruler_g3.numbers import _BP, _RATE_CN, _RATE_EN, _T     # 同 stats.build_stats
    h = {"dxm": dxm, "admission_mentions": admission_mentions, "T": _T, "BP": _BP, "rates": (_RATE_CN, _RATE_EN)}
    pnames = {str(o["admission_id"]): _primary_name(build_real({"admission_id": str(o["admission_id"]), "docs": o["docs"]}).ds, dxm) for o in lines}
    dx_meta = dx_map_check(pnames, codes)
    pre = []
    for obj in lines:
        rid = str(obj["admission_id"])
        adm = build_real({"admission_id": rid, "docs": obj["docs"]})
        rec = from_admission({"admission_id": rid, "docs": obj["docs"]}, rid)
        pname = pnames.get(rid)
        c3 = codes.get(rid) or ""
        if not c3:
            c3 = (dx_meta["use_exact"] and pname and icd.code3_exact(pname)) or (dx_meta["use_stem"] and pname and icd.code3_stem(pname)) or ""
        ch, blk = c3[:1] or None, icd.block_of(c3)
        st = ("*",) + ((ch,) if ch else ()) + ((blk,) if blk else ()) + ((c3,) if len(c3) == 3 else ())
        pre.append((rid, adm, st, pname, rec.age, rule_reading(vocab, adm, h)))
    return pre


def summarize(per, three, c, k):
    out = []
    n = len(per)
    out.append(("admissions", n))
    for (t, a, dt), v in sorted(three.items()):
        out.append((f"three_{t}_{a}_{dt}", v))
    for key, v in sorted(c.items()):
        out.append((key, v))
    rule_vocab_dx = {x for _, _, _, rr, _ in per for x in rr["dx"]}
    choice = {}
    for t in TYPES:
        jac, cov = {}, {}
        for m in METHODS:
            js = []
            for _, _, _, rr, sr in per:
                a, b = _sets(t, rr, sr[m])
                j = jaccard(a, b)
                if j is not None:
                    js.append(j)
            jac[m] = js
            nm = sum(c[f"norm_{t}_{m}_{g}_n"] for g in ("inev", "notinev"))
            mp = sum(c[f"norm_{t}_{m}_{g}_mapped"] for g in ("inev", "notinev"))
            cov[m] = mp / nm if nm else 0.0
            out += [(f"{t}_{m}_coverage", round(cov[m], 4)), (f"{t}_{m}_jaccard_n", len(js)),
                    (f"{t}_{m}_jaccard_p25", _q(js, 0.25)), (f"{t}_{m}_jaccard_p50", _q(js, 0.5)), (f"{t}_{m}_jaccard_p75", _q(js, 0.75))]
        both = sum(c[f"norm_{t}_both_{g}"] for g in ("inev", "notinev"))
        conf = sum(c[f"norm_{t}_conflict_{g}"] for g in ("inev", "notinev"))
        out.append((f"{t}_conflict_rate", round(conf / both, 4) if both else "NA"))
        for g in ("inev", "notinev"):
            b_ = c[f"norm_{t}_both_{g}"]
            out.append((f"{t}_conflict_rate_{g}", round(c[f"norm_{t}_conflict_{g}"] / b_, 4) if b_ else "NA"))
        mj, my = (statistics.median(jac[m]) if jac[m] else 0.0 for m in METHODS)
        pick = "jia" if mj >= my or abs(mj - my) < TIE else "yi"
        other = "yi" if pick == "jia" else "jia"
        if cov[pick] < cov[other] - COVER_GAP:
            pick = other
        choice[t] = pick
        out.append((f"{t}_chosen", pick))
        if t == "diagnosis":
            for m in METHODS:
                hit = [x for _, _, _, _, sr in per for x in sr[m]["dx"]]
                out.append((f"diagnosis_{m}_in_rule_vocab", round(sum(1 for x in hit if x in rule_vocab_dx) / len(hit), 4) if hit else "NA"))
    for t in TYPES:
        for m in METHODS:
            out += main_readings(t, m, per, k)
        out += attribution(t, choice[t], per)
    out += extras(per, choice)
    out += verdicts(out, choice)
    return out


def _sets(t, rr, x):
    if t == "exam":
        return rr["exams_nonlab"], x["exams"]      # 0.2：两边都只要非检验类
    if t == "symptom":
        return rr["syms"], x["syms"]
    if t == "diagnosis":
        return set(rr["dx"]), x["dx"]
    return ({("BP" if i in ("SBP", "DBP") else i) for i in rr["vitals"]}, {("BP" if i in ("SBP", "DBP") else i) for i in x["vitals"]})


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main_readings(t, m, per, k):
    out, p = [], f"{t}_{m}"
    if t == "exam":
        # 0.2：规则只取非检验类（rr["exams_nonlab"]）；S2 的检查事件在 s2_reading 里已去掉检验类和入院前段落
        ra, sa = _mean([len(rr["exams_nonlab"]) for _, _, _, rr, _ in per]), _mean([len(sr[m]["exams"]) for _, _, _, _, sr in per])
        ov, sp, sk = strata_compare([(layer_keys(st), rr["exams_nonlab"], sr[m]["exams"]) for st, _, _, rr, sr in per], 10, k, 20)
        out += [(f"{p}_kinds_rule_mean", round(ra, 4)), (f"{p}_kinds_s2_mean", round(sa, 4)), (f"{p}_kinds_ratio", round(sa / ra, 4) if ra else "NA"),
                (f"{p}_top10_overlap_p50", _q(ov, 0.5)), (f"{p}_top10_overlap_p25", _q(ov, 0.25)), (f"{p}_top10_strata", len(ov)),
                (f"{p}_spearman20_p50", _q(sp, 0.5)), (f"{p}_spearman20_strata", len(sp)), (f"{p}_strata_skipped", sk)]
        hd = [len(sr[m]["exams_hrds"]) / len(sr[m]["exams"]) for _, _, _, _, sr in per if sr[m]["exams"]]
        out.append((f"{p}_share_in_hr_ds_p50", _q(hd, 0.5)))
        out.append((f"{p}_rule_all_kinds_mean", round(_mean([len(rr["exams"]) for _, _, _, rr, _ in per]), 4)))
        # 化验单列（只描述，化验没准入）：规则的检验类 vs S2 的化验事件
        la, lb = _mean([len(rr["labs"]) for _, _, _, rr, _ in per]), _mean([len(sr[m]["labs"]) for _, _, _, _, sr in per])
        lj = [j for j in (jaccard(rr["labs"], sr[m]["labs"]) for _, _, _, rr, sr in per) if j is not None]
        out += [(f"lab_{m}_kinds_rule_mean", round(la, 4)), (f"lab_{m}_kinds_s2_mean", round(lb, 4)), (f"lab_{m}_kinds_ratio", round(lb / la, 4) if la else "NA"),
                (f"lab_{m}_jaccard_p50", _q(lj, 0.5))]
    elif t == "symptom":
        ov, _, sk = strata_compare([(layer_keys(st), rr["chief"], sr[m]["chief"]) for st, _, _, rr, sr in per], 5, k)
        cr = _mean([1.0 if rr["chief"] else 0.0 for _, _, _, rr, _ in per])
        cs = _mean([1.0 if sr[m]["chief"] else 0.0 for _, _, _, _, sr in per])
        ra, sa = _mean([len(rr["syms"]) for _, _, _, rr, _ in per]), _mean([len(sr[m]["syms"]) for _, _, _, _, sr in per])
        out += [(f"{p}_chief_top5_overlap_p50", _q(ov, 0.5)), (f"{p}_chief_top5_strata", len(ov)), (f"{p}_strata_skipped", sk),
                (f"{p}_chief_cov_rule", round(cr, 4)), (f"{p}_chief_cov_s2", round(cs, 4)),
                (f"{p}_syms_rule_mean", round(ra, 4)), (f"{p}_syms_s2_mean", round(sa, 4)), (f"{p}_syms_ratio", round(sa / ra, 4) if ra else "NA")]
    elif t == "diagnosis":
        ra, sa = _mean([len(rr["dx"]) for _, _, _, rr, _ in per]), _mean([len(sr[m]["dx"]) for _, _, _, _, sr in per])
        rows = []
        for st, pname, _, rr, sr in per:
            ch = tuple(s for s in st if len(s) == 1 and s != "*")
            rows.append((ch, set(rr["dx"]) - {pname}, sr[m]["dx"] - {pname}))
        ov, _, sk = strata_compare(rows, 10, k)
        out += [(f"{p}_count_rule_mean", round(ra, 4)), (f"{p}_count_s2_mean", round(sa, 4)), (f"{p}_count_ratio", round(sa / ra, 4) if ra else "NA"),
                (f"{p}_comorb_top10_overlap_p50", _q(ov, 0.5)), (f"{p}_comorb_strata", len(ov)), (f"{p}_strata_skipped", sk)]
        diffs = []
        for term in PAST_TERMS:
            r_ = _mean([1.0 if term in rr["past"] else 0.0 for _, _, _, rr, _ in per])
            s_ = _mean([1.0 if term in sr[m]["past"] else 0.0 for _, _, _, _, sr in per])
            diffs.append(abs(s_ - r_))
            out += [(f"{p}_past_{term}_rule", round(r_, 4)), (f"{p}_past_{term}_s2", round(s_, 4))]
        out.append((f"{p}_past_max_diff", round(max(diffs), 4)))
    else:
        for item in ("T", "P", "R", "SBP", "DBP"):
            rv = [rr["vitals"][item] for _, _, age, rr, _ in per if item in rr["vitals"] and (age is None or age >= 18)]
            sv = [sr[m]["vitals"][item] for _, _, age, _, sr in per if item in sr[m]["vitals"] and (age is None or age >= 18)]
            out.append((f"{p}_{item}_n_rule", len(rv)))
            out.append((f"{p}_{item}_n_s2", len(sv)))
            for q in (0.1, 0.5, 0.9):
                a_, b_ = _q(rv, q), _q(sv, q)
                out.append((f"{p}_{item}_q{int(q * 100)}_diff", round(b_ - a_, 4) if "NA" not in (a_, b_) else "NA"))
        rn = [v for _, _, _, rr, _ in per for v in rr["notes"].values()]
        sn = [sr[m]["notes"].get(i, False) for _, _, _, rr, sr in per for i in rr["notes"]]
        r_, s_ = _mean([1.0 if v else 0.0 for v in rn]), _mean([1.0 if v else 0.0 for v in sn])
        out += [(f"{p}_note_share_rule", round(r_, 4)), (f"{p}_note_share_s2", round(s_, 4)), (f"{p}_note_share_diff", round(s_ - r_, 4)),
                (f"{p}_note_n", len(rn)), (f"{p}_note_share_rule_se", round(math.sqrt(r_ * (1 - r_) / len(rn)), 4) if rn else "NA")]
    return out


def _status_cat(statuses, pre=False):
    """S2 在同一住院里对同一个词的状态归成一类：absent 没抽到；planned、negated、uncertain、history、excluded、not_patient；
    pre=True 时只在入院记录入院前段落出现的记 pre_only；其余 other。按这个顺序取第一个命中的。"""
    if not statuses:
        return "absent"
    plain = {s[4:] if s.startswith("pre_") else s for s in statuses}
    if pre and all(s.startswith("pre_") for s in statuses):
        return "pre_only"
    for cat in ("planned", "negated", "uncertain", "history", "excluded", "not_patient"):
        if cat in plain:
            return cat
    return "other"


def attribution(t, m, per):
    """方案批文的活口：差出来的那部分是 S2 剔掉了规则误抓、还是补上了规则漏认。只出计数。"""
    out, c = [], collections.Counter()
    if t == "exam":
        for _, _, _, rr, sr in per:
            x = sr[m]
            for key in rr["exams_nonlab"] - x["exams"]:
                c["rule_only_s2_" + _status_cat(x["exam_status"].get(key), pre=True)] += 1
            for key in x["exams"] - rr["exams_nonlab"]:
                tones = rr["exam_tone"].get(key, set())
                c["s2_only_rule_tone_" + ("none" if not tones else "".join(sorted(tones)))] += 1
    elif t == "symptom":
        for _, _, _, rr, sr in per:
            for key in rr["syms"] - sr[m]["syms"]:
                c["rule_only_s2_" + _status_cat(sr[m]["sym_status"].get(key))] += 1
            for key in sr[m]["syms"] - rr["syms"]:
                # 原文这两节里写到了这个词、规则按否定去掉了，还是原文根本没有这个词（S2 的词来自模型名称）
                c["s2_only_" + ("rule_negated" if key in rr["sym_text"] else "not_in_text")] += 1
    elif t == "diagnosis":
        for _, pname, _, rr, sr in per:
            c["rule_only"] += len(set(rr["dx"]) - sr[m]["dx"])
            c["s2_only"] += len(sr[m]["dx"] - set(rr["dx"]))
            # 0.2：既往史公开词。规则判"有"的：离前面最近的否定词多远（none / le10 / gt10）× S2 怎么记的
            for term in PAST_TERMS:
                if term in rr["past"]:
                    s2 = "pos" if term in sr[m]["past"] else _status_cat(sr[m]["past_status"].get(term))
                    b = rr["past_bucket"].get(term, "none")
                    c[f"past_{term}_{b}_{s2}"] += 1
                    c[f"past_all_{b}_{s2}"] += 1
                elif term in sr[m]["past"]:
                    c[f"past_{term}_s2_only"] += 1
                    c["past_all_s2_only"] += 1
    else:
        for _, _, _, rr, sr in per:
            a, b = set(rr["vitals"]), set(sr[m]["vitals"])
            c["rule_only_items"] += len(a - b)
            c["s2_only_items"] += len(b - a)
            c["both_items_value_equal"] += sum(1 for i in a & b if abs(rr["vitals"][i] - sr[m]["vitals"][i]) < 1e-6)
            c["both_items"] += len(a & b)
    for key, v in sorted(c.items()):
        out.append((f"attr_{t}_{m}_{key}", v))
    return out


def extras(per, choice):
    """规则做不到的几样，只描述（方案第 3 节"另报"）。"""
    out = []
    ms, me = choice["symptom"], choice["exam"]
    notes = [n for _, _, _, _, sr in per for n in sr[ms]["prog_syms"].values()]
    out += [("extra_prog_notes_with_symptom", len(notes)), ("extra_prog_symptoms_per_note_p50", _q(notes, 0.5))]
    planned = [len(sr[me]["exam_planned"]) for _, _, _, _, sr in per]
    gaps, done = [], 0
    for _, _, _, _, sr in per:
        for key, d0 in sr[me]["exam_planned"].items():
            d1 = sr[me]["exam_done_day"].get(key)
            if d1 is not None and d1 >= d0:
                done += 1
                gaps.append(d1 - d0)
    out += [("extra_exam_planned_per_adm_mean", round(_mean(planned), 4)), ("extra_exam_planned_total", sum(planned)),
            ("extra_exam_planned_done_later", done), ("extra_exam_plan_to_done_days_p50", _q(gaps, 0.5))]
    return out


def verdicts(rows, choice):
    """按方案第 5 节的线，用选中的做法判。成立 1，不成立 0，读数缺 NA。"""
    v = dict(rows)
    out = []

    def get(key):
        x = v.get(key, "NA")
        return None if x == "NA" else float(x)

    def add(name, ok):
        out.append((name, "NA" if ok is None else int(ok)))
    m = choice["exam"]
    r, o, s = get(f"exam_{m}_kinds_ratio"), get(f"exam_{m}_top10_overlap_p50"), get(f"exam_{m}_spearman20_p50")
    add("verdict_exam_kinds_ratio", None if r is None else LINES["exam"]["kinds_ratio"][0] <= r <= LINES["exam"]["kinds_ratio"][1])
    add("verdict_exam_top10_overlap", None if o is None else o >= LINES["exam"]["top10_overlap"])
    add("verdict_exam_spearman20", None if s is None else s >= LINES["exam"]["spearman20"])
    m = choice["symptom"]
    o, cr, cs = get(f"symptom_{m}_chief_top5_overlap_p50"), get(f"symptom_{m}_chief_cov_rule"), get(f"symptom_{m}_chief_cov_s2")
    add("verdict_symptom_chief_top5_overlap", None if o is None else o >= LINES["symptom"]["chief_top5_overlap"])
    add("verdict_symptom_chief_cov", None if cr is None or cs is None else cs >= cr - LINES["symptom"]["chief_cov_drop"])
    m = choice["diagnosis"]
    r, o, pd = get(f"diagnosis_{m}_count_ratio"), get(f"diagnosis_{m}_comorb_top10_overlap_p50"), get(f"diagnosis_{m}_past_max_diff")
    add("verdict_diagnosis_count_ratio", None if r is None else LINES["diagnosis"]["count_ratio"][0] <= r <= LINES["diagnosis"]["count_ratio"][1])
    add("verdict_diagnosis_comorb_top10_overlap", None if o is None else o >= LINES["diagnosis"]["comorb_top10_overlap"])
    add("verdict_diagnosis_past_diff", None if pd is None else pd <= LINES["diagnosis"]["past_diff"])
    m = choice["vital"]
    for item in ("T", "P", "R", "SBP", "DBP"):
        ds = [get(f"vital_{m}_{item}_q{q}_diff") for q in (10, 50, 90)]
        add(f"verdict_vital_{item}", None if None in ds else all(abs(d) <= LINES["vital"][item] for d in ds))
    nd = get(f"vital_{m}_note_share_diff")
    add("verdict_vital_note_share", None if nd is None else abs(nd) <= LINES["vital"]["note_share_diff"])
    for t in TYPES:
        vs = [x for name, x in out if name.startswith(f"verdict_{t}_")]
        out.append((f"verdict_{t}_all", "NA" if "NA" in vs else int(all(vs))))
    return out


def read_events(path, rids):
    by = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("event_type") in READ_TYPES and str(e.get("admission_id")) in rids:
                by.setdefault(str(e["admission_id"]), []).append(e)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--dx-map", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    os.umask(0o077)
    codes = {}
    with open(a.dx_map, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("code3"):
                codes[str(r["admission_id"])] = r["code3"]
    with open(a.path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    events = read_events(a.events, {str(o["admission_id"]) for o in lines})
    rows = [("version", VERSION)] + compare(lines, codes, events, a.k)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    log("s2t_compare_done", admissions=len(lines), events=sum(len(v) for v in events.values()))


if __name__ == "__main__":
    guarded(main)
