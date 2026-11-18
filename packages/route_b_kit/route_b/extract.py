# -*- coding: utf-8 -*-
"""【消融靶子，不是路线 B 的抽取器】老板 11-10 定调：路线 B 的事实一律由 8B 提问检索获得（agentic.py），
这个正则版只用来出 kb_v0，和 kb_v1_agentic 对比"写正则抠半结构化文档"会漏什么、切碎什么。

规则抽取：从一份路径正文里抽 schema.PATHWAY_FIELDS，以及 GAP_FIELDS 里路径偶尔会写的部分。
每条事实的 quote 都是原文里的一段（逐字），再交 provenance.check_fact 核对。不用模型，不认病种。

抽不到就不出这条事实。第一版里，GAP_FIELDS 抽不到的由 8B 追问补，标 model_prior。
"""
import re

from . import provenance as pv
from .sections import DAY_RE, REL_DAY_RE, form_days, split_sections

ICD_RE = re.compile(r"ICD-?10[：:]\s*([A-Z]\d{2}(?:\.[0-9xX]+)?)")
# 县级版有"（四）标准住院日。\n3-4天。""标准住院日：\n病情复杂多变，7-21天"这类写法：关键词和数之间允许隔几个非数字字
LOS_RE = re.compile(r"标准住院日[^\d（(]{0,12}?[≤<＜]?\s*(\d+)\s*(?:[-－~～至]\s*(\d+))?\s*天")
SYM_RE = re.compile(r"(?:症状|临床表现)[：:]([^。\n]{2,200})")
SIGN_RE = re.compile(r"(?:体征|查体)[：:]([^。\n]{2,200})")
REQ_RE = re.compile(r"(?:建议)?必[需须]的?检查项目[：:]?")
OPT_RE = re.compile(r"根据(?:患者)?病情(?:需要)?[，,]?\s*(?:可)?选择(?:的检查项目)?[：:]?")
WITHIN_RE = re.compile(r"(\d+)\s*(?:[-－~～至]\s*(\d+))?\s*天")
SURG_DAY_RE = re.compile(r"手术日为?(?:入院)?(?:第)?\s*[≤<]?\s*(\d+)\s*(?:[-－~～至]\s*(\d+))?\s*天")
ANES_RE = re.compile(r"麻醉方式[：:]\s*([^。\n；;]{2,60})")
PROC_RE = re.compile(r"(?:手术方式|术式)[：:]\s*([^。\n；;]{2,60})")
PROC_TARGET_RE = re.compile(r"行([^，。；\n（(]{2,30}?(?:术|治疗|透析|化疗))")
POSTOP_RE = re.compile(r"术后(?:住院)?恢复[。.，,：:\s]*(?:恢复)?\s*[≤<]?\s*(\d+)\s*(?:[-－~～至]\s*(\d+))?\s*天")
SURG_TITLE_RE = re.compile(r"手术日[（(]([^）)]{2,40})[）)]")
ABX_RE = re.compile(r"[^。\n]*(?:用至|停用|停药|疗程|术后\s*\d+\s*小时内)[^。\n]*")
# "出"后面不能是入（出入量）、现（出现）、院（出院）——11-10 在 J15、J21 上读出过"3天出院""2－3天出现"
TURN_RE = re.compile(r"(\d+)\s*(?:个)?(?:小时|天|日|工作日)\s*(?:内)?\s*(?:出报告|出结果|回报|报告|出(?![入现院\s]))")
IMG_RE = re.compile(r"(?:超声|B超|彩超|CT|MRI|MR|X线|胸片|平片|内镜|胃镜|肠镜|造影)[^。；\n]{0,25}(?:示|见|可见|显示|表现为)[^。；\n]{2,60}")
ITEM_SPLIT = re.compile(r"[、，,；;]|\s{2,}|\n")
NUMBERED = re.compile(r"(?m)^\s*(?:\d+[.．、]|[（(]\d+[）)])\s*([^\n]{2,120})")


def _fact(value, section, quote, source="pathway"):
    return {"value": value, "source": source, "section": section, "quote": quote}


def _sec_of(secs, pos):
    for s in secs:
        if s["start"] <= pos < s["end"]:
            return s
    return None


def _block_after(text, m, stop_res, limit=400):
    """m 之后到下一个停止标记（下一小项编号、可选检查、下一小节）为止的一段。"""
    a = m.end()
    b = min(len(text), a + limit)
    for rx in stop_res:
        n = rx.search(text, a, b)
        if n and n.start() > a:
            b = min(b, n.start())
    return text[a:b]


def _items(block):
    out = []
    for x in ITEM_SPLIT.split(block):
        x = re.sub(r"^\s*(?:\(\d+\)|（\d+）|\d+[.．、])\s*", "", x).strip(" 。:：")
        if 2 <= len(x) <= 30:
            out.append(x)
    return out


def extract(text, exam_alias=None):
    """返回 {字段: [fact, ...]}。exam_alias 是 {写法: 评测台检查键}，给了就给每个检查项标 bench_key。"""
    secs, form = split_sections(text)
    body_end = form if form is not None else len(text)
    facts = {}

    def put(field, fact):
        facts.setdefault(field, []).append(fact)

    def by_kind(*kinds):
        return [s for s in secs if s["kind"] in kinds]

    for m in LOS_RE.finditer(text, 0, body_end):
        put("los_max", _fact(int(m.group(2) or m.group(1)), "los", m.group(0)))
        break
    for s in by_kind("dx_basis")[:1]:
        seg = text[s["start"]:s["end"]]
        m = SYM_RE.search(seg)
        if m:
            put("chief_symptoms", _fact(m.group(1).strip(), "dx_basis", m.group(0)))
        m = SIGN_RE.search(seg)
        if m:
            put("signs", _fact(m.group(1).strip(), "dx_basis", m.group(0)))
        for im in IMG_RE.finditer(seg):
            put("exam_findings", _fact(im.group(0), "dx_basis", im.group(0)))
    stops = (re.compile(r"\n\s*\d+[.．、]"), OPT_RE, re.compile(r"[（(][一二三四五六七八九十]{1,3}[）)]"))
    for field, rx in (("exams_required", REQ_RE), ("exams_optional", OPT_RE)):
        m = rx.search(text, 0, body_end)
        if m:
            block = _block_after(text, m, stops)
            items = _items(block)
            if items:
                s = _sec_of(secs, m.start())
                f = _fact(items, s["kind"] if s else "other", text[m.start():m.end() + len(block)].strip())
                if exam_alias is not None:
                    f["bench_key"] = [next((k for a, k in exam_alias.items() if a in it), None) for it in items]
                put(field, f)
    for s in by_kind("workup", "preop")[:1]:
        head = text[s["start"]:min(s["end"], s["start"] + 40)].split("\n")[0]   # 标题行："术前准备（术前评估）1-2天"
        m = WITHIN_RE.search(head)
        if m:
            put("workup_within_days", _fact(int(m.group(2) or m.group(1)), s["kind"], m.group(0)))
    m = SURG_DAY_RE.search(text, 0, body_end)
    if m:
        put("surgery_day_max", _fact(int(m.group(2) or m.group(1)), "surgery_day", m.group(0)))
    else:
        # 没写天数、只写时机的（"手术日（入院当天急诊手术，或保守治疗失败后手术）"）：记文字
        m = SURG_TITLE_RE.search(text, 0, body_end)
        if m:
            put("surgery_day_max", _fact(m.group(1), "surgery_day", m.group(0)))
    m = ANES_RE.search(text, 0, body_end)
    if m:
        put("anesthesia", _fact(m.group(1).strip(), "surgery_day", m.group(0)))
    m = PROC_RE.search(text, 0, body_end)
    if not m:
        t = by_kind("target")
        m = PROC_TARGET_RE.search(text, t[0]["start"], t[0]["end"]) if t else None
    if m:
        put("procedure", _fact(m.group(1).strip(), "target", m.group(0)))
    m = POSTOP_RE.search(text, 0, body_end)
    if m:
        put("postop_days", _fact(m.group(0)[m.group(0).index(m.group(1)):].strip(), "postop", m.group(0)))
    for s in by_kind("antibiotic", "drugs")[:2]:
        m = ABX_RE.search(text, s["start"], s["end"])
        if m:
            put("antibiotic_rule", _fact(m.group(0).strip(), s["kind"], m.group(0).strip()))
            break
    for field, kind in (("discharge_criteria", "discharge"), ("variations", "variation")):
        for s in by_kind(kind)[:1]:
            seg = text[s["start"]:s["end"]]
            for m in NUMBERED.finditer(seg):
                put(field, _fact(m.group(1).strip(), kind, m.group(1).strip()))
    for m in TURN_RE.finditer(text):
        ctx = text[max(0, m.start() - 6):m.end() + 2]
        if "出入" in ctx:
            continue
        put("exam_turnaround_days", _fact(m.group(0), "other", m.group(0)))
    # 表头：引文用原文里匹配到的那几个字（"住院第4～6天"），不自己拼
    if form is not None:
        rx = DAY_RE if form_days(text, form) else REL_DAY_RE
        seen = set()
        for m in rx.finditer(text, form):
            key = pv.norm(m.group(0))
            if key not in seen:
                seen.add(key)
                put("daily_form", _fact(m.group(0), "form", m.group(0)))
    for fs in facts.values():
        for f in fs:
            pv.check_fact(f, text)
    return facts, {"sections": [(s["no"], s["kind"], s["title"][:12]) for s in secs], "form": form is not None,
                   "numbering_gaps": [s["no"] for s in secs if s.get("gap_before")],
                   "icd10": (ICD_RE.search(text) or [None, None])[1]}
