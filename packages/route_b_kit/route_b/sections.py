# -*- coding: utf-8 -*-
"""临床路径正文切分。

卫健委（含中华医学会转载的县级版）路径的正文是"临床路径标准住院流程"下按（一）（二）……编号的小节，
后面跟"临床路径表单"，表单按"住院第 N 天""住院第 N-M 天""出院日"分列。这里只按编号和表头切，不认病种。
"""
import re

CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

# 小节标题：（一）……（十二），全角或半角括号都认。不要求在行首：网页抽出来的文本常没有换行
# （11-10 在中华医学会 15 个页面上核过，行首版一节都切不出）。正文里零散的"（一）"靠"编号必须连续递增"挡掉。
SEC_RE = re.compile(r"[（(]([一二三四五六七八九十]{1,3})[）)]\s*([^\s，。、；（(]{2,20})")
FORM_RE = re.compile(r"临床路径表单")
DAY_RE = re.compile(r"住院第?\s*(\d+)\s*(?:[-－~～至]\s*(\d+))?\s*[天日]")
# 2017 县医院适用版有的表单按手术日排："手术日""术后第 N 天""出院日"（中华医学会冠心病、肠梗阻两页）
REL_DAY_RE = re.compile(r"术前第?\s*(\d+)\s*[天日]|手术日|术后第?\s*(\d+)\s*(?:[-－~～至]\s*(\d+))?\s*[天日]|出院日")

# 小节名里的关键词 → 统一的小节代号（不同版本标题措辞不一样，按关键词认）
SEC_KIND = (
    ("适用对象", "target"),
    ("诊断依据", "dx_basis"),
    ("治疗方案", "treatment"),
    ("标准住院日", "los"),
    ("进入路径", "entry"),
    ("术前准备", "preop"),
    ("入院后", "workup"),
    ("检查", "workup"),
    ("抗菌", "antibiotic"),
    ("药物", "drugs"),
    ("手术日", "surgery_day"),
    ("术后", "postop"),
    ("出院标准", "discharge"),
    ("变异", "variation"),
    ("费用", "cost"),
)


def cn_to_int(s):
    """一 → 1，十 → 10，十二 → 12，二十 → 20。"""
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + CN_NUM[s[1]]
    if len(s) >= 2 and s[1] == "十":
        return CN_NUM[s[0]] * 10 + (CN_NUM[s[2]] if len(s) == 3 else 0)
    return CN_NUM[s]


def kind_of(title):
    for kw, k in SEC_KIND:
        if kw in title:
            return k
    return "other"


def split_sections(text):
    """返回 (小节列表, 表单起点)。小节是 dict：no 编号、title 标题、kind 代号、start/end 在原文里的位置（含标题行）。
    编号回到（一）就当成另一层（比如"（一）适用对象"下面没有子编号的情况已排除），只取第一轮单调递增的编号。"""
    form_m = FORM_RE.search(text)
    stop = form_m.start() if form_m else len(text)
    # 先收全部候选标题，再取最长的编号链：从（一）起，每步编号 +1 或 +2；一样长取跳号少的。
    # 允许 +2 是因为原文偶有印错（中华医学会 2 型糖尿病县级版把"（四）标准住院日"印成"（一）"），
    # 跳过的那节并进前一节，gap_before 记下来，汇报时单列；正文里零散的"（三）"会让链变短或多跳号，选不上。
    cands = [(cn_to_int(m.group(1)), m) for m in SEC_RE.finditer(text, 0, stop)]
    best = [None] * len(cands)          # (链长, -跳号数, 前一个下标)
    for i, (no, _) in enumerate(cands):
        if no == 1:
            best[i] = (1, 0, -1)
        for j in range(i):
            if best[j] and no - cands[j][0] in (1, 2):
                cand = (best[j][0] + 1, best[j][1] - (no - cands[j][0] == 2), j)
                if best[i] is None or cand[:2] > best[i][:2]:
                    best[i] = cand
    chain, i = [], max((k for k in range(len(cands)) if best[k]), key=lambda k: best[k][:2], default=-1)
    while i >= 0:
        chain.append(i)
        i = best[i][2]
    secs, last = [], 0
    for i in reversed(chain):
        no, m = cands[i]
        secs.append({"no": no, "title": m.group(2).strip(), "kind": kind_of(m.group(2)), "start": m.start(), "gap_before": no == last + 2})
        last = no
    for i, s in enumerate(secs):
        s["end"] = secs[i + 1]["start"] if i + 1 < len(secs) else stop
    return secs, (form_m.start() if form_m else None)


def form_days(text, form_start):
    """表单里出现的住院日表头，按出现顺序去重：[(起, 止)]，按入院第几天。没有"住院第 N 天"写法时返回 []，
    改用 form_days_relative。"""
    if form_start is None:
        return []
    out, seen = [], set()
    for m in DAY_RE.finditer(text, form_start):
        d = (int(m.group(1)), int(m.group(2) or m.group(1)))
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def form_days_relative(text, form_start):
    """按手术日排的表头：[("pre", n) | ("op", 0) | ("post", 起, 止) | ("discharge",)]，按出现顺序去重。"""
    if form_start is None:
        return []
    out, seen = [], set()
    for m in REL_DAY_RE.finditer(text, form_start):
        s = m.group(0)
        if s.startswith("术前"):
            d = ("pre", int(m.group(1)))
        elif s == "手术日":
            d = ("op", 0)
        elif s == "出院日":
            d = ("discharge",)
        else:
            d = ("post", int(m.group(2)), int(m.group(3) or m.group(2)))
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out
