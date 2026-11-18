"""E-事件 和 E-口吻：找检查、检验、病理、取材的每一次提及，并判定口吻。

口吻五类：
  R 结果：提及之后紧跟结果触发词（示、提示、回报、结果、报告、显示、证实、可见），或紧跟冒号再接内容；
  P 计划/待回：同一分句里提及之前有计划触发词（拟、完善、预约、申请、择期、建议、进一步……），
     且触发词与提及之间没有结果触发词（"完善血常规、凝血功能、双下肢血管彩超"三项都算计划）；
     或提及之后有"待回、未回、尚未、已送检"一类词；或"待"紧挨在提及前面（0.2："待病理结果回报"）；
  H 既往/入院前：同一分句里有既往或入院前的口吻（X年前、既往、外院、门诊、急诊、入院前……）；
  D 已做未写结果：今日行、已行、顺利完成……；
  N 只是提到。
优先级 H > R > P > D > N。这一处和 10-18 方案写的"历史 > 计划 > 结果"不同：真实病历里
"入院后完善胸部CT示……"非常多，"完善"之后紧跟"示"说的是已经做完、有了结果，按计划算会制造大量假的 K3。
另外记 recheck（"复查、再次、再行"）。
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .text import HIST_RE, clause_span

_CFG = json.loads((Path(__file__).parent / "config" / "exams.json").read_text(encoding="utf-8"))
EXAMS = {e["key"]: e for e in _CFG["exams"]}
_ALIAS = {}
for _e in _CFG["exams"]:
    for _a in _e["aliases"]:
        _ALIAS[_a] = _e["key"]
_REGION = {}
for _r, _ws in _CFG["regions"].items():
    for _w in _ws:
        _REGION[_w] = _r


def _alias_pat(a):
    p = re.escape(a)
    if re.match(r"[A-Za-z]", a[0]):
        p = r"(?<![A-Za-z])" + p
    if re.match(r"[A-Za-z]", a[-1]):
        p = p + r"(?![A-Za-z])"
    return p


EXAM_RE = re.compile("|".join(_alias_pat(a) for a in sorted(_ALIAS, key=len, reverse=True)))
REGION_RE = re.compile("|".join(re.escape(w) for w in sorted(_REGION, key=len, reverse=True)))
# "病理征""病理反射""病理性"不是病理检查
NOT_PATH_AFTER = re.compile(r"\s*(?:征|反射|性|改变|生理)")

RESULT_AFTER = re.compile(
    r"^[^，,。；;\n]{0,10}?(?:示|提示|显示|回报|报告[:：示]|报告提示|结果[:：为示]|结果回报|结果提示|结果显示|证实|所见|可见|检查见|下见|见[^解])"
    r"|^[^，,。；;\n]{0,4}?[:：]\s*[^\s，,。；;\n]"
    r"|^\s*(?:正常|升高|增高|降低|减低|偏高|偏低|阴性|阳性|未见异常|无异常|无明显异常|\d)")
PENDING_AFTER = re.compile(r"^[^，,。；;\n]{0,12}?(?:待回|待回报|待出|未回|未出|尚未|待报|已送检|已预约|已申请|待做|待查|待完善|结果待)")
PLAN_TRIGGER = re.compile(
    r"拟|计划|建议|完善|预约|申请|安排|择期|进一步|需行|需要|需查|需|应行|必要时|可考虑|考虑行|可行|明日|明天|择日|定于|准备|待|送检|送")
# 计划触发词之后、提及之前，如果出现了结果触发词，说明计划已经做完（"完善胸部CT示……，血常规"不算计划）
_RESULT_WORD = re.compile(r"示|提示|显示|回报|报告|结果|证实|可见")
DONE_NEAR = re.compile(r"(?:今日|今天|今晨|已于|已行|已完成|顺利|于.{0,8}行|行了|完成)")
RECHECK_BEFORE = re.compile(r"(?:复查|再次|再行|复行|重复|复)[^，,。；;\n]{0,4}$")
# 0.2："待病理结果回报""待MRCP结果回报"：0.1 先看到后面的"结果回报"判成结果；"待"紧挨在前面就是待回
PENDING_BEFORE = re.compile(r"(?:待|等待|尚待)[^，,。；;\n]{0,4}$")


@dataclass(frozen=True)
class Mention:
    doc_idx: int
    start: int
    end: int
    key: str
    cls: str
    sched: bool
    region: object
    tone: str        # R P H D N
    recheck: bool
    section: str


def _region_before(text, s, a):
    window = text[max(a, s - 8):s]
    found = list(REGION_RE.finditer(window))
    return _REGION[found[-1].group(0)] if found else None


# 真实入院记录里这些段落说的都是入院以前的事
PRE_ADMISSION_SECTIONS = {"现病史", "既往史", "个人史", "婚育史", "月经史", "家族史", "过敏史"}


def tone_of(text, s, e):
    """返回 (tone, recheck)。"""
    a, b = clause_span(text, s, e)
    pre, post = text[a:s], text[e:b]
    if HIST_RE.search(text[a:b]):
        return "H", bool(RECHECK_BEFORE.search(pre))
    recheck = bool(RECHECK_BEFORE.search(pre))
    if PENDING_AFTER.search(post) or PENDING_BEFORE.search(pre):
        return "P", recheck
    if RESULT_AFTER.search(post):
        return "R", recheck
    trig = list(PLAN_TRIGGER.finditer(pre))
    if trig and not _RESULT_WORD.search(pre[trig[-1].end():]):
        return "P", recheck
    if DONE_NEAR.search(text[a:b]):
        return "D", recheck
    return "N", recheck


def mentions_in(doc):
    """一篇文书里全部检查类提及。入院记录里入院前段落的提及一律记 H。"""
    out = []
    text = doc.text
    for m in EXAM_RE.finditer(text):
        key = _ALIAS[m.group(0)]
        if key == "病理" and NOT_PATH_AFTER.match(text, m.end()):
            continue
        e = EXAMS[key]
        a, _ = clause_span(text, m.start(), m.end())
        region = _region_before(text, m.start(), a) if e["region"] else None
        tone, recheck = tone_of(text, m.start(), m.end())
        sec = doc.section_name_at(m.start())
        if doc.doc_type == "hospital_record" and sec in PRE_ADMISSION_SECTIONS:
            tone = "H"
        out.append(Mention(doc.idx, m.start(), m.end(), key, e["cls"], e["sched"], region, tone, recheck, sec))
    return out


def same_exam(a, b):
    if a.key != b.key:
        return False
    if a.region and b.region and a.region != b.region:
        return False
    return True


def admission_mentions(adm):
    return {d.idx: mentions_in(d) for d in adm.docs}
