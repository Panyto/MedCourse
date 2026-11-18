"""E-诊断状态：诊断清单解析、诊断名归一、每次提及的状态。

做法是"以清单为词典"：先从本次住院的诊断清单类段落（入院诊断/初步诊断、鉴别诊断、出院诊断、修正诊断）
里取出诊断名，归一后作为这次住院的诊断词典；再在每篇文书里找这些名字的每一次出现，按前后几个字判状态。
不用外部疾病词典，不用模型；清单以外、病程里自由写出的新诊断名抓不到（宁可漏，不可错）。

状态：
  疑似  考虑、疑似、拟诊、待排、不除外、"？"、可能、待查
  确诊  确诊、证实、诊断明确、明确诊断、诊断成立、符合
  排除  排除、除外、不支持、不考虑
  既往  X病史、既往X（入院前已确立）
  否认  否认X、无X
  列出  诊断清单里不带修饰的条目
  出院  出院诊断清单里不带修饰的条目
  鉴别  首次病程鉴别诊断里列出、没写排除的条目
  复述  出院记录"入院诊断""入院情况"里的提及（复述入院时的判断，不算状态变化）
  提及  其余
"""
import re
from dataclasses import dataclass

from .text import clause_span

LIST_SECTIONS = ("入院诊断", "出院诊断", "修正诊断")
# 出院记录里复述入院时情况的段落：其中的诊断状态是入院时的，不算住院过程中的状态变化
RESTATE_SECTIONS = {"入院诊断", "入院情况"}
DIFF_SECTION = "鉴别诊断"

HEDGE = re.compile(r"[？?]|待查|待排除?|可能性?大?|考虑|疑似|不除外|待除外|可疑|拟诊|不能排除|不能除外")
NUM_ITEM = re.compile(r"(?:^|(?<=[\s；;。，,：:？?）)]))(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*[\.、．)）]\s*")
NON_DX = re.compile(r"医师|签名|日期|\d{4}\s*[-/年]|记录|患者|\[")
_PAREN = re.compile(r"[（(][^）)]{0,20}[）)]")
_STRIP = re.compile(r"[？?\s，,。；;：:、.．“”\"'·]|待查|待排除?|可能性?大?|考虑为?|疑似|不除外|待除外|可疑|拟诊|不能排除|不能除外")


def norm_dx(s):
    s = _PAREN.sub("", s)
    s = _STRIP.sub("", s)
    return s


def _split_items(seg_text, seg_start):
    """诊断清单段落 → [(raw, start, end)]。有编号按编号分，没有编号按分号、逗号、换行分。"""
    marks = list(NUM_ITEM.finditer(seg_text))
    spans = []
    if marks:
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(seg_text)
            spans.append((m.end(), end))
    else:
        a = 0
        for i, ch in enumerate(seg_text + "\n"):
            if ch in "；;，,\n。":
                spans.append((a, i))
                a = i + 1
    out = []
    for a, b in spans:
        raw = seg_text[a:b]
        # 条目内部遇到换行或句号就截止（后面常是签名、日期）
        cut = re.search(r"[\n。]", raw)
        if cut:
            raw = raw[:cut.start()]
        raw = raw.strip(" \t　；;，,、")
        if not raw or len(raw) > 40 or NON_DX.search(raw):
            continue
        # 一条里用"、"或"，"并列多个诊断的，不拆（宁可漏）
        lead = len(seg_text[a:b]) - len(seg_text[a:b].lstrip(" \t　"))
        out.append((raw, seg_start + a + lead, seg_start + a + lead + len(raw)))
    return out


@dataclass(frozen=True)
class DxItem:
    doc_idx: int
    section: str
    raw: str
    norm: str
    start: int
    end: int
    hedged: bool
    excluded: bool = False   # 只对鉴别诊断条目有意义


EXCL_IN_ITEM = re.compile(r"可排除|已排除|予以排除|排除|不支持|不考虑|暂不考虑|可除外|除外(?!诊断)")


def list_items(doc):
    """诊断清单段落里的条目。"""
    out = []
    for sec in doc.sections:
        if sec.name in LIST_SECTIONS:
            for raw, a, b in _split_items(sec.text, sec.start):
                n = norm_dx(raw)
                if len(n) >= 2:
                    out.append(DxItem(doc.idx, sec.name, raw, n, a, b, bool(HEDGE.search(raw))))
    return out


DIFF_WITH = re.compile(r"与\s*([^，,。；;\n]{2,40}?)\s*(?:相)?鉴别")


def differential_items(doc):
    """首次病程"鉴别诊断"段落里的条目：编号条目取冒号或逗号前的名字；另认"需与X、Y鉴别"。"""
    out = []
    for sec in doc.sections:
        if sec.name != DIFF_SECTION:
            continue
        text = sec.text
        marks = list(NUM_ITEM.finditer(text))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.end():end]
            name_m = re.match(r"\s*([^：:，,。；;\n]{2,20})", body)
            if not name_m:
                continue
            raw = name_m.group(1).strip()
            n = norm_dx(raw)
            if len(n) < 2 or NON_DX.search(raw):
                continue
            a = sec.start + m.end() + name_m.start(1)
            out.append(DxItem(doc.idx, DIFF_SECTION, raw, n, a, a + len(raw), True,
                              excluded=bool(EXCL_IN_ITEM.search(body))))
        if not marks:
            for m in DIFF_WITH.finditer(text):
                for part in re.split(r"[、及和与]", m.group(1)):
                    n = norm_dx(part)
                    if len(n) >= 2:
                        a = sec.start + m.start(1) + m.group(1).find(part)
                        out.append(DxItem(doc.idx, DIFF_SECTION, part, n, a, a + len(part), True))
    return out


def dx_match(a, b):
    """两个归一后的诊断名是否指同一病：相等，或较短的一个（≥3 字）包含在较长的一个里。"""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 3 and short in long_


def inventory(adm):
    """这次住院的诊断词典：{norm: [DxItem…]}。"""
    inv = {}
    for d in adm.docs:
        items = list_items(d)
        if d.subtype == "首次病程":
            items += differential_items(d)
        for it in items:
            inv.setdefault(it.norm, []).append(it)
    return inv


SUSP_BEFORE = re.compile(r"(?:考虑|疑似|疑为|拟诊|怀疑|不除外|待排除?|初步诊断为?|倾向于?)\s*(?:为)?\s*$")
SUSP_AFTER = re.compile(r"^\s*(?:[？?]|可能|待查|待排|不除外|不能除外|不能排除|可疑)")
CONF_BEFORE = re.compile(r"(?:确诊为?|证实为?|明确诊断为?|诊断为|诊断明确为|符合)\s*$")
CONF_AFTER = re.compile(r"^\s*(?:诊断)?(?:明确|确诊|成立|诊断明确|诊断成立|可确诊)")
EXCL_BEFORE = re.compile(r"(?:排除|除外|不支持|不考虑|暂不考虑|可排除|否定)\s*(?:为)?\s*$")
EXCL_AFTER = re.compile(r"^\s*(?:可|已|基本|予以?|暂)?(?:排除|除外|不支持|不考虑)")
DENY_BEFORE = re.compile(r"(?:否认|无|未见|未发现|没有)[^，,。；;\n]{0,6}$")
HIST_AFTER = re.compile(r"^\s*(?:病史|史)")
HIST_BEFORE = re.compile(r"(?:既往|有)\s*$")


@dataclass(frozen=True)
class DxMention:
    doc_idx: int
    pos: int
    norm: str
    start: int
    end: int
    state: str


def _context_state(text, s, e):
    a, b = clause_span(text, s, e)
    pre, post = text[max(a, s - 12):s], text[e:min(b, e + 12)]
    if DENY_BEFORE.search(pre):
        return "否认"
    if EXCL_BEFORE.search(pre) or EXCL_AFTER.search(post):
        return "排除"
    if CONF_BEFORE.search(pre) or CONF_AFTER.search(post):
        return "确诊"
    if SUSP_BEFORE.search(pre) or SUSP_AFTER.search(post):
        return "疑似"
    if HIST_AFTER.search(post) or HIST_BEFORE.search(pre):
        return "既往"
    return "提及"


def mentions(adm, inv=None):
    """每篇文书里每个词典诊断名的每一次出现及其状态。按 (pos, start) 排序。"""
    inv = inventory(adm) if inv is None else inv
    names = sorted(inv, key=len, reverse=True)
    out = []
    for d in adm.docs:
        items = {(it.start, it.end): it for it in list_items(d)}
        diffs = {(it.start, it.end): it for it in (differential_items(d) if d.subtype == "首次病程" else [])}
        taken = []
        for n in names:
            for m in re.finditer(re.escape(n), d.text):
                s, e = m.start(), m.end()
                if any(a <= s < b or a < e <= b for a, b in taken):
                    continue  # 较长的名字已经占了这个位置
                taken.append((s, e))
                sec = d.section_name_at(s)
                state = None
                if sec in LIST_SECTIONS:
                    it = next((v for (a, b), v in items.items() if a <= s < b + 1), None)
                    if it is not None:
                        if it.hedged:
                            state = "疑似"
                        else:
                            state = "出院" if sec == "出院诊断" else "列出"
                elif sec == DIFF_SECTION and d.subtype == "首次病程":
                    it = next((v for (a, b), v in diffs.items() if a <= s < b + 1), None)
                    if it is not None:
                        state = "排除" if it.excluded else "鉴别"
                if d.doc_type == "discharge_summary" and sec in RESTATE_SECTIONS:
                    state = "复述"
                if state is None:
                    state = _context_state(d.text, s, e)
                out.append(DxMention(d.idx, d.pos, n, s, e, state))
    out.sort(key=lambda x: (x.pos, x.start))
    return out


SUSPECT_STATES = {"疑似", "鉴别"}
CONFIRM_STATES = {"确诊", "出院"}
