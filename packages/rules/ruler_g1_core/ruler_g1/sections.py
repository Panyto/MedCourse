"""按段落标题切分文书。

标题只在行首（或表格单元格 " | " 之后）识别，允许字间空格（"体 格 检 查"）、
允许 "#."、"一、"、"1." 一类编号前缀；标题后面必须是冒号，或者标题独占一行/一格，
或者紧接生命体征"T：36.5℃"（v0.3 新增）。
这样正文里顺带提到的"主诉""诊断"不会被当成标题。
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

_CONFIG = Path(__file__).parent / "config" / "section_titles.json"


def _load_titles():
    raw = json.loads(_CONFIG.read_text(encoding="utf-8"))
    variants = []
    for canon, names in raw.items():
        if canon.startswith("_"):
            continue
        for name in names:
            variants.append((name, canon))
    # 长写法优先，避免"辅助检查结果"被"辅助检查"截断
    variants.sort(key=lambda x: -len(x[0]))
    return variants


_VARIANTS = _load_titles()
_CANON_OF = {}
_ALTS = []
for _name, _canon in _VARIANTS:
    _pat = r"\s*".join(re.escape(ch) for ch in _name)
    _ALTS.append(_pat)
    _CANON_OF[re.sub(r"\s+", "", _name)] = _canon

_PREFIX = r"(?:[#＃]\s*\.?\s*|[一二三四五六七八九十]+\s*[、.．]\s*|\d{1,2}\s*[、.．]\s*)?"
# 标题之后可接：冒号；行尾或表格分隔；或直接接生命体征（"体格检查T：36.5℃"，包 2 r0 A10 见于 1,566 份入院记录）
_VITAL_NEXT = r"(?=[ \t　]*(?:T|体温)[ \t　]*[:：]?[ \t　]*\d)"
TITLE_RE = re.compile(
    r"(?:^|(?<=\n)|(?<=\|))[ \t　]*" + _PREFIX
    + r"(?P<title>" + "|".join(_ALTS) + r")"
    + r"[ \t　]*(?:[:：]|(?=[ \t　]*(?:\n|\||$))|" + _VITAL_NEXT + r")"
)


@dataclass(frozen=True)
class Section:
    name: str
    start: int
    end: int
    text: str


def split_sections(text):
    """返回段落列表。第一个标题之前的部分记为"抬头"。"""
    matches = list(TITLE_RE.finditer(text))
    stop = matches[0].start() if matches else len(text)
    out = [Section("抬头", 0, stop, text[:stop])]
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        canon = _CANON_OF[re.sub(r"\s+", "", m.group("title"))]
        out.append(Section(canon, m.end(), end, text[m.end():end]))
    return out


def section_at(sections, pos):
    for s in sections:
        if s.start <= pos < s.end:
            return s.name
    # 落在标题文字本身上的位置，归到其后的段落
    for s in sections:
        if pos < s.start:
            return s.name
    return "未知"


FIRST_PROGRESS_RE = re.compile(r"首\s*次\s*病\s*程\s*记\s*录")
POSTOP_RE = re.compile(r"术\s*后\s*首\s*次\s*病\s*程")


def is_first_progress(text, window=300):
    """首次病程识别：开头 window 字内有"首次病程记录"，且不是"术后首次病程记录"。"""
    head = text[:window]
    return bool(FIRST_PROGRESS_RE.search(head)) and not POSTOP_RE.search(head)
