"""把一篇文书切成抽取用的切片。入院、出院、首程按章节切；病程多半没有章节，去掉抬头后整篇算一片；太长的按句切。

切片数量和长度的真实分布由 seg_diag 量，不预设"都在 100–400 字"。
"""
import re

from ..rules import setup

setup(("BENCH_G2",))
from ruler_g1.sections import split_sections  # noqa: E402

MAX_SEG = 600
MIN_SEG = 4
MIN_HALVE = 120
SKIP = {"签名"}
_SENT = re.compile(r"[^。；;！？!?\n]+[。；;！？!?\n]*")
_STAMP = re.compile(r"(?:20\d{2}|19\d{2})\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*日?(?:\s*\d{1,2}\s*[:：时]\s*\d{1,2})?")
_HEAD_MAX = 120


def _pieces(text, start, limit):
    """按句切，每片不超过 limit；一句本身超长时硬切。返回 [(起, 止)]，都是文书里的位置。"""
    out, a, b = [], None, None
    for m in _SENT.finditer(text):
        s, e = start + m.start(), start + m.end()
        if a is None:
            a, b = s, e
        elif e - a <= limit:
            b = e
        else:
            out.append((a, b))
            a, b = s, e
        while b - a > limit:
            out.append((a, a + limit))
            a += limit
    if a is not None:
        out.append((a, b))
    return out


def _useful(text):
    return len(re.sub(r"[\s，,。；;：:、.．|｜/\\-]", "", text)) >= MIN_SEG


def _note_body_start(text, stop):
    """病程的抬头：第一个书写时间戳所在行到行尾。读不到时间戳就当没有抬头。"""
    m = _STAMP.search(text[:min(stop, _HEAD_MAX)])
    if not m:
        return 0
    nl = text.find("\n", m.end())
    return nl + 1 if 0 <= nl < stop else m.end()


def segments(doc, limit=MAX_SEG):
    """doc：AdmissionJSONL 的一篇。返回 [{"section","start","end","text"}]，位置相对整篇。"""
    text = doc.get("text") or ""
    out = []
    for sec in split_sections(text):
        if sec.name in SKIP:
            continue
        start, end = sec.start, sec.end
        name = sec.name
        if name == "抬头":
            if doc.get("doc_type") != "progress_note":
                continue
            start, name = _note_body_start(text, end), "正文"
        for a, b in _pieces(text[start:end], start, limit):
            if _useful(text[a:b]):
                out.append({"section": name, "start": a, "end": b, "text": text[a:b]})
    return out


def halve(seg):
    """切片顶到事件上限或输出被截断时，从中间的句界切成两半重抽。太短或切不开返回 None。"""
    text = seg["text"]
    if len(text) < MIN_HALVE:
        return None
    mid, best = len(text) // 2, None
    for m in _SENT.finditer(text):
        if 0 < m.end() < len(text) and (best is None or abs(m.end() - mid) < abs(best - mid)):
            best = m.end()
    if best is None:
        best = mid
    return ({"section": seg["section"], "start": seg["start"], "end": seg["start"] + best, "text": text[:best]},
            {"section": seg["section"], "start": seg["start"] + best, "end": seg["end"], "text": text[best:]})
