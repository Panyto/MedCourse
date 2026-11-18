"""确定性校验：引文必须是切片原文的子串，否则丢弃。比较时忽略空白，位置换算回原文。数值和单位不在引文里的只清掉字段，事件留下。

v0.3（S2G3 之后）：引文对不上时多一步名称回退。名称（去空白、至少 2 个字）在切片原文里，就取包含它的那个分句（逗号、句号之间）
当引文，事件标 ground="name"，单独计数（grounded_by_name）。S2G3 既往史"否认高血压、糖尿病"这类列表，模型给后面几项拼出
"否认糖尿病"当引文，不是原文子串，整条被丢掉，既往史每片的七类事件比 v1 少 62%。名称本身必须在原文里，模型编出来的名称照旧丢掉。
丢掉的按类型计数（dropped_type:*）。
"""
import re

_WS = re.compile(r"\s+")
_PART_END = re.compile(r"[。；;！？!?\n，,]")


def _strip_map(text):
    chars, pos = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            pos.append(i)
    return "".join(chars), pos


def locate(evidence, text):
    """evidence 在 text 里的 (起, 止)，找不到返回 None。"""
    needle = _WS.sub("", evidence or "")
    if not needle:
        return None
    hay, pos = _strip_map(text)
    i = hay.find(needle)
    if i < 0:
        return None
    return pos[i], pos[i + len(needle) - 1] + 1


def name_span(name, text, evidence, tail=False):
    """名称回退：名称在 text 里时，返回包含它的那个分句的 (起, 止)；出现多次取和引文共有字最多的一处（并列取靠前的）。没有返回 None。"""
    nm = _WS.sub("", name or "")
    if len(nm) < 2:
        return None
    hay, pos = _strip_map(text)
    ev = set(_WS.sub("", evidence or ""))
    best, best_key = None, None
    i = hay.find(nm)
    while i >= 0:
        s0, e0 = pos[i], pos[i + len(nm) - 1] + 1
        a = 0
        for m in _PART_END.finditer(text, 0, s0):
            a = m.end()
        m = _PART_END.search(text, e0)
        b = (m.start() if m else len(text)) if tail else e0
        key = (len(ev & set(text[a:b])), -i)
        if best_key is None or key > best_key:
            best, best_key = (a, b), key
        i = hay.find(nm, i + 1)
    if best is None:
        return None
    a, b = best
    while a < b and text[a].isspace():
        a += 1
    return (a, b) if a < b else None


def ground(cand, seg, tally):
    """返回落到整篇文书位置的事件；引文和名称都对不上返回 None。tally 只记计数。"""
    out = dict(cand)
    span = locate(cand.get("evidence"), seg["text"])
    if span is None:
        # 引文从分句开头到名称为止（带上前面的否定词）；有数值的到分句结尾，数值才留得住
        span = name_span(cand.get("name"), seg["text"], cand.get("evidence"), tail=cand.get("value") is not None)
        if span is None:
            tally["dropped_not_substring"] = tally.get("dropped_not_substring", 0) + 1
            k = f"dropped_type:{cand.get('event_type')}"
            tally[k] = tally.get(k, 0) + 1
            return None
        out["evidence"], out["ground"] = seg["text"][span[0]:span[1]], "name"
        tally["grounded_by_name"] = tally.get("grounded_by_name", 0) + 1
        k = f"grounded_by_name_type:{cand.get('event_type')}"
        tally[k] = tally.get(k, 0) + 1
    out["ev_start"], out["ev_end"] = seg["start"] + span[0], seg["start"] + span[1]
    ev = _WS.sub("", out["evidence"])
    for field, code in (("value", "value_cleared"), ("unit", "unit_cleared")):
        v = cand.get(field)
        if v is not None and _WS.sub("", str(v)) not in ev:
            out[field] = None
            tally[code] = tally.get(code, 0) + 1
    if _WS.sub("", cand.get("name") or "") not in ev:
        tally["name_not_in_evidence"] = tally.get("name_not_in_evidence", 0) + 1
    return out


def dedupe(events):
    seen, out = set(), []
    for e in events:
        k = (e["event_type"], e["name"], e.get("value"), e["ev_start"])
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out
