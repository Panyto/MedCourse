"""不依赖任何规则包的结构读数：篇数、类型数、总字数、时间线、文书头日期是否倒退。"""
import datetime as dt
import re

from ..safe import SafeError

_DATE = re.compile(r"(20\d{2}|19\d{2})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})")
HEAD_FIRST = 40     # v0.1 的窗口。合成行的日期都在这里，这一段的读法不变
HEAD_MAX = 120      # 真实病程抬头更长：fold 0 前 40 字 0 篇有日期，前 120 字 11,602 篇全有（S0_date_availability_diagnostic）
# 窗口放大后会碰到正文开头提到的既往或计划日期，所以 40 字以后只认书写时间戳这三种写法，带字段名或介词的不认
_CLOCK = re.compile(r"\s*\d{1,2}\s*[:：时]\s*\d{1,2}")
_TITLE = re.compile(r"[^\n。；;，,]{0,12}?(?:病程|查房|记录|小结|讨论|会诊)")
_FIELD = re.compile(r"(?:入\s*院|出\s*院|出\s*生|手\s*术|麻\s*醉)\s*(?:日\s*期|时\s*间)?\s*[:：]?\s*$")
_PREP = re.compile(r"[于在至自从拟约为行到]\s*$")


def _valid(m):
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def head_date_rule(text):
    """返回 (文书头日期, 规则)。规则：first40、stamp（后面跟时刻）、title（后面跟文书标题）、line（独占行首）；读不出是 (None, None)。"""
    m = _DATE.search(text[:HEAD_FIRST])
    if m:
        x = _valid(m)
        return (x, "first40") if x else (None, None)
    for m in _DATE.finditer(text[:HEAD_MAX]):
        x = _valid(m)
        before = text[:m.start()]
        if x is None or _FIELD.search(before) or _PREP.search(before):
            continue
        if _CLOCK.match(text, m.end()):
            return x, "stamp"
        if _TITLE.match(text, m.end()):
            return x, "title"
        if not before.rstrip(" \t　") or before.rstrip(" \t　").endswith("\n"):
            return x, "line"
    return None, None


def head_date(text):
    return head_date_rule(text)[0]


def date_order(docs):
    """按 day_offset 排好的文书，文书头日期有没有倒退。返回 (可比, 倒退)。入院记录的抬头日期和出院记录的
    出院日期口径不同，这里只比病程。day_offset 为空的文书不比（守则 2.1：空值只是排放位置）。"""
    seq = [d for d in docs if d.get("doc_type") == "progress_note" and d.get("day_offset") is not None]
    seq = sorted(seq, key=lambda d: int(d["day_offset"]))
    dates = [x for x in (head_date(d["text"]) for d in seq) if x is not None]
    if len(dates) < 2:
        return False, False
    return True, any(b < a for a, b in zip(dates, dates[1:]))


def kinds_of(doc):
    return doc.get("doc_type_cn") or doc.get("role") or doc.get("doc_type")


def run(rows, config):
    per = []
    for x in rows:
        if x["kind"] == "care7":
            rec = x["record"]
            chars = sum(len(str(v)) for v in rec.values() if v is not None)
            per.append({"row": x["row"], "id": x["id"], "n_docs": 1, "n_doc_types": 1, "total_chars": chars,
                        "timeline_share": 0.0, "date_chk": False, "date_viol": False})
            continue
        docs = x["docs"]
        if not docs:
            raise SafeError("admission_without_docs", row=x["row"], id=x["id"])
        chk, viol = date_order(docs)
        per.append({"row": x["row"], "id": x["id"], "n_docs": len(docs),
                    "n_doc_types": len({kinds_of(d) for d in docs}),
                    "total_chars": sum(len(d["text"]) for d in docs),
                    "timeline_share": sum(d.get("day_offset") is not None for d in docs) / len(docs),
                    "date_chk": chk, "date_viol": viol})
    return per, []
