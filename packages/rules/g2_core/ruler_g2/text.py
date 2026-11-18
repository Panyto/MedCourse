"""句子、分句、日期等公用小工具。"""
import datetime as dt
import re

SENT_SEPS = "。；;！!\n"
CLAUSE_SEPS = "。；;！!\n，,"

# 既往或入院前的口吻：出现在同一分句里，就说明说的是这次住院以前的事。
# 不收"2015年于……"这类日期写法：出院记录诊疗经过里"于2025年3月4日行……术"说的就是这次住院。
HIST_RE = re.compile(
    r"(?:\d+|[一二三四五六七八九十半两几数]+)\s*(?:余|多)?\s*(?:个)?\s*(?:年|月|周|星期|天|日)\s*(?:余|多)?\s*(?:以)?前"
    r"|既往|曾经?|外院|院外|当地|前次|上次|本次入院前|入院前|门诊|急诊|在.{0,10}医院")


def sentence_span(text, s, e=None):
    """[s, e) 所在句子的起止（不跨 SENT_SEPS）。"""
    e = s if e is None else e
    a = max(text.rfind(ch, 0, s) for ch in SENT_SEPS) + 1
    ends = [x for x in (text.find(ch, e) for ch in SENT_SEPS) if x >= 0]
    return a, (min(ends) if ends else len(text))


def clause_span(text, s, e=None):
    e = s if e is None else e
    a = max(text.rfind(ch, 0, s) for ch in CLAUSE_SEPS) + 1
    ends = [x for x in (text.find(ch, e) for ch in CLAUSE_SEPS) if x >= 0]
    return a, (min(ends) if ends else len(text))


def sentences(text):
    """返回 [(start, end)]，不含分隔符，跳过空句。"""
    out, a = [], 0
    for i, ch in enumerate(text):
        if ch in SENT_SEPS:
            if text[a:i].strip():
                out.append((a, i))
            a = i + 1
    if text[a:].strip():
        out.append((a, len(text)))
    return out


FULL_DATE = re.compile(r"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})(?!\d)")
MONTH_DAY = re.compile(r"(?<![\d年/.\-])(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")


def parse_date(y, m, d):
    try:
        return dt.date(int(y), int(m), int(d))
    except ValueError:
        return None


def find_dates(text, ref_date=None):
    """全日期，以及（有参照日期时）只有月日的写法。返回 [(start, end, date)]。
    只有年份的写法不算日期：去标识化不平移年份，拿来和平移后的日期比没有意义。"""
    out = []
    for m in FULL_DATE.finditer(text):
        d = parse_date(*m.groups())
        if d:
            out.append((m.start(), m.end(), d))
    if ref_date is not None:
        taken = [(a, b) for a, b, _ in out]
        for m in MONTH_DAY.finditer(text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            d = parse_date(ref_date.year, *m.groups())
            if d is None:
                continue
            # 跨年：比参照日期晚半年以上的，按上一年算
            if (d - ref_date).days > 180:
                d = parse_date(ref_date.year - 1, *m.groups())
            if d:
                out.append((m.start(), m.end(), d))
    return sorted(out)


def char_ngrams(text, n=5):
    s = re.sub(r"\s+", "", text)
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def overlaps(a0, a1, b0, b1, slack=0):
    return a0 < b1 + slack and b0 < a1 + slack
