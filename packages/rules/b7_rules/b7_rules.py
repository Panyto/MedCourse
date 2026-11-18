"""B7：语料级改写之后，正文与记录当前的性别、年龄是否矛盾。只用规则与词表，不用模型。由总经理维护。

用法：
    from b7_rules import check_record
    hits = check_record(record)   # record 是 7 字段 dict（英文键或中文键均可，同 ruler_g1.render_summary7）

依赖 ruler_g1 核心 v0.4（g1-rules-0.4），需在 sys.path 上。
设计原则同尺子：宁可漏抽，不可错抽。

规则（每条命中带规则名、字段、字段内字符位置、命中值）：
  AGE-HIST     正文出现"N 年"类时长，N > 年龄 + 1。排除：日历年份（前面紧挨数字）、"N年M月"日期、
               "第N年""N年级""每N年"、"N年后/N年内"（多为将来或频次）、"随访/复查"语境；
               所在句含家属词则跳过。"余/多/+"取下限 N。
  PED-ADULT    年龄 < 14 岁，正文出现成人特有词（高血压、冠心病、2 型糖尿病、吸烟、饮酒、绝经、退休）。
               同一分句（按 。；，！？换行 切）里该词前面有否定或劝阻词则跳过；所在句含家属词或孕产语境则跳过；
               "被动吸烟""二手烟"不算。
  SEX-CONTENT  核心 v0.4 的 G1-2（抬头性别 ↔ 性别特有内容）。
  SEX-BODY     核心 v0.4 的 G1-1（抬头性别 ↔ 正文性别）。
  AGE-BODY     核心 v0.4 的 G1-3（抬头年龄 ↔ 正文年龄，差 ≥2 岁）。
"""
import re

from ruler_g1 import evaluate_document, render_summary7
from ruler_g1.adapters import _KEYS
from ruler_g1.extractors import cn_to_float

B7_VERSION = "b7-rules-0.1"

BODY_FIELDS = ("cc", "hpi", "hc", "di")
FIELD_LABEL = {"cc": "主诉", "hpi": "现病史", "hc": "住院经过", "di": "出院医嘱"}

FAMILY = re.compile(
    r"父|母|兄|弟|姐|妹|子女|儿子|女儿|配偶|爱人|丈夫|妻|家族|家人|家中|家属|亲属|"
    r"祖父|祖母|爷爷|奶奶|外公|外婆|姥姥|姥爷|舅|姨|叔|伯|姑")
PREG_CONTEXT = re.compile(r"孕|妊娠|产妇|胎")

SENT_SPLIT = "。；;！!？?\n"
CLAUSE_SPLIT = SENT_SPLIT + "，,"

NUM = r"(\d{1,3}(?:\.\d)?|[一二两三四五六七八九十]{1,3})"
YEARS = re.compile(r"(?<![\d.])" + NUM + r"\s*(余|多|\+)?\s*年(?!级|度|龄)")
AFTER_EXCLUDE = re.compile(r"\s*(?:\d{1,2}\s*月|后|内|以内|之内)")
BEFORE_EXCLUDE = re.compile(r"(?:第|每|随访|复查|再)\s*$")

ADULT_TERMS = re.compile(r"高血压|冠心病|(?:2|Ⅱ|II|二)\s*型\s*糖尿病|吸烟|饮酒|绝经|退休")
NOT_ASSERTED = re.compile(r"否认|无|未|不|非|没有|避免|禁|戒|忌|勿|限|少")
PASSIVE_SMOKE = re.compile(r"被动吸烟|二手烟")

AGE_RX = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(岁|周岁|个月|月|周|天|日)?")
CHILD_AGE_YEARS = 14.0


def _get(record, field):
    for k in _KEYS[field]:
        if k in record and record[k] is not None:
            return str(record[k]).strip()
    return ""


def parse_age_years(raw):
    """"65岁""65""8个月""3天" → 年数；解析不到返回 None。"""
    m = AGE_RX.search(raw or "")
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2) or "岁"
    if unit in ("岁", "周岁"):
        return v
    if unit in ("个月", "月"):
        return v / 12.0
    if unit == "周":
        return v / 52.0
    return v / 365.0


def _span(text, pos, seps):
    a = max(text.rfind(ch, 0, pos) for ch in seps) + 1
    ends = [x for x in (text.find(ch, pos) for ch in seps) if x >= 0]
    return a, (min(ends) if ends else len(text))


def age_hist_hits(field, text, age):
    out = []
    for m in YEARS.finditer(text):
        n = cn_to_float(m.group(1))
        if n is None or n <= age + 1:
            continue
        if AFTER_EXCLUDE.match(text, m.end()) or BEFORE_EXCLUDE.search(text[:m.start()]):
            continue
        a, b = _span(text, m.start(), SENT_SPLIT)
        sentence = text[a:b]
        if FAMILY.search(sentence) or re.search(r"随访|复查", sentence):
            continue
        out.append({"rule": "AGE-HIST", "field": FIELD_LABEL[field], "start": m.start(),
                    "end": m.end(), "value": n})
    return out


def ped_adult_hits(field, text, age):
    if age >= CHILD_AGE_YEARS:
        return []
    out = []
    for m in ADULT_TERMS.finditer(text):
        if PASSIVE_SMOKE.search(text[max(0, m.start() - 2):m.end()]):
            continue
        ca, _ = _span(text, m.start(), CLAUSE_SPLIT)
        if NOT_ASSERTED.search(text[ca:m.start()]):
            continue
        a, b = _span(text, m.start(), SENT_SPLIT)
        sentence = text[a:b]
        if FAMILY.search(sentence) or PREG_CONTEXT.search(sentence):
            continue
        out.append({"rule": "PED-ADULT", "field": FIELD_LABEL[field], "start": m.start(),
                    "end": m.end(), "value": m.group(0)})
    return out


CORE_RULES = {"G1-2": "SEX-CONTENT", "G1-1": "SEX-BODY", "G1-3": "AGE-BODY"}


def core_hits(record):
    res = evaluate_document(render_summary7(record), doc_type="summary7")
    out = []
    for c in res["conflicts"]:
        if c.pair_id in CORE_RULES:
            out.append({"rule": CORE_RULES[c.pair_id], "field": c.right_section,
                        "start": c.right_start, "end": c.right_end, "value": c.pair_id,
                        "offsets": "rendered_summary7"})
    return out


def check_record(record):
    """返回命中列表；年龄解析不到时只跑核心规则，并带一条 {"rule": "AGE-UNPARSED"}。"""
    hits = []
    age = parse_age_years(_get(record, "age"))
    if age is None:
        hits.append({"rule": "AGE-UNPARSED", "field": "年龄", "start": None, "end": None,
                     "value": _get(record, "age")})
    else:
        for f in BODY_FIELDS:
            text = _get(record, f)
            hits += age_hist_hits(f, text, age)
            hits += ped_adult_hits(f, text, age)
    hits += core_hits(record)
    return hits
