"""年龄、性别、妊娠、药名上下文、剂量和频次的解析。宁可漏抽，不可错抽。"""
import re

# ---------- 年龄、性别 ----------

_AGE_PATS = [
    (re.compile(r"^(\d+(?:\.\d+)?)\s*岁\s*(?:(\d+)\s*(?:个)?月)?\s*(?:(\d+)\s*[天日])?"), "ymd"),
    (re.compile(r"^(\d+)\s*(?:个)?月\s*(?:(\d+)\s*[天日])?"), "md"),
    (re.compile(r"^(\d+)\s*(?:周|w)", re.I), "w"),
    (re.compile(r"^(\d+)\s*(?:天|日|d)(?![a-z])", re.I), "d"),
    (re.compile(r"^(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)?\s*(?:old)?\s*$", re.I), "y"),
]
AGE_HDR = re.compile(r"年\s*龄\s*[：:]\s*([^\s　，,；;]{1,12})")
SEX_HDR = re.compile(r"性\s*别\s*[：:]\s*(男|女)")


def parse_age_range(s):
    """年龄 → (下限, 上限)，单位岁。认不出返回 (None, None)。
    病历按整岁、整月、整周写，"2岁"指满 2 岁不满 3 岁：下限 2、上限 3；"8个月"是 8/12 到 9/12；
    "3周"是 21/365 到 28/365；写到天、或带小数的，上下限相同。LLM-CARe 的 Age 栏常只有数字，按岁算。
    年龄规则（"<18 岁"之类）用下限；儿童剂量的体重用上限（宁可漏，不可错）。"""
    if s is None:
        return None, None
    s = str(s).strip()
    for rx, kind in _AGE_PATS:
        m = rx.match(s)
        if not m:
            continue
        if kind == "ymd":
            y = float(m.group(1))
            if m.group(3):
                v = y + (int(m.group(2)) / 12 if m.group(2) else 0) + int(m.group(3)) / 365
                return v, v
            if m.group(2):
                lo = y + int(m.group(2)) / 12
                return lo, lo + 1 / 12
            return (y, y) if "." in m.group(1) else (y, y + 1)
        if kind == "md":
            lo = int(m.group(1)) / 12
            if m.group(2):
                v = lo + int(m.group(2)) / 365
                return v, v
            return lo, lo + 1 / 12
        if kind == "w":
            return int(m.group(1)) * 7 / 365, (int(m.group(1)) + 1) * 7 / 365
        if kind == "d":
            v = int(m.group(1)) / 365
            return v, v
        if kind == "y":
            v = float(m.group(1))
            if v > 120:
                return None, None
            return (v, v) if "." in m.group(1) else (v, v + 1)
    return None, None


def parse_age(s):
    """'65岁'、'1岁3月'、'8个月'、'15天'、'3周'、'1' → 岁（浮点，取下限）。认不出返回 None。"""
    return parse_age_range(s)[0]


def parse_sex(s):
    if s is None:
        return None
    t = str(s).strip().lower()
    if t.startswith("女") or t.startswith("female") or t in ("f",):
        return "F"
    if t.startswith("男") or t.startswith("male") or t in ("m",):
        return "M"
    return None


def header_age_range(text):
    """文书前 400 字里的"年龄：""性别："→ (年龄下限, 年龄上限, 性别)。"""
    head = (text or "")[:400]
    m = AGE_HDR.search(head)
    lo, hi = parse_age_range(m.group(1)) if m else (None, None)
    m = SEX_HDR.search(head)
    sex = parse_sex(m.group(1)) if m else None
    return lo, hi, sex


def header_age_sex(text):
    lo, _, sex = header_age_range(text)
    return lo, sex


# ---------- 诊断段、妊娠 ----------

DX_HDR = re.compile(r"(?:入院诊断|初步诊断|出院诊断|修正诊断|补充诊断|最后诊断|术前诊断|术后诊断|确定诊断|目前诊断|主要诊断)\s*[：:]")
_NEXT_HDR = re.compile(r"\n\s*(?![\d（(一二三四五六七八九十])[一-鿿]{2,10}\s*[：:]")


def dx_sections(text):
    out = []
    for m in DX_HDR.finditer(text or ""):
        seg = text[m.end(): m.end() + 300]
        n = _NEXT_HDR.search(seg)
        out.append(seg[: n.start()] if n else seg)
    return "\n".join(out)


PREG_POS = re.compile(r"妊娠|孕\s*\d+\s*(?:\+\s*\d+\s*)?周|宫内孕|早孕|中孕|晚孕|孕\s*\d+\s*产\s*\d+|G\s*\d+\s*P\s*\d+")
PREG_END = re.compile(r"产后|分娩|剖宫产|剖腹产|顺产|娩出|引产|(?<!先兆)流产|清宫|异位妊娠|宫外孕|输卵管妊娠|葡萄胎|终止妊娠|"
                      r"胎死宫内|妊娠史|(?<!先兆)早产|产褥|胎盘残留|足月产|助产|产钳|已产|新生儿")


def is_pregnant(dx_text, end_scope_text, sex, age):
    """诊断里写了妊娠，且诊断和住院经过、出院医嘱（或病程、出院记录）里没有分娩、流产、终止妊娠一类字样。"""
    if sex == "M":
        return False
    if age is not None and not (10 <= age <= 60):
        return False
    if not PREG_POS.search(dx_text or ""):
        return False
    if PREG_END.search(dx_text or "") or PREG_END.search(end_scope_text or ""):
        return False
    return True


# ---------- 药名上下文 ----------

CLAUSE_END = re.compile(r"[。；;！!？?\n]")
NEG_PREFIX = re.compile(r"过敏|否认|禁用|禁止|忌用|避免|不宜|不能用|不可用|未使用|未用|未予|拒绝|慎用|不用|其父|其母|父亲|母亲|家属|配偶|爱人|儿子|女儿|哥哥|姐姐|弟弟|妹妹")
NEG_SUFFIX = re.compile(r"^.{0,6}?(?:过敏|禁用|慎用|皮试)")
STOP_CTX = re.compile(r"停用|停服|停药|已停|换用|改用|换为|改为|调整为")
TOPICAL_AFTER = re.compile(r"^[^，,。；;\s]{0,8}?(?:滴眼|眼液|眼膏|眼用|滴耳|耳用|乳膏|软膏|凝胶|外用|外涂|涂抹|涂擦|洗液|栓|阴道|含漱|喷鼻|鼻喷|滴鼻|搽剂|酊|贴膏)")
TOPICAL_BEFORE = re.compile(r"(?:外用|外涂|局部|滴眼|点眼)\s*$")
NOT_DRUG_AFTER = {
    "heparin": re.compile(r"^(?:帽|封管)"),
    "insulin": re.compile(r"^(?:抵抗|释放|水平|瘤|分泌|自身抗体|样|测定)"),
}


def clause_bounds(text, start, end):
    a = 0
    for m in CLAUSE_END.finditer(text, 0, start):
        a = m.end()
    m = CLAUSE_END.search(text, end)
    b = m.start() if m else len(text)
    return a, b


def mention_context(text, start, end, drug_id):
    a, b = clause_bounds(text, start, end)
    prefix = text[a:start]
    after = text[end:b]
    clause = text[a:b]
    nd = NOT_DRUG_AFTER.get(drug_id)
    return {
        "not_drug": bool(nd and nd.search(text[end:end + 8])),
        "negated": bool(NEG_PREFIX.search(prefix) or NEG_SUFFIX.search(after)),
        "stop": bool(STOP_CTX.search(clause)),
        "topical": bool(TOPICAL_AFTER.search(text[end:end + 12]) or TOPICAL_BEFORE.search(text[max(0, start - 4):start])),
        "clause_end": b,
    }


# ---------- 剂量、频次 ----------

_UNIT = {"mg": 1.0, "g": 1000.0, "μg": 0.001, "ug": 0.001, "µg": 0.001, "mcg": 0.001,
         "毫克": 1.0, "克": 1000.0, "微克": 0.001}
MASS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:[~～\-－—至到]\s*\d+(?:\.\d+)?\s*)?(mg|μg|µg|ug|mcg|毫克|微克|克|g)(?!\s*tt)", re.I)
PER_BAD = re.compile(r"^\s*[/／每]\s*(?:kg|千克|公斤|ml|mL|毫升|L(?![a-z])|升|m2|㎡|h|小时|min|分)", re.I)
PER_UNIT = re.compile(r"^\s*[/／]\s*(?:粒|片|袋|支|瓶)")
PKG = re.compile(r"^\s*[×xX*]\s*\d+\s*(?:粒|片|袋|支|瓶|盒)?")
DAILY_AFTER = re.compile(r"^\s*[/／]\s*(?:日|天|晚|d(?![a-z]))", re.I)
DAILY_BEFORE = re.compile(r"(?:每日|每天|一日|一天|日|全日)\s*(?:总量|剂量|总剂量)?\s*(?:为|约|共)?\s*$")
_CN = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "半": 0.5}
COUNT = re.compile(r"(\d+(?:\.\d+)?|[一二两三四半])\s*(?:粒|片|袋|支)(?!\s*[×xX*])")

# 英文频次前面不能紧跟字母，但可以紧跟剂量单位（"0.5gtid"、"10mgqd"）
_L = r"(?:(?<![A-Za-z])|(?<=\d[gG])|(?<=\d[mM][gG]))"
_R = r"(?![A-Za-z])"
FREQ = [
    (re.compile(_L + r"q\.?\s*4\s*h" + _R, re.I), 6), (re.compile(_L + r"q\.?\s*6\s*h" + _R, re.I), 4),
    (re.compile(_L + r"q\.?\s*8\s*h" + _R, re.I), 3), (re.compile(_L + r"q\.?\s*12\s*h" + _R, re.I), 2),
    (re.compile(_L + r"q\.?\s*24\s*h" + _R, re.I), 1),
    (re.compile(_L + r"q\.?i\.?d\.?" + _R, re.I), 4), (re.compile(_L + r"t\.?i\.?d\.?" + _R, re.I), 3),
    (re.compile(_L + r"b\.?i\.?d\.?" + _R, re.I), 2), (re.compile(_L + r"q\.?o\.?d\.?" + _R, re.I), 0.5),
    (re.compile(_L + r"q\.?d\.?" + _R, re.I), 1), (re.compile(_L + r"q\.?n\.?" + _R, re.I), 1),
    (re.compile(_L + r"q\.?w\.?" + _R, re.I), 1 / 7), (re.compile(_L + r"st(?:at)?" + _R + r"(?!\s*[-－段])", re.I), "once"),
    (re.compile(r"每\s*4\s*小时"), 6), (re.compile(r"每\s*6\s*小时"), 4), (re.compile(r"每\s*8\s*小时"), 3),
    (re.compile(r"每\s*12\s*小时"), 2), (re.compile(r"每\s*24\s*小时"), 1),
    (re.compile(r"(?:每日|每天|一日|一天|日)\s*([一二两三四1234])\s*次"), "n"),
    (re.compile(r"([一二两三四1234])\s*次\s*[/／每]\s*(?:日|天|d)"), "n"),
    (re.compile(r"每晚|睡前"), 1), (re.compile(r"早晚各\s*[一1]\s*次"), 2), (re.compile(r"隔日"), 0.5),
    (re.compile(r"每周\s*([一二两三1234])\s*次"), "n7"), (re.compile(r"顿服|单次|一次性"), "once"),
]


def _cn_num(s):
    return float(s) if s[0].isdigit() else float(_CN[s])


def find_freq(window):
    best = None
    for rx, val in FREQ:
        m = rx.search(window)
        if not m:
            continue
        if val == "n":
            v = _cn_num(m.group(1))
        elif val == "n7":
            v = _cn_num(m.group(1)) / 7
        else:
            v = val
        if best is None or m.start() < best[0]:
            best = (m.start(), v, m.group(0))
    return (best[1], best[2]) if best else (None, None)


# 药名和剂量之间出现顿号或并列词，或者逗号后面紧跟着另一个药的剂型字再接剂量，剂量多半属于后面另一个药
# （"氨氯地平、氯化钾缓释片1g tid"，"阿托伐他汀，复方丹参滴丸270mg"）：不抽。"X片，每日3次，每次0.5mg"照抽。
SEP_BEFORE_DOSE = re.compile(r"、|及|和|与|联合|加用|并用|序贯|改为|换为|改用|换用")
OTHER_DRUG_BEFORE_DOSE = re.compile(r"[，,][^，,]*[一-鿿]{2,}(?:片|胶囊|注射液|颗粒|针|丸|散|口服液|混悬液|乳|栓|膏|剂)\s*$")
FREQ_SPAN = 20       # 频次先在剂量后面 20 字内、下一个剂量之前找，找不到再看药名和剂量之间


def _other_drug_before(pre):
    return bool(SEP_BEFORE_DOSE.search(pre) or OTHER_DRUG_BEFORE_DOSE.search(pre))


def _freq_region(window, m):
    rest = window[m.end(): m.end() + FREQ_SPAN]
    n = MASS.search(rest)
    return rest[: n.start()] if n else rest


def parse_dose(window, prefix_tail=""):
    """window：药名后面到分句末、下一个药名或 40 字为止的文字。返回 dict：single_mg、daily_mg、freq、text；认不出剂量返回 None。"""
    for m in MASS.finditer(window):
        if _other_drug_before(window[: m.start()]):
            return None
        tail = window[m.end():]
        if PER_BAD.match(tail):
            continue
        mg = float(m.group(1)) * _UNIT.get(m.group(2).lower(), _UNIT.get(m.group(2), 1.0))
        before = (prefix_tail + window[: m.start()])[-8:]
        if DAILY_AFTER.match(tail) or DAILY_BEFORE.search(before):
            return {"single_mg": None, "daily_mg": mg, "freq": None, "text": m.group(0)}
        strength = PER_UNIT.match(tail) or PKG.match(tail)
        if strength:
            rest = tail[strength.end():]
            c = COUNT.search(rest[:20])
            if not c:
                return None
            single = mg * _cn_num(c.group(1))
        else:
            single = mg
        freq, ftxt = find_freq(_freq_region(window, m))
        if freq is None:
            freq, ftxt = find_freq(window[: m.start()])
        if freq == "once":
            daily = single
        elif isinstance(freq, (int, float)):
            daily = single * freq
        else:
            daily = None
        return {"single_mg": single, "daily_mg": daily, "freq": ftxt, "text": m.group(0)}
    return None
