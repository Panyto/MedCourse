"""关1 R 层抽取器：只用规则与词表，不用模型。

每条事实带段落名和字符位置，便于人工核对与埋雷定位。
设计原则：宁可漏抽，不可错抽。抽不到只会降低 n_checkable；错抽会直接制造假冲突。
"""
import json
import re
from pathlib import Path

from .models import Fact
from .sections import section_at, split_sections

RULE_VERSION = "g1-rules-0.5"

# ---------- 公用 ----------

CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_float(s):
    """阿拉伯数字或不超过"九十九"的中文数字；"半"为 0.5。解析失败返回 None。"""
    if s == "半":
        return 0.5
    try:
        return float(s)
    except ValueError:
        pass
    if not s or any(ch not in CN_DIGIT and ch != "十" for ch in s):
        return None
    if "十" not in s:
        return float(CN_DIGIT[s]) if len(s) == 1 else None
    left, _, right = s.partition("十")
    tens = CN_DIGIT.get(left, None) if left else 1
    ones = CN_DIGIT.get(right, None) if right else 0
    if tens is None or ones is None:
        return None
    return float(tens * 10 + ones)


# 不描述患者本人的段落：这些段落里的性别、年龄一律不抽
NON_PATIENT_SECTIONS = {"家族史", "婚育史", "月经史", "个人史", "既往史", "过敏史"}

SUBJ = r"(?:患者|病人|患儿|\[患者\])"
CLAUSE_START = r"(?:^|(?<=[。；;\n|：:，,]))[ \t　]*"


def _fact(meta, sections, start, end, field, value, rule):
    return Fact(meta[0], meta[1], meta[2], section_at(sections, start), start, end,
                field, value, rule)


# ---------- 抬头：性别、年龄、日期、住院天数 ----------

HEAD_SEX = re.compile(r"性\s*别\s*[:：]?[\s|]*(男|女)(?=[\s|，,；;。]|$)")
HEAD_AGE = re.compile(r"年\s*龄\s*[:：]?[\s|]*(≥\s*)?(\d{1,3})\s*(岁|个月|月|天|日)")
HEAD_DATE = re.compile(
    r"(入|出)\s*院\s*(?:时\s*间|日\s*期)\s*[:：]?[\s|]*"
    r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})")
HEAD_LOS = re.compile(r"(?:住\s*院\s*天\s*数|实\s*际\s*住\s*院)\s*[:：]?[\s|]*(\d{1,4})\s*天")


def age_in_years(num, unit):
    if unit == "岁":
        return float(num)
    if unit in ("个月", "月"):
        return float(num) / 12.0
    return float(num) / 365.0


def extract_header(text, meta, sections):
    out = []
    m = HEAD_SEX.search(text)
    if m:
        out.append(_fact(meta, sections, m.start(1), m.end(1), "header_sex", m.group(1), "sex.head"))
    m = HEAD_AGE.search(text)
    if m:
        ge = 1 if m.group(1) else 0
        yrs = age_in_years(m.group(2), m.group(3))
        out.append(_fact(meta, sections, m.start(2), m.end(3), "header_age", f"{yrs:g}|ge={ge}", "age.head"))
    for m in HEAD_DATE.finditer(text):
        field = "admit_date" if m.group(1) == "入" else "discharge_date"
        if any(f.field == field for f in out):
            continue
        val = f"{int(m.group(2)):04d}-{int(m.group(3)):02d}-{int(m.group(4)):02d}"
        out.append(_fact(meta, sections, m.start(2), m.end(4), field, val, "date.head"))
    m = HEAD_LOS.search(text)
    if m:
        out.append(_fact(meta, sections, m.start(1), m.end(1), "los_days", m.group(1), "los.head"))
    return out


# ---------- 正文：患者本人的性别、年龄 ----------

AGE_TAIL = r"(?!\s*(?:时|起|开始|以前|以后|之前|之后|前|后|左右|余|多))"
BODY_SEX_RULES = [
    # 患者，男，……　／　[患者]，女性，……
    ("sex.body.subj", re.compile(SUBJ + r"[，,、：:\s]*(男|女)性?(?=[，,、。；;\s\d≥]|$)")),
    # 分句开头的 "男，45岁"
    ("sex.body.clause", re.compile(CLAUSE_START + r"(男|女)性?[，,、\s]*≥?\s*\d{1,3}\s*岁")),
    # 女性患者、男患儿
    ("sex.body.noun", re.compile(r"(男|女)性?患(?:者|儿)")),
]
BODY_AGE_RULES = [
    ("age.body.subj", re.compile(SUBJ + r"[，,、：:\s]*(?:(?:男|女)性?[，,、\s]*)?(≥\s*)?(\d{1,3})\s*(岁|个月|天)" + AGE_TAIL)),
    ("age.body.clause", re.compile(CLAUSE_START + r"(?:男|女)性?[，,、\s]*(≥\s*)?(\d{1,3})\s*(岁|个月|天)" + AGE_TAIL)),
]


def extract_body_demographics(text, meta, sections):
    out = []
    seen = set()
    for rule, rx in BODY_SEX_RULES:
        for m in rx.finditer(text):
            sec = section_at(sections, m.start(1))
            if sec in NON_PATIENT_SECTIONS or sec == "抬头":
                continue
            key = ("sex", m.start(1))
            if key in seen:
                continue
            seen.add(key)
            out.append(_fact(meta, sections, m.start(1), m.end(1), "body_sex", m.group(1), rule))
    for rule, rx in BODY_AGE_RULES:
        for m in rx.finditer(text):
            sec = section_at(sections, m.start(2))
            if sec in NON_PATIENT_SECTIONS or sec == "抬头":
                continue
            key = ("age", m.start(2))
            if key in seen:
                continue
            seen.add(key)
            ge = 1 if m.group(1) else 0
            yrs = age_in_years(m.group(2), m.group(3))
            out.append(_fact(meta, sections, m.start(2), m.end(3), "body_age", f"{yrs:g}|ge={ge}", rule))
    return out


# ---------- 时长 ----------

NUM = r"(\d{1,3}(?:\.\d+)?|[一二两三四五六七八九十]{1,3}|半)"
FZ = r"(余|多|\+|许)?"
UNIT_DAYS = {"年": 365.0, "月": 30.0, "周": 7.0, "星期": 7.0, "天": 1.0, "日": 1.0,
             "小时": 1 / 24, "h": 1 / 24, "分钟": 1 / 1440}
# v0.5：模糊词可在单位前（"30余年""7+年""3个多月"）或单位后（"2年余"）；单位后可带"半"（"1年半"）
# 分组：1 数字，2 单位前模糊，3 "个"后模糊，4 单位，5 半，6 单位后模糊
DUR_CORE = NUM + r"\s*" + FZ + r"\s*(?:个)?\s*(余|多)?\s*(年|月|周|星期|天|日|小时|h|分钟)(半)?\s*" + FZ
# 前面不能紧挨数字、日期成分或"+"（"孕38+2周"的"2周"）；后面不能接日期成分或"份"
PRE_BLOCK = r"(?<![\d.年月/\-+])"
DURATION = re.compile(PRE_BLOCK + DUR_CORE + r"(?!\s*\d{1,2}\s*[日号])(?!份)")
ONSET = re.compile(PRE_BLOCK + DUR_CORE + r"\s*(?:之|以)?前")
# v0.5："入院前3天"
ONSET_PRE = re.compile(r"(?:入院|就诊|来院|来我院|入我院)\s*前\s*" + DUR_CORE)
QUOTE = re.compile(r"因\s*[“\"「]([^”\"」]{1,80})[”\"」]")
# v0.5：孕周、停经时长不是病程
GESTATION_BEFORE = re.compile(r"(?:孕|妊娠|停经|胎龄|孕龄)\s*$")
# v0.5：主诉里从确诊、发现或治疗起算的时长；现病史可以从更早的首发症状写起
CHIEF_ANCHOR = re.compile(r"确诊|诊断|发现|查出|体检|术后|化疗|放疗|靶向|免疫治疗|治疗后|透析|复查|随访")
# v0.5：现病史里"高血压病史10年""确诊糖尿病5年"这类不带"前"的病史时长，也算起病证据
HIST_ANCHOR = re.compile(r"病史|史|患|确诊|诊断|发现|罹患|透析")
HIST_AFTER = re.compile(r"\s*(?:病史|史)")
# v0.5："近3个月出现乏力""近1年来"
ONSET_NEAR = re.compile(r"近\s*" + DUR_CORE)
DX_EVENT = re.compile(r"确诊|诊断|发现|查出|体检")
ONSET_VERB = re.compile(r"出现|突发|突然|感到|自觉|发现|开始|起病|诱因|发生|摔伤|受伤|跌倒|扭伤|发作")
# v0.5："3天前症状加重""近1天来急剧加重"说的是病情加重，不是起病；"再发""再次出现"是新一次发作，仍算起病
WORSEN = re.compile(r"加重|加剧|恶化")
RECUR = re.compile(r"再发|再次|(?<!反)复发|又出现|又发|再度")  # "反复发作"里的"复发"不算
FIRST_VERB = re.compile(r"出现|起病|发病|开始")
# v0.5："5年前行胃癌切除术"是治疗事件，不是起病：分句里有治疗类词、没有起病动词时，按病史时长处理
TREAT_EVENT = re.compile(r"行|手术|术后|切除|化疗|放疗|治疗|置入|植入|移植|透析")
# v0.5：主诉里"于1周前入院治疗"的"1周"是入院时间，不是病程
CHIEF_NOT_DUR_AFTER = re.compile(r"\s*(?:之|以)?前\s*(?:于|在|来|到)?\s*(?:我院|本院|外院|医院)?\s*(?:入院|住院|就诊|来院|收住|收治)")
ADMIT_WORD = re.compile(r"来院|来我院|就诊|入院|收入|收住|收治|住院|复查|出院|门诊")
RELDAY = re.compile(r"今晨|今早|今日|今天|今上午|今中午|今下午|今晚|今夜|昨晨|昨晚|昨夜|昨日|昨天|前天|前日")
RELDAY_DAYS = {"今": 0.5, "昨": 1.0, "前": 2.0}
ONSET_DATE = re.compile(r"(?<!\d)(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*(?:[月\-/.]\s*(\d{1,2})\s*[日号]?|月)")
CLAUSE_SEPS = "。；;\n，,"


def _duration_value(m):
    n = cn_to_float(m.group(1))
    if n is None:
        return None
    unit = m.group(4)
    if unit == "年" and n > 120:
        return None
    days = n * UNIT_DAYS[unit] + (0.5 * UNIT_DAYS[unit] if m.group(5) else 0.0)
    fuzzy = 1 if (m.group(2) or m.group(3) or m.group(6)) else 0
    return days, fuzzy


def _clause(text, s, e):
    a = max(text.rfind(ch, 0, s) for ch in CLAUSE_SEPS) + 1
    ends = [x for x in (text.find(ch, e) for ch in CLAUSE_SEPS) if x >= 0]
    return a, (min(ends) if ends else len(text))


def _gestational(text, s):
    return bool(GESTATION_BEFORE.search(text[max(0, s - 4):s]))


def _durations_in(span_text, offset, rx, meta, sections, field, rule, first_only=False, extra_fn=None, skip_fn=None):
    out = []
    for m in rx.finditer(span_text):
        if _gestational(span_text, m.start()):
            continue
        if skip_fn and skip_fn(span_text, m):
            continue
        v = _duration_value(m)
        if v is None:
            continue
        extra = extra_fn(span_text, m) if extra_fn else ""
        s, e = offset + m.start(), offset + m.end()
        out.append(_fact(meta, sections, s, e, field, f"{v[0]:g}|fuzzy={v[1]}{extra}", rule))
        if first_only:
            break
    return out


def _chief_anchor(span_text, m):
    """anchor=1：主诉从确诊、发现或治疗起算。worse=1：这个时长说的是加重或再发（"腹痛加重3天"），
    它前面同一分句里、上一个时长之后出现了加重、再发类词。"""
    a, b = _clause(span_text, m.start(), m.end())
    prev = [x.end() for x in DURATION.finditer(span_text, a, m.start()) if x.end() <= m.start()]
    before = span_text[max([a] + prev):m.start()]
    worse = 1 if (WORSEN.search(before) or RECUR.search(before)) else 0
    return f"|anchor={1 if CHIEF_ANCHOR.search(span_text[a:b]) else 0}|worse={worse}"


def _chief_not_duration(span_text, m):
    return bool(CHIEF_NOT_DUR_AFTER.match(span_text, m.end()))


def _is_worse(clause):
    """分句说的是病情加重（不是起病、也不是再发）。"出现头晕并逐渐加重"里起病动词在前，仍算起病。"""
    if RECUR.search(clause):
        return False
    w = WORSEN.search(clause)
    if not w:
        return False
    f = FIRST_VERB.search(clause)
    return not (f and f.start() < w.start())


def _evidence_tag(kind):
    """kind=onset：真正的起病时间；kind=hist：不带"前"的病史时长，只能增加相容，不能单独让这一对可比；
    kind=worse：加重时间，只有主诉本身写的是加重时长（worse=1）才拿来比；
    "N年前行手术"这类治疗事件记作 kind=hist。
    dx=1：证据所在分句本身就是确诊、发现类事件，主诉的"确诊口径"放宽对它不适用。"""
    def tag(span_text, m):
        a, b = _clause(span_text, m.start(), m.end())
        clause = span_text[a:b]
        k = kind
        if kind == "onset":
            if _is_worse(clause):
                k = "worse"
            elif TREAT_EVENT.search(clause) and not ONSET_VERB.search(clause):
                k = "hist"
        return f"|kind={k}|dx={1 if DX_EVENT.search(clause) else 0}"
    return tag


def _onset_evidence(span_text, offset, meta, sections, prefix):
    """现病史（或出院记录入院情况引号外）里一切能说明"病了多久"的证据，供 G1-4 使用。"""
    out = []
    onset_tag = _evidence_tag("onset")
    out += _durations_in(span_text, offset, ONSET, meta, sections, "onset_any", prefix + ".all", extra_fn=onset_tag)
    out += _durations_in(span_text, offset, ONSET_PRE, meta, sections, "onset_any", prefix + ".pre", extra_fn=onset_tag)
    out += _durations_in(span_text, offset, ONSET_NEAR, meta, sections, "onset_any", prefix + ".near", extra_fn=onset_tag)
    # 不带"前"的病史时长
    hist_tag = _evidence_tag("hist")
    for m in DURATION.finditer(span_text):
        if _gestational(span_text, m.start()):
            continue
        if re.match(r"\s*(?:之|以)?前", span_text[m.end():m.end() + 3]) or span_text[max(0, m.start() - 1):m.start()] == "近":
            continue
        a, _ = _clause(span_text, m.start(), m.end())
        if not (HIST_ANCHOR.search(span_text[a:m.start()]) or HIST_AFTER.match(span_text, m.end())):
            continue
        v = _duration_value(m)
        if v is None:
            continue
        out.append(_fact(meta, sections, offset + m.start(), offset + m.end(), "onset_any",
                         f"{v[0]:g}|fuzzy={v[1]}{hist_tag(span_text, m)}", prefix + ".hist"))
    # "今晨突发胸痛"
    for m in RELDAY.finditer(span_text):
        a, b = _clause(span_text, m.start(), m.end())
        clause = span_text[a:b]
        if not ONSET_VERB.search(clause) or ADMIT_WORD.search(clause):
            continue
        out.append(_fact(meta, sections, offset + m.start(), offset + m.end(), "onset_any",
                         f"{RELDAY_DAYS[m.group(0)[0]]:g}|fuzzy=1{onset_tag(span_text, m)}", prefix + ".relday"))
    # "2024年1月出现咳嗽"：换算要用入院日期，留给比对器
    for m in ONSET_DATE.finditer(span_text):
        a, b = _clause(span_text, m.start(), m.end())
        clause = span_text[a:b]
        if not ONSET_VERB.search(clause) or ADMIT_WORD.search(clause):
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        if not (1 <= mo <= 12):
            continue
        prec = "day" if m.group(3) else "month"
        d = int(m.group(3)) if m.group(3) else 1
        out.append(_fact(meta, sections, offset + m.start(), offset + m.end(), "onset_date",
                         f"{y:04d}-{mo:02d}-{d:02d}|prec={prec}{onset_tag(span_text, m)}", prefix + ".date"))
    return out


def extract_durations(text, meta, sections):
    out = []
    for sec in sections:
        if sec.name == "主诉":
            out += _durations_in(sec.text, sec.start, DURATION, meta, sections, "chief_duration", "dur.chief",
                                 extra_fn=_chief_anchor, skip_fn=_chief_not_duration)
        elif sec.name == "现病史":
            # 第一个"N…前"只作展示（v0.4 的 G1-4 口径），v0.5 的 G1-4 用 onset_any 和 onset_date
            out += _durations_in(sec.text, sec.start, ONSET, meta, sections, "onset_ago", "dur.onset", first_only=True)
            out += _onset_evidence(sec.text, sec.start, meta, sections, "dur.onset")
        elif sec.name == "入院情况":
            # 出院记录：引号里的主诉 ↔ 引号外的起病证据
            q = QUOTE.search(sec.text)
            if not q:
                continue
            qs = sec.start + q.start(1)
            out += _durations_in(q.group(1), qs, DURATION, meta, sections, "chief_duration", "dur.chief.quoted",
                                 extra_fn=_chief_anchor, skip_fn=_chief_not_duration)
            rest = sec.text[:q.start()] + " " * (q.end() - q.start()) + sec.text[q.end():]
            out += _durations_in(rest, sec.start, ONSET, meta, sections, "onset_ago", "dur.onset.discharge", first_only=True)
            out += _onset_evidence(rest, sec.start, meta, sections, "dur.onset.discharge")
    return out


# ---------- 侧别 ----------

_ORGANS = json.loads((Path(__file__).parent / "config" / "organs.json").read_text(encoding="utf-8"))
_ORGAN_OF = {}
for canon, names in _ORGANS.items():
    if canon.startswith("_"):
        continue
    for n in names:
        _ORGAN_OF[n] = canon
MULTI_LESION_ORGANS = frozenset(_ORGANS.get("_multi", []))
_ORGAN_ALT = "|".join(re.escape(n) for n in sorted(_ORGAN_OF, key=len, reverse=True))
LATERALITY = re.compile(r"(左右|双|左|右)\s*(?:侧)?\s*(" + _ORGAN_ALT + r")")
# 同一分句里出现这些词，说明说的是既往或另一件事
LAT_HISTORY_MARK = re.compile(r"术后|切除|既往|病史|曾|史[，,。；;]|陈旧")
# v0.5：阴性所见（"右侧卵巢未见异常"）说的不是病变在哪一侧
LAT_NEG_MARK = re.compile(r"未见|未及|未触及|未闻及|无异常|无明显|无压痛|无肿大|无畸形|无红肿|无叩痛|正常|阴性")
# v0.5：体表定位线（"左锁骨中线第5肋间"）不是病变部位
LAT_LANDMARK_AFTER = re.compile(r"\s*(?:中线|旁线|上线|下线)")


def extract_laterality(text, meta, sections):
    out = []
    for m in LATERALITY.finditer(text):
        if LAT_LANDMARK_AFTER.match(text, m.end()):
            continue
        side = {"左右": "双", "双": "双", "左": "左", "右": "右"}[m.group(1)]
        organ = _ORGAN_OF[m.group(2)]
        # 所在分句（前后都看，"左肾切除术后"的标记在器官词之后）
        a = max(text.rfind(ch, 0, m.start()) for ch in "。；;\n，,") + 1
        ends = [text.find(ch, m.end()) for ch in "。；;\n，,"]
        b = min([x for x in ends if x >= 0] or [len(text)])
        # v0.5：分句不越过段落标题。v0.4 里"现病史："这几个字本身含"病史"，现病史第一句的侧别全被当成既往
        for sec in sections:
            if sec.start <= m.start() < sec.end:
                a, b = max(a, sec.start), min(b, sec.end)
                break
        clause = text[a:m.start()] + text[m.end():b]
        hist = 1 if LAT_HISTORY_MARK.search(clause) else 0
        neg = 1 if LAT_NEG_MARK.search(clause) else 0
        out.append(_fact(meta, sections, m.start(), m.end(), "laterality",
                         f"{side}|{organ}|hist={hist}|neg={neg}", "lat.v3"))
    return out


# ---------- 性别特异内容（G1-2） ----------

FEMALE_ONLY = re.compile(r"孕\s*\d+\s*产\s*\d+|G\s*\d+\s*P\s*\d+|初潮|月经周期|经期|末次月经|LMP|绝经|痛经|白带|剖宫产|顺产")
MALE_ONLY = re.compile(r"前列腺|睾丸|阴囊|阴茎|包皮")
OTHER_PERSON = re.compile(r"配偶|妻|爱人|丈夫|其夫|女儿|儿媳|母|姐|妹|父|兄|弟|儿子|子女")
# v0.3：出生史里的孕产次、分娩方式说的是患者母亲（儿科病历的个人史常写"系G1P1，足月顺产"）
BIRTH_TOKEN = re.compile(r"孕\s*\d+\s*产\s*\d+|G\s*\d+\s*P\s*\d+|剖宫产|顺产")
BIRTH_CONTEXT = re.compile(r"胎|出生|足月|早产|过期产|新生儿|生后|娩出|产程|羊水|Apgar|阿氏评分")
CHILD_AGE_YEARS = 14.0


def _header_age_years(text):
    m = HEAD_AGE.search(text)
    return age_in_years(m.group(2), m.group(3)) if m else None


def extract_sex_specific(text, meta, sections):
    out = []
    age = _header_age_years(text)
    child = age is not None and age < CHILD_AGE_YEARS
    for sec in sections:
        if sec.name in {"家族史"}:
            continue
        for rx, field in ((FEMALE_ONLY, "female_only_content"), (MALE_ONLY, "male_only_content")):
            for m in rx.finditer(sec.text):
                a = max(sec.text.rfind(ch, 0, m.start()) for ch in "。；;\n") + 1
                b_candidates = [sec.text.find(ch, m.end()) for ch in "。；;\n"]
                b = min([x for x in b_candidates if x >= 0] or [len(sec.text)])
                sentence = sec.text[a:b]
                if OTHER_PERSON.search(sentence):
                    continue
                if BIRTH_TOKEN.fullmatch(m.group(0)) and (child or BIRTH_CONTEXT.search(sentence)):
                    continue
                s = sec.start + m.start()
                out.append(_fact(meta, sections, s, sec.start + m.end(), field, m.group(0), "sexspec.v3"))
    return out


# ---------- 入口 ----------

def extract_facts(text, admission_id="SYNTH", doc_seq=0, doc_type="hospital_record"):
    meta = (admission_id, doc_seq, doc_type)
    sections = split_sections(text)
    facts = []
    facts += extract_header(text, meta, sections)
    facts += extract_body_demographics(text, meta, sections)
    facts += extract_durations(text, meta, sections)
    facts += extract_laterality(text, meta, sections)
    facts += extract_sex_specific(text, meta, sections)
    return facts, sections
