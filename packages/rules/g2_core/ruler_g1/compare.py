"""关1 比对器：纯程序判定，不用模型。

evaluate_document() 返回：
  facts        抽取出的事实
  conflicts    冲突列表
  checkable    {pair_id: n_checkable}
  mild         {pair_id: 轻差次数}（目前只有 G1-3 的 1 岁差）
"""
import datetime as dt
import re
from collections import defaultdict

from .extractors import MULTI_LESION_ORGANS, extract_facts
from .models import Conflict

PAIRS = ("G1-1", "G1-2", "G1-3", "G1-4", "G1-5", "G1-6")

# 各对适用的文书类型。first_progress 由调用方用 sections.is_first_progress 判定后传入。
PAIR_DOC_TYPES = {
    "G1-1": {"hospital_record", "discharge_summary", "first_progress", "summary7"},
    "G1-2": {"hospital_record", "summary7"},
    "G1-3": {"hospital_record", "discharge_summary", "first_progress", "summary7"},
    # v0.5：G1-4 改为"证据口径"（主诉最长时长 ↔ 现病史里任何一个起病证据），原 G1-4b 并入，不再单列
    "G1-4": {"hospital_record", "discharge_summary", "summary7"},
    "G1-5": {"hospital_record", "discharge_summary", "summary7"},
    "G1-6": {"discharge_summary"},
}
# summary7：LLM-CARe 论文格式（性别、年龄、主要诊断、主诉、现病史、住院经过、出院医嘱），
# 由 adapters.render_summary7() 渲染成带标题的文本后再进尺子。

# G1-5 的比对段落对：每一对里，同一器官一边只写左、另一边只写右才算冲突
LAT_SECTION_PAIRS = {
    "hospital_record": [("主诉", "入院诊断"), ("主诉", "专科情况"), ("现病史", "入院诊断")],
    "discharge_summary": [("入院情况", "出院诊断"), ("入院诊断", "出院诊断")],
    "summary7": [("主诉", "出院诊断"), ("现病史", "出院诊断")],
}

# G1-4 时长容差：区间 [v, v×1.5(带余/多/+)]，再两侧各放宽 ×0.8/×1.25 和 ±min(1 天, v/2)
FUZZY_FACTOR = 1.5
REL_LO, REL_HI, ABS_DAYS = 0.8, 1.25, 1.0
AGE_CONFLICT_YEARS = 2.0
AGE_MILD_YEARS = 1.0


def _conf(pair, a, b, axis):
    return Conflict(pair, a.section, a.start, a.end, b.section, b.start, b.end, axis)


def _parse_kv(value):
    head, *rest = value.split("|")
    kv = dict(x.split("=", 1) for x in rest if "=" in x)
    return head, kv


def _interval(days, fuzzy):
    hi = days * (FUZZY_FACTOR if fuzzy else 1.0)
    slack = min(ABS_DAYS, 0.5 * days)  # 小时级时长不能放宽一整天
    return days * REL_LO - slack, hi * REL_HI + slack


# G1-4 多段主诉：判断主诉某一段说的症状，现病史里带时长的句子有没有提到（按两字片段重合，去掉通用词）
_LINK_FILLER = re.compile(
    r"无明显诱因|诱因|发现|确诊|诊断|出现|反复|间断|持续|伴有|伴|及|和|或|患者|于|约|余|多|近|来|前|后|再发|加重|"
    r"首次|突发|突然|逐渐|渐|明显|自觉|感|有|性|入院|就诊|我院|门诊|升高|降低|增高|减低|异常|不适|增多|减少|下降|"
    r"小时|分钟|年|月|周|天|日|[0-9０-９一二三四五六七八九十半两个]")
_CHIEF_SEPS = "。；;\n，,“\"「"
_SENT_SEPS = "。；;\n"


def _bigrams(s):
    s = re.sub(r"[^\u4e00-\u9fff]", " ", _LINK_FILLER.sub(" ", s))
    return {s[i:i + 2] for i in range(len(s) - 1) if " " not in s[i:i + 2]}


def _chief_topic(text, part, parts):
    a = max(text.rfind(ch, 0, part.start) for ch in _CHIEF_SEPS) + 1
    a = max([a] + [p.end for p in parts if p.end <= part.start])
    return text[a:part.start]


def _sentence(text, s, e):
    a = max(text.rfind(ch, 0, s) for ch in _SENT_SEPS) + 1
    ends = [x for x in (text.find(ch, e) for ch in _SENT_SEPS) if x >= 0]
    return text[a:(min(ends) if ends else len(text))]


def _date(s):
    y, m, d = (int(x) for x in s.split("-"))
    return dt.date(y, m, d)


def evaluate_document(text, admission_id="SYNTH", doc_seq=0, doc_type="hospital_record"):
    facts, sections = extract_facts(text, admission_id, doc_seq,
                                    "progress_note" if doc_type == "first_progress" else doc_type)
    by = defaultdict(list)
    for f in facts:
        by[f.field].append(f)
    conflicts, mild = [], defaultdict(int)
    checkable = {p: 0 for p in PAIRS}
    applies = {p: doc_type in PAIR_DOC_TYPES[p] for p in PAIRS}

    head_sex = by["header_sex"][0] if by["header_sex"] else None
    head_age = by["header_age"][0] if by["header_age"] else None

    # G1-1 抬头性别 ↔ 正文性别
    if applies["G1-1"] and head_sex:
        for b in by["body_sex"]:
            checkable["G1-1"] += 1
            if b.value_norm != head_sex.value_norm:
                conflicts.append(_conf("G1-1", head_sex, b, "有无"))

    # G1-2 抬头性别 ↔ 性别特异内容（每份文书至多记 1 次可比、1 次冲突）
    if applies["G1-2"] and head_sex:
        checkable["G1-2"] += 1
        wrong = by["female_only_content"] if head_sex.value_norm == "男" else by["male_only_content"]
        if wrong:
            conflicts.append(_conf("G1-2", head_sex, wrong[0], "有无"))

    # G1-3 抬头年龄 ↔ 正文年龄
    if applies["G1-3"] and head_age:
        hv, hkv = _parse_kv(head_age.value_norm)
        hv, hge = float(hv), hkv.get("ge") == "1"
        for b in by["body_age"]:
            bv, bkv = _parse_kv(b.value_norm)
            bv, bge = float(bv), bkv.get("ge") == "1"
            checkable["G1-3"] += 1
            if hge and bge:
                continue
            if hge:
                diff = max(0.0, hv - bv)
            elif bge:
                diff = max(0.0, bv - hv)
            else:
                diff = abs(hv - bv)
            if diff >= AGE_CONFLICT_YEARS:
                conflicts.append(_conf("G1-3", head_age, b, "数值"))
            elif diff >= AGE_MILD_YEARS:
                mild["G1-3"] += 1

    # G1-4 主诉病程 ↔ 现病史的起病证据（v0.5 证据口径）
    # 证据：任何"N…前"、"入院前N天"、不带"前"的病史时长、"今晨/昨日"、带起病动词的日期（用入院日期换算）。
    # 任何一条证据与主诉相容就不算冲突；主诉从确诊、发现或治疗起算（anchor=1）时，比它长的证据也算相容。
    if applies["G1-4"] and by["chief_duration"]:
        evidence = []  # (fact, lo, hi, kind, dx)
        for onset in by["onset_any"]:
            ov, okv = _parse_kv(onset.value_norm)
            evidence.append((onset, *_interval(float(ov), okv.get("fuzzy") == "1"),
                             okv.get("kind", "onset"), okv.get("dx") == "1"))
        admit = by["admit_date"][0] if by["admit_date"] else None
        for od in by["onset_date"] if admit else []:
            dv, dkv = _parse_kv(od.value_norm)
            try:
                ad, d0 = _date(admit.value_norm), _date(dv)
            except ValueError:
                continue
            if dkv.get("prec") == "month":
                nxt = dt.date(d0.year + (d0.month == 12), d0.month % 12 + 1, 1)
                far, near = (ad - d0).days, (ad - nxt).days + 1
                if near < 0:
                    continue
                lo, hi = near * REL_LO - ABS_DAYS, far * REL_HI + ABS_DAYS
            else:
                days = (ad - d0).days
                if days < 0:
                    continue
                lo, hi = _interval(float(days), False)
            evidence.append((od, lo, hi, "onset", dkv.get("dx") == "1"))
        # 主诉的各段：写加重、再发的段（worse=1）不作主诉病程，除非整条主诉只有这一种
        all_parts = by["chief_duration"]
        parts = [f for f in all_parts if _parse_kv(f.value_norm)[1].get("worse") != "1"] or all_parts
        parts = sorted(parts, key=lambda f: float(_parse_kv(f.value_norm)[0]), reverse=True)
        chief = parts[0]
        if len(parts) > 1:
            # 多段主诉（"发现血压升高10年，头晕3天"）：最长那段在现病史里既没有同样长的证据，
            # 带时长的句子也没提到它的症状，说明现病史只写了这次发作，改用下一段比
            c_lo = _interval(float(_parse_kv(chief.value_norm)[0]), False)[0]
            topic = _bigrams(_chief_topic(text, chief, all_parts))
            longer = any(hi >= c_lo for _, lo, hi, k, _ in evidence if k != "worse")
            linked = any(topic & _bigrams(_sentence(text, e[0].start, e[0].end)) for e in evidence)
            if not longer and not linked:
                chief = parts[1]
        cv, ckv = _parse_kv(chief.value_norm)
        c_lo, c_hi = _interval(float(cv), ckv.get("fuzzy") == "1")
        chief_worse = ckv.get("worse") == "1"
        # 主诉写的是加重时长（"腹痛加重3天"）时，更早的起病同样不算冲突
        anchored = ckv.get("anchor") == "1" or chief_worse
        # 加重时间只在主诉本身写的是加重时长时才参与比较（"发热2天，加重1天"不能拿"近1天加重"来圆）
        if not chief_worse:
            evidence = [e for e in evidence if e[3] != "worse"]
        # 至少要有一条真正的起病证据才可比；病史时长只能帮着判"相容"
        if any(e[3] == "onset" for e in evidence):
            checkable["G1-4"] += 1
            compatible = any(not (c_hi < lo or hi < c_lo) for _, lo, hi, _, _ in evidence)
            if not compatible and anchored:
                # 主诉从确诊、发现或治疗起算：更早的症状起病不算冲突；但证据本身也是确诊、发现类事件时不放宽，
                # 病史时长（"高血压10年"）也不拿来放宽
                compatible = any(hi >= c_lo for _, lo, hi, k, dx in evidence if k == "onset" and not dx)
            first = next(e for e in evidence if e[3] == "onset")
            if not compatible:
                conflicts.append(_conf("G1-4", chief, first[0], "数值"))
            elif not chief_worse:
                # 加重段比整个病程还长（"…2个月，加重1年"），而现病史里没有任何一条证据对得上它
                for w in all_parts:
                    wv, wkv = _parse_kv(w.value_norm)
                    if wkv.get("worse") != "1":
                        continue
                    w_lo, w_hi = _interval(float(wv), wkv.get("fuzzy") == "1")
                    if w_lo > c_hi and not any(not (w_hi < lo or hi < w_lo) for _, lo, hi, _, _ in evidence):
                        conflicts.append(_conf("G1-4", w, first[0], "数值"))
                        break

    # G1-5 侧别（v0.3：只在被比的两段里看"双/左右"，全身查体里的"双肺""双下肢"不再连带排除）
    if applies["G1-5"]:
        lat = []
        for f in by["laterality"]:
            side, organ, *rest = f.value_norm.split("|")
            kv = dict(x.split("=", 1) for x in rest)
            lat.append((f, side, organ, kv.get("hist") == "1" or kv.get("neg") == "1"))
        for sx, sy in LAT_SECTION_PAIRS.get(doc_type, []):
            in_pair = [x for x in lat if x[0].section in (sx, sy)]
            organs = {x[2] for x in in_pair}
            for organ in sorted(organs):
                # v0.5：两侧常各有病灶的血管、脑区不比
                if organ in MULTI_LESION_ORGANS:
                    continue
                mentions = [x for x in in_pair if x[2] == organ]
                if any(x[1] == "双" for x in mentions):
                    continue
                # v0.5：既往（hist）和阴性所见（neg）不参与比对
                fx = [x for x in mentions if x[0].section == sx and not x[3]]
                fy = [x for x in mentions if x[0].section == sy and not x[3]]
                if not fx or not fy:
                    continue
                set_x = {x[1] for x in fx}
                set_y = {x[1] for x in fy}
                checkable["G1-5"] += 1
                if len(set_x) == 1 and len(set_y) == 1 and set_x != set_y:
                    conflicts.append(_conf("G1-5", fx[0][0], fy[0][0], "侧别"))

    # G1-6 住院天数 ↔ 日期差（计天惯例：出院日期 − 入院日期，见包 1 A8）
    if applies["G1-6"] and by["los_days"] and by["admit_date"] and by["discharge_date"]:
        try:
            actual = (_date(by["discharge_date"][0].value_norm) - _date(by["admit_date"][0].value_norm)).days
        except ValueError:
            actual = None
        if actual is not None:
            checkable["G1-6"] += 1
            los = int(by["los_days"][0].value_norm)
            # v0.5：当天入出院记 1 天是常规写法
            if los != actual and not (actual == 0 and los == 1):
                conflicts.append(_conf("G1-6", by["los_days"][0], by["admit_date"][0], "数值"))

    return {"facts": facts, "conflicts": conflicts, "checkable": checkable,
            "mild": dict(mild), "sections": [s.name for s in sections]}
