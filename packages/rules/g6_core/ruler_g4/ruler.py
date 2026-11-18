"""关4 尺子（g4-rules-0.2；0.1 → 0.2 的改动见 README 第 9 节）：写全了没有（要素覆盖）、写得具体不具体（信息单元密度）、是不是靠空话和套话凑的（空句、跨病历套话句、段间重复）。

全部是规则，不调模型；和 LLM-CARe 生成时用的信号（它的 9 条完整性清单和 LLM 评委）脱钩（关卡谱 X1）。
"""
import re
import unicodedata

from ruler_g3.adapters import care_get, doc_kind
from ruler_g3.engine import mentions as g3_mentions
from ruler_g3.adapters import from_care7
from ruler_g3.lexicon import lexicon
from ruler_g3.parse import mention_context

from .terms import (DISEASES, DRUG_EXTRA, EXAMS, GENERAL, PROC_CUT, PROC_NAMED, PROC_STOP, RESULT_WORDS, SYMPTOMS,
                    TREAT_CLASSES)

FIELDS = ("cc", "hpi", "hc", "di")          # 参与计数的四段叙述（性别、年龄、诊断是给定属性，不算）
MIN_SENT = 6                                # 少于 6 个汉字的句子不参与空句和套话
WINDOW = 25                                 # 检查名后面看多少字找结果
TPL_MIN_RECORDS = 3                         # 同一句（规范化后）出现在本池至少 3 份不同病历里，算套话句
DUP_N = 10                                  # 7 字段段间重复：后段里 ≥10 字的片段在前段出现过

_CHAR = re.compile(r"[一-鿿A-Za-z0-9]")
_CJK = re.compile(r"[一-鿿]")
_SENT_END = re.compile(r"[。；;！!？?\n]")
_PUNCT_WS = re.compile(r"[\s\W_]+", re.U)
_DIGITS = re.compile(r"\d")

# ---------- 带单位的数 ----------
_DATE = re.compile(r"\d{4}\s*[-/.年]\s*\d{1,2}(?:\s*[-/.月]\s*\d{1,2}\s*[日号]?)?|\d{1,2}\s*月\s*\d{1,2}\s*[日号]")
_ASCII_U = r"(?:℃|°C|bpm|mmHg|kPa|%|g/dL|g/L|mg/dL|mg/L|mmol/L|μmol/L|µmol/L|umol/L|U/L|IU/L|u/L|ng/mL|ng/ml|pg/mL|pg/ml|μg/L|ug/L|mIU/L|μIU/mL|uIU/mL" \
           r"|[×xX*]\s*10\^?\s*(?:9|12)\s*/\s*L|10\^?(?:9|12)/L|L/min|cm|mm|mL|ml|kg|mg|μg|µg|ug|mcg|IU|g|U|L)(?![A-Za-z])"
_CN_U = r"(?:次\s*[/／]\s*分|分钟|小时|个月|星期|毫克|毫升|微克|千克|公斤|克|片|粒|支|袋|枚|滴|次|级|分|度|天|日|周|月|年|h|min)"
_NUM = re.compile(r"(?<![\d.A-Za-z])(\d+(?:\.\d+)?)(?:\s*[-~～—至到]\s*\d+(?:\.\d+)?)?\s*(?:余|多)?\s*(?:个)?\s*(" + _ASCII_U + "|" + _CN_U + r")")
_GRADE = re.compile(r"(?:[ⅠⅡⅢⅣⅤ]+|(?<![A-Za-z])(?:IV|I{1,3}|V)(?![A-Za-z]))\s*(?:级|度|期|型)")
_CN_TIME = re.compile(r"([一二两三四五六七八九十半数几]+)\s*(?:余|多)?\s*(?:个)?\s*(分钟|小时|天|日|周|星期|月|年)")
IU_TYPES = ("NUM", "DRUG", "EXAM", "SYM", "DX", "PROC")
DRUG_EXTRA_SET = set(DRUG_EXTRA)

# ---------- 0.2：时间锚点（不是信息单元，单独计数；H4-4 用） ----------
# 日期（去标识化后日期整体平移，但写法保留）、日期占位符、"第 N 天""术后 N 天""次日"一类的相对日。
_DATE_PH = re.compile(r"\[[^\[\]\n]{0,10}(?:日期|时间|DATE|Date|date)[^\[\]\n]{0,10}\]")
_RELDAY = re.compile(r"第\s*(?:\d+|[一二两三四五六七八九十]+)\s*(?:天|日)|术后\s*(?:\d+|[一二两三四五六七八九十]+)\s*(?:天|日)|次日|翌日|次晨")

# ---------- 0.2：出院医嘱里的"如出现……及时就诊"警示语，里面的症状、检查不算这个病人的信息 ----------
_WARN = re.compile(r"(?:如|若|一旦|如果|如有)[^。；;！!？?\n]{0,40}?(?:及时|立即|随时|尽快|即刻|马上)[^。；;！!？?\n]{0,6}"
                   r"(?:就诊|复诊|就医|来院|到院|急诊|医院|随诊)")
TIME_UNITS = {"分钟", "小时", "个月", "星期", "天", "日", "周", "月", "年", "h", "min"}


def _term_rx(terms):
    terms = sorted(set(terms), key=len, reverse=True)
    parts = []
    for t in terms:
        p = re.escape(t)
        if re.match(r"^[A-Za-z0-9\-/+.]+$", t):
            p = r"(?<![A-Za-z])" + p + r"(?![A-Za-z])"
        parts.append(p)
    return re.compile("|".join(parts))


_TYPE_OF = {}
for _t in SYMPTOMS:
    _TYPE_OF[_t] = "SYM"
for _t in EXAMS:
    _TYPE_OF.setdefault(_t, "EXAM")
for _t in PROC_NAMED:
    _TYPE_OF.setdefault(_t, "PROC")
for _t in TREAT_CLASSES:
    _TYPE_OF.setdefault(_t, "PROC")
for _t in DISEASES:
    _TYPE_OF.setdefault(_t, "DX")
for _t in DRUG_EXTRA:
    _TYPE_OF.setdefault(_t, "DRUG")
_TERMS = _term_rx(_TYPE_OF)
_RESULT = _term_rx(RESULT_WORDS)
_PROC_SHU = re.compile(r"([一-鿿]{1,14})术")


def nchars(text):
    return len(_CHAR.findall(text or ""))


def norm_key(s, dx=""):
    """句子的规范化写法：NFKC、去空白和标点、数字都变 0、本份病历的诊断名换成 Δ。用于跨病历比对套话句。"""
    s = unicodedata.normalize("NFKC", s or "")
    if dx and 2 <= len(dx) <= 30:
        s = s.replace(dx, "Δ")
    s = _PUNCT_WS.sub("", s)
    return _DIGITS.sub("0", s).lower()


def _overlaps(a, b, spans):
    return any(a < e and s < b for s, e in spans)


def info_units(text, dx=""):
    """一段文字里的信息单元。返回 [{type, key, start, end}]，可以有重复的 key（去重在上一层做）。
    dx：本份病历的主要诊断字串（给定属性），它在正文里出现的地方不算信息单元。"""
    text = text or ""
    out = []
    dates = [(m.start(), m.end()) for m in _DATE.finditer(text)]
    dxs = [(m.start(), m.end()) for m in re.finditer(re.escape(dx), text)] if dx and 2 <= len(dx) <= 30 else []
    for m in _NUM.finditer(text):
        if _overlaps(m.start(), m.end(), dates):
            continue
        unit = re.sub(r"\s+", "", m.group(2))
        out.append({"type": "NUM", "key": f"{m.group(1)}{unit}", "start": m.start(), "end": m.end(), "unit": unit})
    for m in _CN_TIME.finditer(text):
        if _overlaps(m.start(), m.end(), [(x["start"], x["end"]) for x in out]):
            continue
        out.append({"type": "NUM", "key": m.group(0), "start": m.start(), "end": m.end(), "unit": m.group(2)})
    for m in _GRADE.finditer(text):
        out.append({"type": "NUM", "key": re.sub(r"\s+", "", m.group(0)), "start": m.start(), "end": m.end(), "unit": "级"})
    # 有名字的手术（"…术"）
    procs = []
    for m in _PROC_SHU.finditer(text):
        run = m.group(1)
        cut = max((i for i, ch in enumerate(run) if ch in PROC_CUT), default=-1)
        name = run[cut + 1:] + "术"
        if len(name) < 3 or name in PROC_STOP or name.endswith("手术") and len(name) <= 4:
            continue
        s = m.end() - len(name)
        procs.append((s, m.end()))
        out.append({"type": "PROC", "key": name, "start": s, "end": m.end()})
    # 症状、检查、具名治疗（一个合并的最长优先词表）
    terms = [(m.start(), m.end(), m.group(0)) for m in _TERMS.finditer(text)
             if not _overlaps(m.start(), m.end(), procs) and not _overlaps(m.start(), m.end(), dxs)]
    for i, (s, e, t) in enumerate(terms):
        kind = _TYPE_OF[t]
        if kind == "EXAM":
            nxt = next((terms[j][0] for j in range(i + 1, len(terms)) if _TYPE_OF[terms[j][2]] == "EXAM"), len(text))
            se = _SENT_END.search(text, e)
            stop = min(nxt, se.start() if se else len(text), e + WINDOW)
            if not _RESULT.search(text, e, stop):
                continue
        out.append({"type": kind, "key": t, "start": s, "end": e})
    # 药名：ruler_g3 的词表；否定的、"肝素帽"一类不算
    lex = lexicon()
    lex_spans = []
    for s, e, alias, did in lex.find(text):
        lex_spans.append((s, e))
        ctx = mention_context(text, s, e, did)
        if ctx["not_drug"] or ctx["negated"]:
            continue
        out.append({"type": "DRUG", "key": did, "start": s, "end": e})
    # 0.2：词表外的维生素、补充剂，和词表命中重叠的不再算一次
    out = [u for u in out if not (u["type"] == "DRUG" and u["key"] in DRUG_EXTRA_SET
                                  and _overlaps(u["start"], u["end"], lex_spans))]
    out = [u for u in out if not _overlaps(u["start"], u["end"], dxs)]
    return sorted(out, key=lambda x: (x["start"], x["end"]))


def sentences(text):
    """按 。；;！!？?和换行切句。返回 [(start, end)]，不含切分符。"""
    out, a = [], 0
    text = text or ""
    for m in _SENT_END.finditer(text):
        if m.start() > a:
            out.append((a, m.start()))
        a = m.end()
    if a < len(text):
        out.append((a, len(text)))
    return out


def time_anchors(text):
    """0.2：时间锚点计数。返回 {"date_n": 日期和日期占位符, "relday_n": 第 N 天一类}。"""
    text = text or ""
    return {"date_n": len(_DATE.findall(text)) + len(_DATE_PH.findall(text)), "relday_n": len(_RELDAY.findall(text))}


def warning_spans(text):
    """0.2：出院医嘱里"如出现……及时就诊"那一句的范围（按句切）。"""
    text = text or ""
    return [(s, e) for s, e in sentences(text) if _WARN.search(text[s:e])]


def analyze_text(text, dx="", field=None):
    """一段文字：信息单元、句子（每句是否空、规范化 key）。
    field == "di"（出院医嘱）时，警示语句子里的症状、检查不算信息单元（0.2）。"""
    text = text or ""
    ius = info_units(text, dx)
    if field == "di":
        warn = warning_spans(text)
        ius = [u for u in ius if not (u["type"] in ("SYM", "EXAM") and _overlaps(u["start"], u["end"], warn))]
    sents = []
    for s, e in sentences(text):
        seg = text[s:e]
        cjk = len(_CJK.findall(seg))
        n_iu = sum(1 for u in ius if u["start"] < e and s < u["end"])
        sents.append({"start": s, "end": e, "chars": nchars(seg), "cjk": cjk, "n_iu": n_iu,
                      "empty": int(cjk >= MIN_SENT and n_iu == 0), "key": norm_key(seg, dx) if cjk >= MIN_SENT else None})
    return {"chars": nchars(text), "iu": ius, "sentences": sents}


def _grams(s, n):
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _covered(text_norm, grams, n):
    if len(text_norm) < n or not grams:
        return 0
    mark = [False] * len(text_norm)
    for i in range(len(text_norm) - n + 1):
        if text_norm[i:i + n] in grams:
            for j in range(i, i + n):
                mark[j] = True
    return sum(mark)


def _plain(s):
    return _PUNCT_WS.sub("", unicodedata.normalize("NFKC", s or ""))


def containment(b, a, n=5):
    """关2 的篇间复制率口径（g2_core ruler_g2.copyrate.containment）：b 的 n-gram 集合有多少比例出现在 a 里。
    这里先去掉空白（与关2 相同），b 不足 n 字返回 0。"""
    sb, sa = re.sub(r"\s+", "", b or ""), re.sub(r"\s+", "", a or "")
    gb = _grams(sb, n)
    if not gb:
        return 0.0
    return len(gb & _grams(sa, n)) / len(gb)


# ---------- 7 字段：要素覆盖 ----------
_DISEASE_WORD = re.compile(r"[癌炎病症瘤]|综合征|结石|梗死|梗塞|衰竭|骨折|损伤|障碍|出血|狭窄|畸形|感染|化疗|复查|维持治疗|迟缓")
_ONSET = re.compile(r"急性|急起|突然|突发|骤然|骤起|逐渐|渐进|缓慢|缓起|隐匿|慢性|反复|间断|间歇|进行性|持续性|阵发性")
_TRIGGER = re.compile(r"诱因|受凉|着凉|劳累|饱餐|饮酒|外伤|情绪激动|活动后|感冒后|饮食不洁|饮食不当|淋雨|无明显原因")
_TREAT = re.compile(r"治疗|手术|吸氧|输液|补液|抗感染|降压|降糖|调脂|降脂|抗凝|抗血小板|护胃|营养支持|对症|化疗|放疗|透析|康复|训练")
_NEG_BEFORE = re.compile(r"(无|否认|未诉|不伴|伴有?|未见|未出现|未再|并)[^。；;，,]{0,8}$")
_PRIOR = re.compile(r"就诊|门诊|急诊|诊断为|诊为|予以?|给予|服用|口服|治疗后|住院|曾在|曾于|外院|当地医院|我院")
_DIS_STATE = re.compile(r"出院时|出院前|目前|现患者|病情(?:好转|稳定|平稳|缓解)|症状(?:缓解|消失|好转|改善|减轻)|生命体征平稳|一般情况(?:可|良好|好)|好转出院|痊愈|治愈")
_FOLLOW = re.compile(r"复查|复诊|随诊|随访|门诊|定期")
_LIFE = re.compile(r"饮食|休息|戒烟|戒酒|限酒|运动|活动|低盐|低脂|糖尿病饮食|避免|劳累|作息")

ON_LIST = ("L1_主诉有症状或病名", "L2_主诉有时长", "L3_起病缓急", "L4_诱因", "L5_主要症状",
           "L6_一般情况五项", "L7_住院检查", "L8_住院治疗", "L9_出院带药均有剂量")
OFF_LIST = ("O1_伴随或阴性症状", "O2_院外诊治经过", "O3_出院情况", "O4_复查随访", "O5_生活饮食指导")


def coverage_care7(record):
    """《病历书写基本规范》第十八条（现病史）、第二十四条（出院记录）里的要素，按 7 字段格式落成规则。
    L1–L9 与 LLM-CARe 的 9 条完整性清单同一批事项（但用规则判，不用评委）；O1–O5 是规范有、它的清单没有的。"""
    cc, hpi, hc, di = (care_get(record, k) for k in FIELDS)
    dx = care_get(record, "dx")
    ius = {k: info_units(v, dx) for k, v in zip(FIELDS, (cc, hpi, hc, di))}
    has = lambda k, t: any(u["type"] == t for u in ius[k])
    gen = {name: any(w in hpi for w in words) for name, words in GENERAL.items()}
    rec = from_care7(record, "cov")
    di_ms = [m for m in g3_mentions(rec) if m["seg"] == "DISCHARGE" and not m["negated"] and not m["topical"]]
    o1 = False
    for u in ius["hpi"]:
        if u["type"] == "SYM" and _NEG_BEFORE.search(hpi[max(0, u["start"] - 10):u["start"]]):
            o1 = True
            break
    exam_any = any(_TYPE_OF.get(m.group(0)) == "EXAM" for m in _TERMS.finditer(hc))
    out = {
        "L1_主诉有症状或病名": has("cc", "SYM") or has("cc", "DX") or bool(_DISEASE_WORD.search(cc)),
        "L2_主诉有时长": any(u["type"] == "NUM" and u.get("unit") in TIME_UNITS for u in ius["cc"]),
        "L3_起病缓急": bool(_ONSET.search(hpi)),
        "L4_诱因": bool(_TRIGGER.search(hpi)),
        "L5_主要症状": has("hpi", "SYM"),
        "L6_一般情况五项": all(gen.values()),
        "L7_住院检查": exam_any,
        "L8_住院治疗": has("hc", "DRUG") or has("hc", "PROC") or bool(_TREAT.search(hc)),
        "L9_出院带药均有剂量": bool(di_ms) and all(m["dose"] for m in di_ms),
        "O1_伴随或阴性症状": o1,
        "O2_院外诊治经过": bool(_PRIOR.search(hpi)),
        "O3_出院情况": bool(_DIS_STATE.search(hc) or _DIS_STATE.search(di)),
        "O4_复查随访": bool(_FOLLOW.search(di)),
        "O5_生活饮食指导": bool(_LIFE.search(di)),
    }
    out = {k: int(v) for k, v in out.items()}
    out["general_n"] = sum(gen.values())
    return out


def analyze_care7(record):
    """一份 7 字段病历的全部关4 读数（套话率要在整池上算，见 template_rates）。"""
    dx = care_get(record, "dx")
    per = {k: analyze_text(care_get(record, k), dx, k) for k in FIELDS}
    keys = set()
    by_type = {}
    for k in FIELDS:
        for u in per[k]["iu"]:
            if (u["type"], u["key"]) not in keys:
                keys.add((u["type"], u["key"]))
                by_type[u["type"]] = by_type.get(u["type"], 0) + 1
    chars = sum(per[k]["chars"] for k in FIELDS)
    empty = sum(s["chars"] for k in FIELDS for s in per[k]["sentences"] if s["empty"])
    # 段间重复：hpi、hc、di 里有多少字落在"前面各段出现过的 ≥10 字片段"上
    seen, dup, later = set(), 0, 0
    for i, k in enumerate(FIELDS):
        t = _plain(care_get(record, k))
        if i > 0:
            dup += _covered(t, seen, DUP_N)
            later += len(t)
        seen |= _grams(t, DUP_N)
    cov = coverage_care7(record)
    anchors = [time_anchors(care_get(record, k)) for k in FIELDS]
    date_n = sum(a["date_n"] for a in anchors)
    relday_n = sum(a["relday_n"] for a in anchors)
    return {
        "chars": chars,
        "iu_distinct": len(keys),
        "iu_by_type": {t: by_type.get(t, 0) for t in IU_TYPES},
        "iu_per_kchar": (1000.0 * len(keys) / chars) if chars else 0.0,
        "empty_chars": empty,
        "empty_share": (empty / chars) if chars else 0.0,
        "dup_chars": dup,
        "dup_share": (dup / later) if later else 0.0,
        "coverage": cov,
        "on_list": sum(cov[k] for k in ON_LIST) / len(ON_LIST),
        "off_list": sum(cov[k] for k in OFF_LIST) / len(OFF_LIST),
        "date_n": date_n,
        "relday_n": relday_n,
        "anchor_n": date_n + relday_n,
        "sent_keys": [(s["key"], s["chars"]) for k in FIELDS for s in per[k]["sentences"] if s["key"]],
        "fields": per,
    }


def template_rates(pool):
    """pool：[(rid, sent_keys)]，sent_keys 来自 analyze_care7 或 analyze_admission。
    同一个规范化句子出现在本池 ≥ TPL_MIN_RECORDS 份不同病历里，算套话句。返回 {rid: (套话字数, 参与比对的字数)}。
    这个读数依赖池的大小，所以只在份数相同的池之间比（任务书写明配对的池）。"""
    df = {}
    for rid, keys in pool:
        for k in {k for k, _ in keys}:
            df[k] = df.get(k, 0) + 1
    out = {}
    for rid, keys in pool:
        tot = sum(c for _, c in keys)
        tpl = sum(c for k, c in keys if df.get(k, 0) >= TPL_MIN_RECORDS)
        out[rid] = (tpl, tot)
    return out


# ---------- 整次住院（关2 各组对 R-core，另报不判） ----------
_FP = re.compile(r"首\s*次\s*病\s*程\s*记\s*录")
_POSTOP = re.compile(r"术\s*后\s*首\s*次\s*病\s*程")
HEADERS = {
    "hr": (("主诉",), ("现病史",), ("既往史",), ("个人史",), ("婚育史", "婚姻史", "月经史", "生育史", "出生史", "喂养史"),
           ("家族史",), ("体格检查",), ("专科检查", "专科情况"), ("辅助检查",), ("初步诊断", "入院诊断")),
    "fp": (("病例特点",), ("诊断依据", "拟诊讨论", "初步诊断依据"), ("鉴别诊断",), ("诊疗计划", "诊疗方案", "治疗计划")),
    "discharge": (("入院情况", "入院时情况"), ("入院诊断",), ("诊疗经过", "住院经过"), ("出院诊断",),
                  ("出院情况", "出院时情况"), ("出院医嘱",)),
}


def doc_role(doc):
    """hr 入院记录 / fp 首次病程 / course 其他病程 / discharge 出院记录。"""
    k = doc_kind(doc)
    if k != "course":
        return k
    if str(doc.get("role") or "") == "FP":
        return "fp"
    head = (doc.get("text") or "")[:300]
    if _FP.search(head) and not _POSTOP.search(head):
        return "fp"
    return "course"


def header_coverage(text, role):
    groups = HEADERS.get(role)
    if not groups:
        return None
    hit = 0
    for names in groups:
        alts = "|".join(r"\s*".join(re.escape(c) for c in n) for n in names)
        if re.search(r"(?:" + alts + r")\s*[:：]", text or ""):
            hit += 1
    return hit / len(groups)


def analyze_admission(adm):
    """整次住院：每篇一行。docs 按列表顺序（导出文件和合成住院里的原顺序），复制率对的是列表里的上一篇。"""
    docs = adm.get("docs") or []
    rows, prev = [], None
    for i, d in enumerate(docs):
        text = d.get("text") or ""
        role = doc_role(d)
        a = analyze_text(text)
        keys = {(u["type"], u["key"]) for u in a["iu"]}
        empty = sum(s["chars"] for s in a["sentences"] if s["empty"])
        rows.append({
            "index": i, "role": role, "chars": a["chars"],
            "iu_distinct": len(keys), "iu_per_kchar": (1000.0 * len(keys) / a["chars"]) if a["chars"] else 0.0,
            "empty_share": (empty / a["chars"]) if a["chars"] else 0.0,
            "copy5_prev": containment(text, prev, 5) if prev is not None else None,
            "copy10_prev": containment(text, prev, 10) if prev is not None else None,
            "header_cov": header_coverage(text, role),
            "sent_keys": [(s["key"], s["chars"]) for s in a["sentences"] if s["key"]],
        })
        prev = text
    return rows
