"""关2 尺子的诊断计数（0.2 新增，总经理编写）。

每个函数只返回类别和数字，不返回任何正文片段，伙计汇总成计数表即可（T50–T56）。
类别名全部来自本文件或尺子自己的词表（细类名、段落名、检查词表的 key、固定的类别），不会带出病历内容。

- classifier_profile(adm)：每篇病程的细类（0.1 和 0.2 两种认法）、标题词位置档、术后/术前字段信号数（T50）
- classifier_acceptance(adms)：校0 的数（任务书 1.1 第 2.6 节）
- k5_profile(adm)：每个正文日期的去向和样子（T52）
- k4_profile(adm)：出院记录"入院情况"里每个触发词的去向（T53）
- k3_profile(result, adm)：每个 K3 冲突的样子（T54）
- mine_miss(kind, clean, new_adm, rec, new_result)：雷没命中的原因（T55）
- legit_alarm(kind, clean, new_adm, rec, new_result)：合法差异误报的样子（T56）
"""
import re

from ruler_g1.sections import is_first_progress
from ruler_g2 import diagnoses as dxm
from ruler_g2 import readings as rd
from ruler_g2.events import admission_mentions
from ruler_g2.text import HIST_RE, clause_span
from ruler_g2.timeline import SUBTYPE_RULES, classify_progress_detail

# ---------- 细类 ----------

POST_FIELDS = ["手术时间", "手术日期", "术中诊断", "术后诊断", "手术方式", "手术名称", "麻醉方式", "手术经过",
               "手术简要经过", "术中所见", "术中见", "术后处理", "注意观察"]
PRE_FIELDS = ["术前诊断", "手术指征", "拟施手术", "拟行手术", "拟施麻醉", "拟行麻醉", "术前准备", "手术者", "简要病情"]


def _field_count(text, names):
    return sum(1 for n in names if re.search(r"\s*".join(re.escape(c) for c in n) + r"\s*[:：]?", text))


def _classify_v01(text):
    """0.1 的认法（只看开头 80 字、按规则顺序），只用于对照。"""
    if is_first_progress(text):
        return "首次病程"
    head = text[:80]
    for name, rx in SUBTYPE_RULES:
        if name in ("术后首次病程", "术前小结"):
            rx = re.compile(rx.pattern.split("|")[0])   # 0.1 没有"术后首程""术前总结"写法
        if rx.search(head):
            return name
    return "其他病程"


def _pos_bin(p):
    if p is None:
        return "none"
    return "0-79" if p < 80 else "80-299"


def _bin3(n):
    return str(n) if n < 3 else "3+"


def classifier_profile(adm):
    out = []
    for d in adm.docs + adm.off_docs:
        if d.doc_type != "progress_note":
            continue
        sub, pos = classify_progress_detail(d.text)
        out.append({"subtype_v02": sub, "subtype_v01": _classify_v01(d.text), "title_pos": _pos_bin(pos),
                    "post_fields": _bin3(_field_count(d.text, POST_FIELDS)),
                    "pre_fields": _bin3(_field_count(d.text, PRE_FIELDS)),
                    "on_timeline": d.pos is not None})
    return out


DS_SURGERY = re.compile(r"麻醉下|术后|手术顺利|行[^，,。；;\n]{1,25}术(?!前)")
# 0.3：校0 第 5 项没命中时再分一层（只作诊断，不改合格线）：出院记录缺失 / 只有宽口径手术操作词 / 都没有
DS_SURGERY_WIDE = re.compile(r"全麻|局麻|腰麻|硬膜外|神经阻滞|镜下|介入|置入|植入|支架|切除|修补|缝合|清创|剖宫|引产|"
                             r"穿刺|置管|成形|吻合|造瘘|固定|取出|分娩|手术")


def classifier_acceptance(adms):
    """adms：fold 0 全部 build_real() 的结果。返回校0 要的数。"""
    n_post = n_pre = n_post_adm = n_both = n_order_ok = n_ds_ok = n_fp1 = n_fp2 = 0
    n_ds_missing = n_ds_wide = 0
    for adm in adms:
        progs = [d for d in adm.docs + adm.off_docs if d.doc_type == "progress_note"]
        n_post += sum(d.subtype == "术后首次病程" for d in progs)
        n_pre += sum(d.subtype == "术前小结" for d in progs)
        fps = [d for d in progs if d.subtype == "首次病程"]
        n_fp1 += len(fps) >= 1
        n_fp2 += len(fps) >= 2
        post = rd._surgery(adm)
        if post is None:
            continue
        n_post_adm += 1
        if adm.ds is not None and DS_SURGERY.search(adm.ds.text):
            n_ds_ok += 1
        elif adm.ds is None:
            n_ds_missing += 1
        elif DS_SURGERY_WIDE.search(adm.ds.text):
            n_ds_wide += 1
        pres = [d for d in adm.docs if d.subtype == "术前小结"]
        if pres:
            n_both += 1
            n_order_ok += pres[0].pos < post.pos
    return {"post_docs": n_post, "pre_docs": n_pre, "post_admissions": n_post_adm,
            "pre_and_post_admissions": n_both, "pre_before_post": n_order_ok,
            "post_with_ds_surgery_words": n_ds_ok, "fp_admissions": n_fp1, "fp_multi_admissions": n_fp2,
            "post_ds_missing": n_ds_missing, "post_ds_wide_words_only": n_ds_wide}


# ---------- K5 ----------

LABEL_FAMILIES = [
    ("签名审签", re.compile(r"签名|签字|审签|审核|审阅|核对|确认")),
    ("记录时间", re.compile(r"记录时间|记录日期|书写时间|完成时间|打印|保存|修改")),
    ("入出院日期", re.compile(r"入院(?:时间|日期)|出院(?:时间|日期)")),
    ("月经孕产", re.compile(r"月经|预产期|LMP|孕")),
    ("复查随诊", re.compile(r"复查|复诊|随访|随诊|门诊|医嘱")),
    ("手术操作", re.compile(r"手术|术|麻醉|穿刺|置管|拔管")),
    ("检查检验", re.compile(r"示|提示|报告|结果|回报|检查|CT|MRI|彩超|B超|血常规|心电图")),
    ("用药治疗", re.compile(r"用药|给予|予|服|输|注射|治疗")),
]


def _label(text, s):
    a = max(text.rfind(ch, 0, s) for ch in "。；;\n") + 1
    pre = text[max(a, s - 15):s]
    for name, rx in LABEL_FAMILIES:
        if rx.search(pre):
            return name
    return "无"


def _delta_bin(n):
    if n <= 0:
        return "≤0"
    if n <= 2:
        return str(n)
    return "3-7" if n <= 7 else ("8-30" if n <= 30 else ">30")


def _line_pos(text, s):
    lines_before = text.count("\n", 0, s)
    lines_after = text.count("\n", s)
    if lines_before == 0:
        return "first_line"
    return "last_3_lines" if lines_after <= 2 else "middle"


def k5_profile(adm):
    out = []
    for d, s, e, x, doc_date, fate in rd.k5_scan(adm):
        out.append({"doc_kind": d.subtype, "fate": fate, "delta": _delta_bin((x - doc_date).days),
                    "section": d.section_name_at(s), "line_pos": _line_pos(d.text, s), "label": _label(d.text, s),
                    "form": "full" if re.match(r"\d{4}", d.text[s:e]) else "month_day"})
    return out


# ---------- K4 ----------

K4_FAMILIES = [("入院后", r"入院后"), ("住院期间", r"住院期间"), ("经治疗", r"经|治疗后"), ("术后术中", r"术后|术中"),
               ("病理", r"病理"), ("复查", r"复查"), ("出院时", r"出院时|好转出院"), ("行某术", r"行")]


def k4_profile(adm):
    """0.3：在整段"入院情况"上数，另记触发词是否落在截断点之后（after_cut）、是否被入院前分句跳过（prior_skip）。"""
    ds = adm.ds
    if ds is None:
        return []
    sec = ds.section("入院情况")
    if sec is None:
        return [{"fate": "no_section"}]
    cut = rd.k4_span(ds)
    out, first = [], True
    n = sec[1] - sec[0]
    for m in rd.K4_RE.finditer(ds.text, sec[0], sec[1]):
        fam = next(name for name, p in K4_FAMILIES if re.match(p, m.group(0)))
        ca, cb = clause_span(ds.text, m.start(), m.end())
        if m.start() >= cut[1]:
            fate = "after_cut"
        elif HIST_RE.search(ds.text[ca:cb]):
            fate = "hist_skip"
        elif rd.k4_skip(ds.text, m):
            fate = "prior_skip"
        else:
            fate = "conflict" if first else "conflict_later"
            first = False
        third = min(2, int(3 * (m.start() - sec[0]) / max(1, n)))
        out.append({"fate": fate, "trigger": fam, "where": ("前", "中", "后")[third],
                    "section_len": "<100" if n < 100 else ("100-299" if n < 300 else "≥300")})
    return out


# ---------- K3 ----------

def k3_profile(result, adm):
    ments = admission_mentions(adm)
    by = {(m.doc_idx, m.start): m for ms in ments.values() for m in ms}
    doc = {d.idx: d for d in adm.docs}
    out = []
    for c in result["readings"]["K3"]["conflicts"]:
        (di1, s1, _), (di2, s2, _) = c["ends"]
        m1, m2 = by.get((di1, s1)), by.get((di2, s2))
        d1, d2 = doc[di1], doc[di2]
        gap = d2.day - d1.day
        a, _ = clause_span(d1.text, s1, s1 + 1)
        out.append({"exam": c["detail"], "result_doc": d1.subtype, "plan_doc": d2.subtype,
                    "result_section": d1.section_name_at(s1),
                    "result_clause_admit": bool(re.search(r"入院后|完善", d1.text[a:s1])),
                    "result_recheck": bool(m1 and m1.recheck),
                    "result_prior_section": bool(m1 and rd.k3_prior(d1, m1)),
                    "same_region": bool(m1 and m2 and m1.region == m2.region),
                    "day_gap": _delta_bin(gap) if gap > 0 else "0"})
    return out


# ---------- 雷没命中的原因 ----------

def _mention_at(adm, doc_idx, start, end):
    for m in admission_mentions(adm)[doc_idx]:
        if m.start < end and start < m.end:
            return m
    return None


def _k2_reason(adm, rec):
    m = None
    for x in admission_mentions(adm)[rec.doc_idx]:
        if x.key == "病理" and rec.start <= x.start < rec.end:
            m = x
            break
    if m is None:
        return "k2_no_pathology_mention"
    if m.tone != "R":
        return f"k2_tone_{m.tone}"
    d = adm.doc(rec.doc_idx)
    ca, cb = clause_span(d.text, m.start, m.end)
    if rd.PRE_BIOPSY_WORDS.search(d.text[ca:cb]):
        return "k2_biopsy_words"
    sp = rd._sampling_doc(adm)
    if sp is None:
        return "k2_no_sampling_doc"
    if all(x.idx != rec.doc_idx for x in rd._before(adm, sp)):
        return "k2_not_before_sampling"
    return "k2_other"


def _b2_reason(adm, rec):
    d = adm.doc(rec.doc_idx)
    name_raw = d.text[rec.start:rec.end]
    dm = dxm.mentions(adm)
    xs = [x for x in dm if dxm.dx_match(x.norm, dxm.norm_dx(name_raw))]
    if not xs:
        return "b2_name_not_in_inventory"
    if any(x.state == "既往" for x in xs):
        return "b2_history_skip"
    susp = next((x for x in xs if x.state in dxm.SUSPECT_STATES), None)
    if susp is None:
        return "b2_no_suspect"
    conf = next((x for x in xs if x.pos > susp.pos and x.state in dxm.CONFIRM_STATES), None)
    if conf is None:
        return "b2_no_confirm"
    ments = admission_mentions(adm)
    for dd in adm.docs:
        if susp.pos < dd.pos <= conf.pos:
            for m in ments[dd.idx]:
                if m.tone == "R":
                    return f"b2_evidence:{m.key}"
    return "b2_other"


def _s3_reason(adm, rec):
    from .mines import _lat
    organ = rec.note.split("×")[0]
    ds_sides = {sd for o, sd, _, _ in _lat(adm.ds, {"出院诊断"}) if o == organ}
    if len(ds_sides) != 1:
        return "s3_ds_mixed_or_missing"
    a_sides = {sd for o, sd, _, _ in _lat(adm.hr, {"主诉", "现病史", "入院诊断"}) if o == organ}
    if adm.fp is not None:
        a_sides |= {sd for o, sd, _, _ in _lat(adm.fp, {"入院诊断"}) if o == organ}
    if not a_sides:
        return "s3_admission_side_missing"
    return "s3_other"


def _k4_reason(adm, rec):
    ds = adm.ds
    sec = ds.section("入院情况") if ds else None
    if sec is None:
        return "k4_section_missing"
    if not (sec[0] <= rec.start < sec[1] + 5):
        return "k4_outside_section"
    if rec.start >= rd.k4_span(ds)[1]:
        return "k4_after_cut"
    hits = list(rd.K4_RE.finditer(ds.text, rec.start, rec.end))
    if not hits:
        return "k4_trigger_unmatched"
    ca, cb = clause_span(ds.text, hits[0].start(), hits[0].end())
    if HIST_RE.search(ds.text[ca:cb]):
        return "k4_hist_clause"
    if rd.k4_skip(ds.text, hits[0]):
        return "k4_prior_clause"
    return "k4_other"


def _k5_reason(adm, rec):
    for d, s, e, x, doc_date, fate in rd.k5_scan(adm):
        if d.idx == rec.doc_idx and s < rec.end and rec.start < e:
            return f"k5_{fate}"
    return "k5_date_not_found"


def _k3_reason(adm, rec):
    m = _mention_at(adm, rec.doc_idx, rec.start, rec.end)
    if m is None:
        return "k3_no_mention"
    if m.tone != "R":
        return f"k3_tone_{m.tone}"
    if not m.sched:
        return "k3_not_scheduled_exam"
    return "k3_other"


def _k1_reason(adm, rec):
    d = adm.doc(rec.doc_idx)
    ms = list(rd.K1_RE.finditer(d.text, rec.start, rec.end))
    if not ms:
        return "k1_pattern_unmatched"
    sa, sb = rd.sentence_span(d.text, ms[0].start(), ms[0].end())
    ca, cb = clause_span(d.text, ms[0].start(), ms[0].end())
    if HIST_RE.search(d.text[sa:sb]) or "史" in d.text[sa:sb]:
        return "k1_hist_sentence"
    if rd.PLAN_CLAUSE.search(d.text[ca:cb]):
        return "k1_plan_clause"
    return "k1_other"


_REASON = {"K1": _k1_reason, "K2": _k2_reason, "K3": _k3_reason, "K4": _k4_reason, "K5": _k5_reason,
           "B2": _b2_reason, "S3": _s3_reason}


def mine_miss(kind, clean, new_adm, rec, new_result):
    """返回一个原因代码。clean、new_result 是 evaluate_admission 的结果。"""
    from .mines import MINE_READING
    rid = MINE_READING[kind]
    if rd.hit_span(new_result, rid, rec.doc_idx, rec.start, rec.end):
        return "hit"
    if clean["readings"][rid]["adm_conflict"]:
        return "clean_conflicted"
    r = new_result["readings"][rid]
    if r["adm_conflict"]:
        return "reported_elsewhere"
    if not r["adm_checkable"]:
        return "not_checkable"
    fn = _REASON.get(kind)
    return fn(new_adm, rec) if fn else "other"


def legit_alarm(kind, clean, new_adm, rec, new_result):
    """合法差异版新报了对应读数的冲突时，冲突的一端是否落在改动处；L3 另给改动处检查词的口吻。"""
    from .mines import LEGIT_READING
    rid = LEGIT_READING[kind]
    if clean["readings"][rid]["adm_conflict"] or not new_result["readings"][rid]["adm_conflict"]:
        return "no_new_alarm"
    where = "at_edit" if rd.hit_span(new_result, rid, rec.doc_idx, rec.start, rec.end) else "elsewhere"
    if kind == "L3":
        m = _mention_at(new_adm, rec.doc_idx, rec.start, rec.end)
        return f"{where}:tone_{m.tone if m else 'none'}"
    return where
