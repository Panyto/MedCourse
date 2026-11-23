"""S1、S2 共用的数据契约 mc-contract-0.1。都是能进 JSONL 的普通 dict，这里定字段和校验，改字段要升版。

PatientState  病人属性和抽样来源
Journey       逐日事件：检查、用药、手术操作、诊断变化、每日体征和症状
DocSpec       一篇文书：类型、第几天、时刻、书写角色、风格码
SectionPlan   这一节要写的事实（当天可见的）、拷贝预算、目标字数
AdmissionJSONL 和 run 9 导出同格式，六关规则包和 bench_core 不用改就能读
Segment、EventRecord  L1 抽取的切片和抽取结果，S2 产出，S3 配成"计划 → 真实段落"

所有 day 都是相对入院的整数，0 是入院当天。校验函数返回错误代码列表，不带正文，可以进日志。

0.2（S2 抽取 v0.2，金标 S2G v1 之后）：事件类型多两个出口，体征（sign）、七类以外（other）；状态 rule_out 拆开，
已排除（excluded）单列，待排归疑似（uncertain）；多一个"不是这个病人"（not_patient）。下游建统计表、金标抽题只用 CORE_TYPES
里状态不是 not_patient 的事件。v1 的事件（rule_out）由 extract.normalize 换成新状态。
"""
import re

CONTRACT_VERSION = "mc-contract-0.2.1"
LEGACY_CONTRACTS = ("mc-contract-0.1", "mc-contract-0.2", None)

SEXES = ("男", "女")
EXAM_KINDS = ("检验", "影像", "功能", "内镜", "造影", "穿刺", "培养", "病理", "取材", "其他")
PROC_KINDS = ("手术", "操作")
DX_STATES = ("疑似", "确诊", "排除")
DOC_TYPES = ("hospital_record", "progress_note", "discharge_summary")
ROLES = ("HR", "FP", "SUP", "PRE", "POST", "DAILY", "DS")
COPY_OPS = ("exact", "numeric_changed", "new")
CORE_TYPES = ("diagnosis", "exam", "lab", "medication", "procedure", "vital", "symptom")
EVENT_TYPES = CORE_TYPES + ("sign", "other")
ASSERTIONS = ("done", "planned", "history", "excluded", "negated", "uncertain", "not_patient")
LEGACY_ASSERTIONS = ("rule_out",)     # mc-contract-0.1 的事件里会有，校验放行，normalize 换掉


def is_core(e):
    """下游（统计表、金标抽题）要的事件：七类核心类型，说的是这个病人。"""
    return e.get("event_type") in CORE_TYPES and e.get("assertion") != "not_patient"


def _day(x, los):
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= los


_HM_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def _hm_ok(s):
    return s is None or (isinstance(s, str) and bool(_HM_RE.match(s)))


def validate_patient(p):
    errs = []
    if p.get("sex") not in SEXES:
        errs.append("patient_sex")
    if not isinstance(p.get("age"), (int, float)) or not 0 <= p["age"] <= 120:
        errs.append("patient_age")
    dx = p.get("primary_dx") or {}
    if not dx.get("name"):
        errs.append("patient_dx")
    if not p.get("dept"):
        errs.append("patient_dept")
    if p.get("pregnant") and p.get("sex") != "女":
        errs.append("patient_pregnant_sex")
    return errs


def validate_journey(j):
    """结构校验：天数在范围内、先后对、手术前后对、id 不重。用药规则另由 planner.med_allowed 过。"""
    errs = validate_patient(j.get("patient") or {})
    los = j.get("los")
    if not isinstance(los, int) or los < 0:
        return errs + ["los"]
    ids = set()

    def fresh(i, code):
        if i in ids or not i:
            errs.append(code)
        ids.add(i)

    ev_day_by_id = {}
    for e in j.get("exams", []):
        fresh(e.get("id"), "exam_id")
        if not (_day(e.get("order_day"), los) and _day(e.get("result_day"), los) and e["order_day"] <= e["result_day"]):
            errs.append("exam_days")
        if not (_hm_ok(e.get("order_hm")) and _hm_ok(e.get("result_hm"))):
            errs.append("exam_hm")
        elif e.get("order_hm") and e.get("result_hm") and e["order_day"] == e["result_day"] and e["order_hm"] > e["result_hm"]:
            errs.append("exam_hm_order")
        if e.get("kind") not in EXAM_KINDS:
            errs.append("exam_kind")
        if e.get("id") and _day(e.get("result_day"), los):
            ev_day_by_id[e["id"]] = (e["result_day"], e.get("result_hm") or "00:00")
    for m in j.get("meds", []):
        fresh(m.get("id"), "med_id")
        if not (_day(m.get("start_day"), los) and _day(m.get("stop_day"), los) and m["start_day"] <= m["stop_day"]):
            errs.append("med_days")
        if not (_hm_ok(m.get("start_hm")) and _hm_ok(m.get("stop_hm"))):
            errs.append("med_hm")
        elif m.get("start_hm") and m.get("stop_hm") and m["start_day"] == m["stop_day"] and m["start_hm"] > m["stop_hm"]:
            errs.append("med_hm_order")
    surg = 0
    for a in j.get("procedures", []):
        fresh(a.get("id"), "proc_id")
        if not (_day(a.get("decide_day"), los) and _day(a.get("do_day"), los) and a["decide_day"] <= a["do_day"]):
            errs.append("proc_days")
        if not (_hm_ok(a.get("decide_hm")) and _hm_ok(a.get("do_hm"))):
            errs.append("proc_hm")
        if a.get("kind") not in PROC_KINDS:
            errs.append("proc_kind")
        surg += a.get("kind") == "手术"
        if a.get("id") and _day(a.get("do_day"), los):
            ev_day_by_id[a["id"]] = (a["do_day"], a.get("do_hm") or "00:00")
    if bool(surg) != bool(j.get("surgical")):
        errs.append("surgical_flag")
    if surg > 1:
        errs.append("surgery_count")
    require_ev = j.get("contract") not in LEGACY_CONTRACTS
    last = {}
    for d in sorted(j.get("dx_events", []), key=lambda x: (x.get("day", -1), x.get("hm") or "00:00")):
        if not _day(d.get("day"), los) or d.get("state") not in DX_STATES:
            errs.append("dx_event")
        if not _hm_ok(d.get("hm")):
            errs.append("dx_hm")
        if d.get("day", 0) >= 1 and d.get("state") in ("确诊", "排除") and (require_ev or "evidence_ids" in d):
            eids = d.get("evidence_ids") or []
            if "evidence_ids" not in d or not eids:
                errs.append("dx_missing_evidence")
            else:
                dts = (d.get("day"), d.get("hm") or "23:59")
                if any((eid not in ev_day_by_id) or (ev_day_by_id[eid] > dts) for eid in eids):
                    errs.append("dx_future_evidence")
        last[d.get("name")] = d.get("state")
    days = [d.get("day") for d in j.get("daily", [])]
    if days != list(range(los + 1)):
        errs.append("daily_days")
    return errs


def validate_docspecs(specs, los):
    errs = []
    if not specs:
        return ["no_docs"]
    if specs[0].get("role") != "HR" or specs[-1].get("role") != "DS":
        errs.append("doc_order")
    for s in specs:
        if s.get("doc_type") not in DOC_TYPES or s.get("role") not in ROLES:
            errs.append("doc_type")
        if not _day(s.get("day"), los):
            errs.append("doc_day")
    if [s.get("k") for s in specs] != list(range(len(specs))):
        errs.append("doc_k")
    if [s.get("day") for s in specs] != sorted(s.get("day") for s in specs):
        errs.append("doc_day_order")
    return errs


def validate_section_plan(p):
    errs = []
    if not p.get("section"):
        errs.append("plan_section")
    for f in p.get("facts", []):
        if not f.get("text"):
            errs.append("plan_fact_text")
    if p.get("copy", {}).get("op") not in (None,) + COPY_OPS:
        errs.append("plan_copy_op")
    return errs


def admission_line(journey, docs):
    """定稿的 AdmissionJSONL 一行。docs 里已有 doc_type、day_offset、text、doc_type_cn、k、role。"""
    return {"admission_id": journey["journey_id"], "docs": docs,
            "meta": {"admit_date": journey["admit_date"], "contract": CONTRACT_VERSION}}


def validate_event(e):
    errs = []
    if e.get("event_type") not in EVENT_TYPES:
        errs.append("event_type")
    if not e.get("name"):
        errs.append("event_name")
    if e.get("assertion") not in (None,) + ASSERTIONS + LEGACY_ASSERTIONS:
        errs.append("event_assertion")
    if not isinstance(e.get("ev_start"), int) or not isinstance(e.get("ev_end"), int) or e["ev_start"] >= e["ev_end"]:
        errs.append("event_span")
    return errs
