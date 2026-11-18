"""S1-A 规划诊断：用同一张 Stats、同一组种子只做规划和编译（不写文书、不调模型），量监督员几轮审查的问题还在不在。只出计数。

python -B -m medcourse.plan_diag --stats stats_v0.json --n 500 --out tables/S1A_plan_diag.csv

输出 metric,value 两列，没有诊断名、科室名、药名，可以进包。

0.1.4 加：主诊断名的来源（诊断名、主干、公开词典、章层）、各章人数、术式来源（具体写法、大类）、检查种类数和补齐数；
编译后数：回报日在第 0 天以后的检查有没有在哪一节当必写事实（results_never_must，必须为 0）、出院记录"入院情况"有没有标只写入院当时。

0.1.5 加：我们给"入院情况"的事实本身带不带 K4 触发词（ds_admission_facts_leak，必须为 0；0.1.4 模板照抄事实就命中 29/497）；
检查目标种类数和没补够的次数；病程里程序写了生命体征的篇数。

0.1.6 加：术式来源多了 key_kw、family_kw（章层、全局层按适应证关键词取）；分侧术式、诊断加了侧别的次数；
结构项（必须为 0）：分侧术式配了同器官却没写侧别的诊断、关键词来源的术式和诊断对不上、诊断名的孕周和 gest_weeks 不一致、
28 周前剖宫产、13 周后清宫、名字里还留着占位符；病程"今日重点"各类别的篇数。

0.1.8 加：分层多了 block（ICD-10 节）；结构项（必须为 0）：科室名带表头字段（dept_label_glued）、带院区字样（dept_campus）、
主诉症状不在主诉症状表里（chief_not_in_table）、麻醉和术式规则对不上（anesthesia_mismatch）、检查写异常却没有结论（exam_placeholder）、
用药退到全局层（level_meds_global）；描述：主诉第一个症状是发热的次数（第七次合成病历 32/47）、主诉症状表为空的次数、
诊断相关检查和写正常的检查各几项；统计表里真实病程按小类的篇数（stats_pb_*，只有计数）。

0.1.9 加：主诉的来源层级（level_chief_name、code3、block、treatment、symptoms、none；0.1.9.1 多 chapter_organ：章和诊断器官一致时用章层主诉表，
不算 chief_from_chapter），chief_table_empty 改指"节层以内没有主诉表"；
审查员第八轮补充的结构项（必须为 0）：诊断相关检查部位和诊断器官对不上（exam_diag_site_mismatch）、没写部位的检查和带部位的同名检查并存
（exam_bare_dup）、同类有创检查两项（exam_invasive_dup）、14 岁以下第 0 天血压超出年龄范围（child_bp_out_of_range）；
描述：鉴别诊断有没有拿到主诉（fp_diff_has_chief ÷ fp_diff）；
结构项（必须为 0）：主诉来自章层或全局层（chief_from_chapter）、常规化验或心电图算成诊断相关（exam_diag_routine）、
检查结果以"提示"开头（exam_result_tishi，会被写成"示提示"）、检查或病理结果写"与××诊断相符"（path_placeholder）、
长期用药出院前停（chronic_stop_before_discharge）、成人出院生命体征超出范围（ds_vitals_adult_off）、手术前的病程给了"已做"的事实
（before_op_done_fact）、介入和内镜手术给了切口或引流的事实（interv_incision_fact）；描述：补的诊断相关检查、一项都没有的住院、
入院血常规有极端值、血红蛋白前后差 ≥30、总胆红素 ≥100、用药大类、长期用药、一般情况的事实；按住院天数分档的病程篇数
（plan_pb_los_*）和真实的（stats_pb_los_*）；统计表里的真实比例（stats_real_*）。

0.1.12 加（v0112）：结构项（必须为 0）：非长期用药途径为空、又没有口服剂型却出院带药（home_no_oral）；化疗、放疗、复查住院的主诊断
走了疑似（tx_primary_suspected）；同组化验部分大于总量（lab_part_gt_total）。描述：主诉来自器官合并层的次数和那个 code3、词
（chief_organ_*）；仍写"不适"的器官层原因（chief_none_organ_reason_*）；没有诊断相关检查的原因（exam_diag_none_reason_*）；
首次血常规各项的分位数层、异常比例层、偏高偏低（cbc_*）；出院带药按途径（home_route_*）。chief_cov_rows 多器官合并层，
另对写"不适"的 code3 和第十一次的 I84、N13（CHIEF_WATCH）出公开肛肠、泌尿主诉词的命中数（chief_probe_*）和去侧别、去部位以后
主诉表的词数、最大计数（chief_lat_*、chief_base_*）。
"""
import argparse
import collections
import csv
import json
import os
import random
import re

from . import compiler, gate, icd, planner, stats as st_mod
from .drug_class import drug_class, indicated, subclass
from .safe import guarded, log

TABLES = ("dept", "los", "surgery", "exams", "meds", "symptoms", "durations", "comorbid")
LEVELS = ("name", "code3", "block", "chapter", "global", "none")
_DEPT_LABEL = re.compile(r"年龄|性别|姓名|床号|床位|住院号|病案号|病历号")
_DEPT_CAMPUS = re.compile(r"(?:南|北|东|西|中|新|老)院(?:区)?|本部|分院")
_FEMALE = re.compile(r"子宫|卵巢|输卵管|宫颈|阴道|外阴|妊娠|分娩|产后|产褥|胎膜|胎盘|月经|盆腔炎|剖宫产|流产")
_MALE = re.compile(r"前列腺|睾丸|附睾|阴囊|精索|阴茎|包皮|鞘膜积液")


def _level(stats, table, strata):
    t = stats.get(table) or {}
    for s in strata:
        if (t.get(s) or {}).get("items"):
            if s.startswith("n:"):
                return "name"
            if s.startswith("blk:"):
                return "block"
            return "global" if s == "*" else ("code3" if len(s) == 3 else "chapter")
    return "none"


def diagnose(stats, n, seed):
    planner.require_stats(stats)
    c = collections.Counter()
    none_codes = set()
    names, codes = set(), set()
    for i in range(n):
        jid = f"MC{seed}-{i:05d}"
        rng = random.Random(planner.seed_of(seed, jid))
        try:
            j = planner.plan_journey(stats, rng, jid, None)
        except Exception as e:  # noqa: BLE001 只记类型
            c[f"exception_{type(e).__name__}"] += 1
            continue
        if j is None:
            c["no_patient"] += 1
            continue
        specs = planner.schedule(j, stats, rng)
        c["plan_check_failed"] += bool(planner.check_plan(j, specs))
        c["planned"] += 1
        p = j["patient"]
        name, code = p["primary_dx"]["name"], p["primary_dx"]["code3"]
        sname = p["primary_dx"].get("stats_name") or name
        names.add(name)
        codes.add(code)
        c["dx_chapter_only"] += len(code) < 3
        c[f"dx_source_{p['primary_dx'].get('source')}"] += 1
        c[f"chapter_{code[:1]}"] += 1
        strata = planner.strata_of(sname, code)
        for t in TABLES:
            # 0.1.8：用药不退到全局层，按规划器实际用的层数
            c[f"level_{t}_{_level(stats, t, [x for x in strata if x != '*'] if t == 'meds' else strata)}"] += 1
        c["surgical"] += j["surgical"]
        # 手术为 0 时分得清是"这些诊断本来少做手术"还是"读不到术式"：期望手术率（千分）和本诊断有没有术式可取
        s = planner._table(stats, "surgery", strata)["items"]
        c["surgery_expected_per_mille_sum"] += round(1000 * s.get("yes", 0) / max(1, s.get("yes", 0) + s.get("no", 0)))
        avail, how = planner.proc_for(stats, strata, sname)
        c["proc_available"] += bool(avail)
        if avail:
            c[f"proc_available_{how}"] += 1
        c["dx_side_added"] += bool(p["primary_dx"].get("side_added"))
        c["placeholder_left"] += any("{" in x for x in [name] + [a["name"] for a in j["procedures"]])
        if p["pregnant"]:
            w = icd.gest_weeks(name)
            c["gest_weeks_mismatch"] += w is not None and w != p["gest_weeks"]
        for a in j["procedures"]:
            c[f"proc_source_{a.get('source')}"] += 1
            c["proc_name_concat"] += a["name"].endswith("手术治疗")
            fam = planner.proc_family(a["name"])
            c["proc_kw_mismatch"] += a.get("source") in ("key_kw", "family_kw") and not planner.family_ok(fam, sname)
            lateral = a.get("side") in ("左", "右", "双")
            c["proc_lateral"] += lateral
            c["proc_lateral_dx_sided"] += lateral and bool(planner.side_of(name))
            c["lateral_proc_dx_unsided_same_organ"] += lateral and not planner.side_of(name) and planner.same_organ(a["name"], name)
            c["cesarean_before_28w"] += fam == "剖宫产" and (p.get("gest_weeks") or 0) < 28
            c["curettage_after_13w"] += fam == "清宫" and (p.get("gest_weeks") or 0) > 13
        c["surgery_without_procedure"] += bool(j["surgical"]) != any(a["kind"] == "手术" for a in j["procedures"])
        kinds = {(e["key"], e["region"] or "") for e in j["exams"]}
        c["exam_kinds_total"] += len(kinds)
        c["exams_filled"] += sum(1 for e in j["exams"] if e.get("filled"))
        # 编译一遍（不写文书）：第 0 天以后回报的检查要在某一节当必写事实；第 0 天的由入院记录"辅助检查"程序写出
        plans = [(s["role"], pl) for s in specs for pl in compiler.compile_doc(j, s, specs, stats, rng)]
        must = {f["key"] for _, pl in plans for f in pl["facts"] if f["must"] and f.get("key")}
        c["results_never_must"] += sum(1 for e in j["exams"] if e["result_day"] > 0 and e["name"] not in must)
        c["ds_admission_only_missing"] += not any(pl.get("admission_only") for _, pl in plans if pl["section"] == "入院情况")
        c["ds_admission_facts_leak"] += any(gate.admission_leaks("。".join(f["text"] for f in pl["facts"])) for _, pl in plans if pl.get("admission_only"))
        c["exam_target_total"] += j.get("exam_target", 0)
        c["exam_fill_short"] += len(kinds) < j.get("exam_target", 0)
        course = [pl for r, pl in plans if r in compiler.COURSE]
        c["course_notes"] += len(course)
        c["course_vitals_prefix"] += sum(1 for pl in course if pl.get("program_prefix"))
        c["course_med_facts"] += sum(1 for pl in course for f in pl["facts"] if f["kind"] == "med")
        c["course_exam_fact"] += sum(1 for pl in course if any(f["kind"] == "exam" for f in pl["facts"]))
        for pl in course:
            for k in pl.get("focus") or ():
                c[f"course_focus_{k}"] += 1
        c["zero_meds"] += not j["meds"]
        # 0.1.8
        c["dept_label_glued"] += bool(_DEPT_LABEL.search(p["dept"]))
        c["dept_campus"] += bool(_DEPT_CAMPUS.search(p["dept"]))
        # 0.1.9：主诉表只到节层
        src = j["chief"].get("source")
        c[f"level_chief_{src}"] += 1
        t_cc = planner._table(stats, "chief_symptoms", planner.fine_strata(strata))["items"]
        c["chief_table_empty"] += not t_cc
        c["chief_not_in_table"] += src in ("name", "code3", "block") and any(x not in t_cc for x in j["chief"]["symptoms"])
        c["chief_from_chapter"] += src in ("chapter", "global")
        c["chief_fever_first"] += j["chief"]["symptoms"][:1] == ["发热"]
        c[f"chief_n_{len(j['chief']['symptoms'])}"] += 1
        c["anesthesia_mismatch"] += sum(1 for a in j["procedures"] if a.get("anesthesia") != planner.anesthesia_for(a["name"]))
        plain = [e for e in j["exams"] if not e["items"] and e["key"] != "病理" and "abnormal" in e]
        c["exam_placeholder"] += sum(1 for e in plain if e["abnormal"] and not e.get("finding"))
        c["exam_diagnostic"] += sum(1 for e in plain if e.get("diagnostic"))
        c["exam_plain_normal"] += sum(1 for e in plain if not e["abnormal"])
        c["meds_total"] += len(j["meds"])
        c["meds_with_dose"] += sum(1 for m in j["meds"] if m["dose"])
        c["exams_total"] += len(j["exams"])
        c["exam_keys_total"] += len({e["key"] for e in j["exams"]})
        c["has_differential"] += bool(j["differentials"])
        c["has_comorbidity"] += bool(p["comorbidities"])
        c["recheck_cbc"] += any(e.get("recheck") for e in j["exams"])
        wrong = _MALE if p["sex"] == "女" else _FEMALE
        c["comorb_wrong_sex_or_age"] += any(wrong.search(x["name"]) or not planner.dx_ok(x["name"], p["sex"], p["age"]) for x in p["comorbidities"])
        c["diff_wrong_sex_or_age"] += any(wrong.search(x) or not planner.dx_ok(x, p["sex"], p["age"]) for x in j["differentials"])
        c["dept_wrong_sex_or_age"] += not planner.dept_ok(p["dept"], p["sex"], p["age"], p["pregnant"])
        txt = compiler.past_text(p)
        denied = txt.split("否认", 1)[1] if "否认" in txt else ""
        dxs = [name] + [x["name"] for x in p["comorbidities"]]
        c["past_denies_listed_dx"] += any(t in denied and any(st_mod.past_matches(t, d) for d in dxs) for t in compiler.NEG_PAST)
        c["pregnant"] += bool(p["pregnant"])
        c["pregnant_age_out_of_range"] += bool(p["pregnant"]) and not planner.PREG_AGE[0] <= p["age"] <= planner.PREG_AGE[1]
        c["pregnant_menopause_text"] += bool(p["pregnant"]) and "绝经" in compiler.menses_text(p)
        for a in j["procedures"]:
            ds, ps = planner.side_of(name), planner.side_of(a["name"])
            c["proc_side_conflict"] += bool(ds and ps and ds != ps and "双" not in (ds, ps))
        v019(c, stats, j, specs, plans, p)
        v0111(c, stats, j, plans, p, strata, none_codes)
        v0112(c, j, p, code3_of(p))
        v020(c, j, p, code3_of(p), denied)
    chief_cov_rows(c, stats, none_codes | set(CHIEF_WATCH))
    pb = stats.get("pb_subtypes") or {}
    for b, v in sorted((pb.get("by_los") or {}).items()):
        c[f"stats_pb_los_{b}_admissions"] = v["admissions"]
        c[f"stats_pb_los_{b}_notes"] = v["notes"]
        for sub_, x in sorted(v["sub"].items()):
            c[f"stats_pb_los_{b}_{sub_}_notes"] = x["notes"]
    for k, v in sorted((stats.get("real_rates") or {}).items()):
        c[f"stats_real_{k}"] = v
    ng = stats.get("note_general") or {}
    for k, v in sorted(ng.items()):
        c[f"stats_note_general_{k}"] = v if k == "n" else round(1000 * v)
    c["stats_pb_admissions"] = pb.get("admissions", 0)
    for i, q in enumerate(pb.get("notes_q") or ()):
        c[f"stats_pb_notes_q{i}"] = q
    for sub_, v in sorted((pb.get("by_subtype") or {}).items()):
        c[f"stats_pb_sub_{sub_}_admissions"] = v["admissions"]
        c[f"stats_pb_sub_{sub_}_notes"] = v["notes"]
        c[f"stats_pb_sub_{sub_}_per_adm_p50"] = v["per_adm_q"][2]
    rows = [("planned_target", n)] + sorted(c.items())
    rows += [("distinct_primary_dx", len(names)), ("distinct_code3_or_chapter", len(codes))]
    rows += [(f"stats_{k}", v) for k, v in sorted((stats.get("surgery_meta") or {}).items())]
    rows += [(f"stats_labs_dropped_{k}", v) for k, v in sorted((stats.get("labs_dropped") or {}).items())]
    rows += [(f"stats_labs_meta_{k}", v) for k, v in sorted((stats.get("labs_meta") or {}).items())]
    rows += [(f"stats_dx_map_{k}", v) for k, v in sorted((stats.get("dx_map_meta") or {}).items())]
    nv = stats.get("note_vitals") or {}
    rows += [("stats_note_vitals_per_mille", round(1000 * nv["rate"]) if nv else "NA")]
    rows += [("stats_proc_family_strata", len(stats.get("proc_family") or {}))]
    rows += [(f"stats_blocked_{ch}_{k}", v) for ch, row in sorted((stats.get("blocked_chapters") or {}).items()) for k, v in sorted(row.items())]
    rows += [(f"stats_s2_switch_{k}", v) for k, v in sorted((stats.get("s2_switch_meta") or {}).items())]
    rows += [
        ("stats_exam_lags_keys", len(stats.get("exam_lags") or {})),
        ("stats_exam_lags_cells", sum(len(v) for v in (stats.get("exam_lags") or {}).values())),
        ("stats_symptom_course_strata", len(stats.get("symptom_course") or {})),
        ("stats_symptom_course_cells", sum(len(v.get("items") or {}) for v in (stats.get("symptom_course") or {}).values())),
    ]
    return rows


def v0111(c, stats, j, plans, p, strata, none_codes):
    """0.1.11 的读数。结构项（必须为 0）：鉴别诊断的事实带"初步诊断"（diff_fact_initial_dx）、同名检查早于手术回报（exam_before_its_proc）、
    没取组织的手术开了术后病理（path_without_tissue）、鉴别诊断和主诊断器官对不上（diff_organ_mismatch，只数查得到器官的）、
    长期药没有适应证（chronic_without_indication，主治一类除外）、糖化血红蛋白或血糖异常却没有糖尿病（dm_marker_without_dm）、
    同类同亚类两种药（same_subclass_pair）、14 岁以下用了儿童层没有的药（child_med_not_in_child_layer）、出院当天才回报阳性培养
    （culture_pos_on_discharge）、病程事实给了二便（course_bowel_fact）。描述：并进手术的检查、检查结论里的编码长名、
    写"不适"的原因和 code3（第 11 条）。"""
    los = j["los"]
    c["diff_fact_initial_dx"] += any("初步诊断" in f["text"] for _, pl in plans if pl["section"] == "鉴别诊断" for f in pl["facts"])
    path = any(e["key"] == "病理" and e["name"] == "术后病理" for e in j["exams"])
    for a in j["procedures"]:
        for e in j["exams"]:
            if planner.proc_has_exam(a["name"], e["key"]):
                c["exam_merged_into_proc"] += 1
                c["exam_before_its_proc"] += e["result_day"] <= a["do_day"] < los
        c["path_without_tissue"] += path and not planner.takes_tissue(a)
    organs = set(p["primary_dx"].get("organs") or ())
    for n in j["differentials"]:
        o = planner.dx_organs(n, "")
        c["diff_organ_mismatch"] += bool(o) and not (o & organs)
    dxt = "；".join([p["primary_dx"]["name"]] + [x["name"] for x in p["comorbidities"]] + list(p["past_history"]))
    code3 = p["primary_dx"].get("code3") or ""
    c["chronic_without_indication"] += sum(1 for m in j["meds"] if m.get("chronic") and not m.get("core") and not indicated(m["drug_id"], dxt, code3))
    c["dm_marker_without_dm"] += any(x["key"] in planner.DIABETES_MARKERS and x["flag"] != "正常" for e in j["exams"] for x in e["items"]) and "糖尿病" not in dxt
    pairs = collections.Counter((m.get("class"), subclass(m["drug_id"])) for m in j["meds"] if subclass(m["drug_id"]))
    c["same_subclass_pair"] += sum(1 for v in pairs.values() if v >= 2)
    if p["age"] < planner.CHILD_AGE:
        c["child_admissions"] += 1
        kid = next((set(planner._sub(stats, "meds", s)) for s in planner.child_layers(strata) if planner._sub(stats, "meds", s)), None)
        if kid is not None:
            c["child_med_not_in_child_layer"] += sum(1 for m in j["meds"] if m["drug_id"] not in kid)
    c["culture_pos_on_discharge"] += any(e["kind"] == "培养" and e.get("abnormal") and los >= 1 and e["result_day"] >= los for e in j["exams"])
    c["course_bowel_fact"] += sum(1 for r, pl in plans if r in compiler.COURSE for f in pl["facts"] if f["kind"] == "general" and f["text"] in compiler.GENERAL_BOWEL)
    c["finding_long_name"] += sum(1 for e in j["exams"] if re.search(r"引起的|不伴有?|伴有", e.get("finding") or ""))
    if j["chief"].get("source") == "none":
        c[f"chief_none_reason_{j['chief'].get('none_reason', '?')}"] += 1
        c[f"chief_none_code_{code3 or '?'}"] += 1
        none_codes.add(code3 or "?")


CHIEF_WATCH = ("I84", "N13")   # 第十一次写"不适"的两个 code3（36 + 13 份），不管这次还写不写都出计数


def code3_of(p):
    c3 = (p.get("primary_dx") or {}).get("code3") or ""
    return c3 if len(c3) == 3 else ""


def v0112(c, j, p, code3):
    """0.1.12 的三个结构项和描述项。"""
    from .drug_class import oral_ok
    tx = bool(p["primary_dx"].get("treatment"))
    c["home_no_oral"] += sum(1 for m in j["meds"] if m["discharge"] and not m.get("chronic") and not oral_ok(m["drug_id"], m["route"]))
    for m in j["meds"]:
        if m["discharge"]:
            c[f"home_route_{m['route'] or 'empty'}"] += 1
    name = p["primary_dx"]["name"]
    c["tx_primary_suspected"] += tx and any(d["name"] == name and d["state"] == "疑似" for d in j["dx_events"])
    for e in j["exams"]:
        by = {x["key"]: x for x in e["items"]}
        c["lab_part_gt_total"] += sum(1 for a, b in planner.PART_OF.items() if a in by and b in by and by[a]["unit"] == by[b]["unit"]
                                      and float(by[a]["value"]) > float(by[b]["value"]))
    ch = j["chief"]
    if ch.get("source") == "organ":
        c[f"chief_organ_{ch.get('organ')}"] += 1
        c[f"chief_organ_code_{code3 or '?'}"] += 1
        for t in ch["symptoms"][:1]:
            c[f"chief_organ_term_{code3 or '?'}_{t}"] += 1
    if ch.get("source") == "none":
        c[f"chief_none_organ_reason_{ch.get('organ_reason', '?')}"] += 1
    ed = j.get("exam_diag") or {}
    if ed.get("none_reason"):
        c[f"exam_diag_none_reason_{ed['none_reason']}"] += 1
    for key, v in sorted((ed.get("cbc") or {}).items()):
        c[f"cbc_{key}_q_{v['q']}"] += 1
        c[f"cbc_{key}_rate_{v['rate']}"] += 1
        c[f"cbc_{key}_flag_{v['flag']}"] += 1


_GRADE_TAIL = re.compile(r"(?:[（(][^）)]{0,12}[）)])?(?:NYHA\s*)?(?:心功能)?[IⅠⅡⅢⅣV1-5一二三四五]+(?:级|期)|(?:极高|很高|高|中|低)危(?:组)?")


def _grade_stem(dx_name):
    s = re.sub(r"[\s，,、；;（）()]", "", dx_name or "")
    if not _GRADE_TAIL.search(s):
        return None
    base = _GRADE_TAIL.sub("", s)
    return base if len(base) >= 2 else ("心功能分级" if "心功能" in s else None)


def v020(c, j, p, code3, denied):
    """0.2.0 的 2 个新增结构项（child_chronic_denial、func_exam_has_dx）和第 5、6、7、10、12、13 条诊断计数。"""
    # 结构项（必须为 0）：14 岁以下写成人慢性病否认清单；功能检查异常结果带"符合××"
    c["child_chronic_denial"] += p["age"] < planner.CHILD_AGE and any(t in denied for t in compiler.NEG_PAST)
    c["func_exam_has_dx"] += sum(1 for e in j["exams"] if e["kind"] == "功能" and "符合" in (e.get("finding") or ""))
    # 第 5 条：首次血常规单项极值计数，及血红蛋白/血小板极值来自哪层
    cbc = [e for e in j["exams"] if e["key"] == "血常规"]
    ed = j.get("exam_diag") or {}
    cbc_src = ed.get("cbc") or {}
    if cbc:
        base = {x["key"]: float(x["value"]) for x in cbc[0]["items"]}
        if "WBC" in base:
            c["plan_wbc_admissions"] += 1
            c["plan_wbc_gt12"] += base["WBC"] > 12
        if "NEUT" in base:
            c["plan_neut_admissions"] += 1
            c["plan_neut_gt85"] += base["NEUT"] > 85
        if "HGB" in base:
            c["plan_hgb_admissions"] += 1
            if base["HGB"] < 90:
                c["plan_hgb_lt90"] += 1
                s = cbc_src.get("HGB") or {}
                c[f"cbc_HGB_lt90_q_{s.get('q', '?')}"] += 1
                c[f"cbc_HGB_lt90_rate_{s.get('rate', '?')}"] += 1
        if "PLT" in base:
            c["plan_plt_admissions"] += 1
            if base["PLT"] < 100:
                c["plan_plt_lt100"] += 1
                s = cbc_src.get("PLT") or {}
                c[f"cbc_PLT_lt100_q_{s.get('q', '?')}"] += 1
                c[f"cbc_PLT_lt100_rate_{s.get('rate', '?')}"] += 1
    # 第 6 条：exam_diag 的 no_organ（按章）和 no_candidate（按细因）
    if ed.get("none_reason") == "no_organ":
        ch_ = ((p.get("primary_dx") or {}).get("code3") or "")[:1] or "?"
        c[f"exam_diag_no_organ_ch_{ch_}"] += 1
    if ed.get("no_cand_detail"):
        c[f"exam_diag_no_cand_{ed['no_cand_detail']}"] += 1
    # 第 7 条：抗菌药第 1 天停、同一分级诊断重复出现
    abx1 = [m for m in j["meds"] if j["los"] >= 3 and m.get("class") == "抗菌药" and not m.get("chronic") and m["stop_day"] <= 1]
    if abx1:
        c["abx_stop_day1"] += 1
        c["abx_stop_day1_surg" if j["surgical"] else "abx_stop_day1_nonsurg"] += 1
    stems = [s for s in (_grade_stem(x) for x in planner.final_diagnoses(j)) if s]
    c["dup_graded_dx"] += len(stems) != len(set(stems))
    # 第 10 条：出院新增诊断比例的规划拆分（合并症总数、仅出院列出的合并症、其中被同步进既往史而在入院就已提及的）
    comorb = p.get("comorbidities") or []
    ds_only = [x for x in comorb if not x.get("at_admission")]
    in_past = [x for x in ds_only if any(st_mod.past_matches(t, x["name"]) for t in st_mod.PAST_TERMS if t in (p.get("past_history") or ()))]
    c["comorb_total"] += len(comorb)
    c["comorb_ds_only"] += len(ds_only)
    c["comorb_ds_only_in_past"] += len(in_past)
    c["plan_ds_only_adm"] += bool(ds_only)
    c["plan_ds_dx_items"] += 1 + len(comorb)
    c["plan_ds_new_share_permille_sum"] += round(1000 * len(ds_only) / (1 + len(comorb)))
    # 第 12 条：核心药来自哪一层、哪个候选池、长期核心药是否缺适应证
    dxt = "；".join([p["primary_dx"]["name"]] + [x["name"] for x in comorb] + list(p.get("past_history") or []))
    for m in j["meds"]:
        if m.get("core"):
            c[f"med_core_cls_layer_{m.get('cls_layer') or '?'}"] += 1
            c[f"med_core_reason_{m.get('core_reason') or '?'}"] += 1
            c[f"med_core_pool_{m.get('drug_pool') or '?'}"] += 1
            c["med_core_chronic_unindicated"] += bool(m.get("chronic") and not indicated(m["drug_id"], dxt, code3))
        else:
            c[f"med_noncore_pool_{m.get('drug_pool') or '?'}"] += 1
    # 第 13 条：未配麻醉的非手术操作次数
    c["proc_anesthesia_empty"] += sum(1 for a in j["procedures"] if not a.get("anesthesia"))


def chief_cov_rows(c, stats, none_codes):
    """第 11 条：写"不适"的那些 code3，真实统计表里各层的主诉覆盖（住院数、有入院记录、有主诉节、匹配到主诉词；压掉 <k 的格子）
    和主诉表的词数。只有计数。"""
    cov, cc = stats.get("chief_cov") or {}, stats.get("chief_symptoms") or {}
    probe, lat, base = (stats.get(x) or {} for x in ("chief_probe", "chief_lat", "chief_base"))
    for code in sorted(none_codes):
        layers = [code] if len(code) != 3 else [code, planner.icd.block_of(code), code[:1]]
        # 0.1.12：器官合并层（stats 0.11）
        layers += [f"org:{o}" for o in sorted(planner.dx_organs("", code))] if len(code) == 3 else []
        for layer in [x for x in layers if x]:
            row = cov.get(layer) or {}
            c[f"chief_cov_{code}_{layer}_n"] = row.get("n", 0)
            for f in ("hr", "cc", "term"):
                c[f"chief_cov_{code}_{layer}_{f}"] = (row.get("items") or {}).get(f, 0)
            c[f"chief_cov_{code}_{layer}_table_terms"] = len((cc.get(layer) or {}).get("items") or {})
            # 0.1.12：公开主诉词的命中数（压掉 <k 的格子，0 表示不到 k）；去侧别、去部位以后能过 k 的词数和最大计数
            for w, n in sorted(((probe.get(layer) or {}).get("items") or {}).items()):
                c[f"chief_probe_{code}_{layer}_{w}"] = n
            for nm, t in (("lat", lat), ("base", base)):
                items = (t.get(layer) or {}).get("items") or {}
                c[f"chief_{nm}_{code}_{layer}_table_terms"] = len(items)
                c[f"chief_{nm}_{code}_{layer}_max"] = max(items.values(), default=0)


def v019(c, stats, j, specs, plans, p):
    """0.1.9 的结构项和描述项。plans：[(role, plan)]，和 specs 一一对应展开。"""
    los = j["los"]
    plain = [e for e in j["exams"] if not e["items"]]
    c["exam_diag_routine"] += sum(1 for e in plain if e.get("diagnostic") and not planner.diag_eligible(e["key"]))
    # 审查员第八轮补充（结构项，必须为 0）：诊断相关检查部位和诊断器官对不上；没写部位的 CT、超声和带部位的同名检查并存；
    # 同类有创检查两项；14 岁以下第 0 天血压超出年龄范围
    organs = set(p["primary_dx"].get("organs") or ())
    c["exam_diag_site_mismatch"] += sum(1 for e in plain if e.get("diagnostic") and not planner.site_ok(e["key"], e["region"], organs))
    regional = {e["key"] for e in j["exams"] if e["region"]}
    c["exam_bare_dup"] += sum(1 for e in j["exams"] if e["key"] in planner.REGIONAL_KEYS and not e["region"] and e["key"] in regional)
    c["exam_bare_left"] += sum(1 for e in j["exams"] if e["key"] in planner.REGIONAL_KEYS and not e["region"])
    grp = collections.Counter(("组织" if e["kind"] in ("取材", "病理") else e["kind"]) for e in j["exams"]
                              if e["kind"] in planner.INVASIVE and e["name"] != "术后病理")
    c["exam_invasive_dup"] += sum(1 for n in grp.values() if n > 1)
    c["exam_invasive"] += sum(grp.values())
    cr = planner.child_range(p["age"])
    if cr:
        d0 = j["daily"][0]
        c["child_admissions"] += 1
        c["child_bp_out_of_range"] += not (cr["SBP"][0] - 8 <= d0["SBP"] <= cr["SBP"][1] + 15)
    if j["differentials"]:
        c["fp_diff_has_chief"] += any(f["kind"] == "patient" for _, pl in plans if pl["section"] == "鉴别诊断" for f in pl["facts"])
        c["fp_diff"] += 1
    c["exam_result_tishi"] += sum(1 for e in j["exams"] if planner.exam_result_text(e).startswith("提示"))
    c["path_placeholder"] += sum(1 for e in j["exams"] if "相符" in planner.exam_result_text(e))
    c["exam_diag_guaranteed"] += sum(1 for e in j["exams"] if e.get("guaranteed"))
    c["exam_diag_none"] += not any(e.get("diagnostic") for e in j["exams"])
    c["exam_culture"] += sum(1 for e in j["exams"] if e["kind"] == "培养")
    c["exam_culture_positive"] += sum(1 for e in j["exams"] if e["kind"] == "培养" and e.get("diagnostic"))
    c["chief_treatment"] += bool(j["chief"].get("kind"))
    # 化验：入院血常规极端、血红蛋白前后差、总胆红素（和 stats_real_* 同口径）
    cbc = [e for e in j["exams"] if e["key"] == "血常规"]
    if cbc:
        base = {x["key"]: float(x["value"]) for x in cbc[0]["items"]}
        c["plan_cbc_admissions"] += 1
        c["plan_cbc_any_extreme"] += st_mod.cbc_extreme(base)
    hgb = [float(x["value"]) for e in j["exams"] for x in e["items"] if x["key"] == "HGB"]
    if len(hgb) >= 2:
        c["plan_hgb_2plus"] += 1
        c["plan_hgb_swing_ge30"] += max(hgb) - min(hgb) >= 30
    tbil = [float(x["value"]) for e in j["exams"] for x in e["items"] if x["key"] == "TBIL"]
    if tbil:
        c["plan_tbil_admissions"] += 1
        c["plan_tbil_ge100"] += max(tbil) >= 100
    labs = [x for e in j["exams"] for x in e["items"]]
    c["lab_items"] += len(labs)
    c["lab_items_abnormal"] += sum(1 for x in labs if x["flag"] != "正常")
    # 用药
    classes = {m.get("class") or drug_class(m["drug_id"]) for m in j["meds"]}
    c["meds_classes_total"] += len(classes)
    c["meds_core"] += any(m.get("core") for m in j["meds"])
    c["meds_chronic"] += sum(1 for m in j["meds"] if m.get("chronic"))
    c["chronic_stop_before_discharge"] += sum(1 for m in j["meds"] if m.get("chronic") and m["stop_day"] < los)
    c["meds_antineoplastic"] += "抗肿瘤药" in classes
    # 出院生命体征（成人）
    last = j["daily"][-1]
    if p["age"] >= 18 and los >= 1:
        c["ds_vitals_adult"] += 1
        c["ds_vitals_adult_off"] += st_mod.ds_vitals_off(last["SBP"], last["P"], last["R"])
    # 手术时间线、术后观察
    post = next((s["k"] for s in specs if s["role"] == "POST"), None)
    by_k = {}
    for (role, pl) in plans:
        by_k.setdefault(pl["k"], []).append(pl)
    for a in j["procedures"]:
        for s in specs:
            if s["role"] in compiler.COURSE and s["day"] == a["do_day"] and post is not None and s["k"] < post:
                c["before_op_notes"] += 1
                c["before_op_done_fact"] += any(f["kind"] == "procedure" for pl in by_k.get(s["k"], []) for f in pl["facts"])
        if compiler.no_incision(a):
            c["proc_no_incision"] += 1
            c["interv_incision_fact"] += any(re.search(r"切口|引流", f["text"]) for _, pl in plans for f in pl["facts"])
    # 一般情况的事实
    course = [pl for r, pl in plans if r in compiler.COURSE]
    c["course_general_facts"] += sum(1 for pl in course for f in pl["facts"] if f["kind"] == "general")
    # 按住院天数分档的病程篇数（和 stats_pb_los_* 比）
    b = st_mod.los_bucket(los)
    c[f"plan_pb_los_{b}_admissions"] += 1
    sup_cnt = 0
    for s in specs:
        if s["role"] in ("SUP", "DAILY", "PRE", "POST"):
            c[f"plan_pb_los_{b}_{s['role']}_notes"] += 1
        if s["role"] == "SUP":
            sup_cnt += 1
    c["plan_sup_multi_adm"] += sup_cnt >= 2
    c["plan_chief_resolved_by_ds"] += bool(
        los >= 2 and j["chief"]["symptoms"] and j["chief"]["symptoms"][0] not in j["daily"][-1]["symptoms"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.umask(0o077)
    with open(a.stats, encoding="utf-8") as f:
        stats = json.load(f)
    rows = diagnose(stats, a.n, a.seed)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    log("plan_diag_done", rows=len(rows))


if __name__ == "__main__":
    guarded(main)
