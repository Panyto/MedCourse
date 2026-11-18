"""产出诊断：读一次流程跑出来的目录（journeys.jsonl、admissions.jsonl），拆开看写出来的和规划差在哪。只出计数，不打印正文。

python -B -m medcourse.out_diag --run 目录 --out tables/xxx_out_diag.csv
python -B -m medcourse.out_diag --real 真实fold0.jsonl --out tables/real_out_diag.csv      # 真实病程的同口径基线，只出计数

1. 检查：规划了几种、写出来被读成结果口吻的几种（同评测台 B_exam 的口径），按检查名分开数。检查名是规则包的公开词表。
2. K4：出院记录"入院情况"命中的第一个触发词属于哪一类（固定类名）。
3. 相邻 tf-idf 拆项：用这批病程自己拟合 IDF（只看三项之间的相对大小，不等于评测台的读数），算相邻余弦：
   全文、去掉抬头、再去掉程序写的生命体征。另报抬头和生命体征占病程字数的比例。
4. 药名：每次住院规划了几种药、写出来几种、一共出现几次。
5. 日常病程、上级查房去掉抬头后的字数分位数。
6. 套话（0.1.6）：病程正文里常见套话（固定类名，病程写法的公开常识）各在几篇病程里出现、一共几次；
   跨住院重复句：去掉数字后同一句话出现在两成以上住院里的，占病程句子的千分比；和同一住院上一篇病程一字不差的句子千分比（含 apply_copy 拷进来的）。
7. 评测台口径（0.1.7，pb_ 开头）：首程之后、出院记录之前的全部病程（同 progress_between，含术前小结、术后首程），数同样的套话、
   重复句、相邻 tf-idf（这批自己拟合 IDF）和正文字数。--real 在真实 fold 0 上算同一组 pb_ 读数，两边直接比；真实模式只读文书，只出计数。
8. 读正文才看得出的硬伤（0.1.8，qa_ 开头，每次住院有没有，只出计数）：第七次合成病历审阅和监督员第七轮找到的
   复读、半句截断、没头的"示……"、体格检查写"未测"、检查占位句、科室名带表头字段、主诉以发热开头、非手术住院写术后、
   既往史和首程或现病史打架、日常病程里出现上级医师的话。
   0.1.9 加 7 项（第八次合成病历审阅）：病理和检查的占位句（新口径，"与主要／术前诊断相符"都算；旧的 placeholder 口径照留）、
   体格检查写病史、提示句写进现病史、常规化验或心电图"提示"主要诊断、手术前的病程写成已做、长期用药写"疗程已足"、
   血红蛋白前后差 ≥30。另加两项描述：病程里的麻醉方式和计划不一致、介入和内镜手术写了切口或引流。
   审查员第八轮补充的三项（描述）：体格检查写了检查结果、同一篇病程"继续"和"加用／停用"同一种药、影像内镜在和诊断器官对不上的部位写"提示／符合"。
"""
import argparse
import collections
import csv
import json
import os
import re

from .rules import setup
from .safe import guarded, log

_VITALS = re.compile(r"体温[\d.]+℃，脉搏\d+次/分，呼吸\d+次/分，血压\d+/\d+mmHg。")
K4_CLASS = (("入院后", "入院后"), ("住院期间", "住院期间"), ("经", "经治疗"), ("治疗后", "治疗后"), ("术后", "术后"), ("术中", "术中"),
            ("病理", "病理"), ("复查", "复查"), ("出院时", "出院时"), ("好转出院", "好转出院"), ("行", "行术"))


# 套话：类名 → 正则（在去掉空白和标点的正文上匹配）
BOILER = (("精神睡眠饮食", r"精神.{0,2}(?:睡眠|食欲|饮食|纳眠).{0,6}(?:可|尚可|好|正常|一般)"), ("二便", r"(?:二便|大小便).{0,2}(?:正常|如常|无异常|可|通畅)"),
          ("病情平稳", r"病情(?:平稳|稳定)"), ("继续目前治疗", r"继续(?:目前|当前|现有|原)(?:治疗|方案)"), ("密切观察", r"密切(?:观察|监测)"),
          ("无特殊不适", r"无(?:特殊|明显)不适"), ("生命体征平稳", r"生命体征(?:平稳|稳定)"), ("一般情况可", r"一般情况(?:可|尚可|良好)"),
          ("神志清", r"神志清"), ("查体同前", r"查体.{0,2}(?:同前|大致同前)"), ("遵上级医师", r"(?:遵|按)上级医师"), ("对症支持", r"对症支持"))
_BOILER_RE = [(c, re.compile(rx)) for c, rx in BOILER]
_STRIP = re.compile(r"[\s，,。；;：:、！？!?（）()]")
_SENT_SPLIT = re.compile(r"[。；;！？!?\n]")
COMMON_SHARE = 0.2


_QA_SEC = re.compile(r"^([\u4e00-\u9fff（）()]{2,12})：(.+)$")
_QA_SKIP = ("姓名", "入院时间", "手术时间", "出院时间")
_QA_SPLIT = re.compile(r"[，,。；;！？!?、]")
_QA_END = re.compile(r"[。！？!?]$|[。！？!?][）)”】]$")
_QA_HEADLESS = re.compile(r"(?:^|[。；])(?:示|提示)(?:未见|可见)|。结果显示")
_QA_PE_UNTESTED = re.compile(r"(?:体温|血压|脉搏|心率|呼吸)(?:频率)?未测")
_QA_POSTOP = re.compile(r"术后(?:恢复|第|切口|伤口|复查|康复|处理|观察|注意|病理|顺利)|手术(?:顺利|过程)|术区|切口(?:愈合|敷料|干燥)|伤口(?:愈合|敷料|干燥)")
_QA_PAST_HTN = re.compile(r"高血压")
_QA_DENY = re.compile(r"否认高血压|既往无特殊|既往体健")
_QA_SENIOR = re.compile(r"(?:副?主任|主治|上级)医师")
# 0.1.9
_QA_PATH_PH = re.compile(r"与(?:主要|术前)?诊断相符|结果与主要诊断")
_QA_PE_HISTORY = re.compile(r"主诉|自述|自诉|诉有|病史|否认|既往")
_QA_HINT_LEAK = re.compile(r"另节|另一节|另有一节|另述|详见另|另行记录")
_QA_ROUTINE = ("血常规|尿常规|粪常规|便常规|肝功能|肾功能|电解质|血糖|血脂|凝血|D-?二聚体|利钠肽|心肌肌钙蛋白|肌钙蛋白|心肌酶|C反应蛋白|降钙素原|"
               "糖化血红蛋白|淀粉酶|血沉|肿瘤标志物|甲状腺功能|动态心电图|心电图")
_QA_BEFORE_OP = re.compile(r"手术(?:过程)?顺利|术中顺利|术后第|术后(?:恢复|切口|患者)|今日(?:在.{0,8}下)?行|已在.{0,8}下行")
_QA_HB = re.compile(r"血红蛋白(?:为|下降至|降至|升至)?(\d+)\s*g/L")
_QA_ANES = re.compile(r"(全麻|全身麻醉|局麻|局部麻醉|腰硬联合麻醉|硬膜外麻醉|静脉麻醉|腰麻)下")
_QA_ANES_NORM = {"全麻": "全身麻醉", "局麻": "局部麻醉"}
_QA_INCISION = re.compile(r"切口|引流")
_QA_PE_RESULT = re.compile(r"心电图|彩超|B超|超声|CT|磁共振|MRI|X线|胸片|平片|血常规|化验|检验|造影")
_QA_EXAM_DX = re.compile(r"((?:头颅|头部|颅脑|颈部|胸部|双肺|肺部|心脏|冠脉|腹部|上腹部|全腹|肝胆|盆腔|子宫附件|泌尿系|双肾|肾脏|膀胱|前列腺|腰椎|下肢|乳腺|血管)?"
                         r"(?:CT|磁共振|超声|X线|心脏超声|胃镜|肠镜|支气管镜))(?:示|检查示)?[：:]?(?:提示|符合)")
QA_FLAGS = ("repeat", "truncated", "headless", "pe_untested", "placeholder", "dept_glued", "chief_fever_first", "nonsurg_postop", "past_conflict", "senior_in_daily",
            "path_placeholder", "pe_history", "hint_leak", "routine_says_dx", "before_op_done", "chronic_course_done", "lab_swing", "anes_conflict", "interv_incision",
            "pe_result", "med_conflict", "exam_site_mismatch", "meta_text", "cross_repeat")
# 0.1.12（第十一次合成病历）：元话语（13、16 号"本次查房记录完整，内容真实，符合病历书写规范"）；同一句（≥20 字）在 ≥3 篇病程里出现
# （08 号一句甲状腺功能连读 6 篇）。先计数，不进闸门
_QA_META = re.compile(r"(?:记录|病程)(?:完整|真实)|无遗漏|内容真实|符合(?:病历)?书写规范")
CROSS_MIN_CHARS, CROSS_MIN_NOTES = 20, 3


def _qa_repeat(body):
    cnt = collections.Counter(x.strip() for x in _QA_SPLIT.split(body) if len(x.strip()) >= 4)
    return any((len(x) >= 8 and n >= 3) or n >= 5 for x, n in cnt.items())


def qa_flags(j, docs):
    """一次住院命中哪些 qa_ 硬伤（集合）。只看合成文书，不出正文。"""
    out = set()
    by = {}
    for d in docs:
        by.setdefault(d.get("doc_type_cn") or d.get("doc_type"), []).append(d["text"])
        for line in d["text"].split("\n"):
            m = _QA_SEC.match(line)
            name, body = (m.group(1), m.group(2)) if m else ("", line if len(line) > 60 else "")
            if not body or name in _QA_SKIP:
                continue
            if len(body) >= 20 and not _QA_END.search(body.strip()):
                out.add("truncated")
            if _qa_repeat(body):
                out.add("repeat")
            if _QA_HEADLESS.search(body):
                out.add("headless")
            if name == "体格检查" and _QA_PE_UNTESTED.search(body):
                out.add("pe_untested")
        if "与主要诊断相符的异常改变" in d["text"]:
            out.add("placeholder")
        if re.search(r"科室：\S*(?:年龄|性别)", d["text"]):
            out.add("dept_glued")
        if (d.get("doc_type_cn") == "日常病程记录") and _QA_SENIOR.search(d["text"]):
            out.add("senior_in_daily")
    hr = "\n".join(by.get("入院记录", []))
    if re.search(r"^主诉：发热", hr, re.M):
        out.add("chief_fever_first")
    if not j.get("surgical") and _QA_POSTOP.search("\n".join(by.get("出院记录", []))):
        out.add("nonsurg_postop")
    past = re.search(r"既往史：(.*)", hr)
    if past and _QA_PAST_HTN.search(past.group(1)) and "否认高血压" not in past.group(1):
        hpi = re.search(r"现病史：(.*)", hr)
        fp = "\n".join(by.get("首次病程记录", []))
        if (hpi and _QA_DENY.search(hpi.group(1))) or _QA_DENY.search(fp):
            out.add("past_conflict")
    out |= qa_flags_v019(j, docs, hr)
    out |= qa_flags_v0112(docs)
    return out


def qa_flags_v0112(docs):
    out = set()
    if any(_QA_META.search(d["text"]) for d in docs):
        out.add("meta_text")
    cnt = collections.Counter()
    for d in docs:
        if d.get("role") in ("SUP", "DAILY"):
            cnt.update({x.strip() for x in re.split(r"[。！？!?\n]", _body(d["text"])) if len(x.strip()) >= CROSS_MIN_CHARS})
    if any(n >= CROSS_MIN_NOTES for n in cnt.values()):
        out.add("cross_repeat")
    return out


def qa_flags_v019(j, docs, hr):
    """0.1.9 的 7 项和 2 项描述。"""
    from .compiler import no_incision
    from .drug_class import is_chronic
    out = set()
    whole = "\n".join(d["text"] for d in docs)
    if _QA_PATH_PH.search(whole):
        out.add("path_placeholder")
    pe = re.search(r"^体格检查：(.*)$", hr, re.M)
    if pe and _QA_PE_HISTORY.search(pe.group(1)):
        out.add("pe_history")
    hpi = re.search(r"^现病史：(.*)$", hr, re.M)
    if hpi and _QA_HINT_LEAK.search(hpi.group(1)):
        out.add("hint_leak")
    dx = re.sub(r"[？?]", "", ((j.get("patient") or {}).get("primary_dx") or {}).get("name") or "")
    if dx and re.search(r"(?:" + _QA_ROUTINE + r")[^。；，,]{0,10}提示" + re.escape(dx), whole):
        out.add("routine_says_dx")
    for m in j.get("meds") or ():
        chronic = m.get("chronic") if "chronic" in m else is_chronic(m.get("drug_id"))
        if chronic and re.search(re.escape(m["name"]) + r"[^。]{0,12}疗程已足", whole):
            out.add("chronic_course_done")
    hbs = [int(x) for x in _QA_HB.findall(whole)]
    if len(hbs) >= 2 and max(hbs) - min(hbs) >= 30:
        out.add("lab_swing")
    procs = j.get("procedures") or ()
    if j.get("surgical") and procs:
        a = procs[0]
        post = next((i for i, d in enumerate(docs) if d.get("role") == "POST"), None)
        for i, d in enumerate(docs):
            if post is not None and i < post and d.get("role") in ("SUP", "DAILY") and d.get("day_offset") == a.get("do_day") and _QA_BEFORE_OP.search(_body(d["text"])):
                out.add("before_op_done")
        plan = a.get("anesthesia")
        found = [_QA_ANES_NORM.get(x, x) for x in _QA_ANES.findall(whole)]
        if (plan and any(x != plan for x in found)) or (not plan and found):
            out.add("anes_conflict")
        if no_incision(a) and _QA_INCISION.search(whole):
            out.add("interv_incision")
    # 审查员第八轮补充的三项（描述）
    if pe and _QA_PE_RESULT.search(pe.group(1)):
        out.add("pe_result")
    for d in docs:
        if d.get("role") in ("SUP", "DAILY"):
            body = _body(d["text"])
            for m in j.get("meds") or ():
                n = re.escape(m["name"])
                if re.search(r"继续[^。；;，,]{0,6}" + n, body) and re.search(r"(?:加用|停用|停)[^。；;，,]{0,4}" + n, body):
                    out.add("med_conflict")
    organs = set(((j.get("patient") or {}).get("primary_dx") or {}).get("organs") or ())
    if organs:
        from .planner import exam_organ
        from ruler_g2.events import _REGION
        for m in _QA_EXAM_DX.finditer(whole):
            phrase = m.group(1)
            key = next((k for k in ("心脏超声", "CT", "磁共振", "超声", "X线", "胃镜", "肠镜", "支气管镜") if phrase.endswith(k)), None)
            reg = _REGION.get(phrase[:-len(key)]) if key and len(phrase) > len(key) else None
            o = exam_organ(key, reg)
            if o is not None and o not in organs:
                out.add("exam_site_mismatch")
    return out


def _sentences(text):
    return [x for x in (_STRIP.sub("", re.sub(r"\d+(?:\.\d+)?", "0", t)) for t in _SENT_SPLIT.split(text)) if len(x) >= 6]


def pb_stats(groups):
    """groups：每次住院一组病程全文（progress_between）。返回 pb_ 开头的计数。"""
    c = {}
    bodies = [[_body(t) for t in g] for g in groups]
    flat_notes = [b for g in bodies for b in g]
    c["pb_admissions"] = len(groups)
    c["pb_notes"] = len(flat_notes)
    any_hit = 0
    for cls, rx in _BOILER_RE:
        c[f"pb_boiler_notes:{cls}"] = 0
    for b in flat_notes:
        flat = _STRIP.sub("", b)
        hit = False
        for cls, rx in _BOILER_RE:
            if rx.search(flat):
                c[f"pb_boiler_notes:{cls}"] += 1
                hit = True
        any_hit += hit
    c["pb_boiler_notes_any"] = any_hit
    sent_adm, rows, prev_rep, n_sent = collections.Counter(), [], 0, 0
    for g in bodies:
        prev, here = None, set()
        for b in g:
            sents = _sentences(b)
            rows.append(sents)
            n_sent += len(sents)
            if prev is not None:
                prev_rep += sum(1 for x in sents if x in prev)
            prev = set(sents)
            here |= prev
        for x in here:
            sent_adm[x] += 1
    n_adm = len(groups)
    common = {x for x, n in sent_adm.items() if n_adm >= 10 and n >= COMMON_SHARE * n_adm}
    c["pb_sentences"] = n_sent
    c["pb_sent_common_x1000"] = round(1000 * sum(1 for r in rows for x in r if x in common) / n_sent) if n_sent and n_adm >= 10 else "NA"
    c["pb_sent_prev_repeat_x1000"] = round(1000 * prev_rep / n_sent) if n_sent else "NA"
    from ruler_g2.similarity import RefTfidf, adjacent_tfidf
    for name, gs in (("full", groups), ("no_header", bodies)):
        model = RefTfidf([t for g in gs for t in g])
        vals = [x for x in (adjacent_tfidf(model, g) for g in gs) if x is not None]
        c[f"pb_tfidf_adj_{name}_x1000"] = round(1000 * sum(vals) / len(vals)) if vals else "NA"
        c[f"pb_tfidf_adj_{name}_n"] = len(vals)
    for name, v in zip(("p25", "p50", "p75"), _q([len(b) for b in flat_notes])):
        c[f"pb_body_chars_{name}"] = v
    return c


def diagnose_real(path, limit=None):
    """真实 fold 0：每次住院取 progress_between 的病程，算 pb_stats。只读文书，只出计数。"""
    setup(("BENCH_G3", "BENCH_G2"))
    from ruler_g2 import build_real
    from ruler_g2.similarity import progress_between
    groups = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            groups.append(progress_between(build_real({"admission_id": str(obj["admission_id"]), "docs": obj["docs"]})))
            if limit and len(groups) >= limit:
                break
    c = pb_stats(groups)
    c["pb_admissions_with_2plus_notes"] = sum(1 for g in groups if len(g) >= 2)
    return sorted(c.items())


def _k4_class(word):
    return next((c for p, c in K4_CLASS if word.startswith(p)), "其他")


def _body(text):
    return text.split("\n", 1)[1] if "\n" in text else ""


def _q(xs):
    xs = sorted(xs)
    return [xs[int(q * (len(xs) - 1))] for q in (0.25, 0.5, 0.75)] if xs else ["NA"] * 3


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def diagnose(run):
    setup(("BENCH_G3", "BENCH_G2"))
    from ruler_g2 import build_real
    from ruler_g2.events import _REGION, admission_mentions
    from ruler_g2.readings import K4_RE, k4_skip, k4_span
    from ruler_g2.similarity import RefTfidf, adjacent_tfidf, progress_between

    journeys = {j["journey_id"]: j for j in _read_jsonl(os.path.join(run, "journeys.jsonl"))}
    c, per_key = collections.Counter(), collections.Counter()
    texts = {"full": [], "no_header": [], "no_header_vitals": []}
    body_chars, head_chars, vital_chars, all_chars = [], 0, 0, 0
    boiler_notes, boiler_hits = collections.Counter(), collections.Counter()
    sent_adm, sent_rows, prev_rep, sent_n, boiler_any = collections.Counter(), [], 0, 0, 0
    for adm_line in _read_jsonl(os.path.join(run, "admissions.jsonl")):
        j = journeys.get(adm_line["admission_id"])
        if j is None:
            c["journey_missing"] += 1
            continue
        c["admissions"] += 1
        docs = adm_line["docs"]
        flags = qa_flags(j, docs)
        c["qa_admissions"] += 1
        for f in QA_FLAGS:
            c[f"qa_adm_{f}"] += f in flags
        adm = build_real({"admission_id": adm_line["admission_id"], "docs": docs})
        ments = admission_mentions(adm)
        read = {(m.key, m.region or "") for d in adm.docs for m in ments[d.idx] if m.tone == "R"}
        planned = {(e["key"], _REGION.get(e["region"], e["region"]) if e["region"] else "") for e in j["exams"] if not e.get("prior")}
        c["exam_planned"] += len(planned)
        c["exam_read"] += len(read)
        c["exam_planned_read"] += len(planned & read)
        for key, _ in planned:
            per_key[f"exam_planned:{key}"] += 1
        for key, _ in planned & read:
            per_key[f"exam_planned_read:{key}"] += 1
        ds, sec = adm.ds, k4_span(adm.ds)
        if sec is not None and sec[1] - sec[0] >= 10:
            c["k4_checkable"] += 1
            hit = next((m for m in K4_RE.finditer(ds.text, sec[0], sec[1]) if not k4_skip(ds.text, m)), None)
            if hit:
                c["k4_hit"] += 1
                c[f"k4_hit:{_k4_class(hit.group(0))}"] += 1
        notes = progress_between(adm)
        texts["full"].append(notes)
        texts["no_header"].append([_body(t) for t in notes])
        texts["no_header_vitals"].append([_VITALS.sub("", _body(t)) for t in notes])
        for t in notes:
            all_chars += len(t)
            head_chars += len(t) - len(_body(t))
            vital_chars += sum(len(m.group(0)) for m in _VITALS.finditer(t))
        prev_set = None
        seen_here = set()
        for d in docs:
            if d.get("role") in ("SUP", "DAILY"):
                body = _body(d["text"])
                body_chars.append(len(body))
                flat = _STRIP.sub("", body)
                for cls, rx in _BOILER_RE:
                    n = len(rx.findall(flat))
                    boiler_hits[cls] += n
                    boiler_notes[cls] += n > 0
                boiler_any += any(rx.search(flat) for _, rx in _BOILER_RE)
                sents = _sentences(body)
                sent_rows.append(sents)
                sent_n += len(sents)
                if prev_set is not None:
                    prev_rep += sum(1 for x in sents if x in prev_set)
                prev_set = set(sents)
                seen_here |= set(sents)
        for x in seen_here:
            sent_adm[x] += 1
        names = [m["name"] for m in j["meds"]]
        whole = "".join(d["text"] for d in docs)
        c["meds_planned"] += len(names)
        c["meds_written"] += sum(1 for n in set(names) if n in whole)
        c["med_mentions"] += sum(whole.count(n) for n in set(names))
    for k, groups in texts.items():
        model = RefTfidf([t for g in groups for t in g])
        vals = [x for x in (adjacent_tfidf(model, g) for g in groups) if x is not None]
        c[f"tfidf_adj_{k}_x1000"] = round(1000 * sum(vals) / len(vals)) if vals else "NA"
        c[f"tfidf_adj_{k}_n"] = len(vals)
    c["progress_header_share_x1000"] = round(1000 * head_chars / all_chars) if all_chars else "NA"
    c["progress_vitals_share_x1000"] = round(1000 * vital_chars / all_chars) if all_chars else "NA"
    for name, v in zip(("p25", "p50", "p75"), _q(body_chars)):
        c[f"course_body_chars_{name}"] = v
    c["course_notes"] = len(body_chars)
    for cls, _ in BOILER:
        c[f"boiler_notes:{cls}"] = boiler_notes[cls]
        c[f"boiler_hits:{cls}"] = boiler_hits[cls]
    c["boiler_notes_any"] = boiler_any
    n_adm = c["admissions"]
    common = {x for x, n in sent_adm.items() if n_adm >= 10 and n >= COMMON_SHARE * n_adm}
    c["course_sentences"] = sent_n
    c["course_sent_common_x1000"] = round(1000 * sum(1 for sents in sent_rows for x in sents if x in common) / sent_n) if sent_n and n_adm >= 10 else "NA"
    c["course_sent_common_distinct"] = len(common)
    c["course_sent_prev_repeat_x1000"] = round(1000 * prev_rep / sent_n) if sent_n else "NA"
    c.update(pb_stats(texts["full"]))
    return sorted(c.items()) + sorted(per_key.items())


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--real", help="真实 fold 0 语料：算同口径的 pb_ 基线，只出计数")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.umask(0o077)
    rows = diagnose_real(a.real, a.limit) if a.real else diagnose(a.run)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(rows)
    log("out_diag_done", rows=len(rows))


if __name__ == "__main__":
    guarded(main)
