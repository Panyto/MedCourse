"""抽取 v0.2 的确定性纠偏（金标 S2G v1 之后）：按事件所在的章节、引文前后的上下文词、评测台的检查词表，改类型、改状态、清名称和数值。
不调模型，不按病种写规则：章节名和上下文词是病历写法（迁移临床 NLP 里 NegEx、ConText 的做法），检查归类用评测台冻结的
检查词表（ruler_g2/config/exams.json 的 cls）。

python -B -m medcourse.extract.normalize --events 目录/events.jsonl --source fold0.jsonl --out 目录/events_norm.jsonl

每条事件多一个 norm 字段，记用到了哪几条规则；counts（同目录 normalize_counts.csv）只有计数。下游（统计表、金标抽题）只用
contract.is_core 为真的事件：七类核心类型，状态不是 not_patient。体征（sign）、七类以外（other）留在文件里，不进下游。

规则来自 S2G v1 金标 300 题的章节计数（`S2G_by_section.csv`）。规则是照着这 300 题写的，在这 300 题上重新计分（gold.rescore）
只能看出有没有误删，准确率要看 S2G v2 的新题。

0.3（S2G3 金标 180 题之后，老板确认的口径 D1–D4，见 `验收_S2G3_pre.md` 第 5.5 节、`验收_S2G3_pkg1.md` 第 2 节）：
- 第二趟没作答的事件（状态空）按上下文补状态：疑似词 → 疑似；同一分句前面有否定词 → 否认；计划章节或计划词 → 计划；其余已发生（fill_missing）。
  S2G3 诊断状态错的 4 题里 2 题是没作答，都在诊断清单里。
- D1：既往史里的诊断和在用的药算已发生，同一分句写了治愈、切除、术后、曾患、停用这类结束词的才算既往（past_dx、past_med）；
  "高血压病史"同样处理（dx_history_suffix）。0.2.1 把既往史里的一律改成既往。
- D2：检查、化验做了、结果正常，状态是已发生，不是否认；同一分句写了未行、未查、拒绝这类没做的除外（exam_normal_done）。
- D4：生命体征只认体温、脉搏或心率、呼吸、血压、血氧，眼压改检查，其余（体重、评分）归七类以外（vital_closed）。
- aux_prior（入院记录"辅助检查"一节改既往）只管检查、化验：S2G3 的逐条规则翻转计数里，它把这一节的诊断改错了 3 题。
- 手术操作的名字是药（规则包药名词表的别名，或化疗、靶向治疗这类）改用药，是病理、免疫组化这类改检查（proc_is_drug、proc_is_exam）；
  康复训练、理疗、高压氧这类补进七类以外；"其他不适""手术"这类笼统说法、"××目标"归七类以外（generic_nonevent）。
规则是照着 S2G3 的 180 题改的，这 180 题从此是开发集，准确率看 S2G4 的新题。
"""
import argparse
import collections
import csv
import json
import os
import re

from ..contract import CORE_TYPES
from ..rules import setup
from ..safe import SafeError, guarded, log

setup(("BENCH_G2", "BENCH_G3"))
from ruler_g2.events import EXAM_RE, EXAMS, _ALIAS  # noqa: E402  评测台冻结的检查词表

DERIVED_ROOT = os.environ.get("MC_DERIVED_ROOT", "/home/weify/pyt/product/derived/")
# 0.2.1（S2G v2 开发检查停下：误删 10/133 = 7.5%）：规则改保守。认不出名字的检查不再归出去，只把笼统说法归出去；
# 删掉 past_symptom；rule_out 先看明确的疑似，再看已排除（"不考虑""不支持"算已排除）
NORM_VERSION = "mc-norm-0.3"

PE_SECTION = re.compile(r"体格检查|专科|查体")
HIST_SECTION = re.compile(r"个人史|婚育史|婚姻史|月经史|生育史|家族史|过敏史|接种史|输血史|外伤史|手术史")
FAMILY_SECTION = re.compile(r"家族史")
PAST_SECTION = re.compile(r"既往史")
HIST_TITLE = re.compile(r"(?:过敏|药物过敏|食物过敏|手术|外伤|手术外伤|输血|家族|个人|婚育|婚姻|月经|生育|接种|预防接种|吸烟|饮酒|疫区|疫水|冶游|性病|传染病|遗传病|慢性病)史$")
PLAN_SECTION = re.compile(r"计划|医嘱|注意事项|处理措施|观察的事项")
DISCUSS_SECTION = re.compile(r"讨论|鉴别|诊断依据")
HPI_SECTION = re.compile(r"现病史")
AUX_SECTION = re.compile(r"辅助检查")
# 名称开头的否定词：去掉，状态改成否认／阴性
NEG_PREFIX = re.compile(r"^(?:无明显|未见明显|未见|未闻及|未触及|未扪及|未及|否认|不伴|没有|无)")
# 上下文词（同一句里、事件之前）
HYPO = re.compile(r"常表现为|常伴|多见于|多表现为|典型表现|可表现为|可出现|可能出现|易出现|可引起|可致|注意有无|警惕|如有|若有|一旦|风险|并发症")
PRIOR = re.compile(r"外院|院外|门诊|急诊|入院前|当地医院|\d+\s*(?:天|日|周|月|年)前|[半一两三四五六七八九十]+(?:个)?(?:天|日|周|月|年)前")
PLAN_CUE = re.compile(r"拟|计划|建议|嘱|明日|明天|择期|择日|定于|准备|予以?完善|进一步")
ONGOING_STOP = re.compile(r"既往|曾|已缓解|缓解后|消失|好转后|治愈")
# 0.2.1：先看"待排""不除外"这类明确的疑似，再看已排除（含"不考虑""不支持"），最后才是"考虑""可能"。
# 0.2 先匹配"考虑"，把"不考虑肺结核"判成了疑似（开发检查里诊断的状态对因此少了 2 题）
UNCERTAIN_STRONG = re.compile(r"待排|待除外|不除外|不排除|[？?]")
EXCLUDED_CUE = re.compile(r"已排除|可排除|予以排除|排除|不考虑|不支持|已除外")
# 0.2.1：笼统的"完善相关检查""入院抽血"才归七类以外。0.2 把认不出名字的检查都归出去，在开发检查里误删成立题 8 次、
# 只挡住不成立题 5 次（彩超、胸片这类写法不在评测台的检查词表里）
GENERIC_EXAM = re.compile(r"^(?:完善|入院|进一步|行)?(?:相关|常规|各项|入院)?(?:检查|化验|抽血|检验)$")
PE_CUE = re.compile(r"查体|体检|体格检查|PE[:：]")
# 不是手术操作的诊疗活动（病历写法的常识）
CARE = re.compile(r"气压|推拿|熨|理疗|针灸|艾灸|拔罐|按摩|康复|训练|功能锻炼|高压氧|氦氖|激光照射|针刺|电针|磁疗|蜡疗|中频|低频|红外|冲击波|熏洗|坐浴|"
                  r"会诊|宣教|健康教育|出院|护理|吸氧|氧疗|心电监护|雾化|饮食|评分|评估")
# 0.3（D1）：既往史里写了结束的，才算既往
ENDED_CUE = re.compile(r"治愈|痊愈|已愈|已切除|切除术后|术后|已康复|已消失|已缓解|曾患|曾有|曾因|曾诊断|曾服|停用|已停|停药")
# 0.3（D2）：检查、化验没做的写法
NOT_DONE_CUE = re.compile(r"未行|未做|未查|未完善|未能|暂未|拒绝|拒查|未予")
# 0.3：同一分句里、事件前面的否定词（补状态用；"、"不断开，"否认高血压、糖尿病"里的糖尿病也算）
NEG_BEFORE = re.compile(r"(?:无|否认|未见|未闻及|未触及|不伴|没有)[^，,。；;：:]*$")
# 0.3（D4）：生命体征只有五项
VITAL_OK = re.compile(r"体温|脉搏|脉率|心率|呼吸|血压|血氧|氧饱和|饱和度|生命体征|SpO2|^T$|^P$|^R$|^BP$|^HR$", re.I)
VITAL_TO_EXAM = re.compile(r"眼压")
# 0.3：手术操作的边界。名字里有这些动作的仍是手术操作
PROC_VERB = re.compile(r"术|置管|置入|植入|切除|穿刺|插管|引流|造瘘|活检|取材")
PROC_DRUG_WORD = re.compile(r"化疗|化学治疗|靶向治疗|免疫治疗|内分泌治疗|缩宫素|催产素")
PROC_EXAM_WORD = re.compile(r"病理|免疫组化|组化|基因检测|FISH|染色体")
GENERIC_NONEVENT = re.compile(r"^(?:其他|其它)?(?:特殊)?(?:不适|异常|情况)$|^(?:手术|外伤|输血|治疗)$|目标$")
LAB_WORD = re.compile(r"培养|核酸|抗体|抗原|DNA|RNA|血气|涂片|药敏")
FREQ = re.compile(r"\s*(?:(?<![A-Za-z])(?:qd|bid|tid|qid|qn|qod|q\d+h|prn|st)(?![A-Za-z])|每日\s*[一二三四1234]\s*次|每\s*\d+\s*小时\s*[一1]?\s*次?)", re.I)
VEHICLE = re.compile(r"氯化钠|葡萄糖|注射用水|生理盐水|林格|(?<![A-Za-z])(?:NS|GS)(?![A-Za-z])")
BP = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")
_SENT_END = re.compile(r"[。；;！？!?\n]")
LAB_CLS = {"检验", "培养"}


def _drug_alias_re():
    """规则包 g3-rules-0.3 的药名词表（公开药名和别名），手术操作的名字是药的改用药。"""
    path = os.path.join(os.environ["BENCH_G3"], "ruler_g3", "data", "drug_lexicon.csv")
    with open(path, encoding="utf-8") as f:
        names = {a for r in csv.DictReader(f) for a in (r.get("aliases") or "").split("|") if len(a) >= 2}
    return re.compile("|".join(re.escape(a) for a in sorted(names, key=len, reverse=True)))


DRUG_RE = _drug_alias_re()


def _clause(text, s, e):
    """事件所在的那一句：从前一个句号到后一个句号。返回 (句首, 句尾)。"""
    a = 0
    for m in _SENT_END.finditer(text, 0, s):
        a = m.end()
    m = _SENT_END.search(text, e)
    return a, (m.start() if m else len(text))


def exam_classes(name):
    """名称里认得出的评测台检查键和它们的类别。"""
    return [(_ALIAS[m.group(0)], EXAMS[_ALIAS[m.group(0)]]["cls"]) for m in EXAM_RE.finditer(name or "")]


def normalize_event(e, text, s, t, doc_type, section):
    """返回纠偏后的事件（新 dict）。text 是能看到的原文（整篇或切片），s、t 是引文在 text 里的位置。"""
    out = dict(e)
    rules = []
    sec = section or ""
    a, b = _clause(text, s, t)
    before = text[a:s]
    clause = text[a:b]
    name = out.get("name") or ""
    typ = out.get("event_type")
    st = out.get("assertion")
    # 1 旧状态：rule_out 拆成已排除和疑似／待排
    if st == "rule_out":
        st = "uncertain" if UNCERTAIN_STRONG.search(clause) or not EXCLUDED_CUE.search(clause) else "excluded"
        rules.append("legacy_rule_out")
    # 1.5（0.3）第二趟没作答的补状态
    if st is None:
        # 否定词可能在引文里、名称前面（名称回退的引文从分句开头到名称为止："否认高血压、糖尿病"）
        ev = text[s:t]
        pre = before + (ev[:ev.find(name)] if name and name in ev else "")
        # 诊断清单的"肺结核？"：问号本身是分句的结尾，要连着看分句后面那一个字
        st = ("uncertain" if UNCERTAIN_STRONG.search(clause + text[b:b + 1]) else "negated" if NEG_BEFORE.search(pre)
              else "planned" if PLAN_SECTION.search(sec) or PLAN_CUE.search(before) else "done")
        rules.append("fill_missing")
    # 2 名称开头的否定词
    m = NEG_PREFIX.match(name)
    if m and len(name) > m.end():
        name, st = name[m.end():], "negated"
        rules.append("neg_prefix")
    # 3 类型。"高血压病史"是诊断（去掉"病史"、状态既往）；"过敏史""手术史"这类标题是病史条目，归七类以外
    if HIST_TITLE.search(name):
        if typ != "other":
            typ = "other"
            rules.append("history_item")
    elif name.endswith("病史") and len(name) > 2 and typ == "diagnosis":
        # 0.3（D1）：现在仍有的慢性病算已发生，写了结束的才算既往
        name, st = name[:-2], ("negated" if st == "negated" else "history" if ENDED_CUE.search(clause) else "done")
        rules.append("dx_history_suffix")
    elif HIST_SECTION.search(sec) and not PAST_SECTION.search(sec):
        if typ == "diagnosis" and FAMILY_SECTION.search(sec):
            st = "not_patient"
            rules.append("family_dx")
        elif typ != "other":
            typ = "other"
            rules.append("history_item")
    # 0.2 的 past_symptom（既往史里的症状归七类以外）在开发检查里只误删、没挡住一题不成立的，0.2.1 删掉
    if typ in ("symptom", "exam") and PE_SECTION.search(sec):
        typ = "sign"
        rules.append("pe_section")
    elif typ == "symptom" and PE_CUE.search(before):
        typ = "sign"
        rules.append("pe_cue")
    if typ == "procedure" and CARE.search(name):
        typ = "other"
        rules.append("care_not_procedure")
    elif typ == "procedure" and not PROC_VERB.search(name) and (PROC_DRUG_WORD.search(name) or DRUG_RE.search(name)):
        typ = "medication"
        rules.append("proc_is_drug")
    elif typ == "procedure" and not PROC_VERB.search(name) and PROC_EXAM_WORD.search(name):
        typ = "exam"
        rules.append("proc_is_exam")
    if typ == "vital" and not VITAL_OK.search(name):
        typ = "exam" if VITAL_TO_EXAM.search(name) else "other"
        rules.append("vital_closed")
    if typ in CORE_TYPES and GENERIC_NONEVENT.search(name):
        typ = "other"
        rules.append("generic_nonevent")
    if typ == "exam":
        cls = {c for _, c in exam_classes(name)}
        if cls and cls <= LAB_CLS:
            typ = "lab"
            rules.append("exam_is_lab")
        elif not cls and LAB_WORD.search(name):
            typ = "lab"
            rules.append("exam_is_lab")
        elif not cls and GENERIC_EXAM.search(name):
            typ = "other"
            rules.append("exam_generic")
    elif typ == "lab":
        cls = {c for _, c in exam_classes(name)}
        if cls and not cls & LAB_CLS:
            typ = "exam"
            rules.append("lab_is_exam")
    # 3.5（0.3，D2）：检查、化验做了、结果正常是已发生
    if typ in ("exam", "lab") and st == "negated" and not NOT_DONE_CUE.search(clause):
        st = "done"
        rules.append("exam_normal_done")
    # 4 状态（七类核心类型；否认、已排除的不动）
    if typ in CORE_TYPES and st not in ("negated", "excluded"):
        if typ not in ("lab", "vital", "medication") and HYPO.search(before):
            st = "not_patient"
            rules.append("hypothetical")
        elif PAST_SECTION.search(sec) and typ in ("diagnosis", "medication") and st in ("done", "history"):
            # 0.3（D1）：既往史里的诊断和在用的药，写了结束的才算既往
            new = "history" if ENDED_CUE.search(clause) else "done"
            if new != st:
                st = new
                rules.append("past_dx" if typ == "diagnosis" else "past_med")
        elif PAST_SECTION.search(sec) and st == "done":
            st = "history"
            rules.append("past_section")
        elif PLAN_SECTION.search(sec) and st in ("done", "uncertain", None):
            st = "planned"
            rules.append("plan_section")
        elif doc_type == "hospital_record" and AUX_SECTION.search(sec) and st == "done" and typ in ("exam", "lab"):
            st = "history"
            rules.append("aux_prior")
        elif typ in ("exam", "lab", "procedure", "medication") and st == "done" and PRIOR.search(before):
            st = "history"
            rules.append("prior_cue")
        elif HPI_SECTION.search(sec) and typ == "symptom" and st == "history" and not ONGOING_STOP.search(clause):
            st = "done"
            rules.append("hpi_ongoing")
        elif typ in ("exam", "lab", "procedure", "medication", "vital") and st == "done" and PLAN_CUE.search(before):
            st = "planned"
            rules.append("plan_cue")
    # 5 数值
    val = out.get("value")
    if typ == "vital" and re.search(r"血压|BP", name, re.I) and (val is None or "/" not in str(val)):
        m = BP.search(text[s:t]) or BP.search(clause)
        if m:
            out["value"], out["unit"] = f"{m.group(1)}/{m.group(2)}", "mmHg"
            rules.append("bp_full")
    if typ == "medication" and val is not None:
        v2 = FREQ.sub("", str(val)).strip()
        if v2 != str(val):
            out["value"] = v2 or None
            rules.append("dose_freq")
        if (out.get("unit") or "").lower() == "ml" and VEHICLE.search(text[s:t]):
            # 溶媒的毫升数被当成剂量：在引文里找药名后面紧跟的剂量，找不到就清空
            m = re.search(re.escape(name) + r"\s*(\d+(?:\.\d+)?)\s*(mg|g|μg|ug|IU|U|万U|万单位|ml|mL)", text[s:t]) if name else None
            out["value"], out["unit"] = (m.group(1), m.group(2)) if m else (None, None)
            rules.append("dose_vehicle")
    out.update(event_type=typ, name=name, assertion=st, norm=rules)
    return out


def normalize_events(events, source):
    """events：抽取产出；source：{住院号: 住院}。返回 (新事件列表, 计数)。"""
    cnt = collections.Counter()
    out = []
    for e in events:
        adm = source.get(str(e["admission_id"]))
        if adm is None:
            cnt["no_source"] += 1
            continue
        doc = adm["docs"][e["doc_idx"]]
        n = normalize_event(e, doc.get("text") or "", e["ev_start"], e["ev_end"], e.get("doc_type") or doc.get("doc_type"), e.get("section"))
        cnt["events"] += 1
        for r in n["norm"]:
            cnt[f"rule:{r}"] += 1
        cnt[f"type_in:{e.get('event_type')}"] += 1
        cnt[f"type_out:{n['event_type']}"] += 1
        cnt[f"assert_out:{n.get('assertion')}"] += 1
        cnt["core_out"] += n["event_type"] in CORE_TYPES and n.get("assertion") != "not_patient"
        out.append(n)
    return out, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.umask(0o077)
    out = os.path.abspath(a.out)
    if not a.synthetic and not out.replace("\\", "/").startswith(DERIVED_ROOT):
        raise SafeError("out_not_in_derived_root")
    with open(a.events, encoding="utf-8") as f:
        events = [json.loads(x) for x in f if x.strip()]
    rids = {str(e["admission_id"]) for e in events}
    source = {}
    with open(a.source, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                adm = json.loads(line)
                if str(adm.get("admission_id")) in rids:
                    source[str(adm["admission_id"])] = adm
    norm, cnt = normalize_events(events, source)
    with open(out, "w", encoding="utf-8") as f:
        for e in norm:
            f.write(json.dumps(dict(e, norm_version=NORM_VERSION), ensure_ascii=False) + "\n")
    with open(os.path.join(os.path.dirname(out), "normalize_counts.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(sorted(cnt.items()))
    log("normalize_done", events=cnt["events"], core=cnt["core_out"])


if __name__ == "__main__":
    guarded(main)
