"""L4 可见性编译器：Journey 和 DocSpec → 每节的 SectionPlan。每篇只给当天能知道的事。

抬头、时刻、日期、诊断清单、化验数值行由程序写（mode=program），模型只写叙述段（mode=llm），省调用，也不可能写错数。
拷贝预算按文书细类和天数间隔给（S0_copyops.csv），不写死单一比例；真正的拷贝由 apply_copy 在写完后执行。

0.1.6（监督员第五轮）：病程每篇开头的两条事实（"症状：X，较前好转""查体：围绕X的体征，较前好转"）天天一个句式；
现在按当天发生的事给"今日重点"（course_focus）：有回报写结果判读，加药停药写理由和疗效，术后写切口和恢复，拟手术写术前准备，
诊断明确或排除写依据，上级查房写分析意见；什么都没发生的日子轮换一个观察点。症状只在有变化时写变化，查体不再每篇都给。
拷贝（apply_copy）不动：真实相邻病程的 5-gram 拷贝在评测台下限以上，W-8B 已经偏少。

0.1.8（第七次合成病历审阅、监督员第七轮）：
- 体格检查目标字数封顶 220、专科情况封顶 100。真实的体格检查是医院长模板，按真实分位数给目标字数，写手手里只有一条事实，
  闸门又要写到目标的四成，只能凑字，6/47 份复读到截断。
- 体格检查告诉写手"生命体征已写在开头"（33/47 份写完生命体征又写"体温未测"）；现病史告诉写手"既往史另有一节"，
  病例特点给既往史（既往史写高血压、首程却写"否认高血压"）。这两条提示只进 LLM 提示词（hint），模板写手不写。
- 非手术住院的出院记录和病程标 nonsurgical，闸门查"术后恢复""手术顺利"这类话（2/47 份非手术住院写了"术后恢复顺利"）。
- 拷贝不拷上级医师的话：0.1.7 只挡"查房"，"主任医师强调……"被拷进了住院医师的日常病程。

0.1.9（第八次合成病历审阅，老板批准的十条）：
- 手术当天、术后首程之前的病程写"今日拟行××"（0.1.8 手术当天上午的查房已写"手术过程顺利"，手术在 19 时）；
  病程给麻醉方式（0.1.8 病程自己编了"全麻下"）；局部麻醉、静脉麻醉的手术（介入、内镜、白内障……，按 0.1.8 已有的麻醉规则判）
  术后观察写术区和穿刺点，不写切口、引流（冠脉造影写了"切口愈合良好"）。
- 拷贝不拷病情趋势句（较前、与前相仿、好转、仍有……）和带症状词的句子（0.1.8 一篇里既"较前明显好转"又拷了"与前日相仿"）。
- 病程按真实比例给"精神、睡眠、饮食""二便"这类一般情况（stats.note_general）：用事实代替套话，不靠禁止。
- 查体事实写"阳性体征较前好转"，不写"咳嗽相关体征"（0.1.8 被原样写出 35 处）。
- 审查员第八轮补充：鉴别诊断给病人的真实主诉和初步诊断，不让写手把鉴别病的症状安在病人头上；拟诊讨论的依据写明是哪几项
  已回报的检查；专科情况拿同一篇写好的体格检查当事实（pe_ref，流程补）；查房重点不"继续"今天才加用的药。
- 现病史的提示不再说"另有一节"（写手照抄成"既往史及过敏史详见另节"）；体格检查的提示加"不写病史和患者自述"。
- 出院记录"入院情况"拿入院记录实际写出的体格检查当事实（流程传 ctx），不再另编一套查体。
- 停药理由按药物大类写；长期用药不写"疗程已足"。
- 化疗、复查这类住院的主诉、现病史按治疗写（planner.chief_text、chief_key）；当天没有症状时不写症状句。

0.1.11：鉴别诊断只给主诉（不写"初步诊断"）；病程不再给二便；专科情况的体格检查改进提示（流程补）。
"""
import re

from .drug_class import stop_reason
from .planner import CHILD_AGE, chief_key, chief_text, exam_result_text, final_diagnoses
from .stats import past_matches

_DIGIT = re.compile(r"\d")
_PLAN_WORD = re.compile(r"拟|计划|明日|明天|待|术前|术后|今日|今天|昨|择期|入院第|第\s*[一二三四五六七八九十\d]+\s*天")
_UNSAFE_NUM = re.compile(r"\d+\s*(?:℃|次/分|mmHg|%|g/L|mmol/L|μmol/L|umol/L|U/L|ng/mL|pg/mL|mg/L|×10|\*10|天|日|周|月|年|小时|h\b|岁|个月)|第\s*\d+|\d+\s*[-/~～]\s*\d+|\d{2,}/\d{2,}")
_SENT = re.compile(r"[^。；;！？!?\n]+[。；;！？!?]?")
# 0.1.7：只挡拷出矛盾的两种：今天已经停了的药（"继续某药"拷进写着"停用某药"的一篇），和这一篇已经写了同一件事的变化句
# （一篇里既"与前相仿"又"较前好转"）。其余照拷：真实相邻病程拷得比我们多（B_copy5 在评测台下限以下）
_TREND = re.compile(r"较前|与前相仿|同前")
_SENIOR = re.compile(r"(?:副?主任|主治|上级)医师")
# 0.1.8：真实是长模板，写手只有一条事实，目标字数按真实给会凑字、复读
SECTION_CAP = {"体格检查": 220, "专科情况": 100, "既往史": 80}
# 0.1.9：体格检查多一句"不写病史和患者自述"（0.1.8 体格检查写病史 14/20）；现病史不再说"另有一节"（写手照抄成"详见另节"）
HINT_PE = "生命体征已经写在这一节开头，不要再写体温、脉搏、呼吸、血压，也不要写“未测”；只写查体所见，不写病史和患者自述，不写化验、心电图和影像结果。"
# 0.1.12（第十一次真未来泄露 O 52/256）：去掉"诊治情况"。56 条命中在现病史，54 条带门诊、急诊、外院这类词，写手照"诊治情况"补了
# 事实里没有的门诊检查结果。改成只写症状经过，入院前查过什么由事实给（compile_doc 的 HR 现病史）；不点名禁止哪种写法
HINT_HPI = "只写这次发病的症状、起病和变化经过，不写既往史和过敏史。"
# 0.1.9：拷贝不拷病情趋势句（同一篇里既"较前好转"又拷来"与前相仿"）；0.21 放开“以改善/以缓解”类治疗目的用语
_COPY_TREND = re.compile(r"较前|与前相仿|同前|相仿|(?<!以)(?<!旨在)(?<!以期)(?:好转|加重|缓解|减轻|改善)|消失|仍有|仍感|无明显变化")
# 0.1.9.1（第九次 S1-B 停止）：拷贝不拷当天的用药动作句。剂量为空的药，"加用某药口服"不带数字，被当成稳定句拷进下一篇，
# 和下一篇的"继续某药治疗"并存（模板 qa_adm_med_conflict 12/499）。"继续某药"是持续状态，照拷
_COPY_ACTION = re.compile(r"加用|停用|停药|改用|换用|改为|调整|减量|加量|开始使用|开始予")
# 0.1.9：一般情况（病程常见写法，公开知识），按 stats.note_general 的比例给
GENERAL_MENTAL = {"better": ("精神、睡眠较前好转，饮食可", "精神可，饮食、睡眠可", "精神好，食欲改善，睡眠可"),
                  "same": ("精神、饮食、睡眠可", "精神尚可，饮食、睡眠一般", "精神可，纳眠可")}
GENERAL_BOWEL = ("大小便正常", "二便如常", "二便正常")
def no_incision(a):
    """局部麻醉、静脉麻醉或未配麻醉的操作（介入、内镜、白内障、清宫、康复……）没有切口和引流。按 planner.ANESTHESIA_RULES 判，
    不另写手术类别表（0.1.9 采纳审查员"不按病种堆规则"的意见；0.2.0 加上麻醉为空的非手术操作）。"""
    return not a.get("anesthesia") or a.get("anesthesia") in ("局部麻醉", "静脉麻醉")


def _proc_done_phrase(a, prefix=""):
    return f"{prefix}在{a['anesthesia']}下行{a['name']}" if a.get("anesthesia") else f"{prefix}行{a['name']}"


def postop_watch(a, n):
    """术后第 n 天要观察的。开放、腔镜手术照 0.1.6 写切口和引流；没有切口的写术区和穿刺点（0.1.8 冠脉造影写了"切口愈合良好"）。"""
    if no_incision(a):
        return "观察术区及穿刺点情况，有无出血等并发症" + ("，指导饮食和活动" if n >= 3 else "")
    return "观察切口愈合及饮食、活动恢复情况" if n >= 3 else "观察切口及引流情况，指导早期活动"


def op_state(j, spec, specs):
    """0.1.9：这一篇写的时候手术做没做。手术当天，术后首程之前的文书算还没做（0.1.8 按天算，手术当天上午的查房写了"手术过程顺利"）。
    返回 (已做的, 已决定还没做的)。"""
    post = next((s["k"] for s in specs if s["role"] == "POST"), None)
    done, pending = [], []
    for a in j["procedures"]:
        if a["do_day"] < spec["day"] or (a["do_day"] == spec["day"] and post is not None and spec["k"] > post):
            done.append(a)
        elif a["decide_day"] <= spec["day"]:
            pending.append(a)
    return done, pending
ROUND = {"主任医师查房记录": "主任医师", "主治医师查房记录": "主治医师", "上级医师查房记录": "上级医师"}
SUBTYPE = {"日常病程记录": "其他病程", "主任医师查房记录": "主任医师查房", "主治医师查房记录": "主治医师查房", "上级医师查房记录": "上级医师查房"}
DEFAULT_CHARS = {"现病史": 220, "既往史": 60, "体格检查": 90, "专科情况": 50, "病例特点": 180, "拟诊讨论": 150, "诊疗计划": 90, "鉴别诊断": 80, "简要病情": 90, "手术指征": 60, "注意事项": 50,
                 "手术简要经过": 160, "术后处理措施": 80, "术后应当特别注意观察的事项": 50, "入院情况": 150, "诊疗经过": 400, "出院情况": 70, "出院医嘱": 90, "正文": 220}
SEC_SUBTYPE = {"HR": "入院记录", "FP": "首次病程", "PRE": "术前小结", "POST": "术后首次病程", "DS": "出院记录"}


def doc_date(j, day):
    import datetime as dt
    return (dt.date.fromisoformat(j["admit_date"]) + dt.timedelta(days=day)).isoformat()


def header(j, spec, specs):
    """程序写的抬头，格式和 G-E′ 同，六关读数都认。"""
    p, day, hm = j["patient"], spec["day"], spec["time"]
    date, t = doc_date(j, day), spec["doc_type_cn"]
    who = f"姓名：[患者]　性别：{p['sex']}　年龄：{p['age_text']}　科室：{p['dept']}"
    if t == "入院记录":
        return f"入院记录\n{who}\n入院时间：{date} {hm}\n"
    if t == "首次病程记录":
        return f"首次病程记录\n{date} {hm}\n姓名：[患者]　年龄：{p['age_text']}　科室：{p['dept']}\n"
    if t == "出院记录":
        hr_hm = next(s["time"] for s in specs if s["role"] == "HR")
        return f"出院记录\n{who}\n入院时间：{j['admit_date']} {hr_hm}　出院时间：{date} {hm}　住院天数：{j['los']}天\n"
    return f"{date} {hm}　{t}\n"


def _fact(kind, text, must=False, key=None, values=()):
    return {"kind": kind, "text": text, "must": must, "key": key, "values": [str(v) for v in values]}


def hpi_workup_facts(v):
    """0.1.12：入院记录现病史的"入院前后查过什么"。规划里没有入院前的检查；入院当天已回报的照给（和"辅助检查"节同源，数值在 allowed 里），
    一项都没有时说明入院前没做过检查。都不是必写。"""
    if v["results"]:
        return [_fact("result", "入院当天已回报：" + "；".join(f"{e['name']}示：{exam_result_text(e)}" for e in v["results"]), False, None,
                      [x["value"] for e in v["results"] for x in e["items"]])]
    return [_fact("plan", "入院前未做过检查")]


def _result_fact(e, must):
    """0.1.5 写成"X示："：结果口吻紧跟检查名，评测台的检查种类只数结果口吻。"""
    vals = [x["value"] for x in e["items"]]
    return _fact("result", f"{e['name']}示：{exam_result_text(e)}", must, e["name"], vals)


def _med_text(m, with_dose):
    tail = f"{m['dose']}{(' ' + m['freq']) if m['freq'] else ''}{m['route']}" if with_dose else ""
    return f"{m['name']}{tail}"


def _med_values(m):
    return re.findall(r"\d+(?:\.\d+)?", m["dose"])


def _vital_sentence(v):
    return f"体温{v['T']}℃，脉搏{v['P']}次/分，呼吸{v['R']}次/分，血压{v['SBP']}/{v['DBP']}mmHg。"


def _vital_values(v):
    return [v["T"], v["P"], v["R"], v["SBP"], v["DBP"]]


def visible(j, day):
    """截至 day 的可见事实。"""
    return {"results": [e for e in j["exams"] if e["result_day"] <= day], "today": [e for e in j["exams"] if e["result_day"] == day],
            "pending": [e for e in j["exams"] if e["order_day"] <= day < e["result_day"]], "new_order": [e for e in j["exams"] if e["order_day"] == day < e["result_day"]],
            "meds": [m for m in j["meds"] if m["start_day"] <= day <= m["stop_day"]], "meds_new": [m for m in j["meds"] if m["start_day"] == day],
            "done": [a for a in j["procedures"] if a["do_day"] <= day], "planned": [a for a in j["procedures"] if a["decide_day"] <= day < a["do_day"]],
            "dx": {d["name"]: d["state"] for d in sorted(j["dx_events"], key=lambda x: x["day"]) if d["day"] <= day}}


def _dx_fact(name, state, must=False, basis=None):
    """诊断默认锁定：确诊的写成已明确，只有真在疑似状态才写考虑。写成考虑又在出院列确诊，会多算一步诊断变化。
    basis（0.1.9，审查员第八轮补充）：写明依据是哪些症状和哪几项已回报的检查（0.1.8 只写"依据症状和已回报的检查"，
    写手编出了没做的"腹部平片示肠管扩张、液平面"）。"""
    why = basis or "症状和已回报的检查"
    text = f"考虑{name}，依据{why}" if state == "疑似" else f"诊断{name}明确，依据{why}"
    return _fact("dx", text, must, name)


# 0.1.9（审查员第八轮补充）：只进 LLM 提示词的几句
HINT_DIFF = "患者的表现只写上面给出的主诉，不要把鉴别疾病的症状写成患者的表现。"
HINT_DISCUSS = "依据只写上面列出的症状和检查，没有列出的检查不写。"
# 0.1.11（第十次合成病历 M6）：体格检查不再当事实给（写手照抄成专科情况），整段放进提示只作对照（流程补在这句后面）；
# 专科情况只写专科部位。0.1.10 只给前 120 字，后面的阳性体征（"指套染血"）没进来，专科情况写成了"指诊未触及明显肿块"
HINT_SPEC = "只写本专科部位的查体，不重复抄写体格检查；和同一篇的体格检查一致，不写相反的体征；不写化验、心电图和影像结果。"


def _dx_line(names):
    return "\n".join(f"{i + 1}.{n}" for i, n in enumerate(names)) + "\n"


NEG_PAST = ("高血压", "糖尿病", "冠心病")


def past_text(p):
    """既往史：有的写"有…病史"，否认的只列诊断清单和既往史里都没有的病（0.2.0 认全称：冠状动脉粥样硬化性心脏病 ↔ 冠心病；
    14 岁以下儿童不列成人慢性病否认清单，无病史时写"既往体健"）。"""
    present = "；".join([p["primary_dx"]["name"]] + [c["name"] for c in p["comorbidities"]] + list(p["past_history"]))
    neg = [] if p["age"] < CHILD_AGE else [t for t in NEG_PAST if not past_matches(t, present)]
    parts = (["有" + "、".join(p["past_history"]) + "病史"] if p["past_history"] else []) + (["否认" + "、".join(neg) + "等慢性病史"] if neg else [])
    return "，".join(parts) or "既往体健"


def past_line(p):
    """病例特点用的一句既往史（0.1.8）。"""
    t = past_text(p)
    return t if t.startswith("既往") else "既往" + t


def menses_text(p):
    """妊娠的写停经周数，不写绝经。"""
    if p.get("pregnant"):
        return f"平素月经规律，现停经{p['gest_weeks']}周。"
    return "已绝经。" if p["age"] >= 55 else "既往月经规律。"


def _admission_dx(j):
    v = visible(j, 0)["dx"]
    p = j["patient"]
    out = [n + ("？" if v.get(n) == "疑似" else "") for n in (p["primary_dx"]["name"],)]
    return out + [c["name"] for c in p["comorbidities"] if c["at_admission"]]


COURSE = ("SUP", "DAILY")
TREND_TEXT = {"好转": "较前好转", "减轻": "较前减轻", "平稳": "与前相仿", "同前": "同前"}
# 没发生什么事的日子轮换的观察点（病程常见写法，公开知识）。{m} 是一种在用药，{s} 是主要症状
QUIET = ("关注{s}的程度和发作情况", "观察目前治疗的反应，有无药物不良反应", "注意主要阳性体征的变化", "向患者及家属交代病情，患者配合治疗")
QUIET_MED = "{m}使用中，注意有无不良反应"


def prev_course_day(spec, specs):
    """这篇之前最近一篇日常病程、上级查房的日子；没有返回 0（第 0 天回报的检查入院记录"辅助检查"已经写了）。"""
    days = [s["day"] for s in specs if s["k"] < spec["k"] and s["role"] in COURSE]
    return max(days) if days else 0


def course_focus(j, spec, specs, rng):
    """这一篇病程的"今日重点"：按上一篇病程以来发生的事，取优先级最高的两条；什么都没发生就轮换一个观察点。
    返回 [(类别, 句子)]，句子是病程里能直接写的话，不带数值。"""
    p, day = j["patient"], spec["day"]
    prev = prev_course_day(spec, specs)
    v = visible(j, day)
    d = j["daily"][day]
    out = []
    if spec["role"] == "SUP":
        dx = p["primary_dx"]["name"]
        state = "诊断明确" if v["dx"].get(dx) != "疑似" else "诊断尚待进一步明确"
        diff = [n for n in j["differentials"] if v["dx"].get(n) == "疑似"]
        # 0.1.9（审查员第八轮补充）：今天才加的药不写"继续"（0.1.8 写出了"继续使用阿托伐他汀，加用阿托伐他汀20mg"）
        med = next((m for m in v["meds"] if not prev < m["start_day"] <= day), None)
        out.append(("round", f"{ROUND[spec['doc_type_cn']]}查房分析：{dx}{state}" + (f"，需与{'、'.join(diff)}鉴别" if diff else "")
                    + (f"，治疗上继续{med['name']}" if med else "，暂不调整治疗")))
    done, pending = op_state(j, spec, specs)
    for a in done:
        if a["do_day"] < day:
            n = day - a["do_day"]
            out.append(("postop", f"术后第{n}天，" + postop_watch(a, n)))
    for a in pending:
        out.append(("preop", ("今日" if a["do_day"] == day else "") + f"拟行{a['name']}，完善术前准备，向患者及家属交代手术风险"))
    back = [e for e in j["exams"] if prev < e["result_day"] <= day]
    if back:
        out.append(("result", f"结合{'、'.join(e['name'] for e in back[:3])}结果分析病情"))
    for x in j["dx_events"]:
        if prev < x["day"] <= day and x["state"] in ("确诊", "排除") and x["day"] > 0:
            out.append(("dx", f"结合检查结果，{x['name']}诊断明确" if x["state"] == "确诊" else f"结合检查结果，排除{x['name']}"))
    stopped = [m for m in j["meds"] if prev <= m["stop_day"] < day]
    if stopped:
        # 0.1.9：停药理由按大类写（0.1.8 一律"疗程已足"，波立维、美托洛尔也这样写）
        by = {}
        for m in stopped:
            by.setdefault(stop_reason(m.get("class")), []).append(m["name"])
        out.append(("med_stop", "；".join(f"{'、'.join(ns)}{r}" for r, ns in by.items())))
    long = [m for m in v["meds"] if day - m["start_day"] >= 3 and m["stop_day"] > day]
    if long:
        m = long[spec["k"] % len(long)]
        out.append(("med_effect", f"已使用{m['name']}{day - m['start_day']}天，" + ("症状改善，治疗有效" if d["trend"] in ("好转", "减轻") else "症状无加重，继续观察疗效")))
    if not out:
        meds = v["meds"]
        s0 = d["symptoms"][0] if d["symptoms"] else None
        pool = [q.format(s=s0) for q in QUIET if s0 or "{s}" not in q] + ([QUIET_MED.format(m=meds[spec["k"] % len(meds)]["name"])] if meds else [])
        out.append(("quiet", pool[rng.randrange(len(pool))]))
    return out[:2]


def target_chars(stats, role, cn, section, rng):
    sub = SEC_SUBTYPE.get(role) or SUBTYPE.get(cn) or "其他病程"
    q = ((stats.get("section_chars") or {}).get(f"{sub}|{section}") or {}).get("q")
    cap = SECTION_CAP.get(section, 10 ** 6)
    if q:
        return min(cap, max(20, int(rng.uniform(q[1], q[3]))))
    return min(cap, DEFAULT_CHARS.get(section, 80))


def copy_budget(stats, spec, prev_day, prev_role=None):
    t = stats.get("copy") or {}
    sub_key = SUBTYPE.get(spec["doc_type_cn"], spec["doc_type_cn"])
    sub = (t.get("文书细类") or {}).get(sub_key) or (t.get("文书细类") or {}).get("其他病程")
    overall = (t.get("全部") or {}).get("全部")
    if not sub or not overall:
        return 0.0
    gap = spec["day"] - prev_day
    g = (t.get("天数间隔") or {}).get("3+" if gap >= 3 else str(max(0, gap)))
    base = sub["reuse"] * ((g["reuse"] / overall["reuse"]) if g and overall["reuse"] else 1.0)
    if prev_role is None:
        return base
    # 0.21：FP→首篇不在 B_tfidf 计算窗口内，重点补齐 C1a 第一跳与 PRE/POST 断层
    if prev_role == "FP":
        return min(0.55, base * 1.65)
    if prev_role in ("PRE", "POST") or spec["role"] in ("PRE", "POST"):
        return min(0.50, base * 1.45)
    return min(0.45, base * 1.22)


def compile_doc(j, spec, specs, stats, rng, ctx=None):
    """返回这篇文书的节列表：[{section, mode, title, facts, target_chars, program_text?, copy?}]。title 为空表示不写小标题。
    ctx（0.1.9，流程传）：{"hr_pe": 入院记录实际写出的体格检查摘录}，出院记录"入院情况"拿它当事实；不传时照 0.1.8。"""
    p, day, role = j["patient"], spec["day"], spec["role"]
    v = visible(j, day)
    d = j["daily"][day]
    plans = []

    def llm(section, facts, title=None):
        plans.append({"section": section, "mode": "llm", "title": section if title is None else title, "facts": facts,
                      "target_chars": target_chars(stats, role, spec["doc_type_cn"], section, rng)})

    def prog(section, text, title=None):
        plans.append({"section": section, "mode": "program", "title": section if title is None else title, "facts": [], "program_text": text})

    syms = "、".join(d["symptoms"])
    ck = chief_key(j)
    treat = j["chief"].get("kind")
    focus_dx = p["primary_dx"].get("underlying") or p["primary_dx"]["name"]
    past = p["past_history"]
    is_child_clean = p["age"] < CHILD_AGE and not past
    if role == "HR":
        prog("主诉", chief_text(j) + "。")
        if treat:
            # 0.1.9：化疗、复查这类住院，现病史写这次来做什么
            hpi = f"入院缘由：{chief_text(j)}" + (f"，目前伴{syms}" if syms else "")
        else:
            hpi = f"主要症状：{syms}，病程{j['chief']['duration']}"
        llm("现病史", [_fact("symptom", hpi, True, ck)] + hpi_workup_facts(v))
        plans[-1]["hint"] = HINT_HPI
        plans[-1]["known_past"] = list(past)
        plans[-1]["child_past"] = is_child_clean
        if is_child_clean:
            llm("既往史", [_fact("past", "既往体健，生长发育同同龄儿", False), _fact("allergy", "否认食物及药物过敏史")])
        else:
            llm("既往史", [_fact("past", past_text(p), bool(past), past[0] if past else None), _fact("allergy", "否认药物过敏史")])
        plans[-1]["known_past"] = list(past)
        plans[-1]["child_past"] = is_child_clean
        prog("个人史", "生于原籍，无特殊嗜好。")
        if p["age"] >= 18:
            prog("婚育史", "适龄结婚，育有子女。" if p["age"] >= 22 else "未婚。")
        if p["sex"] == "女" and 12 <= p["age"]:
            prog("月经史", menses_text(p))
        prog("家族史", "否认家族遗传病史。")
        if syms:
            pe_fact = f"查体有和{syms}相符的阳性体征"
        else:
            pe_fact = "浅表淋巴结未触及肿大，心肺腹查体未见明显异常"
        plans.append({"section": "体格检查", "mode": "llm", "title": "体格检查", "program_prefix": _vital_sentence(d), "facts": [_fact("exam", pe_fact)],
                      "target_chars": target_chars(stats, role, spec["doc_type_cn"], "体格检查", rng), "values": _vital_values(d), "hint": HINT_PE})
        llm("专科情况", [_fact("exam", f"{p['dept']}专科查体，和{'主要症状' if syms else focus_dx}相符")])
        # 0.1.9（审查员第八轮补充）：流程写完体格检查后把它的摘录加进这一节的事实（pe_ref），两节不能写出相反的体征
        plans[-1]["pe_ref"] = True
        plans[-1]["hint"] = HINT_SPEC
        res = [e for e in v["results"]]
        if res:
            prog("辅助检查", "；".join(f"{e['name']}：{exam_result_text(e)}" for e in res) + "。")
        prog("入院诊断", "\n" + _dx_line(_admission_dx(j)))
    elif role == "FP":
        res = [_result_fact(e, False) for e in v["results"]]
        llm("病例特点", [_fact("patient", f"{p['sex']}，{p['age_text']}，因{chief_text(j)}入院", True, ck), _fact("past", past_line(p))]
            + [_fact("vital", _vital_sentence(d), False, None, _vital_values(d))] + res)
        plans[-1]["known_past"] = list(past)
        plans[-1]["child_past"] = is_child_clean
        prog("初步诊断", "\n" + _dx_line(_admission_dx(j)))
        names = "、".join(e["name"] for e in v["results"])
        basis = chief_text(j) + (f"及{names}结果" if names else "")
        llm("拟诊讨论", [_dx_fact(p["primary_dx"]["name"], v["dx"].get(p["primary_dx"]["name"]), True, basis)] + res
            + ([_fact("plan", "待" + "、".join(e["name"] for e in v["pending"]) + "回报后进一步明确")] if v["pending"] else []))
        plans[-1]["hint"] = HINT_DISCUSS
        if j["differentials"]:
            # 0.1.9（审查员第八轮补充）：给写手病人的真实主诉和初步诊断（0.1.8 只给"需要与××鉴别"，写手把脑梗死的症状安在了心衰病人头上）
            # 0.1.11：只给主诉，不再写"初步诊断××"。"初步诊断"是评测台读"疑似"的触发词，拟诊讨论已写明确的主诊断在这里被读回疑似，
            # 出院再读成确诊，诊断变化步数 0.58 → 1.03（pkg10 验收第 2 节，本机复现去掉后回到 0.58）
            llm("鉴别诊断", [_fact("patient", f"患者因{chief_text(j)}入院")]
                + [_fact("dx", f"需要与{n}鉴别：患者目前没有{n}的典型症状和体征，待进一步检查明确", True, n) for n in j["differentials"]])
            plans[-1]["hint"] = HINT_DIFF
        plan = [_fact("plan", "完善相关检查：" + "、".join(e["name"] for e in v["pending"]), False, None)] if v["pending"] else []
        plan += [_fact("med", "用药：" + _med_text(m, True), False, m["name"], _med_values(m)) for m in v["meds_new"]]
        plan += [_fact("plan", f"拟行{a['name']}") for a in op_state(j, spec, specs)[1]]
        llm("诊疗计划", plan or [_fact("plan", "对症治疗，密切观察病情")])
    elif role in COURSE:
        # 0.1.4 按上一篇病程之后的变化写。0.1.3 每篇都列在用药、天天重复待回报的检查，相邻两篇太像；
        # 检查只给当天回报的，病程又不是天天写，中间回报的检查一直进不了病程
        prev = prev_course_day(spec, specs)
        # 0.1.6：重点放最前面；症状有消失的写消失，没变化的写程度；查体只在异常、术后或一半的日子给
        facts = []
        if role == "SUP":
            facts.append(_fact("round", f"今日{ROUND[spec['doc_type_cn']]}查房", True, ROUND[spec["doc_type_cn"]]))
        focus = course_focus(j, spec, specs, rng)
        facts += [_fact("focus", t) for _, t in focus]
        gone = [x for x in j["daily"][prev]["symptoms"] if x not in d["symptoms"]] if prev < day else []
        if gone:
            facts.append(_fact("symptom", f"{'、'.join(gone)}已消失" + (f"，仍有{syms}" if syms else "")))
        elif syms:
            facts.append(_fact("symptom", f"{syms}{TREND_TEXT.get(d['trend'], '')}"))
        # 0.1.9：一般情况按真实病程的比例给（精神睡眠饮食、二便），用事实代替套话
        ng = stats.get("note_general") or {}
        if rng.random() < (ng.get("精神睡眠饮食") or 0):
            facts.append(_fact("general", rng.choice(GENERAL_MENTAL["better" if d["trend"] in ("好转", "减轻") else "same"])))
        # 0.1.11：二便不再给。写手自己就写二便（第十次 A256 0.15），程序再按真实比例加一份，合起来 0.32，超过真实 0.18；
        # 精神睡眠饮食写手写得少，照给。留一次抽签，后面的规划不跟着变
        rng.random()
        abnormal = d["T"] >= 37.3 or d["SBP"] >= 160 or d["SBP"] < 90
        if d["symptoms"] and (abnormal or any(k == "postop" for k, _ in focus) or rng.random() < 0.5):
            # 0.1.9：写"阳性体征较前好转"，不写"咳嗽相关体征"（0.1.8 被原样写出 35 处）；不按症状另配部位表
            facts.append(_fact("exam", f"查体：阳性体征{TREND_TEXT.get(d['trend'], '')}"))
        facts += [_result_fact(e, True) for e in j["exams"] if prev < e["result_day"] <= day]
        # 0.1.9：手术当天术后首程之前的病程算还没做；做了的写麻醉方式
        done, pending = op_state(j, spec, specs)
        facts += [_fact("procedure", _proc_done_phrase(a, "今日"), True, a["name"]) for a in done if a["do_day"] == day]
        for a in done:
            if a["do_day"] < day:
                facts.append(_fact("procedure", _proc_done_phrase(a, "已")))
        facts += [_fact("plan", f"今日拟行{a['name']}，尚未实施" if a["do_day"] == day else f"已决定第{a['do_day']}天行{a['name']}，尚未实施") for a in pending]
        ordered = [e for e in j["exams"] if prev < e["order_day"] <= day < e["result_day"]]
        facts += [_fact("pending", "已开具" + "、".join(e["name"] for e in ordered) + "，待回报")] if ordered else []
        started = [m for m in j["meds"] if prev < m["start_day"] <= day]
        stopped = [m for m in j["meds"] if prev <= m["stop_day"] < day]
        facts += [_fact("med", "加用" + _med_text(m, True), True, m["name"], _med_values(m)) for m in started]
        facts += [_fact("med", "停用" + "、".join(m["name"] for m in stopped))] if stopped else []
        # 0.1.5：在用药按文书序号轮换提两种（加用了药就再提一种），第一种必写。0.1.4 只提一种，药名次数从 11.95 掉到 4.9
        rest = [m for m in v["meds"] if m not in started]
        for n, m in enumerate([rest[(spec["k"] + i) % len(rest)] for i in range(min(len(rest), 1 if started else 2))]):
            with_dose = rng.random() < (stats.get("p_dose_written") or 0.25) * 1.5
            facts.append(_fact("med", "继续" + _med_text(m, with_dose) + "治疗", n == 0 and not started, m["name"], _med_values(m)))
        # 0.1.5：生命体征按真实病程写体征的比例写，体温、血压异常的日子一定写（0.1.4 每篇都由程序写全四项）。
        # 这是为了像真实病程；对相邻 tf-idf 的影响不一定是降：夹具上的模板去掉体征句反而更相像（体征数字每天不同），看 out_diag 的拆项
        nv = stats.get("note_vitals")
        vitals = abnormal or rng.random() < (nv["rate"] if nv else 0.5)
        plans.append({"section": "正文", "mode": "llm", "title": "", "program_prefix": _vital_sentence(d) if vitals else "", "facts": facts, "values": _vital_values(d),
                      "target_chars": target_chars(stats, role, spec["doc_type_cn"], "正文", rng), "copy": {"op": "exact", "budget": 0.0},
                      "focus": [k for k, _ in focus]})
    elif role == "PRE":
        a = next(x for x in j["procedures"] if x["kind"] == "手术")
        pre_susp = v["dx"].get(p["primary_dx"]["name"]) == "疑似"
        llm("简要病情", [_fact("patient", f"{p['sex']}，{p['age_text']}，因{chief_text(j)}入院", True, ck)] + [_result_fact(e, False) for e in v["results"]])
        plans[-1]["copy"] = {"op": "exact", "budget": 0.0}
        prog("术前诊断", "\n" + _dx_line([p["primary_dx"]["name"] + ("？" if pre_susp else "")]))
        llm("手术指征", [_fact("dx", f"{'考虑' if pre_susp else ''}{p['primary_dx']['name']}，有手术指征，无绝对禁忌", True, p["primary_dx"]["name"])])
        prog("拟施手术名称和方式", a["name"] + "。")
        if a.get("anesthesia"):
            prog("拟施麻醉方式", a["anesthesia"] + "。")
        llm("注意事项", [_fact("plan", "术中术后可能的风险，已向家属交代")])
    elif role == "POST":
        a = next(x for x in j["procedures"] if x["kind"] == "手术")
        date = doc_date(j, day)
        prog("手术时间", f"{date} {spec['time'][:2]}时前完成。")
        prog("术中诊断", p["primary_dx"]["name"] + "。")
        prog("手术方式", a["name"] + "。")
        if a.get("anesthesia"):
            prog("麻醉方式", a["anesthesia"] + "。")
        anes_txt = f"麻醉：{a['anesthesia']}，" if a.get("anesthesia") else ""
        llm("手术简要经过", [_fact("procedure", f"手术：{a['name']}，{anes_txt}手术顺利，不写具体数值", True, a["name"])])
        llm("术后处理措施", [_fact("med", "术后用药：" + "、".join(_med_text(m, False) for m in v["meds"][:4]))] if v["meds"] else [_fact("plan", "监测生命体征，对症支持治疗")])
        plans[-1]["copy"] = {"op": "exact", "budget": 0.0}
        llm("术后应当特别注意观察的事项", [_fact("plan", "观察生命体征，" + postop_watch(a, 0))])
    elif role == "DS":
        d0 = j["daily"][0]
        # 0.1.4：0.1.3 这一节只给两条短事实、字数却按真实写，模型补"入院后完善检查"一类过渡句（K4）。现在给足入院当天的事实，
        # 标 admission_only：写手提示只写入院当时，闸门查"入院后"一类说法，允许出现的数值只到第 0 天
        adm = [_fact("patient", f"因{chief_text(j)}入院", True, ck)]
        adm += [_fact("symptom", "入院时症状：" + "、".join(d0["symptoms"]))] if d0["symptoms"] else []
        adm.append(_fact("vital", "入院查体：" + _vital_sentence(d0), False, None, _vital_values(d0)))
        pe = (ctx or {}).get("hr_pe")
        # 0.1.9：拿入院记录实际写出的体格检查当事实（0.1.8 只给"专科查体与主要症状相符"，写手另编了一套，和入院记录矛盾）
        adm.append(_fact("exam", "入院时体格检查：" + pe, False, None, re.findall(r"\d+(?:\.\d+)?", pe)) if pe
                   else _fact("exam", f"{p['dept']}专科查体与主要症状相符"))
        llm("入院情况", adm + [_result_fact(e, False) for e in visible(j, 0)["results"]])
        plans[-1]["admission_only"] = True
        prog("入院诊断", "\n" + _dx_line(_admission_dx(j)))
        # 最后一篇病程之后才回报的检查只有出院记录能写，必写
        last = max([s["day"] for s in specs if s["role"] in COURSE], default=0)
        course = [_fact("result", f"第{e['result_day']}天{e['name']}：{exam_result_text(e)}", e["result_day"] > last, e["name"], [x["value"] for x in e["items"]])
                  for e in sorted(j["exams"], key=lambda x: x["result_day"])]
        # 0.1.5 写日历日期：S4b 读的是诊疗经过里"日期＋行……术"，0.1.4 写"第 N 天行……术"，有手术也测不了
        course += [_fact("procedure", _proc_done_phrase(a, doc_date(j, a["do_day"])), True, a["name"]) for a in j["procedures"]]
        course += [_fact("med", "使用" + _med_text(m, rng.random() < (stats.get("p_dose_written") or 0.25)), False, m["name"], _med_values(m)) for m in j["meds"][:6]]
        changes = [x for x in j["dx_events"] if x["state"] != "疑似" and x["day"] > 0]
        course += [_fact("dx", f"第{x['day']}天{x['name']}{'明确' if x['state'] == '确诊' else '排除'}") for x in changes]
        llm("诊疗经过", course)
        prog("出院诊断", "\n" + _dx_line(final_diagnoses(j)))
        state = f"本次{treat}过程顺利" if treat in ("化疗", "放疗") else f"症状{d['trend']}"
        llm("出院情况", [_fact("symptom", state), _fact("vital", _vital_sentence(d), False, None, _vital_values(d))])
        home = [m for m in j["meds"] if m["discharge"]]
        llm("出院医嘱", [_fact("med", "出院带药：" + _med_text(m, True), True, m["name"], _med_values(m)) for m in home] + [_fact("plan", "注意休息，定期门诊复查")])
    done_ops = op_state(j, spec, specs)[0] if j["surgical"] else ()
    all_no_incision = bool(j["surgical"] and j["procedures"] and all(no_incision(a) for a in j["procedures"]))
    for pl in plans:
        pl["k"], pl["day"] = spec["k"], day
        pl["allowed"] = sorted(_allowed(j, day, pl))
        if pl["mode"] == "llm" and not j["surgical"] and role in ("DS",) + COURSE:
            pl["nonsurgical"] = True
        if pl["mode"] == "llm" and j["surgical"] and not done_ops and role in COURSE + ("PRE",):
            pl["before_op"] = True
        if pl["mode"] == "llm" and all_no_incision:
            pl["no_incision"] = True
    return plans


def _allowed(j, day, plan):
    """这一节正文里允许出现的带单位数值：本节事实 + 当天可见的全部结果和体征。只写入院当时的节只到第 0 天。"""
    vday = 0 if plan.get("admission_only") else (day if plan["section"] not in ("诊疗经过",) else j["los"])
    v = visible(j, vday)
    out = set()
    for f in plan["facts"]:
        out.update(f["values"])
    out.update(str(x) for x in plan.get("values", ()))
    for e in v["results"]:
        out.update(x["value"] for x in e["items"])
    for m in v["meds"]:
        out.update(_med_values(m))
    for dd in j["daily"][:vday + 1]:
        out.update(str(x) for x in _vital_values(dd))
    return out


def assemble(j, spec, specs, plans, texts):
    """header + 每节的标题和正文。texts：{section: 写好的文字}。"""
    parts = [header(j, spec, specs)]
    for pl in plans:
        body = pl.get("program_text") if pl["mode"] == "program" else (pl.get("program_prefix", "") + texts.get(pl["section"], ""))
        if pl["title"]:
            sep = "" if body.startswith("\n") else ""
            parts.append(f"{pl['title']}：{sep}{body}".rstrip() + "\n")
        else:
            parts.append(body.rstrip() + "\n")
    return "".join(parts)


def active_med_nums(j, day):
    """当天仍在使用的药名和对应剂量/频次数字集合，供 stable_sentences/apply_copy 放行在用药物剂量句。"""
    active = [m for m in j.get("meds") or () if m["start_day"] <= day <= m["stop_day"]]
    names = [m["name"] for m in active if m.get("name")]
    nums = {x for m in active for x in re.findall(r"\d+(?:\.\d+)?", f"{m.get('dose', '')} {m.get('freq', '')}")}
    return names, nums


def _digit_ok(s, allowed_nums):
    if not _DIGIT.search(s):
        return True
    if not allowed_nums or _UNSAFE_NUM.search(s):
        return False
    med_names, dose_nums = allowed_nums
    if not med_names or not dose_nums or not any(m in s for m in med_names):
        return False
    found = re.findall(r"\d+(?:\.\d+)?", s)
    return bool(found) and set(found).issubset(dose_nums)


def stable_sentences(text, banned=(), allowed_nums=None):
    """上一篇里能原样拷的句子：够长、不带不安全数字（在用药剂量除外）、没有日期和计划口吻；banned 里的词不能出现。
    0.1.9：病情趋势句也不拷（0.21 放行“以改善/以缓解”目的句）。0.1.9.1：加用、停用等用药动作句不拷。"""
    out = []
    for m in _SENT.finditer(text):
        s = m.group(0).strip()
        if (len(s) >= 8 and _digit_ok(s, allowed_nums) and not _PLAN_WORD.search(s) and "查房" not in s and not _SENIOR.search(s)
                and not _COPY_TREND.search(s) and not _COPY_ACTION.search(s) and not any(b and b in s for b in banned)):
            out.append(s)
    return out


def copy_banned(j, day):
    """拷贝时不能出现的词：今天已经不用的药，和这次住院的全部症状词（0.1.9：症状句不拷，症状的变化只由当天的事实写）。"""
    syms = {s for d in j["daily"] for s in d["symptoms"]} | set(j["chief"]["symptoms"])
    return inactive_meds(j, day) + sorted(syms)


def trend_clash(sentence, text):
    """拷进来的变化句和这一篇已经写的同一件事打架：变化词前面的几个字（"咳嗽相关体征"）这一篇里已经有了。"""
    m = _TREND.search(sentence)
    subject = sentence[:m.start()][-6:] if m else ""
    return len(subject) >= 2 and subject in text


def inactive_meds(j, day):
    """这一天已经停了或还没开始的药名。0.1.6 会把上一篇的"继续某药治疗"拷进写着"停用某药"的这一篇。"""
    active = {m["name"] for m in j["meds"] if m["start_day"] <= day <= m["stop_day"]}
    return sorted({m["name"] for m in j["meds"]} - active)


def _sent_drop_reason(s, text, banned, med_set, allowed_nums, safe_fn):
    if len(s) < 8:
        return "copy_drop_short"
    if not _digit_ok(s, allowed_nums):
        return "copy_drop_digit"
    if _PLAN_WORD.search(s) or (safe_fn is not None and not safe_fn(s)):
        return "copy_drop_plan_word"
    if "查房" in s or _SENIOR.search(s):
        return "copy_drop_round"
    if _COPY_TREND.search(s):
        return "copy_drop_trend"
    if _COPY_ACTION.search(s):
        return "copy_drop_action"
    if med_set and any(b and b in s for b in med_set):
        return "copy_drop_banned_med"
    if any(b and b in s for b in banned):
        return "copy_drop_banned_sym"
    if s in text:
        return "copy_drop_already_in"
    if trend_clash(s, text):
        return "copy_drop_trend_clash"
    return None


def apply_copy(text, prev_text, budget, rng, banned=(), diag=None, banned_meds=None, allowed_nums=None, safe_fn=None):
    """把上一篇的稳定句子原样拷进这一篇，直到拷进来的字符占比接近预算。程序执行，所以数字和可见性不受影响。返回 (新文字, 拷了多少字)。
    0.2.0（第 9 条）：diag 给了时，按顺序记上一篇每个候选句被哪条过滤挡掉，只出计数，不改抽样。"""
    if not prev_text or budget <= 0 or not text:
        return text, 0
    med_set = set(banned_meds or ())
    if diag is not None:
        diag["copy_pairs"] = diag.get("copy_pairs", 0) + 1
    pool = []
    for m in _SENT.finditer(prev_text):
        s = m.group(0).strip()
        if diag is not None:
            diag["copy_cand_sents"] = diag.get("copy_cand_sents", 0) + 1
        reason = _sent_drop_reason(s, text, banned, med_set, allowed_nums, safe_fn)
        if reason is not None:
            if diag is not None:
                diag[reason] = diag.get(reason, 0) + 1
            continue
        if diag is not None:
            diag["copy_pool_ok"] = diag.get("copy_pool_ok", 0) + 1
        pool.append(s)
    rng.shuffle(pool)
    added, chosen = 0, []
    for s in pool:
        if (added + len(s)) / (len(text) + added + len(s)) > budget:
            if diag is not None:
                diag["copy_drop_budget"] = diag.get("copy_drop_budget", 0) + 1
            continue
        chosen.append(s)
        added += len(s)
        if diag is not None:
            diag["copy_kept_sents"] = diag.get("copy_kept_sents", 0) + 1
    if not chosen:
        return text, 0
    sents = [m.group(0) for m in _SENT.finditer(text)]
    at = max(0, len(sents) - 1)
    return "".join(sents[:at]) + "".join(chosen) + "".join(sents[at:]), added
