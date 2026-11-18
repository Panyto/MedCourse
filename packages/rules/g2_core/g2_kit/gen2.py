"""新写法 G-E′（g2-core 0.4 新增）：增量状态驱动，不喂前几篇全文。

五组（gen.py）共有的问题：相邻病程 tf-idf 0.68～0.71，真实 0.23。五组都把前几篇全文放进提示词。
G-E′ 每篇只给：
  1. 患者信息；
  2. 目前病情（截至前一天）：诊断现状、此前已回报的检查、进行中的处置、仍在等的检查、已定未做的处置，都用自然语言短句，
     不用"=疑似（依据：…）"那种格式（G-D 把它抄进了 15/189 次住院的正文）；
  3. 今天的新情况：今天回报的检查、新安排的检查、实施的处置、决定的处置、诊断变化；
  4. 上一篇文书末尾的处理意见（程序截取，不调模型）。
当天回报的检查和当天实施的手术、操作必须写进正文，缺了就重采样，最多再采 2 次。
时刻不再写死：按文书类型从合理区间里抽（每个槽位固定种子），同一天内按文书顺序递增。

消融组 G-E′+前文（ARM id "G-E2P"）：G-E′ 的全部内容，再加前几篇全文（和五组一样，超长时先放下最早的）。
用来回答"抄写是不是喂前文全文造成的"。

这里的提示词由总经理编写，伙计逐字使用，不得改写；要改写进 GAPS。
"""
import hashlib
import random
import re

from ruler_g2.events import EXAM_RE, _ALIAS
from ruler_g2.text import char_ngrams

from g2_kit.gen import (HINT, MAX_MODEL_LEN, MAX_TOKENS, SYSTEM, _clean, _key_of, _states_as_of, doc_date,
                        estimate_tokens, render_prior)

ARMS2 = ("G-E2", "G-E2P")
ARM_NAME = {"G-E2": "G-E′", "G-E2P": "G-E′+前文"}
MAX_RESAMPLE = 2
VERSION = "gen2-1.0"

# ---------- 槽位过滤 ----------

PERINATAL_NAME = re.compile(r"新生儿|早产儿|围产期|围生期|胎粪|脐炎")
FEMALE_NAME = re.compile(r"子宫|卵巢|输卵管|宫颈|阴道|外阴|妊娠|分娩|产后|产褥|胎膜|胎盘|胎位|臀先露|月经|盆腔炎|女性盆腔|剖宫产|流产|异位妊娠|葡萄胎")
MALE_NAME = re.compile(r"前列腺|睾丸|附睾|阴囊|精索|阴茎|包皮|鞘膜积液|男性不育")
FEMALE_CODE = re.compile(r"^(?:O\d\d|N7\d|N8\d|N9[0-8]|C5[1-8]|D2[5-8]|D06|D39)")
MALE_CODE = re.compile(r"^(?:N4[0-9]|N5[01]|C6[0-3]|D29|D40)")


def slot_filter(slot):
    """返回槽位不合格的原因列表（空表示合格）。合格与否对所有组一起用：不合格的槽位在所有组的比较里都去掉。"""
    why = []
    dx, code, age, sex = slot["dx_name"], slot.get("dx_code3") or "", slot["age"], slot["sex"]
    # O 章（妊娠、分娩）的"胎儿宫内窘迫"是母亲的诊断，不算围产期
    if (code.startswith("P") or (PERINATAL_NAME.search(dx) and not code.startswith("O"))) and age > 1:
        why.append(f"围产期或新生儿诊断配 {age} 岁")
    if (FEMALE_CODE.match(code) or FEMALE_NAME.search(dx)) and sex != "女":
        why.append(f"女性专属诊断配{sex}性")
    if (MALE_CODE.match(code) or MALE_NAME.search(dx)) and sex != "男":
        why.append(f"男性专属诊断配{sex}性")
    if code.startswith("O") and not 12 <= age <= 55:
        why.append(f"妊娠相关诊断配 {age} 岁")
    return why


# ---------- 时刻 ----------

# 各类文书的时刻区间（分钟）
TIME_RANGE = {"入院记录": (8 * 60, 16 * 60), "主任医师查房记录": (8 * 60, 10 * 60 + 30),
              "主治医师查房记录": (8 * 60, 10 * 60 + 30), "上级医师查房记录": (8 * 60, 10 * 60 + 30),
              "日常病程记录": (8 * 60 + 30, 16 * 60 + 30), "术前小结": (14 * 60, 18 * 60),
              "术后首次病程记录": (11 * 60, 20 * 60), "出院记录": (9 * 60, 11 * 60 + 30)}
FP_AFTER_HR = (60, 300)   # 首次病程在入院后 1～5 小时（规范要求 8 小时内）
SAME_DAY_GAP = 15
LATEST = 23 * 60 + 50


def _hhmm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def doc_times(slot):
    """{k: "HH:MM"}。每个槽位固定种子；同一天内按文书顺序严格递增（至少隔 15 分钟）。"""
    rng = random.Random(int(hashlib.sha256(f"{slot['slot_id']}|times".encode()).hexdigest()[:12], 16))
    out, last_day, last, hr_t = {}, None, -1, None
    for spec in slot["skeleton"]:
        t, day = spec["doc_type_cn"], spec["day"]
        if t == "首次病程记录" and hr_t is not None:
            lo, hi = hr_t + FP_AFTER_HR[0], hr_t + FP_AFTER_HR[1]
        else:
            lo, hi = TIME_RANGE.get(t, TIME_RANGE["日常病程记录"])
        m = lo + 5 * rng.randrange((hi - lo) // 5 + 1)
        if day == last_day and m < last + SAME_DAY_GAP:
            m = last + SAME_DAY_GAP
        m = min(m, LATEST)
        if t == "入院记录":
            hr_t = m
        out[spec["k"]] = _hhmm(m)
        last_day, last = day, m
    return out


def template2(slot, spec, times):
    """同 gen.template，只是时刻来自 doc_times。出院记录的入院时间与入院记录一致。"""
    t, day, k = spec["doc_type_cn"], spec["day"], spec["k"]
    date, hm = doc_date(slot, day), times[k]
    sex, age, dept = slot["sex"], slot["age"], slot["dept"]
    hr_k = next((x["k"] for x in slot["skeleton"] if x["doc_type_cn"] == "入院记录"), None)
    hr_hm = times.get(hr_k, "10:00")
    if t == "入院记录":
        return (f"入院记录\n姓名：[患者]　性别：{sex}　年龄：{age}岁　科室：{dept}\n入院时间：{date} {hm}\n"
                "主诉：…\n现病史：…\n既往史：…\n个人史：…\n婚育史：…\n月经史：…（男性写\"/\"）\n家族史：…\n"
                "体格检查T：…℃ P：…次/分 R：…次/分 BP：…/…mmHg …\n专科情况：…\n辅助检查：…\n入院诊断：…")
    if t == "首次病程记录":
        return (f"首次病程记录\n{date} {hm}\n姓名：[患者]　年龄：{age}岁　科室：{dept}\n"
                "病例特点：…\n初步诊断：…\n拟诊讨论：…\n鉴别诊断：…\n诊疗计划：…")
    if t == "术前小结":
        return (f"{date} {hm}　术前小结\n简要病情：…\n术前诊断：…\n手术指征：…\n拟施手术名称和方式：…\n"
                "拟施麻醉方式：…\n注意事项：…")
    if t == "术后首次病程记录":
        return (f"{date} {hm}　术后首次病程记录\n手术时间：{date} …\n术中诊断：…\n手术方式：…\n麻醉方式：…\n"
                "手术简要经过：…\n术后处理措施：…\n术后应当特别注意观察的事项：…")
    if t == "出院记录":
        return (f"出院记录\n姓名：[患者]　性别：{sex}　年龄：{age}岁　科室：{dept}\n"
                f"入院时间：{slot['admit_date']} {hr_hm}　出院时间：{date} {hm}　住院天数：{slot['los']}天\n"
                "入院情况：…\n入院诊断：…\n诊疗经过：…\n出院诊断：…\n出院情况：…\n出院医嘱：…")
    if t in ("主任医师查房记录", "主治医师查房记录", "上级医师查房记录"):
        return f"{date} {hm}　{t}\n…"
    return f"{date} {hm}　日常病程记录\n…"


HEADER_TIME = [re.compile(r"入院时间[:：]\s*\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2})"),
               re.compile(r"^\s*首次病程记录\s*\n\s*\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2})"),
               re.compile(r"出院时间[:：]\s*\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2})"),
               re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s+(\d{2}:\d{2})")]


def header_time(doc_type_cn, text):
    """从正文抬头读出时刻；读不到返回 None。"""
    i = {"入院记录": 0, "首次病程记录": 1, "出院记录": 2}.get(doc_type_cn, 3)
    m = HEADER_TIME[i].search(text[:300])
    return m.group(1) if m else None


# ---------- 状态与增量 ----------

def _names(oid, ids):
    ns = [oid[i]["name"] for i in ids if i in oid]
    return "、".join(ns)


def _dx_sentence(j, oid):
    name, st = j["diagnosis"], j["state"]
    basis = j.get("basis") or []
    if st == "疑似":
        return f"考虑{name}，尚未明确。"
    if basis == ["既往"]:
        return f"{name}为既往已确诊的疾病。"
    by = _names(oid, basis)
    by = f"（根据{by}结果）" if by else ""
    return f"{name}已于第{j['day']}天{'明确' if st == '确诊' else '排除'}{by}。"


def _side(a):
    return "" if a.get("side") in (None, "无") else f"（{a['side']}侧）"


def presentation_text(draft, full=True):
    p = draft["presentation"]
    if not full:
        return f"主诉：{p['chief_complaint']}"
    return (f"主诉：{p['chief_complaint']}\n现病史要点：{p['history']}\n既往史要点：{p['past_history']}\n"
            f"入院查体：{p['exam']}")


def render_state(draft, day, max_prev_results=8):
    """截至前一天（day−1）的病情。第 0 天没有"前一天"，返回空串。"""
    if day <= 0:
        return ""
    oid = {o["id"]: o for o in draft["observations"]}
    lines = [f"【目前病情（截至住院第{day - 1}天）】"]
    cur = _states_as_of(draft, day - 1)
    if cur:
        lines.append("诊断：" + "".join(_dx_sentence(j, oid) for j in cur.values()))
    prev = sorted((o for o in draft["observations"] if o["result_day"] < day), key=lambda x: x["result_day"])
    if prev:
        lines.append("此前已回报的检查（供你保持前后一致，正文不必逐条重抄）：")
        for o in prev[-max_prev_results:]:
            lines.append(f"- {o['name']}（第{o['result_day']}天）：{o['result']}")
    acts = sorted((a for a in draft["actions"] if a["do_day"] < day), key=lambda x: x["do_day"])
    if acts:
        lines.append("已实施的处置：")
        for a in acts:
            if a["kind"] in ("手术", "操作"):
                extra = f"，今天为术后第{day - a['do_day']}天" if a["kind"] == "手术" else ""
                lines.append(f"- 第{a['do_day']}天已行{a['name']}{_side(a)}{extra}")
            else:
                lines.append(f"- 第{a['do_day']}天起：{a['name']}{_side(a)}")
    pend = [o for o in draft["observations"] if o["order_day"] < day < o["result_day"]]
    if pend:
        lines.append("仍在等待结果的检查：" + "、".join(f"{o['name']}（第{o['order_day']}天安排）" for o in pend))
    plan = [a for a in draft["actions"] if a["plan_day"] < day < a["do_day"]]
    if plan:
        lines.append("已决定、尚未实施的处置：" + "、".join(f"{a['name']}{_side(a)}（拟第{a['do_day']}天）" for a in plan))
    return "\n".join(lines)


def render_today(draft, day, before_surgery=False):
    """before_surgery：这篇写在当天手术之前（同一天后面还有术后首次病程），当天的手术写成"今天拟行"。"""
    oid = {o["id"]: o for o in draft["observations"]}
    lines = [f"【今天（住院第{day}天）的新情况】"]
    back = [o for o in draft["observations"] if o["result_day"] == day]
    new = [o for o in draft["observations"] if o["order_day"] == day < o["result_day"]]
    done = [a for a in draft["actions"] if a["do_day"] == day and not (before_surgery and a["kind"] == "手术")]
    later = [a for a in draft["actions"] if a["do_day"] == day and before_surgery and a["kind"] == "手术"]
    decided = [a for a in draft["actions"] if a["plan_day"] == day < a["do_day"]]
    jud = [j for j in sorted(draft["judgments"], key=lambda x: x["day"]) if j["day"] == day]
    if back:
        lines.append("今天回报的检查：")
        lines += [f"- {o['name']}：{o['result']}" for o in back]
    if new:
        lines.append("今天新安排、结果未回的检查：" + "、".join(o["name"] for o in new))
    if done:
        lines.append("今天实施的处置：" + "、".join(a["name"] + _side(a) for a in done))
    if later:
        lines.append("今天拟行的手术（本篇写在手术之前）：" + "、".join(a["name"] + _side(a) for a in later))
    if decided:
        lines.append("今天决定的处置：" + "、".join(f"{a['name']}{_side(a)}（拟第{a['do_day']}天实施）" for a in decided))
    if jud:
        if day == 0:
            susp = [j["diagnosis"] for j in jud if j["state"] == "疑似"]
            rest = [j for j in jud if j["state"] != "疑似"]
            s = ""
            if susp:
                s += f"入院初步考虑{susp[0]}" + (f"，需要鉴别{'、'.join(susp[1:])}。" if len(susp) > 1 else "。")
            s += "".join(_dx_sentence(j, oid) for j in rest)
            lines.append("诊断：" + s)
        else:
            lines.append("诊断变化：" + "".join(_dx_sentence(j, oid) for j in jud))
    if len(lines) == 1:
        lines.append("今天没有新回报的检查，没有新的处置，诊断没有变化。")
    return "\n".join(lines)


def render_course(draft):
    """出院记录用：整次住院经过（出院时一切都已发生）。"""
    oid = {o["id"]: o for o in draft["observations"]}
    lines = ["【整次住院经过】", "出院诊断：" + "；".join(draft["final_diagnoses"]), "检查（按回报日）："]
    for o in sorted(draft["observations"], key=lambda x: x["result_day"]):
        lines.append(f"- {o['name']}（第{o['result_day']}天回报）：{o['result']}")
    lines.append("处置：")
    for a in sorted(draft["actions"], key=lambda x: x["do_day"]):
        lines.append(f"- 第{a['do_day']}天：{a['name']}{_side(a)}")
    jud = sorted(draft["judgments"], key=lambda x: x["day"])
    susp0 = [j["diagnosis"] for j in jud if j["day"] == 0 and j["state"] == "疑似"]
    s = (f"入院时考虑{susp0[0]}" + (f"，需要鉴别{'、'.join(susp0[1:])}。" if len(susp0) > 1 else "。")) if susp0 else ""
    s += "".join(_dx_sentence(j, oid) for j in jud if j["state"] != "疑似")
    lines.append("诊断经过：" + s)
    return "\n".join(lines)


# ---------- 上一篇的处理意见 ----------

PLAN_HEAD = re.compile(
    r"(?:^|\n)[ \t　#*＊]*(?:[一二三四五六七八九十]+[、.．]|\d{1,2}[、.．])?[ \t　*＊]*"
    r"(?:下一步诊疗计划|诊疗计划|分析与处理意见|处理意见|术后处理措施|处理措施|治疗措施|诊疗意见|注意事项|出院医嘱|入院诊断|初步诊断)"
    r"[ \t　*＊]*[:：]?")
PLAN_MAX = 300


def plan_excerpt(prev_text):
    """上一篇末尾的处理意见：取最后一个计划类标题之后的内容；没有就取最后 200 字。压掉多余空白，截到 300 字以内的句末。"""
    heads = list(PLAN_HEAD.finditer(prev_text))
    s = prev_text[heads[-1].end():] if heads else prev_text[-200:]
    s = re.sub(r"[*＊#]+", "", s)
    s = re.sub(r"[ \t　]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s).strip()
    if len(s) > PLAN_MAX:
        cut = max(s.rfind(ch, 0, PLAN_MAX) for ch in "。；;\n")
        s = s[:cut + 1] if cut > 50 else s[:PLAN_MAX]
    return s


# ---------- 逐篇提示词 ----------

def doc_messages2(arm, slot, k, prior, draft, times, count_tokens=estimate_tokens):
    """返回 (messages, max_tokens, info)。prior：已写好的前几篇 [{k, doc_type_cn, day, text}]。
    G-E2 只用 prior 的最后一篇截取处理意见；G-E2P 另把 prior 全文放进去。"""
    assert arm in ARMS2
    spec = slot["skeleton"][k]
    t, day = spec["doc_type_cn"], spec["day"]
    date = doc_date(slot, day)
    is_ds = spec["role"] == "DS"
    blocks = ["【患者信息】\n" f"性别：{slot['sex']}；年龄：{slot['age']}岁；科室：{slot['dept']}\n入院日期：{slot['admit_date']}"]
    full_pres = t in ("入院记录", "首次病程记录", "术前小结", "出院记录")
    blocks.append("【入院时情况】\n" + presentation_text(draft, full_pres))
    if is_ds:
        blocks.append(render_course(draft))
    else:
        st = render_state(draft, day)
        if st:
            blocks.append(st)
        later_post = any(x["role"] == "POST" and x["day"] == day and x["k"] > k for x in slot["skeleton"])
        blocks.append(render_today(draft, day, before_surgery=later_post))
    if prior:
        p = prior[-1]
        what = "的入院诊断" if p["doc_type_cn"] == "入院记录" else "末尾的处理意见"
        blocks.append(f"【上一篇文书（住院第{p['day']}天《{p['doc_type_cn']}》）{what}】\n" + plan_excerpt(p["text"]))
    head = (f"你正在为一位虚构患者按时间顺序逐篇书写同一次住院的病历。现在请写：住院第{day}天（{date}）的《{t}》。\n\n"
            + "\n\n".join(blocks))
    reqs = [f"站在书写当天的时间点写：只写到{date}为止已经发生、已经知道的情况。"]
    if not is_ds and t not in ("入院记录", "首次病程记录"):
        reqs.append("正文围绕今天的新情况写；此前已回报的检查和入院时的病史不要逐条重抄，需要时用一句话带过。")
    if not is_ds and t not in ("入院记录", "首次病程记录"):
        reqs.append("和上一篇的处理意见衔接：写明计划中的事今天做了没有、结果如何。")
    reqs.append("今天回报的检查结果和今天实施的手术、操作，正文里必须写到。" if not is_ds
                else "诊疗经过要写到全部手术、操作和主要检查结果。")
    reqs.append("严格按下面的格式只输出这一篇文书的正文，不要输出其他文书，不要加任何说明，抬头的日期和时刻照写。")
    if t in HINT:
        reqs.append(HINT[t])
    tail = "【书写要求】\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(reqs, 1)) + "\n\n【格式】\n" + template2(slot, spec, times)
    max_tokens = MAX_TOKENS[t]
    info = {"day": day, "n_prior": len(prior), "dropped_prior": 0}
    if arm == "G-E2":
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + "\n\n" + tail}]
        if count_tokens(msgs[0]["content"] + msgs[1]["content"]) > MAX_MODEL_LEN - max_tokens - 200:
            raise ValueError(f"{slot['slot_id']} 第{k}篇 提示词超长")
        return msgs, max_tokens, info
    # G-E2P：加前几篇全文，超长时先放下最早的非入院记录
    budget = MAX_MODEL_LEN - max_tokens - 200
    keep = list(range(len(prior)))

    def build(idx):
        pr = [prior[i] for i in idx]
        mid = ("\n\n【已写好的前几篇文书】\n" + render_prior(pr)) if pr else "\n\n【已写好的前几篇文书】\n（这是第一篇）"
        if pr and len(pr) < len(prior):
            mid += f"\n（为控制长度，另有 {len(prior) - len(pr)} 篇较早的文书没有列出）"
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + mid + "\n\n" + tail}]

    msgs = build(keep)
    while count_tokens(msgs[0]["content"] + msgs[1]["content"]) > budget:
        drop = next((i for i in keep[:-1] if prior[i]["doc_type_cn"] != "入院记录"), None)
        if drop is None:
            drop = keep[0] if len(keep) > 1 else None
        if drop is None:
            raise ValueError(f"{slot['slot_id']} 第{k}篇 提示词超长")
        keep.remove(drop)
        info["dropped_prior"] += 1
        msgs = build(keep)
    return msgs, max_tokens, info


def seed_for(slot_id, arm, k, attempt):
    """每次请求的随机种子：槽位、组、篇序号、第几次采样决定。传给 vLLM 的 seed 参数。"""
    return int(hashlib.sha256(f"{slot_id}|{arm}|{k}|{attempt}".encode()).hexdigest()[:8], 16)


# ---------- 覆盖检查 ----------

APPROACH = re.compile(r"^(?:左|右|双)?(?:侧)?(?:腹腔镜下?|宫腔镜下?|胸腔镜下?|关节镜下?|膀胱镜下?|输尿管镜下?|内镜下|内窥镜下?|显微镜下|"
                      r"经皮(?:肝)?(?:穿刺)?|经尿道|经阴道|经腹|经口|开放性?|微创|介入)?")


def _obs_present(o, text):
    key = _key_of(o["name"])
    if key and any(_ALIAS[m.group(0)] == key for m in EXAM_RE.finditer(text)):
        return True
    name = _clean(o["name"])
    return len(name) >= 2 and name[:4] in _clean(text)


def _act_present(a, text):
    t = _clean(text)
    for core in {re.sub(r"^(左|右|双)(侧)?", "", a["name"]), APPROACH.sub("", a["name"])}:
        core = _clean(core)
        if len(core) >= 2 and core[:4] in t:
            return True
    g = char_ngrams(_clean(APPROACH.sub("", a["name"])), 3)
    return bool(g) and len(g & char_ngrams(t, 3)) / len(g) >= 0.6


def coverage_missing(draft, slot, k, text):
    """这篇必须写到而没写到的项目。非出院记录：当天回报的检查、当天实施的手术和操作。
    出院记录：全部手术和操作、出院主要诊断。返回 ["检查:腹部CT", "处置:…", "诊断:…"]。"""
    spec = slot["skeleton"][k]
    day = spec["day"]
    miss = []
    if spec["role"] == "DS":
        for a in draft["actions"]:
            if a["kind"] in ("手术", "操作") and not _act_present(a, text):
                miss.append("处置:" + a["name"])
        dx = draft["final_diagnoses"][0]
        if _clean(dx)[:4] not in _clean(text):
            miss.append("诊断:" + dx)
        return miss
    for o in draft["observations"]:
        if o["result_day"] == day and not _obs_present(o, text):
            miss.append("检查:" + o["name"])
    for a in draft["actions"]:
        if a["do_day"] == day and a["kind"] in ("手术", "操作") and not _act_present(a, text):
            miss.append("处置:" + a["name"])
    return miss


def pick_attempt(attempts):
    """attempts：[(text, missing_list)]，按采样顺序。取第一个全覆盖的；都不全就取缺得最少的（并列取最早的）。返回下标。"""
    for i, (_, miss) in enumerate(attempts):
        if not miss:
            return i
    return min(range(len(attempts)), key=lambda i: (len(attempts[i][1]), i))


# ---------- 格式检查 ----------

STATE_MARKERS = ("今天的新情况", "目前病情（截至", "此前已回报的检查", "上一篇文书", "供你保持", "不必逐条重抄",
                 "今天新安排、结果未回", "今天决定的处置", "本篇写在手术之前", "整次住院经过", "尚未明确。")


def state_leak(text):
    """提示词里状态表的字样被抄进正文：返回命中的字样列表。"""
    return [m for m in STATE_MARKERS if m in text]


def format_checks2(slot, docs, times):
    """每篇：抬头时刻是否照写；同一天内（按文书顺序）时刻是否倒挂；状态表字样是否外泄。"""
    out, last = [], {}
    for x in sorted(docs, key=lambda z: z["k"]):
        hm = header_time(x["doc_type_cn"], x["text"])
        inv = False
        if hm is not None:
            prev = last.get(x["day"])
            inv = prev is not None and hm < prev
            last[x["day"]] = hm
        out.append({"k": x["k"], "time_copied": hm == times[x["k"]], "same_day_inversion": inv,
                    "state_leak": state_leak(x["text"])})
    return out
