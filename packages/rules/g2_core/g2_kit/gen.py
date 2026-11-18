"""关2 的五组生成：槽位骨架、事实底稿（提示词 + 程序校验）、可见性控制、逐篇提示词、按底稿算的"泄漏"读数。

五组（4.2 节；G-E 是本轮新加的）：
  G-A 逐篇直接续写：性别、年龄、科室、出院主要诊断 + 前几篇 + "现在写第 N 天的 XX"
  G-B 属性前置：G-A + 入出院日期、住院天数、整次住院的文书安排和手术日
  G-C 全知底稿：G-B + 整份事实底稿
  G-D 底稿 + 可见性控制：只给第 N 天以前（含当天）已发生的事、当天的判断状态和已作的计划；出院记录那一篇给全部
  G-E 只控可见性、没有判断层（NHS 管线式）：同 G-D，但不给"医生判断"一层；出院记录那一篇加给出院诊断
五组的共同要求里都有"站在书写当天的时间点写"，差别只在"每篇能看到什么"。
G-C、G-D、G-E 用同一份底稿。

这里的提示词由总经理编写，伙计逐字使用，不得改写；要改写进 GAPS。
"""
import datetime as dt
import hashlib
import json
import re

from ruler_g2 import diagnoses as dxm
from ruler_g2.events import EXAM_RE, _ALIAS, tone_of
from ruler_g2.readings import K1_RE, PLAN_CLAUSE
from ruler_g2.text import char_ngrams, clause_span, sentence_span

ARMS = ("G-A", "G-B", "G-C", "G-D", "G-E")

SYSTEM = "你是一名三级甲等医院的住院医师，有十年临床经验，熟悉国家卫生健康委员会《病历书写基本规范》，擅长书写规范、完整的住院病历。"

ROLE_TYPE = {"HR": "入院记录", "FP": "首次病程记录", "PRE": "术前小结", "POST": "术后首次病程记录", "DS": "出院记录"}
ROUND_TYPE = {"主任医师查房": "主任医师查房记录", "主治医师查房": "主治医师查房记录", "上级医师查房": "上级医师查房记录"}
MAX_TOKENS = {"入院记录": 2000, "首次病程记录": 1800, "出院记录": 2000, "术前小结": 1000, "术后首次病程记录": 1000,
              "主任医师查房记录": 900, "主治医师查房记录": 900, "上级医师查房记录": 900, "日常病程记录": 900}
DRAFT_MAX_TOKENS = 3000
MAX_MODEL_LEN = 10240


# ---------- 槽位 ----------

def fictional_admit_date(slot_id):
    h = int(hashlib.sha256(slot_id.encode()).hexdigest()[:6], 16)
    return (dt.date(2025, 1, 1) + dt.timedelta(days=h % 300)).isoformat()


def skeleton_from_core(core_adm):
    """真实核心集 → 骨架（只有文书类型和第几天，没有正文）。"""
    out = []
    for k, d in enumerate(core_adm.docs):
        if d.role in ROLE_TYPE:
            t = ROLE_TYPE[d.role]
        elif d.subtype in ROUND_TYPE:
            t = ROUND_TYPE[d.subtype]
        else:
            t = "日常病程记录"
        out.append({"k": k, "role": d.role, "doc_type_cn": t, "day": int(d.day)})
    return out


def make_slot(slot_id, core_adm, sex, age, dept, dx_name, dx_code3):
    sk = skeleton_from_core(core_adm)
    los = sk[-1]["day"]
    post = next((x for x in sk if x["role"] == "POST"), None)
    return {"slot_id": slot_id, "admission_id": core_adm.admission_id, "sex": sex, "age": int(age), "dept": dept,
            "dx_name": dx_name, "dx_code3": dx_code3, "chapter": dx_code3[:1], "los": los,
            "surgical": post is not None, "surgery_day": post["day"] if post else None,
            "admit_date": fictional_admit_date(slot_id), "skeleton": sk}


def doc_date(slot, day):
    return (dt.date.fromisoformat(slot["admit_date"]) + dt.timedelta(days=day)).isoformat()


# ---------- 格式模板 ----------

def template(slot, spec):
    t, day = spec["doc_type_cn"], spec["day"]
    date = doc_date(slot, day)
    sex, age, dept = slot["sex"], slot["age"], slot["dept"]
    if t == "入院记录":
        return (f"入院记录\n姓名：[患者]　性别：{sex}　年龄：{age}岁　科室：{dept}\n入院时间：{date} 10:00\n"
                "主诉：…\n现病史：…\n既往史：…\n个人史：…\n婚育史：…\n月经史：…（男性写\"/\"）\n家族史：…\n"
                "体格检查T：…℃ P：…次/分 R：…次/分 BP：…/…mmHg …\n专科情况：…\n辅助检查：…\n入院诊断：…")
    if t == "首次病程记录":
        return (f"首次病程记录\n{date} 15:00\n姓名：[患者]　年龄：{age}岁　科室：{dept}\n"
                "病例特点：…\n初步诊断：…\n拟诊讨论：…\n鉴别诊断：…\n诊疗计划：…")
    if t == "术前小结":
        return (f"{date} 16:00　术前小结\n简要病情：…\n术前诊断：…\n手术指征：…\n拟施手术名称和方式：…\n"
                "拟施麻醉方式：…\n注意事项：…")
    if t == "术后首次病程记录":
        return (f"{date} 17:00　术后首次病程记录\n手术时间：{date} …\n术中诊断：…\n手术方式：…\n麻醉方式：…\n"
                "手术简要经过：…\n术后处理措施：…\n术后应当特别注意观察的事项：…")
    if t == "出院记录":
        return (f"出院记录\n姓名：[患者]　性别：{sex}　年龄：{age}岁　科室：{dept}\n"
                f"入院时间：{slot['admit_date']} 10:00　出院时间：{date} 10:00　住院天数：{slot['los']}天\n"
                "入院情况：…\n入院诊断：…\n诊疗经过：…\n出院诊断：…\n出院情况：…\n出院医嘱：…")
    if t in ("主任医师查房记录", "主治医师查房记录", "上级医师查房记录"):
        return f"{date} 09:00　{t}\n…"
    return f"{date} 09:30　日常病程记录\n…"


HINT = {"主任医师查房记录": "正文写明查房医师职称、病情变化、查体、检查结果、分析与处理意见。",
        "主治医师查房记录": "正文写明查房医师职称、病情变化、查体、检查结果、分析与处理意见。",
        "上级医师查房记录": "正文写明查房医师职称、病情变化、查体、检查结果、分析与处理意见。",
        "日常病程记录": "正文写病情变化、查体、检查结果和处理意见。"}


# ---------- 事实底稿 ----------

def _schedule_text(slot):
    return "；".join(f"第{x['day']}天 {x['doc_type_cn']}" for x in slot["skeleton"])


def draft_messages(slot):
    los, dx = slot["los"], slot["dx_name"]
    if slot["surgical"]:
        surg = f"本次住院在第{slot['surgery_day']}天手术。"
        rule5 = f"有且只有一项 kind 为\"手术\"的处置，do_day 为 {slot['surgery_day']}，plan_day ≤ {slot['surgery_day']}。"
    else:
        surg = "本次住院不做手术。"
        rule5 = "不安排 kind 为\"手术\"的处置。"
    user = f"""请为下面这位虚构患者设计本次住院的"事实底稿"，供后续逐篇书写病历使用。底稿要符合临床常规，前后自洽。

【患者与住院信息】
性别：{slot['sex']}；年龄：{slot['age']}岁；科室：{slot['dept']}
出院主要诊断：{dx}
住院天数：{los}天（第0天入院，第{los}天出院）
{surg}
本次住院的文书安排：{_schedule_text(slot)}

【底稿内容】只输出一个 JSON 对象，不要输出 JSON 以外的任何文字。字段如下：
- "final_diagnoses"：出院诊断列表，第一条必须是"{dx}"，可以再加 0–3 个合并诊断。
- "presentation"：入院时表现，对象，含 "chief_complaint"（主诉，如"右上腹痛3天"）、"history"（现病史要点）、"past_history"（既往史要点）、"exam"（入院查体要点）。
- "observations"：检查和检验列表，每项含 "id"（o1、o2…）、"name"（检查名称，如"腹部CT"）、"kind"（影像/检验/内镜/病理/功能/其他）、"order_day"（第几天安排）、"result_day"（第几天出结果）、"result"（结果，一句话）。入院前或急诊已经做完的检查，order_day 和 result_day 都写 0。
- "judgments"：医生判断的变化，每项含 "day"、"diagnosis"、"state"（疑似/确诊/排除）、"basis"（依据的检查 id 列表；入院前已确诊的慢性病写 ["既往"]）。第0天要列出初步考虑的诊断，以及至少 2 个需要鉴别的诊断，状态都写"疑似"。
- "actions"：处置列表，每项含 "id"（a1、a2…）、"kind"（手术/操作/治疗/会诊/其他）、"name"、"side"（左/右/双/无）、"plan_day"（第几天决定）、"do_day"（第几天实施）。

【必须满足的规则】程序会逐条检查，不满足会退回重写：
1. 所有天数都是 0 到 {los} 之间的整数；result_day ≥ order_day；do_day ≥ plan_day。
2. 病理检查的 order_day 不早于取材的那项手术或操作的 do_day。
3. 每条"确诊"或"排除"的 basis 至少一项；所引检查的 result_day 不晚于这条判断的 day（"既往"除外）。
4. "{dx}"在第{los}天或更早有一条"确诊"。
5. {rule5}
6. observations 3–15 项，judgments 3–15 条，actions 1–8 项。"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def draft_retry_messages(messages, bad_output, errors):
    return messages + [{"role": "assistant", "content": bad_output},
                       {"role": "user", "content": "上面的底稿有以下问题：\n" + "\n".join(f"- {e}" for e in errors)
                        + "\n请修改后重新输出完整的 JSON 对象，不要输出其他文字。"}]


def parse_json_object(s):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None


O_KINDS = {"影像", "检验", "内镜", "病理", "功能", "其他"}
A_KINDS = {"手术", "操作", "治疗", "会诊", "其他"}
STATES = {"疑似", "确诊", "排除"}


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate_draft(draft, slot):
    """返回错误列表（空表示通过）。"""
    errs = []
    los, dx = slot["los"], slot["dx_name"]
    if not isinstance(draft, dict):
        return ["不是 JSON 对象"]
    for k in ("final_diagnoses", "presentation", "observations", "judgments", "actions"):
        if k not in draft:
            errs.append(f"缺少字段 {k}")
    if errs:
        return errs
    fd = draft["final_diagnoses"]
    if not isinstance(fd, list) or not fd or fd[0] != dx:
        errs.append(f"final_diagnoses 第一条必须是\"{dx}\"")
    pr = draft["presentation"]
    if not isinstance(pr, dict) or not all(isinstance(pr.get(k), str) and pr.get(k).strip()
                                           for k in ("chief_complaint", "history", "past_history", "exam")):
        errs.append("presentation 四个字段都要有内容")
    obs, jud, act = draft["observations"], draft["judgments"], draft["actions"]
    if not isinstance(obs, list) or not 3 <= len(obs) <= 15:
        errs.append("observations 要 3–15 项")
        obs = obs if isinstance(obs, list) else []
    if not isinstance(jud, list) or not 3 <= len(jud) <= 15:
        errs.append("judgments 要 3–15 条")
        jud = jud if isinstance(jud, list) else []
    if not isinstance(act, list) or not 1 <= len(act) <= 8:
        errs.append("actions 要 1–8 项")
        act = act if isinstance(act, list) else []
    oid = {}
    for o in obs:
        if not isinstance(o, dict):
            errs.append("observations 里有不是对象的项")
            continue
        i = o.get("id")
        if not isinstance(i, str) or i in oid:
            errs.append(f"检查 id 缺失或重复：{i}")
            continue
        oid[i] = o
        od, rd = o.get("order_day"), o.get("result_day")
        if not (_int(od) and _int(rd) and 0 <= od <= rd <= los):
            errs.append(f"{i} 的天数不对（要 0 ≤ order_day ≤ result_day ≤ {los}）")
        if o.get("kind") not in O_KINDS:
            errs.append(f"{i} 的 kind 不在 {sorted(O_KINDS)} 里")
        if not isinstance(o.get("result"), str) or not o.get("result").strip():
            errs.append(f"{i} 没有 result")
        if not isinstance(o.get("name"), str) or not o.get("name").strip():
            errs.append(f"{i} 没有 name")
    surg = []
    for a in act:
        if not isinstance(a, dict):
            errs.append("actions 里有不是对象的项")
            continue
        pdy, ddy = a.get("plan_day"), a.get("do_day")
        if not (_int(pdy) and _int(ddy) and 0 <= pdy <= ddy <= los):
            errs.append(f"{a.get('id')} 的天数不对（要 0 ≤ plan_day ≤ do_day ≤ {los}）")
        if a.get("kind") not in A_KINDS:
            errs.append(f"{a.get('id')} 的 kind 不在 {sorted(A_KINDS)} 里")
        if a.get("kind") == "手术":
            surg.append(a)
    if slot["surgical"]:
        if len(surg) != 1 or surg[0].get("do_day") != slot["surgery_day"]:
            errs.append(f"要有且只有一项手术，do_day = {slot['surgery_day']}")
    elif surg:
        errs.append("本次住院不做手术，不能有 kind 为\"手术\"的处置")
    sampling_days = [a.get("do_day") for a in act if isinstance(a, dict) and a.get("kind") in ("手术", "操作") and _int(a.get("do_day"))]
    for i, o in oid.items():
        if o.get("kind") == "病理" and _int(o.get("order_day")):
            if not any(o["order_day"] >= d for d in sampling_days):
                errs.append(f"{i} 是病理，order_day 要不早于某项手术或操作的 do_day")
    confirmed_main = False
    for j in jud:
        if not isinstance(j, dict):
            errs.append("judgments 里有不是对象的项")
            continue
        day, st, basis = j.get("day"), j.get("state"), j.get("basis")
        if not (_int(day) and 0 <= day <= los):
            errs.append(f"判断\"{j.get('diagnosis')}\"的 day 不对")
            continue
        if st not in STATES:
            errs.append(f"判断\"{j.get('diagnosis')}\"的 state 要是疑似/确诊/排除")
            continue
        if st in ("确诊", "排除"):
            if not isinstance(basis, list) or not basis:
                errs.append(f"第{day}天\"{j.get('diagnosis')}\"={st}，basis 不能为空")
                continue
            for b in basis:
                if b == "既往":
                    continue
                o = oid.get(b)
                if o is None:
                    errs.append(f"第{day}天\"{j.get('diagnosis')}\"引了不存在的检查 {b}")
                elif _int(o.get("result_day")) and o["result_day"] > day:
                    errs.append(f"第{day}天\"{j.get('diagnosis')}\"引的 {b} 第{o['result_day']}天才出结果")
        if j.get("diagnosis") == dx and st == "确诊":
            confirmed_main = True
    if not confirmed_main:
        errs.append(f"\"{dx}\"要有一条\"确诊\"")
    day0_susp = [j for j in jud if isinstance(j, dict) and j.get("day") == 0 and j.get("state") == "疑似"]
    if len(day0_susp) < 3:
        errs.append("第0天要列出初步考虑的诊断和至少 2 个鉴别诊断（疑似）")
    return errs


# ---------- 可见性 ----------

def _states_as_of(draft, day):
    cur = {}
    for j in sorted(draft["judgments"], key=lambda x: x["day"]):
        if j["day"] <= day:
            cur[j["diagnosis"]] = j
    return cur


def _o_name(oid_map, ids):
    return "、".join(oid_map[i]["name"] for i in ids if i in oid_map) or "既往"


def render_full(draft):
    oid = {o["id"]: o for o in draft["observations"]}
    lines = ["最终诊断（本次住院出院时确立）：" + "；".join(draft["final_diagnoses"])]
    p = draft["presentation"]
    lines.append(f"入院时表现：主诉：{p['chief_complaint']}；现病史要点：{p['history']}；既往史要点：{p['past_history']}；入院查体：{p['exam']}")
    lines.append("检查与检验：")
    for o in sorted(draft["observations"], key=lambda x: (x["order_day"], x["result_day"])):
        lines.append(f"- {o['name']}：第{o['order_day']}天安排，第{o['result_day']}天出结果；结果：{o['result']}")
    lines.append("医生判断的变化：")
    for j in sorted(draft["judgments"], key=lambda x: x["day"]):
        basis = "" if j["state"] == "疑似" else f"（依据：{_o_name(oid, j.get('basis') or [])}）"
        lines.append(f"- 第{j['day']}天：{j['diagnosis']}={j['state']}{basis}")
    lines.append("处置：")
    for a in sorted(draft["actions"], key=lambda x: (x["plan_day"], x["do_day"])):
        side = "" if a.get("side") in (None, "无") else f"（{a['side']}侧）"
        lines.append(f"- {a['kind']}：{a['name']}{side}，第{a['plan_day']}天决定，第{a['do_day']}天实施")
    return "\n".join(lines)


def render_visible(draft, day, with_judgments=True, final=False):
    oid = {o["id"]: o for o in draft["observations"]}
    lines = []
    if final:
        lines.append("出院诊断：" + "；".join(draft["final_diagnoses"]))
    p = draft["presentation"]
    lines.append(f"入院时表现：主诉：{p['chief_complaint']}；现病史要点：{p['history']}；既往史要点：{p['past_history']}；入院查体：{p['exam']}")
    done = [o for o in draft["observations"] if o["result_day"] <= day]
    pend = [o for o in draft["observations"] if o["order_day"] <= day < o["result_day"]]
    lines.append("已回报的检查：" + ("" if done else "无"))
    for o in sorted(done, key=lambda x: x["result_day"]):
        lines.append(f"- {o['name']}（第{o['result_day']}天回报）：{o['result']}")
    lines.append("已安排、结果未回的检查：" + ("" if pend else "无"))
    for o in sorted(pend, key=lambda x: x["order_day"]):
        lines.append(f"- {o['name']}（第{o['order_day']}天安排）")
    if with_judgments:
        cur = _states_as_of(draft, day)
        lines.append("截至今天的诊断判断：")
        for name, j in cur.items():
            basis = "" if j["state"] == "疑似" else f"（依据：{_o_name(oid, j.get('basis') or [])}）"
            lines.append(f"- {name}={j['state']}{basis}")
    acts_done = [a for a in draft["actions"] if a["do_day"] <= day]
    acts_plan = [a for a in draft["actions"] if a["plan_day"] <= day < a["do_day"]]
    lines.append("已实施的处置：" + ("" if acts_done else "无"))
    for a in acts_done:
        side = "" if a.get("side") in (None, "无") else f"（{a['side']}侧）"
        lines.append(f"- {a['kind']}：{a['name']}{side}（第{a['do_day']}天）")
    lines.append("已决定、尚未实施的处置：" + ("" if acts_plan else "无"))
    for a in acts_plan:
        side = "" if a.get("side") in (None, "无") else f"（{a['side']}侧）"
        lines.append(f"- {a['kind']}：{a['name']}{side}（拟第{a['do_day']}天实施）")
    return "\n".join(lines)


# ---------- 逐篇提示词 ----------

def estimate_tokens(s):
    """保守估计：汉字和全角符号按 1 个 token，其余字符按 0.4 个。真实比例更低（Qwen2.5 中文约 0.7），宁可少放前文也不截断。"""
    cjk = sum(1 for ch in s if ord(ch) > 0x2E80)
    return int(cjk + 0.4 * (len(s) - cjk)) + 8


def render_prior(prior):
    """prior：[{k, doc_type_cn, day, text}]（已写好的前几篇）。"""
    return "\n\n".join(f"【第{i + 1}篇｜住院第{x['day']}天｜{x['doc_type_cn']}】\n{x['text'].strip()}"
                       for i, x in enumerate(prior))


def _attr_block(arm, slot, spec):
    base = f"性别：{slot['sex']}；年龄：{slot['age']}岁；科室：{slot['dept']}"
    if arm == "G-A":
        return f"{base}\n本次住院出院主要诊断：{slot['dx_name']}"
    if arm in ("G-B", "G-C"):
        s = (f"{base}\n本次住院出院主要诊断：{slot['dx_name']}\n入院日期：{slot['admit_date']}；"
             f"出院日期：{doc_date(slot, slot['los'])}；住院天数：{slot['los']}天\n"
             f"本次住院的文书安排：{_schedule_text(slot)}\n")
        s += f"手术：第{slot['surgery_day']}天" if slot["surgical"] else "手术：无"
        return s
    return f"{base}\n入院日期：{slot['admit_date']}"


def doc_messages(arm, slot, k, prior, draft=None, count_tokens=estimate_tokens):
    """返回 (messages, max_tokens, info)。info 记被放下的前文篇数、可见性所用的天数。"""
    spec = slot["skeleton"][k]
    t, day = spec["doc_type_cn"], spec["day"]
    date = doc_date(slot, day)
    is_ds = spec["role"] == "DS"
    blocks = ["【患者信息】\n" + _attr_block(arm, slot, spec)]
    if arm == "G-C":
        blocks.append("【事实底稿（整次住院）】\n" + render_full(draft))
    elif arm == "G-D":
        blocks.append(f"【截至今天（住院第{day}天）已知的情况】\n" + render_visible(draft, day, True, final=is_ds))
    elif arm == "G-E":
        blocks.append(f"【截至今天（住院第{day}天）已知的情况】\n" + render_visible(draft, day, False, final=is_ds))
    head = (f"你正在为一位虚构患者按时间顺序逐篇书写同一次住院的病历。现在请写：住院第{day}天（{date}）的《{t}》。\n\n"
            + "\n\n".join(blocks))
    tail = ("【书写要求】\n"
            f"1. 站在书写当天的时间点写：只写到{date}为止已经发生、已经知道的情况。\n"
            "2. 与前面已写好的文书保持一致。\n"
            "3. 严格按下面的格式只输出这一篇文书的正文，不要输出其他文书，不要加任何说明。"
            + (f"\n4. {HINT[t]}" if t in HINT else "") + "\n\n"
            "【格式】\n" + template(slot, spec))
    max_tokens = MAX_TOKENS[t]
    budget = MAX_MODEL_LEN - max_tokens - 200
    keep = list(range(len(prior)))
    dropped = 0

    def build(idx):
        pr = [prior[i] for i in idx]
        mid = ("\n\n【已写好的前几篇文书】\n" + render_prior(pr)) if pr else "\n\n【已写好的前几篇文书】\n（这是第一篇）"
        if pr and len(pr) < len(prior):
            mid += f"\n（为控制长度，另有 {len(prior) - len(pr)} 篇较早的文书没有列出）"
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": head + mid + "\n\n" + tail}]

    msgs = build(keep)
    while count_tokens(msgs[0]["content"] + msgs[1]["content"]) > budget:
        # 先放下最早的非入院记录；只剩入院记录和最近一篇时，再放下入院记录
        drop = next((i for i in keep[:-1] if prior[i]["doc_type_cn"] != "入院记录"), None)
        if drop is None:
            drop = keep[0] if len(keep) > 1 else None
        if drop is None:
            raise ValueError(f"{slot['slot_id']} 第{k}篇 提示词超长")
        keep.remove(drop)
        dropped += 1
        msgs = build(keep)
    return msgs, max_tokens, {"dropped_prior": dropped, "n_prior": len(prior), "day": day}


def finalize_doc(slot, k, output):
    """模型输出 → 一篇合成文书记录。去掉代码块标记和首尾空白。"""
    spec = slot["skeleton"][k]
    text = re.sub(r"^```\w*\s*|\s*```$", "", output.strip())
    return {"slot_id": slot["slot_id"], "k": k, "role": spec["role"], "doc_type_cn": spec["doc_type_cn"],
            "day": spec["day"], "date": doc_date(slot, spec["day"]), "text": text}


# ---------- 按底稿算的泄漏（只对 G-C、G-D、G-E） ----------

DONE_MARK = re.compile(r"行|已行|术后|术中|顺利|实施|完成")
PLAN_MARK = re.compile(r"拟|计划|准备|择期|明日|明天|将|待|术前|建议|预约|决定")


def _key_of(name):
    m = EXAM_RE.search(name)
    return _ALIAS[m.group(0)] if m else None


def _clean(s):
    return re.sub(r"[\s，,。；;：:、]", "", s)


def oracle_leaks(draft, slot, docs):
    """docs：finalize_doc 的结果列表。返回 (逐篇列表, 住院层汇总)。出院记录不算（那时一切都已发生）。
    O：未回报的检查结果被写出（结果内容 6-gram 覆盖 ≥0.5；或检查名带结果口吻、而当天没有同类已回报的检查）
    A：未实施的手术或操作被写成已做
    B：底稿里还没确诊（或排除）的诊断，被写成确诊（或排除）"""
    obs, acts = draft["observations"], draft["actions"]
    per = []
    for x in docs:
        if x["role"] == "DS":
            continue
        n, text = x["day"], x["text"]
        hit = {"k": x["k"], "O": 0, "A": 0, "B": 0}
        visible_keys = {_key_of(o["name"]) for o in obs if o["result_day"] <= n}
        for o in obs:
            if o["result_day"] <= n:
                continue
            res = _clean(o["result"])
            if len(res) >= 8:
                g = char_ngrams(res, 6)
                if g and len(g & char_ngrams(text, 6)) / len(g) >= 0.5:
                    hit["O"] += 1
                    continue
            key = _key_of(o["name"])
            if key and key not in visible_keys:
                for m in EXAM_RE.finditer(text):
                    if _ALIAS[m.group(0)] != key:
                        continue
                    tone, _ = tone_of(text, m.start(), m.end())
                    if tone == "R":
                        hit["O"] += 1
                        break
        for a in acts:
            if a["do_day"] <= n or a["kind"] not in ("手术", "操作"):
                continue
            core = re.sub(r"^(左|右|双)(侧)?", "", a["name"])
            found = False
            for m in re.finditer(re.escape(core[:6]), text) if len(core) >= 4 else []:
                ca, cb = clause_span(text, m.start(), m.end())
                clause = text[ca:cb]
                if DONE_MARK.search(clause) and not PLAN_MARK.search(clause):
                    found = True
                    break
            if not found and a["kind"] == "手术":
                for m in K1_RE.finditer(text):
                    ca, cb = clause_span(text, m.start(), m.end())
                    if not PLAN_CLAUSE.search(text[ca:cb]):
                        found = True
                        break
            if found:
                hit["A"] += 1
        firsts = {}
        for j in sorted(draft["judgments"], key=lambda z: z["day"]):
            if j["state"] in ("确诊", "排除"):
                firsts.setdefault((j["diagnosis"], j["state"]), j["day"])
        for (name, st), d0 in firsts.items():
            if d0 <= n or len(name) < 2:
                continue
            for m in re.finditer(re.escape(name), text):
                if dxm._context_state(text, m.start(), m.end()) == st:
                    hit["B"] += 1
                    break
        per.append(hit)
    summary = {k: int(any(h[k] for h in per)) for k in ("O", "A", "B")}
    summary["any"] = int(any(summary.values()))
    summary["n_docs"] = len(per)
    return per, summary


# ---------- C3 格式（只查合成病历） ----------

def format_checks(slot, docs):
    """每篇：正文里的日期有没有晚于本篇日期（计划口吻、出院医嘱除外）；出院记录抬头的住院天数和入院时间是否照写。"""
    from ruler_g1.sections import split_sections
    from ruler_g2.readings import K5_PLAN
    from ruler_g2.text import find_dates
    out = []
    for x in docs:
        d0 = dt.date.fromisoformat(x["date"])
        excl = [(sec.start, sec.end) for sec in split_sections(x["text"]) if sec.name == "出院医嘱"]
        bad_date = 0
        for s, e, d in find_dates(x["text"], d0):
            if any(a <= s < b for a, b in excl):
                continue
            ca, cb = clause_span(x["text"], s, e)
            if (d - d0).days >= 1 and not K5_PLAN.search(x["text"][ca:cb]):
                bad_date += 1
        ok_head = True
        if x["role"] == "DS":
            ok_head = (f"住院天数：{slot['los']}天" in x["text"].replace(" ", "")) and (slot["admit_date"] in x["text"])
        out.append({"k": x["k"], "future_dates": bad_date, "header_ok": ok_head})
    return out
