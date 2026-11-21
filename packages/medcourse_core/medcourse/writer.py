"""L5 写手 v0：分节写，只给当天计划。提示词由总经理编写，伙计逐字使用，要改写进 GAPS。

LLMWriter 接本机模型（chat 或基座的 raw）；TemplateWriter 不调模型，把事实直接连成句子，用来跑通流程和自检。

0.1.9：提示词模板一字没改（--no-filler、--no-filler2 两臂的那一条也没改）；体格检查、现病史的 hint 文字在 compiler 里改了。
clean 多去掉开头的"患者××："。
"""
import re

from .llm import WRITE, Client
from .planner import seed_of

SYSTEM = (
    "你是一名三级甲等医院的住院医师，严格遵守国家卫健委《病历书写基本规范》。\n"
    "文风与行文铁律：\n"
    "1. 采用客观、简洁、省主语的临床书面语，紧扣给定事实展开，严禁编写“综上所述”“积极配合治疗”“心理疏导”“建议关注身心健康”等总结性客服腔。\n"
    "2. 严禁虚构事实中未给出的检查、用药、数值、部位或病史；严禁自己罗列超过 3 项的“无……、无……、无……”阴性症状长串。\n"
    "3. 严禁输出医院信息系统表单括号或免责套话（如“高（ ）低（ ）”“（护理常规……）”“出院记录一式两份”）。\n"
    "【标准临床文风短示例（仅作行文借鉴，内容均为教学示意，严禁照抄其中具体疾病与数值）】\n"
    "示例甲（内科现病史风格）：患者3天前受凉后出现咳嗽、咳黄黏痰，伴发热，体温最高达38.2℃，偶有胸闷。无咯血、胸痛及呼吸困难。起病以来精神、食欲稍差，睡眠可，大小便正常，体重无明显变化。入院前未做过检查。\n"
    "示例乙（外科查房病程风格，注：仅查房记录写医师查房头衔，日常病程记录直叙病情）：今日主治医师查房：术后第2天，患者诉切口疼痛较前减轻，已肛门排气，无腹胀及发热。生命体征平稳，腹软，切口敷料干洁，无渗血渗液，腹腔引流管在位通畅，引出少量淡血性液。继续头孢呋辛抗感染及补液支持，鼓励早期下床活动。"
)

RETRY_GUIDANCE = {
    "postop_day_mismatch": "上稿写错了术后天数：请严格按事实中给出的精确“术后第K天”书写（手术当日禁止写成术后第1天）。",
    "before_op_done": "上稿提前写了已手术或术后切口/引流管状态：当前尚未手术，严禁写术后、切口干洁或引流管通畅。",
    "nonsurg_postop": "患者本次住院未行任何手术，严禁在正文中写术后、切口、伤口清洁或拆线等外科术后内容。",
    "no_incision_leak": "本术式为经自然腔道或穿刺介入无切口操作，正文中严禁出现“切口”或“引流管”字样，请只写术区或穿刺点情况。",
    "senior_in_daily": "本篇为住院医师《日常病程记录》，严禁出现“主任医师/主治医师/上级医师查房”字样。",
    "pe_dup_vitals": "生命体征已由程序写在本节开头，正文中禁止再次写体温/脉搏/呼吸/血压数值或“已记录”字样。",
    "prompt_leak": "上稿出现了提示词元话语或格式符号（如【】、延续情况、本节要点等），请只输出纯正病历正文。",
    "lab_abn_called_normal": "上稿在存在异常化验指标时写了“未见明显异常/正常”：请如实指出异常项及偏高/偏低方向。",
    "result_unsourced": "上稿以结果口吻提到了当天尚未回报的检查：尚未回报的检查只能写“待回报”或“拟行”，不写结果。",
    "dx_settled_early": "上稿对尚处于疑似/待排状态的诊断下了确诊或排除定论：必须使用“考虑”“待排”“拟诊？”等不确定表述。",
    "past_contradict": "上稿与患者已知既往史或年龄相冲突：如已有慢病史严禁写“既往体健”或“否认”，儿童不写成人慢性病否认清单。",
    "fact_missing": "上稿遗漏了事实中的必写关键词或主要症状/术后天数，请确保全部核心事实落入正文。",
    "number_unsourced": "上稿写出了事实未授权的带单位数值，请只使用给定数值或省去未给定数值。",
    "future_date": "上稿出现了晚于当前书写日的非计划性日期，请严格站在书写当日时间视角。",
    "pe_result": "体格检查/专科情况禁止写入实验室化验、心电图或影像学检查结果，只写查体所见。",
    "pe_history": "体格检查/专科情况禁止写入“主诉/自述/既往/否认”等病史词。",
    "admission_leak": "出院记录“入院情况”只能写入院当时所见，严禁出现“入院后”“住院期间”“经治疗后”等词汇。",
    "too_short": "上稿字数明显不足，请围绕给定事实充实客观临床叙述以达到要求长度。",
    "repeat": "上稿出现局部语句重复，请流畅表达、禁止复读相同分句。",
}
_FENCE = re.compile(r"^```\w*\s*|\s*```$")
# 0.1.4：出院记录"入院情况"在出院日写，但只能写入院当时（0.1.3 两条对话版臂 19/20、20/20 写进了住院后的事）
ADMISSION_ONLY = "5. 这一节只写患者入院当时的情况（主诉、入院查体、入院时已有的检查结果），不写入院以后的检查、治疗和病情变化，不用“入院后”“住院期间”“经治疗”这类说法。\n"
# 0.1.6 W-8B 对照臂（--prev-note）：病程给上一篇病程的正文，要求不照搬。只有带 prev_text 的节（日常病程、上级查房）用；不带时提示词和 0.1.5 一字不差
PREV_NOTE = "【上一篇病程】\n{prev}\n"
PREV_RULE = "5. 上一篇病程里写过的句子不要照搬，写今天的新情况；意思相同的话换一种说法。\n"
PREV_CHARS = 600
# 0.1.7 W-8B 对照臂（--no-filler）：第六次 out_diag 里病程最常见的四句（密切观察 45/55、病情平稳 35/55、生命体征平稳 31/55、无特殊不适 28/55）。
# 只有带 no_filler 的节（日常病程、上级查房）用；不带时提示词和 0.1.5 一字不差
NO_FILLER = "5. 只写上面给出的事实。上面没给生命体征，就不写生命体征怎样；不要每篇都写“密切观察”“病情平稳”“无特殊不适”这类笼统的话。\n"
# 0.1.8 对照臂 D（--no-filler2）：C20 里点了名的"无特殊不适"反而从 0.45 升到 0.65（监督员第七轮）。这一版不点任何词的名
NO_FILLER2 = "5. 只写上面给出的事实。上面没给生命体征，就不写生命体征怎样；结尾不写泛泛的总结句。\n"
# 0.21.2 / 0.21.3（S3 回译微调）：数值与日期占位符规则（匹配 ISO/中文年月日/月日，仅当计划中存在 <Nk>/<Dk> 时启用）
PLACEHOLDER_RULE = "6. 事实中的占位符（如<N1>、<D1>）在正文中原样保留，不要自行改成数字。\n"
_UNIT_POW = re.compile(r"(?:[×xX*]\s*)?10\s*\^\s*\d+")
_DATE_TOK = re.compile(
    r"(?<![A-Za-z0-9_.])(?:19|20)\d{2}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}\s*[日号]?"
    r"|(?<![\d年/.\-])\d{1,2}\s*月\s*\d{1,2}\s*[日号]"
)
_NUM_TOK = re.compile(r"(?<![A-Za-z0-9_<^.\d])\d+(?:\.\d+)?(?!\s*\^|[\d.]|型)")
_ANY_PH = re.compile(r"<(?:N|D)\d+>")


def mask_text_with_map(text, dmap, nmap, allow_new=True):
    """按首次出现顺序把日期和数值替换成 <D1>..、<N1>..，保护 ×10^9/L 幂次单位不被拆开。"""
    s = text or ""
    if not s:
        return s
    kept_u = []

    def _keep_u(m):
        kept_u.append(m.group(0))
        return f"\x00U{len(kept_u) - 1}\x00"

    def _sub_d(m):
        v = m.group(0)
        if v not in dmap:
            if not allow_new:
                return v
            dmap[v] = f"<D{len(dmap) + 1}>"
        return dmap[v]

    def _sub_n(m):
        v = m.group(0)
        if v not in nmap:
            if not allow_new:
                return v
            nmap[v] = f"<N{len(nmap) + 1}>"
        return nmap[v]

    s = _UNIT_POW.sub(_keep_u, s)
    s = _DATE_TOK.sub(_sub_d, s)
    s = _NUM_TOK.sub(_sub_n, s)
    return re.sub(r"\x00U(\d+)\x00", lambda m: kept_u[int(m.group(1))], s)


def mask_plan(plan):
    """为 SectionPlan 构造日期/数值占位副本及双向映射表，不改动原 plan。"""
    dmap, nmap = {}, {}
    out = dict(plan)
    out["facts"] = [dict(f, text=mask_text_with_map(f.get("text") or "", dmap, nmap, allow_new=True)) for f in plan.get("facts") or ()]
    if out.get("prev_text"):
        out["prev_text"] = mask_text_with_map(out["prev_text"], dmap, nmap, allow_new=True)
    out["has_placeholders"] = bool(dmap or nmap)
    return out, dmap, nmap


def unmask_text(text, dmap, nmap, strip_unknown=True):
    """把模型输出的 <Dk>/<Nk> 确定性回填为真实日期和数值；strip_unknown=True 时清除未定义占位符。"""
    if not text:
        return ""
    inv = {ph: orig for orig, ph in dmap.items()}
    inv.update({ph: orig for orig, ph in nmap.items()})
    return _ANY_PH.sub(lambda m: inv.get(m.group(0), "" if strip_unknown else m.group(0)), text)


def section_prompt(j, spec, plan):
    p = j["patient"]
    facts = "\n".join(f"- {f['text']}" for f in plan["facts"]) or "- 无特殊"
    prev = plan.get("prev_text")
    return (f"请为一位虚构患者写《{spec['doc_type_cn']}》的“{plan['section']}”一节，书写时间是住院第{spec['day']}天。\n"
            f"【患者】{p['sex']}，{p['age_text']}，{p['dept']}\n"
            + (PREV_NOTE.format(prev=prev[-PREV_CHARS:]) if prev else "")
            + f"【这一节要写的事实】\n{facts}\n"
            "【要求】\n"
            "1. 只写上面给出的事实，不要新增检查、用药和任何数值。\n"
            "2. 上面写到的数值、药名、检查名照抄。\n"
            "3. 站在书写当天的时间点写，不写以后才知道的事。\n"
            f"4. 约 {plan['target_chars']} 字，写连贯的病历文字，不要分条，不要写小标题。\n"
            + (ADMISSION_ONLY if plan.get("admission_only") else "")
            + (PREV_RULE if prev else "")
            + (NO_FILLER if plan.get("no_filler") else "")
            + (NO_FILLER2 if plan.get("no_filler2") else "")
            + (f"5. {plan['hint']}\n" if plan.get("hint") else "")
            + (PLACEHOLDER_RULE if plan.get("has_placeholders") else "")
            + _retry_block(plan.get("retry_fails"))
            + "只输出这一节的正文。")


def _retry_block(retry_fails):
    if not retry_fails:
        return ""
    tips = [RETRY_GUIDANCE[c] for c in dict.fromkeys(retry_fails) if c in RETRY_GUIDANCE]
    return ("【上一次生成校验未通过，本轮必须纠正】\n" + "\n".join(f"- {t}" for t in tips) + "\n") if tips else ""


def raw_prompt(j, spec, plan):
    p = j["patient"]
    facts = "；".join(f["text"] for f in plan["facts"]) or "无特殊"
    when = "（只写入院当时）" if plan.get("admission_only") else ""
    return (f"以下是一份规范的住院病历（虚构）。\n患者{p['sex']}，{p['age_text']}，{p['dept']}。\n"
            f"《{spec['doc_type_cn']}》住院第{spec['day']}天。本节要点{when}：{facts}。\n{plan['section']}：")


def clean(text, section):
    """去掉代码块标记、开头重复小标题及句末标点前后的粘连分隔符（0.21.3）。"""
    s = _FENCE.sub("", (text or "").strip())
    s = re.sub(rf"^\s*(?:患者)?{re.escape(section)}\s*[:：]\s*", "", s)
    s = re.sub(r"(?<=[。！？!?])[，,、；;:：\s]+", "", s)
    s = re.sub(r"[，,、；;]+(?=[。！？!?])", "", s)
    return s.strip()


class LLMWriter:
    def __init__(self, base_url, model, raw=False, num_placeholder=False, splice_copy=False):
        self.client = Client(base_url, model, raw=raw)
        self.raw = raw
        self.num_placeholder = num_placeholder
        self.splice_copy = splice_copy
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0

    def write(self, j, spec, plan, attempt):
        # 0.1.8：事实本身很长（一串化验数值）时按事实长度留足，0.1.7 按目标字数给，病例特点 197 字处截断
        use_plan, dmap, nmap = mask_plan(plan) if self.num_placeholder else (plan, {}, {})
        need = sum(len(f["text"]) for f in use_plan["facts"])
        max_tokens = min(1600, max(int(use_plan["target_chars"] * 1.6), int(need * 1.5)) + 120)
        seed = seed_of(j["journey_id"], spec["k"], plan["section"], attempt) % (2 ** 31)
        if self.raw:
            r = self.client.generate(prompt=raw_prompt(j, spec, use_plan), max_tokens=max_tokens, sampling=WRITE, seed=seed, stop=["\n\n", "【", "《"])
        else:
            r = self.client.generate(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": section_prompt(j, spec, use_plan)}], max_tokens=max_tokens, sampling=WRITE, seed=seed)
        self.calls += 1
        self.prompt_tokens += r["prompt_tokens"]
        self.completion_tokens += r["completion_tokens"]
        self.seconds += r["seconds"]
        out = unmask_text(r["text"], dmap, nmap, strip_unknown=False) if self.num_placeholder else r["text"]
        return clean(out, plan["section"])


class TemplateWriter:
    """不调模型。事实连成句子。不代表写作质量，只用来验流程和闸门。"""
    splice_copy = True

    def __init__(self):
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0

    def write(self, j, spec, plan, attempt):
        self.calls += 1
        out = []
        for f in plan["facts"]:
            t = f["text"].rstrip("。；;")
            out.append(t + "。")
        return "".join(out) or "无特殊。"
