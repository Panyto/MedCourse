"""L5 写手 v0：分节写，只给当天计划。提示词由总经理编写，伙计逐字使用，要改写进 GAPS。

LLMWriter 接本机模型（chat 或基座的 raw）；TemplateWriter 不调模型，把事实直接连成句子，用来跑通流程和自检。

0.1.9：提示词模板一字没改（--no-filler、--no-filler2 两臂的那一条也没改）；体格检查、现病史的 hint 文字在 compiler 里改了。
clean 多去掉开头的"患者××："。
"""
import re

from .llm import WRITE, Client
from .planner import seed_of

SYSTEM = "你是一名三级甲等医院的住院医师，有十年临床经验，熟悉国家卫生健康委员会《病历书写基本规范》，擅长书写规范的住院病历。"
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
            + "只输出这一节的正文。")


def raw_prompt(j, spec, plan):
    p = j["patient"]
    facts = "；".join(f["text"] for f in plan["facts"]) or "无特殊"
    when = "（只写入院当时）" if plan.get("admission_only") else ""
    return (f"以下是一份规范的住院病历（虚构）。\n患者{p['sex']}，{p['age_text']}，{p['dept']}。\n"
            f"《{spec['doc_type_cn']}》住院第{spec['day']}天。本节要点{when}：{facts}。\n{plan['section']}：")


def clean(text, section):
    """去掉代码块标记和开头重复的小标题。0.1.9 连"患者既往史："一起去掉（第八次 6/20 份写成"既往史：患者既往史：……"）。"""
    s = _FENCE.sub("", (text or "").strip())
    s = re.sub(rf"^\s*(?:患者)?{re.escape(section)}\s*[:：]\s*", "", s)
    return s.strip()


class LLMWriter:
    def __init__(self, base_url, model, raw=False):
        self.client = Client(base_url, model, raw=raw)
        self.raw = raw
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0

    def write(self, j, spec, plan, attempt):
        # 0.1.8：事实本身很长（一串化验数值）时按事实长度留足，0.1.7 按目标字数给，病例特点 197 字处截断
        need = sum(len(f["text"]) for f in plan["facts"])
        max_tokens = min(1600, max(int(plan["target_chars"] * 1.6), int(need * 1.5)) + 120)
        seed = seed_of(j["journey_id"], spec["k"], plan["section"], attempt) % (2 ** 31)
        if self.raw:
            r = self.client.generate(prompt=raw_prompt(j, spec, plan), max_tokens=max_tokens, sampling=WRITE, seed=seed, stop=["\n\n", "【", "《"])
        else:
            r = self.client.generate(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": section_prompt(j, spec, plan)}], max_tokens=max_tokens, sampling=WRITE, seed=seed)
        self.calls += 1
        self.prompt_tokens += r["prompt_tokens"]
        self.completion_tokens += r["completion_tokens"]
        self.seconds += r["seconds"]
        return clean(r["text"], plan["section"])


class TemplateWriter:
    """不调模型。事实连成句子。不代表写作质量，只用来验流程和闸门。"""

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
