"""评委：把一次住院渲染成评委输入；三条逐项问题（QS 状态接续、QK 知识边界、QB 诊断轨迹）和一个笼统打分（JH）；解析。

QS/QK/QB 是我们替对方做的"最强的大模型评委"：直接问、正着问、给定义和例子、要引文。
JH 模仿现有工作里常见的"让模型给时间一致性打 1–5 分"。
提示词由总经理编写，伙计逐字使用。
"""
import json
import re

TYPE_CN = {"入院记录": "入院记录", "首次病程": "首次病程记录", "主任医师查房": "主任医师查房记录",
           "主治医师查房": "主治医师查房记录", "上级医师查房": "上级医师查房记录", "术前小结": "术前小结",
           "术后首次病程": "术后首次病程记录", "出院记录": "出院记录"}


def render_for_judge(adm, only_idx=None):
    """时间轴顺序；每篇一个统一的抬头。only_idx：只渲染这些文书（窗口条件）。真实、合成用同一种格式。"""
    parts = []
    docs = [d for d in adm.docs if only_idx is None or d.idx in only_idx]
    for i, d in enumerate(docs):
        t = TYPE_CN.get(d.subtype, "日常病程记录")
        parts.append(f"【第{i + 1}篇｜住院第{d.day}天｜{t}】\n{d.text.strip()}")
    return "\n\n".join(parts)


_HEAD = "下面是同一次住院按时间顺序排列的若干篇病历文书，每篇开头标明了它是住院第几天写的。\n\n{docs}\n\n"
_TAIL_Q = ('只输出一个 JSON 对象，不要输出其他文字：{{"结论": "有" 或 "无", "第几篇": 数字或 null, '
           '"引文": "能说明问题的一句原文，没有就写空字符串"}}')

Q_PROMPTS = {
    "QK": _HEAD + ("请检查：有没有哪一篇文书写出了它的书写时间点还不可能知道的信息？例如：手术日之前的文书写了术后情况或术中所见；"
                   "取材（手术、活检、穿刺）之前写出了病理结果；某项检查在后面的文书里才安排，前面的文书却已经写出了它的结果；"
                   "出院记录的\"入院情况\"里写了住院以后才得到的检查结果或治疗经过；正文里出现晚于本篇书写日的日期（计划、预约除外）。"
                   "入院以前或急诊、门诊已经做过的检查，入院时写出结果是正常的，不算。\n\n") + _TAIL_Q,
    "QS": _HEAD + ("请检查：这些文书之间，病人的性别、年龄、病变部位的左右侧、手术名称或手术方式（如腹腔镜与开腹）、手术日期、药物过敏史，"
                   "有没有前后不一致、而且文中没有交代原因的地方？年龄相差 1 岁不算。\n\n") + _TAIL_Q,
    "QB": _HEAD + ("请检查：诊断的变化有没有缺少依据或交代的地方？例如：某个诊断从\"考虑、疑似\"变成\"确诊、明确\"，"
                   "但这一篇和之前的文书里都没有任何支持它的新的检查结果或病理；或者已经明确的诊断，后来没有说明就又写成疑似、排除或不再出现。\n\n") + _TAIL_Q,
}
H_PROMPT = _HEAD + ("请就\"时间上的前后一致性\"给这组文书打分：各篇所写的病情、检查、诊断、治疗是否符合时间先后，前后是否衔接、有没有矛盾。\n"
                    "5 分：完全一致；4 分：基本一致，有个别小问题；3 分：有明显的不一致；2 分：多处不一致；1 分：严重混乱。\n\n"
                    '只输出一个 JSON 对象，不要输出其他文字：{{"分数": 1 到 5 的整数, "理由": "一句话"}}')

# 雷 → 应该由哪一条问题看见
MINE_QUESTION = {"S1": "QS", "S3": "QS", "S4a": "QS", "S4b": "QS", "K1": "QK", "K2": "QK", "K3": "QK",
                 "K4": "QK", "K5": "QK", "B2": "QB"}


def q_messages(qid, adm, only_idx=None):
    return [{"role": "user", "content": Q_PROMPTS[qid].format(docs=render_for_judge(adm, only_idx))}]


def h_messages(adm, only_idx=None):
    return [{"role": "user", "content": H_PROMPT.format(docs=render_for_judge(adm, only_idx))}]


def _obj(s):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except json.JSONDecodeError:
            return None
    return None


_YES = {"有", "是", "存在", "yes", "true", "有问题"}
_NO = {"无", "否", "不存在", "没有", "no", "false", "无问题"}


def parse_q(output):
    """返回 {"verdict": True(有问题)/False/None(解析不了), "doc": int|None, "quote": str}。"""
    o = _obj(output or "")
    v, doc, quote = None, None, ""
    if isinstance(o, dict):
        raw = o.get("结论")
        if isinstance(raw, bool):
            v = raw
        elif isinstance(raw, str):
            r = raw.strip().lower()
            v = True if r in _YES else (False if r in _NO else None)
        d = o.get("第几篇")
        doc = d if isinstance(d, int) and not isinstance(d, bool) else None
        q = o.get("引文")
        quote = q if isinstance(q, str) else ""
    if v is None:
        m = re.search(r"结论[\"'”]?\s*[:：]\s*[\"'“]?\s*(有|无)", output or "")
        if m:
            v = m.group(1) == "有"
    return {"verdict": v, "doc": doc, "quote": quote}


def parse_h(output):
    o = _obj(output or "")
    if isinstance(o, dict):
        x = o.get("分数")
        if isinstance(x, (int, float)) and not isinstance(x, bool) and 1 <= x <= 5:
            return int(round(x))
        if isinstance(x, str) and re.fullmatch(r"\s*[1-5]\s*", x):
            return int(x)
    m = re.search(r"分数[\"'”]?\s*[:：]\s*([1-5])", output or "")
    return int(m.group(1)) if m else None


def paired(clean, mined):
    """成对计分。clean、mined 是 parse_q 的 verdict（True=有问题）。
    返回 dict：host_unclean（干净版就判有问题）、upper（解析不了时对评委有利：埋雷版记有、干净版记无）、
    lower（反过来）、point（任何一边解析不了记 None）。前三个是 0/1。"""
    def det(c, m):
        return int(bool(m) and not c)
    up = det(False if clean is None else clean, True if mined is None else mined)
    lo = det(True if clean is None else clean, False if mined is None else mined)
    pt = None if clean is None or mined is None else det(clean, mined)
    return {"host_unclean": int(clean is True), "upper": up, "lower": lo, "point": pt}


def locate_quote(quote, text, min_len=6):
    """引文能不能在原文里找到（去掉空白后的子串，或其中 ≥min_len 字的一段）。返回 (start, end) 或 None。"""
    q = re.sub(r"\s+", "", quote or "")
    if len(q) < min_len:
        return None
    flat, idx = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            flat.append(ch)
            idx.append(i)
    flat = "".join(flat)
    for L in (len(q), min(len(q), 20), min_len):
        for a in range(0, len(q) - L + 1, max(1, L // 2)):
            k = flat.find(q[a:a + L])
            if k >= 0:
                return idx[k], idx[k + L - 1] + 1
    return None
