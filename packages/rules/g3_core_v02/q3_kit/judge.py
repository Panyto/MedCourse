"""关3 评委（H3-2）：LLM-CARe 自己的 5 条"医学正确性"标准，加关1 G1-E 的 C14，逐条问 32B。

提示词、标准原文、解析、采样参数都照搬已冻结的代码，不改一个字：
- JCARE_PROMPT、parse_jcare：关1 G1-D 的 g1d_j32_common.py；
- json_dicts、parse_p2（认 32B 常写错的键 "Meats Requirement"）：关1 G1-E 的 g1e_judge.py；
- K13–K17：LLM-CARe 公开代码 llmcare.py 的 "correctness" 五条（K13 与 G1-D 的 C13 同文）；
- X14：G1-E 的 C14；
- 渲染：render7.render_summary7（与 G1-D、G1-E 相同）。
请求体一律用 request_bytes(body) 序列化后原样发出，日志记 body_sha256(body)；合成病历的请求体和原话全文落盘，
真实病历（R7）的只记 sha256 和元数据，提示词和原话只在内存里。
"""
import hashlib
import json
import re

from .render7 import render_summary7

MODEL = "Qwen2.5-32B-Instruct-AWQ"
TEMPERATURE = 0
TOP_P = 1.0
TOP_K = -1
REPETITION_PENALTY = 1.05
MAX_TOKENS = 512
FORMAT_ATTEMPTS = 3

JCARE_PROMPT = """Below is an electronic medical record consisting of multiple sections:
{record}
Please evaluate whether the record satisfies the following requirement: {requirement}. Identify any conflicts or implausible statements across sections.
Respond in the following JSON format:
{{"Meets Requirement": true/false}}"""

CRITERIA_Q3 = (
    ("K13", "The discharge diagnosis should not conflict with the patient's sex or age."),
    ("K14", "The discharge diagnosis should not conflict with symptoms in the chief complaint; the chief complaint should not contain symptoms that are clearly inconsistent with the discharge diagnosis."),
    ("K15", "The discharge diagnosis should not conflict with symptoms in the history of present illness; the history should not contain symptoms that are clearly inconsistent with the discharge diagnosis."),
    ("K16", "The discharge diagnosis should not conflict with examinations and tests in the hospital course; the hospital course should not contain findings that are clearly inconsistent with the discharge diagnosis."),
    ("K17", "The discharge diagnosis should not conflict with medications in the discharge instructions; the discharge instructions should not contain medications that are clearly inconsistent with the discharge diagnosis."),
    ("X14", "Checks whether any content anywhere in the record (history, menstrual and childbearing history, physical examination, diagnoses, procedures, medications) is impossible or clearly implausible for the patient's stated sex and age."),
)
CARE_IDS = ("K13", "K14", "K15", "K16", "K17")


def seed_for(phase, job_id, attempt):
    raw = f"Q3_J32|{phase}|{job_id}|{attempt}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def request_body(prompt, seed):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def request_bytes(body):
    """发给服务的字节就是这一串：键排序、不转义中文、紧凑分隔。日志里的 sha256 对的也是这一串。"""
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def body_sha256(body):
    return hashlib.sha256(request_bytes(body)).hexdigest()


def prompt_for(record, requirement):
    return JCARE_PROMPT.format(record=render_summary7(record), requirement=requirement)


def build_jobs(cases, phase, synthetic):
    """cases：mines.build_care7_cases 的输出（或任何带 case_id 和 7 字段 record 的 dict）。
    每个 case × 6 条标准一个 job。synthetic 只决定原话能不能落盘，不进提示词。"""
    jobs = []
    for c in cases:
        for cid, req in CRITERIA_Q3:
            jobs.append({"job_id": f"{phase}|{c['case_id']}|{cid}", "phase": phase, "case_id": c["case_id"],
                         "criterion_id": cid, "synthetic": int(bool(synthetic)),
                         "prompt": prompt_for(c["record"], req)})
    return jobs


# ---------- 解析：原样抄自 G1-D g1d_j32_common.py ----------

def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "true/false":
            return None
        if stripped == "true":
            return True
        if stripped == "false":
            return False
    return None


def parse_jcare(text):
    """Find every valid JSON object and take the last valid answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    decoder = json.JSONDecoder()
    found = []
    seen = set()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        span = (index, index + end)
        if span in seen or not isinstance(obj, dict):
            continue
        seen.add(span)
        values = []
        for key, value in obj.items():
            normalized = re.sub(r"[\s_]", "", str(key)).lower()
            if normalized == "meetsrequirement":
                parsed = _coerce_bool(value)
                if parsed is not None:
                    values.append(parsed)
        if len(values) == 1:
            found.append((span, values[0]))
    if not found:
        return None, "failed", 0, 0, 0
    values = [row[1] for row in found]
    return values[-1], "last_valid_json", 0, int(len(set(values)) > 1), len(values)


# ---------- 解析别名：原样抄自 G1-E g1e_judge.py（G1-D P2 修订） ----------

def json_dicts(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    decoder = json.JSONDecoder()
    found, spans = [], set()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        span = (index, index + end)
        if isinstance(value, dict) and span not in spans:
            spans.add(span)
            found.append(value)
    return found


def parse_p2(text):
    parsed = parse_jcare(text)
    if parsed[0] is not None:
        return parsed
    objects = json_dicts(text)
    if len(objects) != 1 or len(objects[0]) != 1:
        return parsed
    key, raw = next(iter(objects[0].items()))
    normalized = re.sub(r"[\s_]", "", str(key)).lower()
    if isinstance(raw, bool):
        value = raw
    elif isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
        value = raw.strip().lower() == "true"
    else:
        value = None
    if normalized in {"meetsrequirement", "meatsrequirement"} and value is not None:
        return value, "P2_alias", 0, 0, 1
    return parsed


# ---------- 计分 ----------

def all_pass(values, ids=CARE_IDS):
    """values：{标准: True(符合)/False/None(解析不了)}。全部 True 才算"通过"；
    有一条 False 就是"不通过"；没有 False 但有 None 返回 None。"""
    vs = [values.get(i) for i in ids]
    if any(v is False for v in vs):
        return False
    if any(v is None for v in vs):
        return None
    return True


def pair_miss(control_values, mined_values, ids=CARE_IDS):
    """H3-2 主读数用的一对病例：对照版 5 条全过才算"可用对"，可用对里雷版也 5 条全过就是"漏判"。
    返回三种口径，各为 "miss"、"caught"、"excluded"：
    - fav（主判定，对评委有利）：雷版解析不了当作"查出来了"；对照版解析不了，雷版查出来时当作通过（多一个查出的对），
      雷版漏掉时当作没通过（这一对不算）。这样漏判率只会被压低，不会因为丢掉解析失败而偏向蓝海；
    - point：两版 5 条都解析成功才算，其余 None；
    - unfav（对评委不利，只报）：雷版解析不了当作漏判；对照版解析不了，雷版漏判时算通过，雷版查出来时不算。"""
    c, m = all_pass(control_values, ids), all_pass(mined_values, ids)
    if c is False:
        return {"fav": "excluded", "point": "excluded", "unfav": "excluded"}
    m_fav = "caught" if m is not True else "miss"
    m_unf = "miss" if m is not False else "caught"
    if c is True:
        fav, unf = m_fav, m_unf
    else:
        fav = m_fav if m_fav == "caught" else "excluded"
        unf = m_unf if m_unf == "miss" else "excluded"
    point = None if (c is None or m is None) else ("miss" if m is True else "caught")
    return {"fav": fav, "point": point, "unfav": unf}


def miss_rate(pairs, key="fav"):
    """pairs：pair_miss 的结果列表。返回 (漏判数, 可用对数)。"""
    used = [p[key] for p in pairs if p[key] in ("miss", "caught")]
    return sum(1 for v in used if v == "miss"), len(used)


def miss_bounds(mined_values, ids=CARE_IDS):
    """雷版"评委漏掉"（全部通过）的点值和上下界：解析不了时，上界当作通过、下界当作不通过。"""
    p = all_pass(mined_values, ids)
    if p is None:
        return {"point": None, "upper": 1, "lower": 0}
    return {"point": int(p), "upper": int(p), "lower": int(p)}


def paired_detect(control_values, mined_values, ids=CARE_IDS):
    """成对发现：对照版全部通过、雷版至少一条不通过。解析不了时给上下界（照 g2_kit.judge.paired 的口径）。"""
    c, m = all_pass(control_values, ids), all_pass(mined_values, ids)

    def det(cp, mp):
        return int(cp is True and mp is False)
    up = det(True if c is None else c, False if m is None else m)
    lo = det(False if c is None else c, True if m is None else m)
    pt = None if c is None or m is None else det(c, m)
    return {"control_fail": int(c is False), "upper": up, "lower": lo, "point": pt}
