"""关4 评委（H4-1）：LLM-CARe 自己的 9 条"内容完整性"标准，逐条问 32B。

除了标准换成 9 条完整性（C01–C09，原文照抄 LLM-CARe 复现代码 run_s2.py 的 COMPLETENESS，即关1 G1-C T07 用的那 9 条）、
种子前缀换成 "Q45_J32"，其余一字不改地照抄 g3_core 0.2 的 q3_kit/judge.py：
- JCARE_PROMPT、parse_jcare：关1 G1-D 的 g1d_j32_common.py；
- json_dicts、parse_p2（认 32B 常写错的键 "Meats Requirement"）：关1 G1-E 的 g1e_judge.py；
- 渲染：render7.render_summary7（与 G1-C、G1-D、G1-E、Q3 相同）；
- 采样参数、request_bytes、body_sha256。
关1 G1-C 的 T07 用同一个提示词判过这 9 条，但那时还没有 parse_p2，格式失败很多（C05 在 S1 上 128/256），所以这一轮全部重判。
合成病历的请求体和原话全文落盘；真实病历（R7 和 R7 上的关键词堆砌版）只记 sha256 和元数据，提示词和原话只在内存里。
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

CRITERIA_Q4 = (
    ("C01", "Includes at least one of the following: symptom, examination result, or disease name; it is not necessary to include all three."),
    ("C02", "Includes the onset time of the main symptom or sign."),
    ("C03", "Mentions the onset pattern, such as sudden onset or chronic progression."),
    ("C04", "States whether there is a cause or trigger; wording like 'no obvious trigger' also satisfies this requirement, or a specific cause can be provided."),
    ("C05", "Includes the main symptoms or signs."),
    ("C06", "Explicitly lists all of the following general conditions: appetite/diet, bowel and urination, mental status, sleep, and weight change."),
    ("C07", "Includes examinations or laboratory tests performed for the patient."),
    ("C08", "Includes treatments received by the patient."),
    ("C09", "Provides dosage and administration instructions for each medication mentioned."),
)
COMP_IDS = tuple(c for c, _ in CRITERIA_Q4)


def seed_for(phase, job_id, attempt):
    raw = f"Q45_J32|{phase}|{job_id}|{attempt}".encode("utf-8")
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
    """cases：带 case_id 和 7 字段 record 的 dict。每个 case × 9 条标准一个 job。
    synthetic 只决定原话能不能落盘，不进提示词。"""
    jobs = []
    for c in cases:
        for cid, req in CRITERIA_Q4:
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

def score(values, none_as, ids=COMP_IDS):
    """values：{标准: True(符合)/False/None(解析不了)}。返回 (通过条数, 计入条数)。
    none_as=True 或 False：解析不了的按这个值算；none_as=None：解析不了的不计入。"""
    k = n = 0
    for i in ids:
        v = values.get(i)
        if v is None:
            if none_as is None:
                continue
            v = none_as
        n += 1
        k += int(v is True)
    return k, n


def mean9(values, none_as, ids=COMP_IDS):
    """9 条平均通过率（H4-1 主读数）。计入条数为 0 时返回 None。"""
    k, n = score(values, none_as, ids)
    return (k / n) if n else None


def all9(values, none_as, ids=COMP_IDS):
    """9 条全过：1 / 0；none_as=None 且有解析不了的，返回 None。"""
    vs = [values.get(i) for i in ids]
    if any(v is None for v in vs):
        if none_as is None:
            return None
        vs = [none_as if v is None else v for v in vs]
    return int(all(v is True for v in vs))


# H4-1 的口径（PREREG_Q45 第 3 节）：
# - fav（主判定，对评委有利）：地板（F-TPL）解析不了算"不符合"，真实（R7）解析不了算"符合"；
# - point：两边都只用解析成功的条；
# - unfav（只报）：地板解析不了算"符合"，真实解析不了算"不符合"。
POLICY = {"fav": {"floor": False, "real": True}, "point": {"floor": None, "real": None},
          "unfav": {"floor": True, "real": False}}
