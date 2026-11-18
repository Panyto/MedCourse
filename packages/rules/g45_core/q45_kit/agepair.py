"""H5-1′：配额改写只改年龄栏以后，正文是不是还是原来那个年龄的人。读者换成 32B，度量换成"本人正文对换来的正文"。

为什么换（Q45 CPU 审查意见"H5-1 为什么作废"）：0.1 用 TF-IDF 年龄模型的概率除以真实病历的同病种平均，合成文本的概率更平，
一除就把小年龄段放大，读数整个翻过来。这里不再用任何概率、不做归一化，只问二选一，再用"换来的正文"扣掉读者自己的偏好。

一个单位 = 一份被改得远的病历（年龄段变了且相差 ≥15 岁，demog.changed_far）：原年龄 old、新年龄 new。
- 本人：把这份病历自己的四段正文给 32B，问"更可能是 old 岁还是 new 岁"，A、B 两种顺序各问一次，选 old 记 1，两次取平均；
- 换来的：同一个三位码里另外随机 2 个槽位的正文（同一行，CARe 对 CARe、真实对真实），问同一个问题（同样的 old、new）；
- D = 本人选 old 的比例 − 换来的选 old 的比例，按槽位配对（stats.paired_mean_diff）。
读者如果天生偏好某个年龄、或者只从病种猜年龄，本人和换来的一样受影响，D 里扣掉了；只有正文里有"这个人"的年龄线索，D 才 > 0。

提示词、采样参数、解析在本模块冻结（PREREG_Q45b 第 6 节）。种子用 judge.seed_for（前缀 Q45_J32，phase 写 agepair）。
"""
import random
import re
import unicodedata

from ruler_g3.adapters import care_get

from . import demog, judge
from .stats import paired_mean_diff

AGE_MAX_TOKENS = 8
K_SWAP = 2
SWAP_SEED = 20261028
ORDERS = ("ON", "NO")        # ON：A=原年龄、B=新年龄；NO：反过来

AGE_PROMPT = """下面是一份住院病历的正文（主诉、现病史、住院经过、出院医嘱），性别、年龄和诊断栏已经去掉。

{text}

只根据上面的正文判断：这位病人更可能是下面哪一个年龄？
A. {a}
B. {b}
只回答一个字母：A 或 B。"""

_SECTIONS = (("cc", "主诉"), ("hpi", "现病史"), ("hc", "住院经过"), ("di", "出院医嘱"))


def narrative(record):
    """四段正文，原样（不规范化、不去数字），每段前面标【段名】。性别、年龄、诊断不进。"""
    return "\n".join(f"【{name}】{(care_get(record, k) or '').strip()}" for k, name in _SECTIONS)


def age_label(value):
    """年龄栏 → 提示词里的写法：≥1 岁写"N岁"（向下取整），<1 岁写"N个月"。认不出返回 None。"""
    a = demog.stated_age(value)
    if a is None:
        return None
    if a < 1:
        return f"{max(1, round(a * 12))}个月"
    return f"{int(a)}岁"


def prompt_for(record, old, new, order):
    a, b = (old, new) if order == "ON" else (new, old)
    return AGE_PROMPT.format(text=narrative(record), a=age_label(a), b=age_label(b))


def request_body(prompt, seed):
    """和评委同一套采样参数（judge.request_body），只把 max_tokens 改成 8。"""
    body = judge.request_body(prompt, seed)
    body["max_tokens"] = AGE_MAX_TOKENS
    return body


def swap_partners(slot, code_of, k=K_SWAP, seed=SWAP_SEED):
    """同一个三位码里、除自己以外随机 k 个槽位（先排序再抽）。code_of={槽位: 三位码}。"""
    code = code_of.get(slot)
    cands = sorted(s for s, c in code_of.items() if c == code and s != slot)
    rng = random.Random(f"{seed}|swap|{slot}")
    return rng.sample(cands, min(k, len(cands)))


def build_jobs(units, texts, code_of, row, synthetic):
    """units：[{unit_id, slot, old, new}]（old、new 是年龄栏原值）；texts={槽位: 7 字段病历}（同一行的正文）。
    每个单位：本人 1 份 + 换来的 K_SWAP 份，每份两种顺序。返回 job 列表（含 prompt）。"""
    jobs = []
    for u in units:
        roles = [("own", u["slot"])] + [(f"swap{i}", s) for i, s in enumerate(swap_partners(u["slot"], code_of))]
        for role, text_slot in roles:
            for order in ORDERS:
                jobs.append({"job_id": f"agepair|{row}|{u['unit_id']}|{role}|{order}", "phase": "agepair",
                             "row": row, "unit_id": u["unit_id"], "slot": u["slot"], "role": role,
                             "text_slot": text_slot, "order": order, "old": u["old"], "new": u["new"],
                             "synthetic": int(bool(synthetic)),
                             "prompt": prompt_for(texts[text_slot], u["old"], u["new"], order)})
    return jobs


def parse_ab(text):
    """模型原话 → "A" / "B" / None。全文里单独出现的字母（前后不挨着别的英文字母）只有 A 或只有 B 才取，两个都有、都没有算失败。"""
    t = unicodedata.normalize("NFKC", text or "").upper()
    found = set(re.findall(r"(?<![A-Z])([AB])(?![A-Z])", t))
    return found.pop() if len(found) == 1 else None


def chose_old(answer, order):
    if answer not in ("A", "B"):
        return None
    return int((answer == "A") == (order == "ON"))


def unit_scores(results):
    """results：[{unit_id, slot, role, order, answer}]（answer 是 parse_ab 的结果，可以是 None）。
    返回 {unit_id: {"slot", "own": 值或 None, "swap": [值或 None, ...]}}；每个值是两种顺序里解析成功的那几次的平均。"""
    acc = {}
    for r in results:
        v = chose_old(r["answer"], r["order"])
        d = acc.setdefault(r["unit_id"], {"slot": r["slot"], "roles": {}})
        d["roles"].setdefault(r["role"], []).append(v)
    out = {}
    for uid, d in acc.items():
        means = {}
        for role, vs in d["roles"].items():
            ok = [v for v in vs if v is not None]
            means[role] = sum(ok) / len(ok) if ok else None
        out[uid] = {"slot": d["slot"], "own": means.get("own"),
                    "swap": [means[k] for k in sorted(means) if k.startswith("swap")]}
    return out


def own_vs_swap(scores, B=1000, seed=20261014):
    """scores：unit_scores 的返回。本人没有读数的单位不用；换来的全都没有读数的单位也不用。
    按槽位：A = 该槽位各单位"本人"的平均，Bm = 该槽位各单位全部"换来的"的平均；D = A − Bm 用 stats.paired_mean_diff。
    返回 {D, ci_lo, ci_hi, slots, units, own_mean, swap_mean, dropped}。"""
    own, swp, units, dropped = {}, {}, 0, 0
    for uid in sorted(scores):
        s = scores[uid]
        sw = [v for v in s["swap"] if v is not None]
        if s["own"] is None or not sw:
            dropped += 1
            continue
        units += 1
        own.setdefault(s["slot"], []).append(s["own"])
        swp.setdefault(s["slot"], []).extend(sw)
    a = {k: sum(v) / len(v) for k, v in own.items()}
    b = {k: sum(v) / len(v) for k, v in swp.items()}
    est, lo, hi, n, mean_b = paired_mean_diff(a, b, B, seed)
    own_mean = sum(a.values()) / len(a) if a else float("nan")
    return {"D": est, "ci_lo": lo, "ci_hi": hi, "slots": n, "units": units, "own_mean": own_mean,
            "swap_mean": mean_b, "dropped": dropped}
