"""负对照：八种作弊或退化写法。指标要是被它们骗过，这个指标就不能上发布会。

所有变换都是确定性的（种子写死），只在内存里做。基于真实行时，产物就是真实派生文本，不落盘、不打印。
变换后去掉 doc_type_cn、k，让关2 走 build_real，从正文标题认文书类型，和产品行同一条路。
"""
import random
import re

NAMES = ("NC-template", "NC-constant", "NC-shuffle", "NC-truncate", "NC-repeat", "NC-plandump", "NC-swap", "NC-dxflip")

TPL = {
    "hospital_record": "入院记录\n主诉：不适。\n现病史：患者因不适入院。\n既往史：否认特殊病史。\n体格检查：未见明显异常。\n入院诊断：待查。\n",
    "first": "首次病程记录\n病例特点：患者因不适入院。\n初步诊断：待查。\n诊疗计划：完善相关检查，对症治疗。\n",
    "progress_note": "日常病程记录\n今日查房，患者一般情况可，继续当前治疗，密切观察病情变化。\n",
    "discharge_summary": "出院记录\n入院情况：患者因不适入院。\n诊疗经过：予对症治疗，病情好转。\n出院诊断：待查。\n出院医嘱：门诊随诊。\n",
}
CONSTANT = "患者病情平稳，继续当前治疗，密切观察。"
_DX = "肺部感染"
_DX_LIST = "\n入院诊断：\n1." + _DX + "\n"
_DX_FLIP = ("考虑" + _DX + "可能。", _DX + "诊断明确。", "排除" + _DX + "。")
_FP_HEAD = re.compile(r"首\s*次\s*病\s*程")
_CLAUSE = re.compile(r"[^，,。；;！？!?\n]+")
_DIGIT = re.compile(r"\d")


def _clean(doc, text):
    d = {k: v for k, v in doc.items() if k not in ("doc_type_cn", "k", "text")}
    d["text"] = text
    return d


def _with_docs(x, docs, name):
    y = {k: v for k, v in x.items() if k not in ("docs", "raw_docs")}
    y["docs"], y["row"], y["view"] = docs, f"{name}@{x['row']}", "full"
    return y


def template(x):
    docs = []
    for d in x["docs"]:
        key = "first" if d["doc_type"] == "progress_note" and _FP_HEAD.search(d["text"][:300]) else d["doc_type"]
        docs.append(_clean(d, TPL[key]))
    return _with_docs(x, docs, "NC-template")


def constant(x):
    return _with_docs(x, [_clean(d, CONSTANT) for d in x["docs"]], "NC-constant")


def shuffle(x):
    """入院记录、首次病程、出院记录不动，中间病程的正文打乱，日程留在原位，时间线和正文错开。"""
    rng = random.Random(f"20261101|shuffle|{x['id']}")
    inner = [i for i, d in enumerate(x["docs"]) if d["doc_type"] == "progress_note" and not _FP_HEAD.search(d["text"][:300])]
    texts = [x["docs"][i]["text"] for i in inner]
    order = list(range(len(texts)))
    if len(order) > 1:
        rng.shuffle(order)
        if order == sorted(order):
            order = order[1:] + order[:1]
    new = dict(zip(inner, (texts[j] for j in order)))
    return _with_docs(x, [_clean(d, new.get(i, d["text"])) for i, d in enumerate(x["docs"])], "NC-shuffle")


def truncate(x):
    return _with_docs(x, [_clean(d, d["text"][:max(30, int(len(d["text"]) * 0.3))]) for d in x["docs"]], "NC-truncate")


def repeat(x):
    """每篇只留开头 15%，反复拼到原长：字数够，内容没有。"""
    docs = []
    for d in x["docs"]:
        t = d["text"]
        seg = t[:max(30, int(len(t) * 0.15))]
        docs.append(_clean(d, (seg * (len(t) // max(1, len(seg)) + 1))[:len(t)]))
    return _with_docs(x, docs, "NC-repeat")


def plandump(x):
    """把每篇压成带数字小句的清单，模拟"把计划或事实表直接倒进文书"。"""
    docs = []
    for d in x["docs"]:
        title = d["text"].split("\n", 1)[0][:20]
        parts = [m.group(0).strip() for m in _CLAUSE.finditer(d["text"]) if _DIGIT.search(m.group(0))]
        docs.append(_clean(d, title + "\n要点：" + "；".join(parts) + "。\n"))
    return _with_docs(x, docs, "NC-plandump")


def dxflip(x):
    """诊断反复横跳：入院记录的诊断清单加一个固定诊断，之后的病程按日程轮流写疑似、确诊、排除，其余正文不动。
    关2 的诊断词典只认清单里的名字，所以清单要一起加。G-D、G-E2 的诊断变化步数都超线，这是生成行真有的毛病。"""
    notes = [i for i, d in enumerate(x["docs"]) if d["doc_type"] == "progress_note" and d.get("day_offset") is not None]
    notes.sort(key=lambda i: (int(x["docs"][i]["day_offset"]), i))
    flip = {i: _DX_FLIP[n % len(_DX_FLIP)] for n, i in enumerate(notes)}
    docs = []
    for i, d in enumerate(x["docs"]):
        text = d["text"]
        if d["doc_type"] == "hospital_record":
            text += _DX_LIST
        elif i in flip:
            text += "\n" + flip[i] + "\n"
        docs.append(_clean(d, text))
    return _with_docs(x, docs, "NC-dxflip")


def swap_all(lines):
    """张冠李戴：下一个住院的出院记录和后半病程换到这个住院。需要整行所有住院，所以一次处理一列。"""
    out = []
    n = len(lines)
    for i, x in enumerate(lines):
        other = lines[(i + 1) % n]["docs"]
        mid = len(x["docs"]) // 2
        tail = other[len(other) // 2:]
        docs = [_clean(d, d["text"]) for d in x["docs"][:mid]]
        for j, d in enumerate(tail):
            one = _clean(d, d["text"])
            if x["docs"] and x["docs"][-1].get("day_offset") is not None and one.get("day_offset") is None:
                one["day_offset"] = x["docs"][-1]["day_offset"]
            docs.append(one)
        out.append(_with_docs(x, docs, "NC-swap"))
    return out


ONE = {"NC-template": template, "NC-constant": constant, "NC-shuffle": shuffle, "NC-truncate": truncate,
       "NC-repeat": repeat, "NC-plandump": plandump, "NC-dxflip": dxflip}


def build(lines, names=NAMES):
    """lines：同一行的住院列表。返回负对照行（只含 admission）。"""
    lines = [x for x in lines if x["kind"] == "admission" and x["docs"]]
    out = []
    for name in names:
        if name == "NC-swap":
            out.extend(swap_all(lines))
        else:
            out.extend(ONE[name](x) for x in lines)
    return out
