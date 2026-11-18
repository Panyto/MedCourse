# -*- coding: utf-8 -*-
"""路线 B 的事实获取：8B 两阶段自主提问检索（老板 11-10 定调；正则版 kb_v0 只当消融靶子）。

0.4（RB2 第一版验收后老板 11-11 批的改版）：
- 引用按句编号：检索到的段落由程序切成带编号的句子（S1、S2……）给 8B 看，8B 作答时先给出处句编号 cite_ids、再给 value，
  引文文字由程序照原文取出——8B 不再手抄引文（第一版 201 条引文里 51 条抄错）。
- 来源由程序判定，不再让 8B 自报 in_text（第一版 510 条里 423 条自报"原文没写"，标准住院日 17 个全是答对了却自报没写）：
  有出处句、且 value 在所引句子里 → grounded；有出处句但 value 不在里面 → quote_failed；没有出处句 → model_prior。
- 每轮两次调用：先定检索还是作答（decide），作答时用必须带 cite_ids、value 的格式（answer）。没检索满轮数之前作答，
  cite_ids 至少一个（只能从已经给它看过的句子编号里选）；检索满全部轮数原文仍只字未提，才允许 cite_ids 留空、按常识答。
- 本病种路径里检索时，检索词去掉病种名称的字词，表单段降权，正文段优先（第一版 K80.0 的"（二）诊断依据"一直没被检出来）。
- value 只写答案本身，用所引句子里的原词，不复述问题。

每个信息需求：8B 写检索词 → BM25 找段落 → 8B 读带编号的句子，决定再检索还是作答（最多 rounds 轮检索）→ 程序核对。
第一阶段问病种共性；第二阶段对选出的每项检查、每个手术追问；最后 8B 自己提最多 3 个追问。
程序不抽字段、不切检查清单、不写归一表：检查名映射到评测台键也由 8B 从候选键里选。
"""
import hashlib
import json
import re

from . import provenance as pv

NO_KEY = "无对应"
NONE_WORDS = ("无", "不适用", "")
FORM_WEIGHT = 0.5
NA = "不适用"
# 0.4.1（老板 11-11）：手术分支门控。8B 先答这个病种做不做手术或有创操作（要引出处），答"不做"的，下面这几项直接记"不适用"，不再追问
SURGICAL_FIELDS = ("anesthesia", "surgery_day_max", "postop_days", "postop_care")
GATE = "surgery_gate"
# 0.5（老板 11-15 批 4 条按字段检索约束）：
# 1) 症状、体征只看正文段落（排除 form 表单段）；
# 2) 可选检查去重已答出的必需检查；
# 3) 主要不做手术的病种问抗菌药时，改问治疗疗程/停药条件，并排除指导原则里围手术期/介入预防用药章节；
# 4) 手术门控改问"主要治疗是不是外科手术或介入治疗"。
NO_FORM_FIELDS = ("chief_symptoms", "signs")
SURG_PROPHY_RE = re.compile(r"围手术期|侵入性诊疗操作|附录\s*[23]|特殊诊疗操作")
ABX_NONSURG_NEED = "抗菌药物的治疗疗程或停药条件（本病主要不做手术，不适用手术预防用药条款；不常规使用抗菌药物就写无）"
_PAREN = re.compile(r"[（(].*?[）)]")

# (字段, 给 8B 的信息需求, 值类型, 检索范围)。0.4.1：标准住院日改成下限、上限两个数（老板 11-11），los_max 取上限
NEEDS_STAGE1 = (
    ("los_max", "标准住院日：写下限和上限两个天数（原文只写“≤N天”或一个数的，min 填 null，max 填 N）", "range", "pathway"),
    ("chief_symptoms", "入院时的典型症状", "list", "pathway"),
    ("signs", "典型体征（查体所见）", "list", "pathway"),
    ("exams_required", "入院后必需做的检查和检验项目", "exams", "pathway"),
    ("exams_optional", "根据病情可选做的检查和检验项目", "exams", "pathway"),
    ("procedure", "住院期间做的手术、操作或主要治疗的名称（没有手术操作就写无）", "text", "pathway"),
    (GATE, "按这份临床路径，本病住院期间的主要治疗是不是外科手术或介入治疗（如切除术、松解吻合术、碎石术、冠状动脉造影/PCI、内镜下手术等）？"
           "主要治疗是手术或介入写 true；以内科药物、康复、化疗、常规血液透析等非手术治疗为主（即使提到并发症或重症时的可选有创操作）写 false", "bool", "pathway"),
    ("anesthesia", "手术或操作的麻醉方式（不做手术操作就写无）", "text", "pathway"),
    ("surgery_day_max", "手术或操作在入院第几天以内做（不做就写无）", "text", "pathway"),
    ("postop_days", "术后住院恢复几天（不做手术就写无）", "text", "pathway"),
    ("antibiotic_rule", "抗菌药物用到什么时候停：治疗疗程或停药条件、预防用药用多久", "text", "both"),
    ("discharge_criteria", "出院标准", "list", "pathway"),
    ("risk_factors", "个人史里的危险因素（吸烟、饮酒、饮食、职业暴露等）", "list", "pathway"),
    ("severity_levels", "病情轻重怎么分档，各档的判定依据", "list", "pathway"),
)

SYSTEM = ("你是临床知识整理助手，只能通过检索查阅给定的公开临床路径和《抗菌药物临床应用指导原则》。检索结果按句编号（S1、S2……）给你。\n"
          "作答规则：\n"
          "1. 只要给你看过的句子里有相关内容，必须在 cite_ids 里写出这些句子的编号，value 用这些句子里的原词作答；\n"
          "2. value 只写答案本身（名称、天数、短语），不复述问题，不加解释；\n"
          "3. 只有检索满全部轮数、原文确实只字未提时，cite_ids 才能留空，这时 value 按你的医学常识回答。")

TYPE_HINT = {
    "int": "value 是一个整数（天数）。",
    "range": "value 写 min（下限天数，没有写 null）和 max（上限天数），都用原文里的数。",
    "bool": "value 写 true 或 false。",
    "text": "value 是一个名称或短语。",
    "list": "value 是列表，每项一个短语，用原词。",
    "exams": "value 是列表，每项写原文里的检查名（name），并从下面的评测台检查键里选最接近的一个（key），没有对应的写“无对应”。评测台检查键：{keys}",
    "postop": "value 写：有没有需要缝合的体表切口（incision），术后是否留置引流管（drain），术后每天观察什么（watch，列表）。",
}
SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])|\n|(?=□)")


def _value_schema(vtype, keys):
    if vtype == "int":
        return {"type": "integer", "minimum": 0, "maximum": 365}
    if vtype == "range":
        return {"type": "object", "required": ["min", "max"],
                "properties": {"min": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 365}, {"type": "null"}]},
                               "max": {"type": "integer", "minimum": 0, "maximum": 365}}}
    if vtype == "bool":
        return {"type": "boolean"}
    if vtype == "list":
        return {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 20}
    if vtype == "exams":
        return {"type": "array", "maxItems": 30, "items": {"type": "object", "required": ["name", "key"],
                "properties": {"name": {"type": "string", "maxLength": 30}, "key": {"enum": list(keys) + [NO_KEY]}}}}
    if vtype == "postop":
        return {"type": "object", "required": ["incision", "drain", "watch"],
                "properties": {"incision": {"type": "boolean"}, "drain": {"type": "boolean"},
                               "watch": {"type": "array", "items": {"type": "string", "maxLength": 30}, "maxItems": 8}}}
    return {"type": "string", "maxLength": 60}


DECIDE_SCHEMA = {"type": "object", "required": ["action", "query"],
                 "properties": {"action": {"enum": ["search", "answer"]}, "query": {"type": "string", "maxLength": 40}}}


def answer_schema(vtype, keys, shown_ids, must_cite):
    """cite_ids 排在 value 前面：先选出处句，再从句子里取答案。编号只能是给它看过的。"""
    cite = {"type": "array", "items": {"enum": list(shown_ids)} if shown_ids else {"type": "string", "maxLength": 0},
            "minItems": 1 if must_cite else 0, "maxItems": 8}
    if not shown_ids:
        cite["maxItems"] = 0
    return {"type": "object", "required": ["cite_ids", "value"],
            "properties": {"cite_ids": cite, "value": _value_schema(vtype, keys)}}


def seed_of(*parts):
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


def _extract_sampling():
    try:
        from medcourse.llm import EXTRACT
        return EXTRACT
    except ImportError:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]


def _gen(client, msgs, schema, seed, max_tokens):
    out = client.generate(sampling=_extract_sampling(), seed=seed, messages=msgs, max_tokens=max_tokens, schema=schema)
    try:
        obj = json.loads(out["text"])
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def ask(client, index, disease, need, vtype, docs, keys=(), seed=0, rounds=3, k=3, max_tokens=600, drop="",
        exclude_kinds=(), exclude_title=None):
    """一个信息需求。返回 (作答 dict 或 None, 过程记录)。作答 dict 里 cites 是 [(编号, 片段 id, 文档, 句子原文)]。"""
    hint = TYPE_HINT[vtype].format(keys="、".join(keys))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"病种：{disease}。\n要查的内容：{need}\n{hint}\n"
                                        f"你最多可以检索 {rounds} 次。先检索，读完句子再决定继续检索还是作答。只回答 JSON。"}]
    seen, sents = set(), {}          # sents：编号 → (片段 id, 文档, 句子)
    trace = {"queries": [], "chunks": [], "searches": 0, "calls": 0, "parse_fail": 0, "prompt_chars": 0}

    def show(hits):
        lines = []
        for h in hits:
            lines.append(f"【片段 {h['id']}｜{h['title']}】")
            for s in sentences(h["text"]):
                sid = f"S{len(sents) + 1}"
                sents[sid] = (h["id"], h["doc"], s)
                lines.append(f"{sid} {s}")
        return "\n".join(lines) or "（没有检索到新的段落）"

    while True:
        final = trace["searches"] >= rounds
        action = "answer"
        if not final:
            trace["prompt_chars"] = max(trace["prompt_chars"], sum(len(m["content"]) for m in msgs))
            obj = _gen(client, msgs, DECIDE_SCHEMA, seed + trace["calls"], 80)
            trace["calls"] += 1
            if obj is None:
                trace["parse_fail"] += 1
                obj = {"action": "search", "query": need}
            action = obj.get("action")
            if action == "answer" and not sents:      # 还没看过任何句子，不能作答
                action = "search"
            if action == "search":
                q = (obj.get("query") or need)[:40]
                hits = index.search(q, k=k, docs=docs, exclude=seen, drop=drop, form_weight=FORM_WEIGHT,
                                    exclude_kinds=exclude_kinds, exclude_title=exclude_title)
                seen.update(h["id"] for h in hits)
                trace["queries"].append(q)
                trace["chunks"].extend(h["id"] for h in hits)
                trace["searches"] += 1
                msgs.append({"role": "assistant", "content": json.dumps({"action": "search", "query": q}, ensure_ascii=False)})
                left = rounds - trace["searches"]
                msgs.append({"role": "user", "content": f"检索结果：\n{show(hits)}\n\n还可以检索 {left} 次。继续检索（换个问法）还是作答？只回答 JSON。"})
                continue
        # 作答
        must = not final
        rule = ("请作答：cite_ids 写出处句编号（至少一个），value 用这些句子里的原词。" if must else
                "检索次数已用完。请作答：看过的句子里有相关内容就写出处句编号并用原词；确实只字未提，cite_ids 留空，按医学常识回答。")
        msgs_a = msgs + [{"role": "user", "content": rule + "只回答 JSON。"}]
        trace["prompt_chars"] = max(trace["prompt_chars"], sum(len(m["content"]) for m in msgs_a))
        obj = _gen(client, msgs_a, answer_schema(vtype, keys, list(sents), must), seed + 100 + trace["calls"], max_tokens)
        trace["calls"] += 1
        if obj is None or "value" not in obj:
            trace["parse_fail"] += 1
            if final:
                return None, trace
            trace["searches"] += 1      # 作答不合格也算用掉一轮，防止死循环
            msgs.append({"role": "user", "content": "上一次作答不合格。"})
            continue
        ids = [i for i in obj.get("cite_ids") or [] if i in sents]
        obj["cites"] = [(i,) + sents[i] for i in ids]
        trace["answered_after_searches"] = trace["searches"]
        return obj, trace


def _check_items(vtype, value):
    """要在所引句子里找到的那部分值。"""
    if vtype == "exams":
        return [e.get("name", "") for e in value or []]
    if vtype == "postop":
        return list((value or {}).get("watch") or [])
    if vtype == "range":
        v = value or {}
        return [x for x in (v.get("max"), v.get("min")) if isinstance(x, int) and not isinstance(x, bool)]
    return value


def na_fact(gate_fact):
    """手术门控答"不做"时，手术相关字段记"不适用"。来源记 not_applicable，出处跟着门控那条走。"""
    return {"value": NA, "source": "not_applicable", "grounded": False, "quote_failed": False, "claimed_in_text": False,
            "gated_by": GATE, "gate_grounded": bool(gate_fact.get("grounded")), "quote": gate_fact.get("quote", ""),
            "cite_ids": [], "quotes": []}


def ground(ans, vtype):
    """来源由程序判定。所引句子是程序照原文取的，不会抄错；只核 value 在不在所引句子里。"""
    value = ans.get("value")
    cites = ans.get("cites") or []
    quotes = [c[3] for c in cites]
    fact = {"value": value, "cite_ids": [c[0] for c in cites], "quotes": quotes, "quote": " / ".join(quotes),
            "claimed_in_text": bool(cites), "grounded": False, "quote_failed": False, "source": "model_prior"}
    if vtype == "range" and isinstance(value, dict):
        fact["value_max"] = value.get("max")     # los_max 取上限（老板 11-11）
    if not cites:
        return fact
    if vtype == "bool":
        # 是非题没法在句子里找"true"，引了句子就算有出处，对不对交金标
        fact.update(grounded=True, source="pathway", doc=sorted({c[2] for c in cites})[0], check="cite_only")
        return fact
    items = _check_items(vtype, value)
    joined = "".join(quotes)
    if isinstance(items, list):
        fact["items_in_quote"] = [sum(1 for x in items if pv.value_in_quote(x, joined)), len(items)]
    ok = pv.value_in_quote(items, joined) if items not in (None, "", []) else (vtype == "postop")
    if not ok:
        fact["quote_failed"] = True          # 引了句子，但答案不在所引句子里
        return fact
    docs = sorted({c[2] for c in cites})
    fact.update(grounded=True, source="guideline" if all(d.startswith("guideline") for d in docs) else "pathway", doc=docs[0])
    return fact


def _is_none(v):
    return v is None or (isinstance(v, str) and v.strip() in NONE_WORDS) or v == []


def _exam_norm(name):
    raw = pv.norm(name).strip("。．.，,；;：:、 ")
    stripped = pv.norm(_PAREN.sub("", str(name or ""))).strip("。．.，,；;：:、 ")
    return stripped or raw


def dedupe_optional(opt_items, req_items):
    """0.5：可选检查去掉已出现在必需检查里的同名项目（忽略括号补充，如"尿常规（包括酮体）"与"尿常规"），不按评测台键去重（保留 18 导联心电图等）。"""
    seen = {_exam_norm(r.get("name")) for r in (req_items or []) if isinstance(r, dict) and r.get("name")}
    seen.discard("")
    kept, dropped = [], 0
    for e in opt_items or []:
        if not isinstance(e, dict):
            continue
        k = _exam_norm(e.get("name"))
        if k and k in seen:
            dropped += 1
            continue
        if k:
            seen.add(k)
        kept.append(e)
    return kept, dropped


def build_disease(client, index, disease, doc_id, sources, keys, key_cls, rounds=3, k=3, extra_questions=3, drop=None,
                  only_fields=None, base_facts=None):
    """一个病种的两阶段提问检索。返回 (facts, log, missing)；missing 是没给出合格答案的需求。
    only_fields：增量重跑只问这些字段（问手术相关字段时自动带上门控）；第二阶段检查追问和自主追问不跑。
    base_facts：增量时取上一版的 procedure、exams_required、surgery_gate。"""
    facts, log, missing = {}, [], []
    path_docs = {doc_id}
    both_docs = {doc_id} | {d for d in sources if d.startswith("guideline")}
    drop = disease if drop is None else drop
    want = None
    if only_fields is not None:
        need_gate = bool(set(only_fields) & set(SURGICAL_FIELDS)) or (
            "antibiotic_rule" in only_fields and not (base_facts or {}).get(GATE))
        want = set(only_fields) | ({GATE} if need_gate else set())

    def gated():
        src = facts.get(GATE) or (base_facts or {}).get(GATE) or [None]
        g = src[0]
        return g if g is not None and g.get("value") is False else None

    def req_exams():
        src = facts.get("exams_required") or (base_facts or {}).get("exams_required") or []
        return [e for f in src for e in (f.get("value") or []) if isinstance(e, dict)]

    def run(field, need, vtype, docs, extra=None):
        if field in SURGICAL_FIELDS and gated():
            facts.setdefault(field, []).append(dict(na_fact(gated()), **(extra or {})))
            log.append({"field": field, "need": need, "gated": True})
            return None
        ex_kinds = ("form",) if field in NO_FORM_FIELDS else ()
        ex_title = None
        if field == "antibiotic_rule" and gated():
            need = ABX_NONSURG_NEED
            ex_title = SURG_PROPHY_RE
        req = req_exams() if field == "exams_optional" else []
        ask_need = need
        if field == "exams_optional" and req:
            names = "、".join(dict.fromkeys(e.get("name", "") for e in req if e.get("name")))
            if names:
                ask_need = f"{need}（已列为必需检查的项目不要重复列出：{names[:160]}）"
        ans, tr = ask(client, index, disease, ask_need, vtype, docs, keys if vtype == "exams" else (),
                      seed_of(doc_id, field, need), rounds, k, drop=drop, exclude_kinds=ex_kinds, exclude_title=ex_title)
        tr.update(field=field, need=ask_need, answered=ans is not None)
        if ex_kinds:
            tr["exclude_kinds"] = list(ex_kinds)
        if ex_title is not None:
            tr["exclude_title"] = ex_title.pattern
        log.append(tr)
        if ans is None:
            missing.append({"field": field, "need": need})
            return None
        deduped = 0
        if field == "exams_optional":
            ans["value"], deduped = dedupe_optional(ans.get("value"), req)
        f = ground(ans, vtype)
        if field == "exams_optional":
            f["deduped_from_required"] = deduped
            if deduped and not ans["value"]:
                f["quote_failed"] = False
        f.update(searches=tr["searches"], queries=tr["queries"], chunks=tr["chunks"], **(extra or {}))
        facts.setdefault(field, []).append(f)
        return f

    for field, need, vtype, scope in NEEDS_STAGE1:
        if want is None or field in want:
            run(field, need, vtype, both_docs if scope == "both" else path_docs)
    if want is not None:
        if "postop_care" in want:
            src = facts if facts.get("procedure") else (base_facts or {})
            proc = (src.get("procedure") or [{}])[0].get("value")
            if gated() or not _is_none(proc):
                run("postop_care", f"{proc}：有没有需要缝合的体表切口，术后是否留置引流管，术后每天重点观察什么", "postop", path_docs, {"procedure": proc})
        return facts, log, missing
    exams, seen = [], set()
    for fld in ("exams_required", "exams_optional"):
        for f in facts.get(fld, []):
            for e in f["value"] or []:
                if e.get("key") and e["key"] != NO_KEY and e["key"] not in seen:
                    seen.add(e["key"])
                    exams.append(e)
    for e in exams:
        run("exam_turnaround_days", f"{e['name']}从开单当天算起，第几天出正式报告（当天出写 0）", "int", path_docs,
            {"exam_key": e["key"], "exam_name": e["name"]})
        if key_cls.get(e["key"]) != "检验":
            run("exam_findings", f"本病患者做{e['name']}，异常时的客观所见（只写征象，不写病名，不写“符合××表现”）", "text", path_docs,
                {"exam_key": e["key"], "exam_name": e["name"]})
    proc = (facts.get("procedure") or [{}])[0].get("value")
    if gated() or not _is_none(proc):
        run("postop_care", f"{proc}：有没有需要缝合的体表切口，术后是否留置引流管，术后每天重点观察什么", "postop", path_docs, {"procedure": proc})
    if extra_questions:
        known = "；".join(f"{fld}={json.dumps(fs[0]['value'], ensure_ascii=False)[:80]}" for fld, fs in facts.items() if fs)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"病种：{disease}。已查到：{known}\n为了按天排出这个病种的整次住院日程（检查哪天开、哪天回报，"
                                            f"用药哪天开始、哪天停，手术前后各做什么），还缺哪些信息？提出最多 {extra_questions} 个要追问的问题。只回答 JSON。"}]
        schema = {"type": "object", "required": ["questions"],
                  "properties": {"questions": {"type": "array", "items": {"type": "string", "maxLength": 60}, "maxItems": extra_questions}}}
        obj = _gen(client, msgs, schema, seed_of(doc_id, "extra"), 300)
        qs = (obj or {}).get("questions") or []
        log.append({"field": "extra_questions", "questions": qs})
        for q in qs[:extra_questions]:
            run("extra_qa", q, "text", both_docs, {"question": q})
    return facts, log, missing
