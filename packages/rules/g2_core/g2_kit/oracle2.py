"""按底稿算的泄漏 v2（g2-core 0.4）。只对给了事实底稿的写法（G-C、G-D、G-E、G-E′）。

v1（gen.oracle_leaks）在 pkg1b 的合成病历上读过：A 类 0 真，O 类假报警集中在两处，B 类 40 条只有 3 条真。
v2 只改判定口径，不改输入输出的形状。三类各自改了什么：
- O：6-gram 路线先从结果里去掉诊断名和检查名，剩下的内容太短就不走这条路线
  （"提示子宫内膜异位症""持续性Ⅲ度房室传导阻滞"这类结果，原先靠疑似诊断名就能凑够覆盖率）；
  检查名+结果口吻路线，分句里有待回、计划、条件、教科书口吻就不算
  （"待头部MRI和颈部血管彩超检查结果回报后""腹部CT可见胰腺水肿或坏死"）。
- A：完成词去掉单字"行"（"进行""需行""确保手术顺利进行"都含"行"）；计划词补上需、考虑、必要时、安排、适时、
  是否、评估、确保等；既往史段落和带"年前、曾、既往"的分句不算（"1年前因胎儿窘迫行剖宫产术"）；
  术后状态词只认"术后第N天、术中见、安返病房、今日行……术"这类，去掉"手术顺利""术后恢复"。
- B：分严格和宽松两档。严格档才进判定：确诊为、证实、诊断明确、已排除、可排除这类定论，且前面没有初步、考虑、
  需、待、不能、若这类词，后面没有可能、待查。宽松档单列计数：符合……表现、目前诊断为……、暂不考虑。
"""
import re

from ruler_g1.sections import split_sections
from ruler_g2.events import EXAM_RE, _ALIAS, tone_of
from ruler_g2.text import HIST_RE, clause_span, sentence_span

from g2_kit.gen import _clean, _key_of, char_ngrams

VERSION = "oracle-2.0"

# ---------- O ----------

O_PENDING = re.compile(
    r"待|等待|未回|未出|尚未|未见报告|结果未|已安排|安排|预计|预约|申请|送检|需|需要|若|如果|如|以明确|以排除|"
    r"有助于|帮助|完善|将|拟|计划|明日|明天|进一步")
O_PRE_SENT_PLAN = re.compile(r"完善|待|等待|安排|预约|申请|送检|计划|包括|拟")
O_SENT_PENDING = re.compile(r"未回报|未出结果|尚未出|待回报|结果将|将在.{0,6}出|明日出|明天出|结果待")
O_GENERIC = re.compile(r"通常|一般|常见|典型|多表现|可表现|多为|往往|鉴别|多伴|常伴|特点为|表现为")
O_HEDGE_WORDS = re.compile(r"提示|考虑|符合|可能性?大?|疑似|待排|性质待定|表现|改变|结果|检查|显示|示")
O_MIN_CONTENT = 8


def _strip_names(s, names):
    for n in sorted(names, key=len, reverse=True):
        if len(n) >= 2:
            s = s.replace(n, "")
    return s


def _dx_names(draft):
    names = {j["diagnosis"] for j in draft.get("judgments", [])}
    names |= set(draft.get("final_diagnoses", []))
    return names


def _common_runs(a, b, k=4):
    """a 里和 b 共有的、长度 ≥k 的连续片段（按出现位置，贪心取最长）。"""
    out, i = [], 0
    while i < len(a):
        best = 0
        for j in range(len(a), i + k - 1, -1):
            if a[i:j] in b:
                best = j - i
                break
        if best >= k:
            out.append(a[i:i + best])
            i += best
        else:
            i += 1
    return out


def _o_content(result, obs_name, dx_names):
    """结果里真正的内容：去掉诊断名（以及和诊断名共有的 ≥4 字片段）、检查名、口吻词之后剩下的字。"""
    s = _strip_names(result, dx_names | {obs_name})
    s = EXAM_RE.sub("", s)
    s = O_HEDGE_WORDS.sub("", s)
    s = _clean(s)
    for n in dx_names:
        for run in _common_runs(s, _clean(n)):
            s = s.replace(run, "")
    return s


def o_hits(draft, n, text):
    """第 n 天这篇正文里，写出了几项第 n 天还没回报的检查结果。返回 [(obs_id, route)]。
    6-gram 路线只数"隐藏结果独有"的 6-gram：已回报结果里也有的 6-gram 不算
    （入院 B 超写"胆囊增大，壁厚"，第 1 天的 CT 也写"胆囊增大，壁厚"，前者不能算后者的泄漏）。"""
    obs = draft["observations"]
    dx = _dx_names(draft)
    text_wo_dx = _clean(_strip_names(text, dx))
    text_grams = char_ngrams(text_wo_dx, 6)
    visible = [o for o in obs if o["result_day"] <= n]
    visible_keys = {_key_of(o["name"]) for o in visible}
    visible_grams = set()
    for o in visible:
        visible_grams |= char_ngrams(_clean(o["result"]), 6)
    diff_spans = [(sec.start, sec.end) for sec in split_sections(text) if sec.name == "鉴别诊断"]
    out, tone_keys_hit = [], set()
    for o in obs:
        if o["result_day"] <= n:
            continue
        content = _o_content(o["result"], o["name"], dx)
        g = char_ngrams(content, 6) - visible_grams if len(content) >= O_MIN_CONTENT else set()
        if len(g) >= 3:
            need = 0.5 if len(g) >= 10 else 0.75
            if len(g & text_grams) / len(g) >= need:
                out.append((o.get("id"), "6-gram"))
                continue
        elif 3 <= len(content) < O_MIN_CONTENT and re.search(r"\d", content) and content in text_wo_dx:
            out.append((o.get("id"), "短结果原样"))
            continue
        key = _key_of(o["name"])
        if not key or key in visible_keys or key in tone_keys_hit:
            continue
        for m in EXAM_RE.finditer(text):
            if _ALIAS[m.group(0)] != key or any(a <= m.start() < b for a, b in diff_spans):
                continue
            tone, _ = tone_of(text, m.start(), m.end())
            if tone != "R":
                continue
            a, b = clause_span(text, m.start(), m.end())
            sa, sb = sentence_span(text, m.start(), m.end())
            if (O_PENDING.search(text[a:b]) or O_PRE_SENT_PLAN.search(text[sa:m.start()])
                    or O_GENERIC.search(text[sa:sb]) or O_SENT_PENDING.search(text[sa:sb])):
                continue
            out.append((o.get("id"), "检查名+结果口吻"))
            tone_keys_hit.add(key)  # 同一类检查（如胸部CT 和胸部CT复查）只记一次
            break
    return out


# ---------- A ----------

A_DONE = re.compile(
    r"已行|已于|已经|已完成|已实施|已做|顺利完成|行了|术中见|术中所见|术后第\s*[\d一二三四五六七八九十]+\s*[天日]|术后\s*\d+\s*(?:天|日|小时|h)|"
    r"安返病房|返回病房|送返病房|(?:今日|今天|今晨|昨日|昨天|于今日|于昨日)(?:在[^，,。；;\n]{0,15}下)?行")
A_PLAN = re.compile(
    r"拟|计划|准备|择期|明日|明天|将|待|术前|建议|预约|决定|需|需要|考虑|必要时|安排|适时|是否|尽快|有必要|评估|"
    r"以便|以利|确保|可行|宜|应|如|若|定于|准备|告知|风险|注意")
A_POST_STATE = re.compile(
    r"术后第\s*[\d一二三四五六七八九十]+\s*[天日]|术后\s*\d+\s*(?:天|日|h|小时)|术中见|术中所见|安返病房|返回病房|送返病房|"
    r"麻醉(?:清醒|苏醒)|(?:今日|今天|于今日)(?:在[^，,。；;\n]{0,15}下)?行[^，,。；;\n]{1,25}术(?!前)")
APPROACH = re.compile(r"^(?:左|右|双)?(?:侧)?(?:腹腔镜下?|宫腔镜下?|胸腔镜下?|关节镜下?|膀胱镜下?|输尿管镜下?|内镜下|内窥镜下?|显微镜下|"
                      r"经皮(?:肝)?(?:穿刺)?|经尿道|经阴道|经腹|经口|开放性?|微创|介入)?")
DONE_GENERIC = re.compile(r"已(?:于[^，,。；;\n]{0,10})?行[^，,。；;\n]{1,25}?术(?!前)")
PAST_SECTIONS = {"现病史", "既往史", "个人史", "婚育史", "月经史", "家族史", "手术史", "过敏史"}


def _past_spans(text):
    return [(s.start, s.end) for s in split_sections(text) if s.name in PAST_SECTIONS]


def _is_past(text, s, e, past):
    if any(a <= s < b for a, b in past):
        return True
    sa, sb = sentence_span(text, s, e)
    return bool(HIST_RE.search(text[sa:sb]))


def a_hits(draft, n, text):
    """第 n 天这篇正文里，几项第 n 天还没做的手术或操作被写成已做。"""
    past = _past_spans(text)
    out = []
    for a in draft["actions"]:
        if a["do_day"] <= n or a["kind"] not in ("手术", "操作"):
            continue
        cores = {re.sub(r"^(左|右|双)(侧)?", "", a["name"]), APPROACH.sub("", a["name"])}
        found = False
        for core in (c for c in cores if len(c) >= 4):
            for m in re.finditer(re.escape(core[:6]), text):
                if _is_past(text, m.start(), m.end(), past):
                    continue
                ca, cb = clause_span(text, m.start(), m.end())
                clause = text[ca:cb]
                if A_DONE.search(clause) and not A_PLAN.search(clause):
                    found = True
                    break
            if found:
                break
        if not found:
            name3 = char_ngrams(APPROACH.sub("", a["name"]), 3)
            for m in DONE_GENERIC.finditer(text):
                if _is_past(text, m.start(), m.end(), past) or not (char_ngrams(m.group(0), 3) & name3):
                    continue
                ca, cb = clause_span(text, m.start(), m.end())
                if not A_PLAN.search(text[ca:m.start()]):
                    found = True
                    break
        if not found and a["kind"] == "手术":
            for m in A_POST_STATE.finditer(text):
                if _is_past(text, m.start(), m.end(), past):
                    continue
                ca, cb = clause_span(text, m.start(), m.end())
                if not A_PLAN.search(text[ca:cb]):
                    found = True
                    break
        if found:
            out.append(a.get("id"))
    return out


# ---------- B ----------

B_STOP_PRE = re.compile(
    r"初步|考虑|疑似|疑为|怀疑|拟诊|倾向|可能|需|需要|待|进一步|以排除|以明确|以便|用以|帮助|有助|评估|是否|若|如果|如|假如|不能|无法|难以|"
    r"未能|不除外|尚不|暂不|不排除|可疑")
B_HEDGE_POST = re.compile(r"^\s*(?:[？?]|可能|待查|待排|不除外|不能除外|不能排除|可疑|待定|待进一步|尚待)")
CONF_PRE = re.compile(r"(?:确诊为?|已确诊|证实为?|证实|明确诊断为?|诊断明确为?|最终诊断为?|病理(?:诊断|报告|证实|示|提示)为?)\s*$")
CONF_POST = re.compile(r"^\s*(?:诊断)?(?:明确|确诊|成立|诊断明确|诊断成立|可确诊|已确诊|已明确)")
EXCL_PRE = re.compile(r"(?:已排除|可排除|可以排除|排除|除外)\s*(?:了)?\s*$")
SOFT_EXCL_PRE = re.compile(r"(?:不支持|否定)\s*$")
SENT_PLAN = re.compile(r"需|需要|进一步|待|建议|完善|安排|拟|计划|以便|有待|是否|若|如果")
EXCL_POST = re.compile(r"^\s*(?:可|已|基本|予以?)?(?:排除|除外)")
WORK_PRE = re.compile(r"(?:目前|现|主要)?诊断(?:为|考虑为)?\s*$")
IMPR_PRE = re.compile(r"(?:符合|支持)\s*$")
TENT_PRE = re.compile(r"(?:暂不考虑|暂不支持|暂可排除|暂排除|暂时(?:可以?)?排除)\s*$")
TENT_POST = re.compile(r"^\s*(?:暂不考虑|暂不支持|暂可排除|暂排除|暂时(?:可以?)?排除)")


def firm_state(text, s, e):
    """诊断名 [s, e) 在正文里的口吻。返回 ("严格", 确诊|排除)、("宽松", 确诊|排除) 或 None。"""
    a, b = clause_span(text, s, e)
    pre = text[max(a, s - 16):s]
    post = text[e:min(b, e + 12)]
    if B_HEDGE_POST.search(post):
        return None
    if TENT_PRE.search(pre) or TENT_POST.search(post):
        return ("宽松", "排除")
    stopped = bool(B_STOP_PRE.search(pre))
    if not stopped:
        if CONF_PRE.search(pre) or CONF_POST.search(post):
            return ("严格", "确诊")
        if EXCL_PRE.search(pre) or EXCL_POST.search(post):
            sa, _ = sentence_span(text, s, e)
            return ("宽松", "排除") if SENT_PLAN.search(text[sa:s]) else ("严格", "排除")
        if SOFT_EXCL_PRE.search(pre):
            return ("宽松", "排除")
        if WORK_PRE.search(pre) or IMPR_PRE.search(pre):
            return ("宽松", "确诊")
    return None


def b_hits(draft, n, text):
    """底稿里第 n 天以后才确诊（或排除）的诊断，这篇里已经写成定论。返回 {"严格": [...], "宽松": [...]}。"""
    firsts = {}
    for j in sorted(draft["judgments"], key=lambda z: z["day"]):
        if j["state"] in ("确诊", "排除"):
            firsts.setdefault((j["diagnosis"], j["state"]), j["day"])
    out = {"严格": [], "宽松": []}
    for (name, st), d0 in firsts.items():
        if d0 <= n or len(name) < 2:
            continue
        best = None
        for m in re.finditer(re.escape(name), text):
            r = firm_state(text, m.start(), m.end())
            if r and r[1] == st:
                best = r[0] if best is None or r[0] == "严格" else best
                if best == "严格":
                    break
        if best:
            out[best].append((name, st, d0))
    return out


def oracle_leaks_v2(draft, slot, docs):
    """docs：finalize_doc 的结果列表。返回 (逐篇列表, 住院层汇总)。出院记录不算。
    汇总里 O、A、B 是判定用的口径（B 只算严格档）；B_loose 单列。"""
    per = []
    for x in docs:
        if x["role"] == "DS":
            continue
        n, text = x["day"], x["text"]
        b = b_hits(draft, n, text)
        per.append({"k": x["k"], "O": len(o_hits(draft, n, text)), "A": len(a_hits(draft, n, text)),
                    "B": len(b["严格"]), "B_loose": len(b["宽松"])})
    summary = {k: int(any(h[k] for h in per)) for k in ("O", "A", "B", "B_loose")}
    summary["any"] = int(summary["O"] or summary["A"] or summary["B"])
    summary["n_docs"] = len(per)
    return per, summary
