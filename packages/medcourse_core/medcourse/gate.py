"""L6 闸门 v0：第 7 节的全量硬规则，加计划对账。只出失败代码和文书序号，不出正文。

节级（写完马上查，过不了就重采样）：带单位的数值要在计划里；计划要写的事实要写到；提示词字样外泄；太短。
住院级（整次住院写完查）：g3-rules-0.3 全部规则和 N1；关1 的 G1-1、2、3、5、6；b7；跨文书 S1、S2、S3、S4b；结果提前。
G1-4、K1–K5、B2 精确率不够，不进住院级闸门（计划 v2 第 7.2 节）。

0.1.4：出院记录"入院情况"这一节（计划标 admission_only）借用 K4 的触发词做节级检查，命中就重写本节。K4 在真实病历上误报多，
是因为长段落里混进了没认出来的"诊疗经过"；这一节由我们单独写，没有这个问题。入院前的分句（"3 年前""门诊"）照 K4 跳过。
评测台的 K4 因此不再是独立读数，M1 的泄露看 oracle2。

0.1.5：漏写的必写事实每条记一次（0.1.4 漏一条就停，重采样选优分不出漏 1 条和漏 3 条）；必写的检查结果要落在结果口吻上
（同评测台 B_exam 的 tone_of），写成"完善 CT 检查"这类计划口吻记 result_tone_missing；"入院情况"三次都带触发词时，
由 strip_admission_leak 剔掉带触发词的分句再交出（流程里计数）。

0.1.8（第七次合成病历审阅、监督员第七轮）：节级多三项。truncated：句尾不是句号、问号、叹号（体格检查复读到 max_tokens 截在
"双肺"，病例特点截在"可见与主要"）；repeat：同一分句 ≥8 字出现 ≥3 次，或 ≥4 字出现 ≥5 次（"双肺底可闻及细湿啰音"复读 25 遍）；
nonsurg_postop：非手术住院写了"术后恢复""手术顺利"这类话。三次都不过时 repair_section 修补再交出：删掉重复的分句、
截回最后一个句号、删掉非手术住院的术后句。strip_admission_leak 剔掉触发分句时，同一句里紧跟着的"示……""结果……"分句一起剔
（0.1.7 留下"。示未见明显异常。"这种没头的半句）。

0.1.9（第八次合成病历审阅）：节级多两项。pe_history：体格检查、专科情况里写了"主诉、自述、病史、否认……"（第八次 14/20 份，
既往史写着冠心病、体格检查写"否认冠心病病史"）；hint_leak：带提示的节写了"详见另节""另述"这类话（我 0.1.8 的提示被写手照抄，2/20）。
三次都不过时 repair_section 删掉那个分句（pe_history）或那句话（hint_leak）。
审查员第八轮补充的"体格检查写检查结果""同一篇继续又加用同一种药"不进闸门：根因在上游（提示、查房重点给了今天才加的药），
在上游改，读数由 out_diag 的 qa_ 单独量（闸门和读数分开，不既改又考）。

0.1.12（第十一次 A256 真未来泄露 O 52/256，体格检查 12 条；老板 11-10 批准，按病历章节边界处理）：体格检查写检查结果改进节级闸门
（pe_result：检查词表里的检查名带结果口吻，不分是不是未来的检查），三次不过删分句和紧跟的从属分句（drop_pe_results）。
上游提示两轮没挡住（qa_adm_pe_result 6 → 7）。M1 的泄露读数仍是 oracle2，独立于这条闸门。

0.1.12.1（第十二次 P30 停在泄露线：入院记录 8 → 0，剩 1 条在查房正文；老板 11-10 批准）：result_unsourced，和 number_unsourced
一脉相承——写手写的节（出院记录除外）里，结果口吻的检查必须是截至当天已回报的（result_keys，流程按规划当天可见的结果给），
不看未来计划；三次不过删分句（drop_results）。比 oracle2 宽：规划里根本没有的检查也拦。
"""
import datetime as dt
import re

from .compiler import (
    BP_PAIR_RE,
    _INC_SKIP_SECTIONS,
    _has_delivery_event,
    exam_ts,
    no_incision,
    norm_unit,
)
from .rules import setup
from .stats import PAST_TERMS, past_matches

setup()
from b7_rules import check_record  # noqa: E402
from q45_kit.realtext import care7_from_admission  # noqa: E402
from ruler_g1 import evaluate_document, extract_facts, is_first_progress, split_sections  # noqa: E402
from ruler_g2 import build_real, evaluate_admission  # noqa: E402
from ruler_g2.events import EXAM_RE, NOT_PATH_AFTER, PRE_ADMISSION_SECTIONS, _ALIAS, tone_of  # noqa: E402
from ruler_g2.text import HIST_RE, clause_span, find_dates  # noqa: E402
from ruler_g2.readings import K4_RE, K5_PLAN, K5_PRE_OP_PLAN, k4_skip  # noqa: E402
from ruler_g3 import evaluate  # noqa: E402
from ruler_g3.adapters import from_admission  # noqa: E402


def _has_any(text, words):
    t = text or ""
    return any(w in t for w in words)


G1_GATE = {"G1-1", "G1-2", "G1-3", "G1-5", "G1-6"}
G2_GATE = ("S1", "S2", "S3", "S4b")
LEAK = ("【", "】", "虚构", "这一节", "只写上面", "事实", "本节要点", "书写要求", "照抄", "上一篇病程", "照搬", "<N", "<D", "延续情况")
_NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(mmol/L|μmol/L|umol/L|g/L|×10\^?9/L|10\^?9/L|mmHg|℃|次/分|U/L|mg/L|ng/mL|pg/mL|%|cm|mm|mg|g|ml|mL)", re.I)
_BP = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})\s*mmHg", re.I)
# 常规写法里的数不算"计划外数值"：溶液浓度、溶媒体积、瞳孔直径、血氧饱和度。切口长度、病灶大小这类仍算（提示词要求不写）
_CONC = re.compile(r"\s*(?:氯化钠|葡萄糖|碳酸氢钠|利多卡因|氯化钾|甘露醇|碘伏|聚维酮碘|乙醇|酒精|硫酸镁|葡萄糖酸钙|布比卡因|罗哌卡因|氨基酸|脂肪乳|过氧化氢)")
_VEHICLE = re.compile(r"(?:氯化钠|葡萄糖|注射用水|生理盐水|NS|GS)(?:注射液|溶液)?[^\d。；;\n]{0,3}$", re.I)
_PUPIL = re.compile(r"瞳孔[^。；;\n]{0,10}$")
_SPO2 = re.compile(r"(?:血氧饱和度|氧饱和度|血氧|SpO2|SaO2)[^\d。；;\n]{0,6}$", re.I)


def routine_number(text, m):
    """这个带单位的数是不是常规写法。"""
    v, unit, before = float(m.group(1)), m.group(2).lower(), text[max(0, m.start() - 14):m.start()]
    if unit == "%":
        return bool(_CONC.match(text, m.end())) or (85 <= v <= 100 and bool(_SPO2.search(before)))
    if unit == "ml":
        return bool(_VEHICLE.search(before))
    return unit == "mm" and v <= 6 and bool(_PUPIL.search(before))
MIN_SHARE = 0.4
_END_OK = "。！？!?"
_CLAUSE_SPLIT = re.compile(r"[，,。；;！？!?\n]")
_POSTOP = re.compile(r"术后(?:恢复|第|切口|伤口|复查|康复|处理|观察|注意|病理|顺利)|手术(?:顺利|过程)|术区|(?:切口|伤口)(?:愈合|敷料|干燥|干洁|清洁|红肿|渗血|渗液|拆线|感染|护理|疼痛)")
REPEAT_LONG, REPEAT_SHORT = (8, 3), (4, 5)
# 0.1.9 / 0.3.2
PE_SECTIONS = ("体格检查", "专科情况")
_PE_HISTORY_WORDS = ("主诉", "自述", "诉有", "病史", "否认", "既往")
_HINT_LEAK_WORDS = ("另节", "另一节", "另有一节", "另述", "详见", "另行记录", "不再赘述", "见上文", "如上所述", "已在开头")
# 0.21 / 0.30 / 0.31 / 0.32：写手端高频 qa_* 章节闸门与兜底修补（PIPELINE_SAFE_FIX 仅保留纯格式修补，语义重违交给上游与重试）
_PE_UNTESTED = re.compile(r"(?:体温|血压|脉搏|心率|呼吸)(?:频率)?未测")
_PE_DUP_VITALS = re.compile(r"(?:体温|脉搏|呼吸|血压)[^。；;\n]{0,14}(?:已记录|已另录|见开头|见前录?|如前所述)|(?:体温|脉搏|血压)\s*[:：]?\s*\d+(?:\.\d+)?\s*(?:℃|次/分|/|mmHg)|(?:^|[，,。；;\s])(?:T|P|R|BP)\s*[:：]?\s*\d")
_SENIOR_WORDS = ("主任医师", "副主任医师", "主治医师", "上级医师")
_BEFORE_OP_HARD = re.compile(
    r"手术(?:过程)?顺利|术中(?:顺利|见|所见)|术后第?\s*[\d一二三四五六七八九十]+\s*(?:天|日|小时|h)|"
    r"术后(?:恢复|切口|伤口|患者)|(?:安返|返回|送返)病房|麻醉(?:清醒|苏醒)|"
    r"(?:切口|伤口|术区)(?:敷料|愈合|干洁|干燥|无渗血|无红肿)|(?:腹腔|胸腔|盆腔|切口|术区|伤口|T管)?引流管(?:在位|通畅|引流通畅|引流出)"
)
_POSTOP_DAY_MENTION = re.compile(r"术后第\s*([0-9]+|[一二三四五六七八九十]+)\s*[天日]")
_NORM_CLAIM_PHRASE = r"(?:均?未见明显异常|均?未见异常|均正常|结果均?正常|在正常范围|未见明显升高|未见明显降低)"
_LAB_ABN_NORM_RE = re.compile(
    r"(?:血常规|尿常规|粪常规|大便常规|肝功能|肾功能|凝血|电解质|心肌酶|肌钙蛋白|血脂|血气|C反应蛋白|降钙素原|糖化血红蛋白|甲状腺功能|辅助检查|实验室检查|上述指标)[^，,。；;\n]{0,12}"
    + _NORM_CLAIM_PHRASE
)
_PANEL_NORM_PAT = {
    "血常规": r"血常规|白细胞|血红蛋白|血小板",
    "尿常规": r"尿常规",
    "粪常规": r"粪常规|大便常规|便常规",
    "肝功能": r"肝功能|转氨酶|胆红素|白蛋白",
    "肾功能": r"肾功能|肌酐|尿素氮",
    "凝血": r"凝血(?:功能|四项|象)?",
    "D二聚体": r"D[-—]?二聚体",
    "电解质": r"电解质|血钾|血钠|血氯|血钙",
    "心肌酶": r"心肌酶|肌钙蛋白|肌酸激酶",
    "利钠肽": r"利钠肽|BNP|NT-?proBNP",
    "血糖": r"血糖",
    "淀粉酶": r"淀粉酶",
    "血脂": r"血脂|胆固醇|甘油三酯",
    "C反应蛋白": r"C反应蛋白|CRP",
    "降钙素原": r"降钙素原|PCT",
    "糖化血红蛋白": r"糖化血红蛋白",
    "甲状腺功能": r"甲状腺功能|甲功",
}
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_int(s):
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in _CN_DIGITS:
        return _CN_DIGITS[s]
    if s.startswith("十") and len(s) == 2 and s[1] in _CN_DIGITS:
        return 10 + _CN_DIGITS[s[1]]
    if len(s) == 2 and s[1] == "十" and s[0] in _CN_DIGITS:
        return _CN_DIGITS[s[0]] * 10
    if len(s) == 3 and s[1] == "十" and s[0] in _CN_DIGITS and s[2] in _CN_DIGITS:
        return _CN_DIGITS[s[0]] * 10 + _CN_DIGITS[s[2]]
    return None
_BEFORE_OP_ACT = re.compile(
    r"(?:于?(?:今日|今天|今晨|昨日|昨天))(?:在[^，,。；;\n]{0,15}下)?行|已在[^，,。；;\n]{0,15}下行|"
    r"(?:已(?:于[^，,。；;\n]{0,10})?(?:顺利)?(?:行|完成|实施|做)|顺利完成|已经(?:完成|行)|行了)"
    r"[^，,。；;\n]{0,25}(?:术|手术|造影|支架|置入|切除|穿刺|引流|活检|内固定|成形|修补|松解|清创|复位)"
)
_BEFORE_OP_EXEMPT = re.compile(
    r"拟|计划|准备|择期|明日|明天|将|待|术前|建议|预约|决定|需|需要|考虑|必要时|安排|适时|是否|评估|"
    r"以便|确保|可行|宜|应|如|若|定于|尚未|告知|风险|可能|并发症|交代|\d+\s*年前|既往|曾"
)
_ENG_PROSE = re.compile(r"(?<![A-Za-z0-9-])([A-Za-z]{4,}(?:\s+[A-Za-z]{4,})+)(?![A-Za-z0-9-])")
_META_TEXT = re.compile(r"(?:记录|病程)(?:完整|真实)|无遗漏|内容真实|符合(?:病历)?书写规范")
_INCISION_WORDS = ("切口", "刀口", "引流")
_POSTPARTUM_WORDS = ("恶露", "乳汁", "哺乳", "产后恢复", "产褥")
_ADULT_CHRON_WORDS = ("高血压", "糖尿病", "冠心病", "冠状动脉粥样硬化", "脑梗死", "脑卒中", "慢阻肺", "慢性阻塞性肺", "房颤", "心房颤动")
_GENERIC_PAST_DENY_WORDS = ("既往无特殊", "既往体健")
FIXABLE = ("truncated", "repeat", "nonsurg_postop", "pe_history", "hint_leak", "pe_result", "result_unsourced", "dx_settled_early",
           "pe_untested", "pe_dup_vitals", "senior_in_daily", "before_op_done", "no_incision_leak", "undelivered_postpartum", "early_preg_fetal",
           "english_prose", "meta_text", "past_contradict", "future_date")
PIPELINE_SAFE_FIX = ("truncated", "repeat", "english_prose", "meta_text", "hint_leak")
SOFT_SECTION_FAILS = frozenset({"too_short", "repeat"})
HARD_SECTION_FAILS = frozenset({
    "number_unsourced", "item_number_mismatch", "item_stale_value", "item_parallel_ambiguous",
    "fact_missing", "result_tone_missing", "dx_polarity_missing",
    "admission_leak", "prompt_leak", "nonsurg_postop", "pe_history", "pe_result",
    "pe_untested", "pe_dup_vitals", "senior_in_daily", "before_op_done",
    "no_incision_leak", "undelivered_postpartum", "early_preg_fetal",
    "result_unsourced", "dx_settled_early", "past_contradict", "future_date",
    "postop_day_mismatch", "lab_abn_called_normal", "english_prose", "meta_text",
    "hint_leak", "empty_after_repair",
})

# 0.2.0（M1 严格档 B 2/1023，老板 11-11 批准）：诊断定论提前（dx_settled_early）。词表自己写，不 import 评测台的 oracle2（闸门和记分牌分家）。
# 当天规划里还是"疑似"的诊断（unsettled_dx，流程按截至当天的可见状态给），写手写成确诊或排除定论、且没有考虑/待/需等缓冲词时命中。
_DX_TENT_POST = re.compile(r"^[？?\s]*(?:[？?]|待排|待查|待明确|尚待|可能|可疑|不除外|不排除|未排|尚未排除)")
_DX_TENT_PRE = re.compile(r"考虑|拟诊|疑似|可疑|待排|待查|待除外|不除外|不排除|尚未排除|未排除|不能排除|难以排除|需排除|需要排除|待进一步|尚待|鉴别")
_DX_FIRM_CONF_PRE = re.compile(r"确诊(?:为)?|明确诊断(?:为)?|已确诊|证实(?:为)?")
_DX_FIRM_CONF_POST = re.compile(r"诊断明确|已确诊|已明确|明确诊断|(?:^|[，,\s])确诊(?![？?])")
_DX_DIAG_BEFORE = re.compile(r"诊断(?:为)?[^，,。；;\n]{0,4}$")
_DX_MINGQUE_AFTER = re.compile(r"^\s*明确(?![？?])")
_DX_FIRM_EXCL_PRE = re.compile(r"已排除|可排除|予以排除|基本排除|排除|除外|不支持|不考虑")
_DX_FIRM_EXCL_POST = re.compile(r"已排除|可排除|予以排除|排除|除外")
_DX_PLAN_SENT_PRE = re.compile(r"待|拟|完善|进一步|需|需要|查明|以便|协助")


def settled_dx_mentions(text, unsettled, settled=()):
    """正文里把当天仍处于疑似状态的诊断写成确诊或排除定论的位置。
    settled：当天已确诊或已排除的更长诊断名（如"社区获得性肺炎"已排除、"肺炎"仍疑似时，不把"排除社区获得性肺炎"判成定论提前）。"""
    t = text or ""
    longer = [(m.start(), m.end()) for o in (settled or ()) if o and len(o) >= 2 for m in re.finditer(re.escape(o), t)]
    out = []
    for name in unsettled or ():
        if not name or len(name) < 2:
            continue
        for m in re.finditer(re.escape(name), t):
            s, e = m.start(), m.end()
            if any(a0 <= s and e <= b0 and (b0 - a0) > (e - s) for a0, b0 in longer):
                continue
            a, b = clause_span(t, s, e)
            pre, post = t[max(a, s - 16):s], t[e:min(b, e + 12)]
            if _DX_TENT_POST.search(post) or _DX_TENT_PRE.search(pre):
                continue
            conf = bool(_DX_FIRM_CONF_PRE.search(pre) or _DX_FIRM_CONF_POST.search(post)
                        or (_DX_DIAG_BEFORE.search(pre) and _DX_MINGQUE_AFTER.search(post)))
            excl = False
            if not conf and (_DX_FIRM_EXCL_PRE.search(pre) or _DX_FIRM_EXCL_POST.search(post)):
                excl = not _DX_PLAN_SENT_PRE.search(t[a:s])
            if conf or excl:
                out.append(m)
    return out


def drop_settled_dx(text, unsettled, settled=()):
    """删掉对未定论诊断下定论的分句；被删分句句末有句号时留给前一个分句。只删不添。"""
    t = text or ""
    starts = [m.start() for m in settled_dx_mentions(t, unsettled, settled)]
    return _drop_clauses_if(t, lambda c: any(c.start() <= s < c.end() for s in starts)) if starts else t


# 0.1.12.1：待回报、计划的说法不算结果。程序给的事实就这样写（"已开具……，待回报""待胸部CT、痰培养回报后进一步明确"），
# 评测台的口吻判断见到"回报后"会判成结果口吻；按分句里的待办词、同一句里提及之前的计划词跳过。
# 不按整句跳过："心电图示窦性心律，余检查待回报"前半句仍是编出来的结果
_PEND_POST = re.compile(r"(?:待回|待回报|待出|未回|未出|尚未|待报|已送检|已预约|已申请|待做|待查|待完善|结果待|回报后|结果将)")
_PEND_PRE = re.compile(r"完善|待|等待|尚待|安排|预约|申请|送检|计划|拟|开具|包括")
_PEND_RESULT_WORDS = ("示", "提示", "显示", "回报", "报告", "结果", "证实", "可见")
_RESULT_IMM_AFTER = re.compile(r"^[^，,。；;\n]{0,6}?(?:示|提示|显示|证实|所见|可见)")


def result_mentions(text, allowed=None):
    """结果口吻的检查名（评测台检查词表）。allowed 给了时只留键不在 allowed 里的（0.1.12.1 result_unsourced）；
    None 表示一律算（0.1.12 体格检查）。病理后面跟"征、反射"这类的不是检查（同评测台 mentions_in）。
    0.21.3 收紧：仅后文有明确待回/回报后字样，或前文有计划触发词且后文未紧接结果触发词时才跳过，防止"进一步治疗"误豁免。"""
    t = text or ""
    out = []
    for m in EXAM_RE.finditer(t):
        key = _ALIAS[m.group(0)]
        if key == "病理" and NOT_PATH_AFTER.match(t, m.end()):
            continue
        if allowed is not None and key in allowed:
            continue
        if tone_of(t, m.start(), m.end())[0] != "R":
            continue
        if allowed is not None:
            a, b = clause_span(t, m.start(), m.end())
            pre, post = t[a:m.start()], t[m.end():b]
            if _PEND_POST.search(post):
                continue
            trigs = list(_PEND_PRE.finditer(pre))
            if trigs and not _has_any(pre[trigs[-1].end():], _PEND_RESULT_WORDS) and not _RESULT_IMM_AFTER.match(post):
                continue
        out.append(m)
    return out


def pe_results(text):
    """0.1.12（老板 11-10 批准，病历章节边界）：体格检查、专科情况只写查体所见。检查名（评测台检查词表）带结果口吻的位置。
    第十一次 A256 体格检查 12 条真未来泄露（尿常规 8、心电图 3），提示里"不写化验、心电图和影像结果"连续两轮没挡住（6 → 7 份）。
    比 oracle2 宽：不管是不是还没回报的检查，查体节里的检查结果一律算。"""
    return result_mentions(text)


def result_keys(exams):
    """当天已回报的检查对应的检查词表键：规划的键，加上名字里能配到的别名键（规划的"心肌肌钙蛋白"，写手写"肌钙蛋白"）。"""
    keys = set()
    for e in exams:
        keys.add(e["key"])
        keys.update(_ALIAS[m.group(0)] for m in EXAM_RE.finditer(e.get("name") or ""))
    return sorted(keys)


def drop_pe_results(text):
    return drop_results(text)


def drop_results(text, allowed=None):
    """删掉带检查结果的分句（allowed 同 result_mentions），连同紧跟着的从属分句（"示……""未见……"），到句号为止。只删不添。"""
    t = text or ""
    starts = [m.start() for m in result_mentions(t, allowed)]
    return _drop_clauses_if(t, lambda c: any(c.start() <= s < c.end() for s in starts), drop_dep=True) if starts else t


def truncated(text):
    t = (text or "").rstrip()
    if not t:
        return False
    if t[-1] in _END_OK:
        return False
    return not (t[-1] in "）)”】" and len(t) > 1 and t[-2] in _END_OK)


def _cjk5(s):
    st = s.strip()
    return st[:5] if len(st) >= 6 and all("\u4e00" <= c <= "\u9fff" for c in st[:5]) else None


def _loop_prefixes(text):
    cnt = {}
    for m in _SENT_PIECE.finditer(text or ""):
        p5 = _cjk5(m.group(0))
        if p5:
            cnt[p5] = cnt.get(p5, 0) + 1
    return {p for p, n in cnt.items() if n >= 6}


def repeated_clauses(text):
    """同一节里重复太多的分句：≥8 字出现 ≥3 次，或 ≥4 字出现 ≥5 次，或连续 ≥6 句共享同一 5 字中文开头。"""
    cnt = {}
    for c in _CLAUSE_SPLIT.split(text or ""):
        c = c.strip()
        if len(c) >= REPEAT_SHORT[0]:
            cnt[c] = cnt.get(c, 0) + 1
    out = {c for c, n in cnt.items() if (len(c) >= REPEAT_LONG[0] and n >= REPEAT_LONG[1]) or n >= REPEAT_SHORT[1]}
    bad_p5 = _loop_prefixes(text)
    if bad_p5:
        for m in _SENT_PIECE.finditer(text or ""):
            s = m.group(0).strip()
            if _cjk5(s) in bad_p5:
                out.add(s)
    return out


def _clean(s):
    return re.sub(r"[\s，,。；;：:、！？!?\-—–（）()]", "", s or "")


_FACT_KEY_ALIASES = {
    "D二聚体": ("D二聚体", "D-二聚体"),
    "利钠肽": ("利钠肽", "脑钠肽", "BNP", "NT-proBNP"),
    "心肌肌钙蛋白": ("心肌肌钙蛋白", "肌钙蛋白", "cTnI"),
    "肌钙蛋白": ("肌钙蛋白", "心肌肌钙蛋白", "cTnI"),
}
_RESULT_NUM_AFTER = re.compile(
    r"^[^，,。；;\n]{0,18}?\d+(?:\.\d+)?\s*(?:mmol/L|μmol/L|umol/L|g/L|×10\^?9/L|10\^?9/L|mmHg|℃|次/分|U/L|mg/L|ng/mL|pg/mL|%|cm|mm|mg|g|ml|mL)",
    re.I,
)
_RESULT_QUAL_AFTER = re.compile(
    r"^[^，,。；;\n]{0,14}?(?:偏高|偏低|升高|降低|升高明显|阳性|阴性|正常|异常|未见异常|未见明显异常|未培养出|培养出|可见|符合|提示)"
)
_DOC_HM_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\s+(\d{2}:\d{2})\b")


def _key_variants(key):
    base = [key] if key else []
    for k2 in _FACT_KEY_ALIASES.get(key or "", ()):
        if k2 not in base:
            base.append(k2)
    return base


def _has_result_evidence(text, key):
    t = text or ""
    pats = []
    for v in _key_variants(key):
        sub = v[:4] if len(v) >= 4 else v
        if sub == "D-二聚":
            pats.append(r"D-?二聚体?")
        else:
            pats.append(re.escape(sub))
    rx = re.compile("|".join(dict.fromkeys(pats)))
    for m in rx.finditer(t):
        if tone_of(t, m.start(), m.end())[0] == "R":
            return True
        a, b = clause_span(t, m.start(), m.end())
        pre, post = t[a:m.start()], t[m.end():b]
        if _PEND_POST.search(post):
            continue
        trigs = list(_PEND_PRE.finditer(pre))
        pending_pre = bool(trigs and not _has_any(pre[trigs[-1].end():], _PEND_RESULT_WORDS))
        if not pending_pre and (_RESULT_NUM_AFTER.search(post) or _RESULT_QUAL_AFTER.search(post)):
            return True
    return False


def _has_dx_firm_polarity(text, name, req_pol):
    t = text or ""
    cands = [name]
    stripped = re.sub(r"[（(][^）)]*[）)]", "", name or "").strip()
    if stripped and len(stripped) >= 2 and stripped not in cands:
        cands.append(stripped)
    for cand in cands:
        for m in re.finditer(re.escape(cand), t):
            s, e = m.start(), m.end()
            a, b = clause_span(t, s, e)
            pre, post = t[max(a, s - 16):s], t[e:min(b, e + 12)]
            if _DX_TENT_POST.search(post) or _DX_TENT_PRE.search(pre):
                continue
            if req_pol == "确诊":
                if (_DX_FIRM_CONF_PRE.search(pre) or _DX_FIRM_CONF_POST.search(post)
                        or (_DX_DIAG_BEFORE.search(pre) and _DX_MINGQUE_AFTER.search(post))
                        or re.search(r"明确(?:诊断)?(?:存在)?$", pre)
                        or re.search(r"^[：:]?\s*(?:诊断)?(?:成立|明确)", post)):
                    return True
            else:
                if ((_DX_FIRM_EXCL_PRE.search(pre) or _DX_FIRM_EXCL_POST.search(post)
                     or re.search(r"基本除外|可除外", pre) or re.search(r"^[，,\s]*(?:可能性极?低|可予?排除)", post))
                        and not _DX_PLAN_SENT_PRE.search(t[a:s])):
                    return True
    return False


def _doc_hm(d):
    if d.get("time"):
        return d["time"]
    m = _DOC_HM_RE.search((d.get("text") or "")[:220])
    return m.group(1) if m else "23:59"


_PANEL_KEY_TOKEN = re.compile(
    "|".join(f"(?P<K{i}>{pat})" for i, (_, pat) in enumerate(_PANEL_NORM_PAT.items()))
    + "|(?P<K_BLANKET>(?:各项|上述|相关)?(?:辅助检查|实验室检查|化验|上述指标|各项指标))"
    + "|(?P<K_OTHER>心电图|心脏超声|超声|胸片|胸部CT|头颅CT|腹部CT|CT|磁共振|MRI)"
)
_PANEL_KEY_ORDER = [k for k in _PANEL_NORM_PAT]
_NORM_CLAIM_RE = re.compile(_NORM_CLAIM_PHRASE)


def _has_abn_lab_called_normal(plan, text):
    if not text:
        return False
    abn_keys = plan.get("abn_lab_keys")
    if abn_keys:
        abn_set = set(abn_keys)
        for m in _NORM_CLAIM_RE.finditer(text):
            sent_a = max(text.rfind(x, 0, m.start()) + 1 for x in "。；;\n")
            ctx = text[max(sent_a, m.start() - 28):m.start()]
            last_k = None
            for pm in _PANEL_KEY_TOKEN.finditer(ctx):
                g = pm.lastgroup
                if g in ("K_BLANKET", "K_OTHER"):
                    last_k = g[2:]
                else:
                    last_k = _PANEL_KEY_ORDER[int(g[1:])]
            if last_k == "BLANKET" or last_k in abn_set:
                return True
        return False
    return bool(plan.get("has_abn_labs") and _LAB_ABN_NORM_RE.search(text))


def future_dates(text, doc_date_str, is_pre_op=False):
    """正文里晚于本篇书写日且非计划/术前口吻的日期 [(start, end, date)]。"""
    if not text or not doc_date_str:
        return []
    doc_d = dt.date.fromisoformat(doc_date_str) if isinstance(doc_date_str, str) else doc_date_str
    out = []
    for s, e, x in find_dates(text, doc_d):
        if (x - doc_d).days < 1:
            continue
        a, b = clause_span(text, s, e)
        clause = text[a:b]
        if K5_PLAN.search(clause) or (is_pre_op and K5_PRE_OP_PLAN.search(clause)):
            continue
        out.append((s, e, x))
    return out


_LAB_ALIAS_PAIRS = (
    ("糖化血红蛋白", "HBA1C"), ("HbA1c", "HBA1C"), ("HBA1C", "HBA1C"),
    ("活化部分凝血活酶时间", "APTT"), ("APTT", "APTT"),
    ("凝血酶原时间", "PT"), ("PT", "PT"),
    ("国际标准化比值", "INR"), ("INR", "INR"),
    ("纤维蛋白原", "FIB"), ("FIB", "FIB"),
    ("D-二聚体", "DD"), ("D—二聚体", "DD"), ("D－二聚体", "DD"), ("D二聚体", "DD"),
    ("C-反应蛋白", "CRP"), ("C—反应蛋白", "CRP"), ("C反应蛋白", "CRP"), ("CRP", "CRP"),
    ("降钙素原", "PCT"), ("PCT", "PCT"),
    ("心肌肌钙蛋白I", "TNI"), ("肌钙蛋白I", "TNI"), ("心肌肌钙蛋白", "TNI"), ("cTnI", "TNI"), ("TnI", "TNI"), ("TNI", "TNI"),
    ("肌酸激酶同工酶", "CKMB"), ("CK-MB", "CKMB"), ("CKMB", "CKMB"),
    ("肌酸激酶", "CK"), ("CK", "CK"),
    ("乳酸脱氢酶", "LDH"), ("LDH", "LDH"),
    ("中性粒细胞比例", "NEUT"), ("中性粒细胞百分比", "NEUT"), ("中性粒细胞", "NEUT"), ("NEUT%", "NEUT"), ("NEUT", "NEUT"),
    ("白细胞计数", "WBC"), ("白细胞", "WBC"), ("WBC", "WBC"),
    ("血红蛋白", "HGB"), ("HGB", "HGB"),
    ("血小板计数", "PLT"), ("血小板", "PLT"), ("PLT", "PLT"),
    ("谷丙转氨酶", "ALT"), ("丙氨酸氨基转移酶", "ALT"), ("ALT", "ALT"),
    ("谷草转氨酶", "AST"), ("天门冬氨酸氨基转移酶", "AST"), ("AST", "AST"),
    ("总胆红素", "TBIL"), ("TBIL", "TBIL"),
    ("白蛋白", "ALB"), ("ALB", "ALB"),
    ("血肌酐", "CREA"), ("肌酐", "CREA"), ("CREA", "CREA"),
    ("尿素氮", "BUN"), ("BUN", "BUN"),
    ("血尿酸", "UA"), ("尿酸", "UA"), ("UA", "UA"),
    ("血清钾", "K"), ("血钾", "K"),
    ("血清钠", "NA"), ("血钠", "NA"),
    ("血清氯", "CL"), ("血氯", "CL"),
    ("血清钙", "CA"), ("血钙", "CA"),
    ("空腹血糖", "GLU"), ("血糖", "GLU"), ("GLU", "GLU"),
    ("血淀粉酶", "AMY"), ("淀粉酶", "AMY"), ("AMY", "AMY"),
    ("总胆固醇", "TC"), ("TC", "TC"),
    ("甘油三酯", "TG"), ("TG", "TG"),
    ("低密度脂蛋白胆固醇", "LDL"), ("低密度脂蛋白", "LDL"), ("LDL-C", "LDL"), ("LDL", "LDL"),
    ("BNP", "BNP"), ("利钠肽", "BNP"),
)
_LAB_ALIAS_MAP = {k.lower(): v for k, v in _LAB_ALIAS_PAIRS}


def _compile_lab_item_re():
    parts = []
    for k, _ in sorted(_LAB_ALIAS_PAIRS, key=lambda kv: (-len(kv[0]), kv[0])):
        p = re.escape(k)
        if k == "血红蛋白":
            p = r"(?<!糖化)(?<!红细胞)" + p
        elif k == "白蛋白":
            p = r"(?<!前)" + p
        elif k in ("C-反应蛋白", "C—反应蛋白", "C反应蛋白"):
            p = r"(?<!超敏)" + p
        if k[0].isascii() and k[0].isalpha():
            p = r"(?<![A-Za-z-])" + p
        if k[-1].isascii() and k[-1].isalpha():
            p = p + r"(?![A-Za-z-])"
        parts.append(p)
    return re.compile("|".join(parts), re.I)


_LAB_ITEM_RE = _compile_lab_item_re()
_LAB_VAL_TOK_RE = re.compile(
    r"(?<![\d.\^])(\d+(?:\.\d+)?)(?![\d.])\s*"
    r"((?:[×x*]\s*)?10\^?\d+/(?:[A-Za-z]+|升)|毫摩尔/升|微摩尔/升|纳克/毫升|皮克/毫升|毫克/升|克/升|单位/升|mmol/L|μmol/L|umol/L|ng/mL|pg/mL|mg/dL|mg/L|g/L|U/L|IU/L|次/分|mmHg|℃|%|秒|s|mg|g|ml|mL)?",
    re.I,
)
_SPAN_STOPS = ("参考", "正常范围", "正常值", "体温", "脉搏", "心率", "呼吸", "血压", "血氧", "瞳孔", "予", "给予", "口服", "静滴")
_DATE_OR_TIME_CH = ("年", "月", "日", "号", "天", "周", "时", "分", "例", "次", "片", "粒", "支", "袋", "期", "型", "级")
_NON_LAB_UNITS = ("℃", "mmhg", "次/分", "cm", "mm")
_CUR_CUES = ("目前", "现", "今日", "今查", "本次", "最新", "出院前", "末次", "仍为", "仍在", "仍", "降至", "升至", "回落", "恢复至")
_HIST_CUES = ("入院", "前次", "初次", "原为", "原", "曾", "既往", "由", "从", "较前", "较入院")
_QUAL_START = ("偏高", "偏低", "升高", "降低", "正常", "未见", "阴性", "阳性", "较前", "恢复", "均", "等", "、", "和", "与", "及", "。")


def _clause_prefix(text, pos, sent_start):
    i = pos - 1
    while i >= sent_start:
        if text[i] in "。！？!?\n；;，,":
            return text[i + 1:pos]
        i -= 1
    return text[sent_start:pos]


def _is_date_or_time_tok(seg, nm, raw_u):
    if raw_u:
        return False
    s, e, tok = nm.start(), nm.end(), nm.group(1)
    prv = seg[s - 1:s] if s > 0 else ""
    nxt = seg[e:e + 1]
    if nxt in _DATE_OR_TIME_CH:
        return True
    if nxt == "个" and seg[e + 1:e + 2] in ("月", "周", "天", "年"):
        return True
    if prv == ":" or nxt == ":":
        return True
    if tok.isdigit() and ((prv in "-/" and (len(tok) == 2 or nxt in "-/")) or (nxt in "-/" and len(tok) == 4)):
        return True
    return False


def _extract_val_matches(seg, accept_roles):
    out = []
    for nm in _LAB_VAL_TOK_RE.finditer(seg):
        head = seg[:nm.start()].rstrip()
        if head.endswith(("^", "×10", "*10", "x10")):
            continue
        raw_u = nm.group(2) or ""
        if _is_date_or_time_tok(seg, nm, raw_u):
            continue
        u_norm = norm_unit(raw_u) if raw_u else ""
        if u_norm in _NON_LAB_UNITS:
            continue
        if u_norm in ("mg", "g", "ml") and not any(r and r.get("unit") == u_norm for r in accept_roles):
            continue
        out.append((nm, nm.group(1), u_norm))
    return out


def audit_item_numbers(plan, text):
    """0.3.7：逐项配对核对 (实体, 数值, 单位) 与复查新旧值角色；“分别为”可一对一配对时按序核对，无法配对则计入未解析并阻断。"""
    roles = plan.get("lab_item_roles") or {}
    t = text or ""
    if not roles or not t:
        return {"fails": [], "parsed": 0, "unparsed": 0, "qualitative_only": 0, "hits": [], "unparsed_details": []}
    ms = list(_LAB_ITEM_RE.finditer(t))
    parsed = unparsed = qualitative_only = 0
    fails, hits, unparsed_details = [], [], []
    occ_by_ik = {}
    idx = 0
    while idx < len(ms):
        m = ms[idx]
        ik = _LAB_ALIAS_MAP.get(m.group(0).lower())
        role = roles.get(ik)
        sent_start = max((t.rfind(ch, 0, m.start()) for ch in "。！？!?\n；;"), default=-1) + 1
        sent_end = len(t)
        for ch in "。！？!?\n；;":
            pos = t.find(ch, m.end(), sent_end)
            if pos != -1:
                sent_end = pos

        chain = [m]
        chain_end = idx
        while chain_end + 1 < len(ms) and ms[chain_end + 1].end() <= sent_end:
            gap = t[ms[chain_end].end():ms[chain_end + 1].start()]
            if gap.strip(" 、,，/及和与同\t"):
                break
            chain_end += 1
            chain.append(ms[chain_end])

        last_m = chain[-1]
        span_end = sent_end
        if chain_end + 1 < len(ms) and ms[chain_end + 1].start() < span_end:
            span_end = ms[chain_end + 1].start()
        for kw in _SPAN_STOPS:
            pos = t.find(kw, last_m.end(), span_end)
            if pos != -1:
                span_end = pos
        tail_seg = t[last_m.end():span_end]
        chain_iks = [_LAB_ALIAS_MAP.get(mj.group(0).lower()) for mj in chain]
        chain_roles = [roles.get(k) for k in chain_iks]
        tail_vals = _extract_val_matches(tail_seg, chain_roles) if tail_seg.strip() else []
        has_fenbie = ("分别" in tail_seg[:tail_vals[0][0].start()]) if tail_vals else (len(chain) >= 2 and "分别" in tail_seg)

        if has_fenbie:
            fb_pos = tail_seg.index("分别")
            pre_fb = tail_seg[:fb_pos].strip(" 均各值结果水平指标检查提示示为是系在：:\t")
            clean_parallel = (
                len(chain) >= 2
                and len(tail_vals) == len(chain)
                and pre_fb == ""
                and all(k in roles for k in chain_iks)
            )
            if clean_parallel:
                pre_item = _clause_prefix(t, chain[0].start(), sent_start)
                between_head = tail_seg[:tail_vals[0][0].start()]
                prev_end = 0
                for j_idx, (k_j, role_j, (nm_j, val_s_j, u_norm_j)) in enumerate(zip(chain_iks, chain_roles, tail_vals)):
                    parsed += 1
                    between_prev = (between_head if j_idx == 0 else "") + tail_seg[prev_end:nm_j.start()]
                    prev_end = nm_j.end()
                    allowed_all_j = {role_j["latest"]} | set(role_j.get("priors") or ())
                    unit_bad = (u_norm_j in ("mg", "g", "ml") and (not role_j.get("unit") or u_norm_j != role_j["unit"])) or (
                        u_norm_j and role_j.get("unit") and u_norm_j != role_j["unit"]
                    )
                    if val_s_j not in allowed_all_j or unit_bad:
                        if "item_number_mismatch" not in fails:
                            fails.append("item_number_mismatch")
                        hits.append(("item_number_mismatch", k_j, val_s_j, u_norm_j, role_j["latest"]))
                        continue
                    has_cur = _has_any(between_prev, _CUR_CUES) or (
                        _has_any(pre_item, _CUR_CUES) and not _has_any(between_prev, _HIST_CUES)
                    )
                    has_hist = _has_any(between_prev, _HIST_CUES) or (
                        not has_cur and _has_any(pre_item, _HIST_CUES)
                    )
                    occ_by_ik.setdefault(k_j, []).append((val_s_j, u_norm_j, has_cur, has_hist))
            else:
                tracked = [(mj, k_j) for mj, k_j in zip(chain, chain_iks) if k_j in roles]
                for mj, k_j in tracked:
                    unparsed += 1
                    unparsed_details.append({"ik": k_j, "item": mj.group(0), "seg": tail_seg[:80], "reason": "parallel_ambiguous"})
                if tracked:
                    if "item_parallel_ambiguous" not in fails:
                        fails.append("item_parallel_ambiguous")
                    hits.append(("item_parallel_ambiguous", ",".join(k_j for _, k_j in tracked), str(len(tail_vals)), "", str(len(chain))))
            idx = chain_end + 1
            continue

        idx += 1
        if not role:
            continue
        span_end = sent_end
        if idx < len(ms) and ms[idx].start() < span_end:
            span_end = ms[idx].start()
        for kw in _SPAN_STOPS:
            pos = t.find(kw, m.end(), span_end)
            if pos != -1:
                span_end = pos
        seg = t[m.end():span_end]
        if not seg.strip():
            qualitative_only += 1
            continue
        pre_item = _clause_prefix(t, m.start(), sent_start)
        val_matches = _extract_val_matches(seg, (role,))
        if not val_matches:
            s_strip = seg.lstrip("：:约为示在仍 \t")
            if not s_strip or any(s_strip.startswith(q) for q in _QUAL_START):
                qualitative_only += 1
            else:
                unparsed += 1
                unparsed_details.append({"ik": ik, "item": m.group(0), "seg": seg[:80], "reason": "no_numeric_token_in_span"})
            continue
        allowed_all = {role["latest"]} | set(role.get("priors") or ())
        prev_end = 0
        for num_idx, (nm, val_s, u_norm) in enumerate(val_matches):
            parsed += 1
            between_prev = seg[prev_end:nm.start()]
            prev_end = nm.end()
            if val_s not in allowed_all or (u_norm and role.get("unit") and u_norm != role["unit"]):
                if "item_number_mismatch" not in fails:
                    fails.append("item_number_mismatch")
                hits.append(("item_number_mismatch", ik, val_s, u_norm, role["latest"]))
                continue
            has_cur = _has_any(between_prev, _CUR_CUES) or (
                num_idx == 0 and _has_any(pre_item, _CUR_CUES) and not _has_any(between_prev, _HIST_CUES)
            )
            has_hist = _has_any(between_prev, _HIST_CUES) or (
                num_idx == 0 and not has_cur and _has_any(pre_item, _HIST_CUES)
            )
            occ_by_ik.setdefault(ik, []).append((val_s, u_norm, has_cur, has_hist))
    for ik, occ in occ_by_ik.items():
        role = roles[ik]
        if not role.get("priors"):
            continue
        last_latest_idx = max((i for i, o in enumerate(occ) if o[0] == role["latest"]), default=-1)
        for i, (val_s, u_norm, has_cur, has_hist) in enumerate(occ):
            if val_s == role["latest"]:
                continue
            if has_cur or (not role.get("is_ds_course") and not has_hist and i > last_latest_idx):
                if "item_stale_value" not in fails:
                    fails.append("item_stale_value")
                hits.append(("item_stale_value", ik, val_s, u_norm, role["latest"]))
    return {
        "fails": fails,
        "parsed": parsed,
        "unparsed": unparsed,
        "qualitative_only": qualitative_only,
        "hits": hits,
        "unparsed_details": unparsed_details,
    }


def check_section(plan, text):
    """节级。返回失败代码列表。"""
    fails = []
    if not text or len(text) < plan["target_chars"] * MIN_SHARE:
        fails.append("too_short")
    if any(w in (text or "") for w in LEAK):
        fails.append("prompt_leak")
    allowed = set(plan.get("allowed") or ())
    ap_raw = plan.get("allowed_pairs")
    allowed_pairs = {(str(v), u) for v, u in ap_raw} if ap_raw is not None else None
    for m in _NUM.finditer(text or ""):
        if routine_number(text, m):
            continue
        val_s, u_norm = m.group(1), norm_unit(m.group(2))
        if (val_s not in allowed) or (allowed_pairs is not None and (val_s, u_norm) not in allowed_pairs):
            fails.append("number_unsourced")
            break
    if "number_unsourced" not in fails and allowed_pairs is not None:
        for m in BP_PAIR_RE.finditer(text or ""):
            if (m.group(1) not in allowed or m.group(2) not in allowed
                    or (m.group(1), "mmhg") not in allowed_pairs or (m.group(2), "mmhg") not in allowed_pairs):
                fails.append("number_unsourced")
                break
    if plan.get("lab_item_roles"):
        fails.extend(audit_item_numbers(plan, text)["fails"])
    t = _clean(text)
    for f in plan["facts"]:
        if not (f["must"] and f.get("key")):
            continue
        if not any(_clean(v)[:4] in t for v in _key_variants(f["key"])):
            fails.append("fact_missing")
            continue
        if f.get("kind") == "result" and not _has_result_evidence(text, f["key"]):
            fails.append("result_tone_missing")
        req_pol = f.get("req_dx_polarity")
        if req_pol is None and f.get("kind") == "dx" and plan.get("role") in ("SUP", "DAILY") and ("诊断明确" in f.get("text", "") or "排除" in f.get("text", "")):
            req_pol = "确诊" if "诊断明确" in f.get("text", "") else "排除"
        if req_pol in ("确诊", "排除") and not _has_dx_firm_polarity(text, f["key"], req_pol):
            fails.append("dx_polarity_missing")
    if plan.get("admission_only") and admission_leaks(text):
        fails.append("admission_leak")
    if text and truncated(text):
        fails.append("truncated")
    if repeated_clauses(text):
        fails.append("repeat")
    if plan.get("nonsurgical") and _POSTOP.search(text or ""):
        fails.append("nonsurg_postop")
    if plan.get("section") in PE_SECTIONS and _has_any(text, _PE_HISTORY_WORDS):
        fails.append("pe_history")
    if plan.get("section") in PE_SECTIONS and pe_results(text):
        fails.append("pe_result")
    if plan.get("section") in PE_SECTIONS and _PE_UNTESTED.search(text or ""):
        fails.append("pe_untested")
    if plan.get("section") == "体格检查" and plan.get("program_prefix") and _PE_DUP_VITALS.search(text or ""):
        fails.append("pe_dup_vitals")
    if plan.get("role") == "DAILY" and _has_any(text, _SENIOR_WORDS):
        fails.append("senior_in_daily")
    if plan.get("before_op") and any(_before_op_clause(c) for c in _CLAUSE_SPLIT.split(text or "") if c.strip()):
        fails.append("before_op_done")
    if plan.get("no_incision") and _has_any(text, _INCISION_WORDS):
        fails.append("no_incision_leak")
    if plan.get("undelivered_preg") and _has_any(text, _POSTPARTUM_WORDS):
        fails.append("undelivered_postpartum")
    if plan.get("early_preg") and "胎动" in (text or ""):
        fails.append("early_preg_fetal")
    if plan.get("result_keys") is not None and result_mentions(text, set(plan["result_keys"])):
        fails.append("result_unsourced")
    if plan.get("unsettled_dx") and settled_dx_mentions(text, plan["unsettled_dx"], plan.get("settled_dx") or ()):
        fails.append("dx_settled_early")
    if plan.get("hint") and _has_any(text, _HINT_LEAK_WORDS):
        fails.append("hint_leak")
    if _ENG_PROSE.search(text or ""):
        fails.append("english_prose")
    if _META_TEXT.search(text or ""):
        fails.append("meta_text")
    if _has_past_contradict(plan, text or ""):
        fails.append("past_contradict")
    if plan.get("doc_date") and future_dates(text or "", plan["doc_date"], is_pre_op=(plan.get("role") == "PRE")):
        fails.append("future_date")
    if plan.get("postop_day") is not None:
        exp_d = plan["postop_day"]
        ms = list(_POSTOP_DAY_MENTION.finditer(text or ""))
        if exp_d == 0:
            if ms:
                fails.append("postop_day_mismatch")
        elif plan.get("role") in ("SUP", "DAILY"):
            if not ms or any(_cn_int(m.group(1)) != exp_d for m in ms):
                fails.append("postop_day_mismatch")
        elif any(_cn_int(m.group(1)) != exp_d for m in ms):
            fails.append("postop_day_mismatch")
    if _has_abn_lab_called_normal(plan, text or ""):
        fails.append("lab_abn_called_normal")
    return fails


def _before_op_clause(c):
    if re.search(r"\d+\s*年前|既往|曾", c):
        return False
    if _BEFORE_OP_HARD.search(c):
        return not _BEFORE_OP_EXEMPT.search(c)
    m = _BEFORE_OP_ACT.search(c)
    return bool(m and not _BEFORE_OP_EXEMPT.search(c))


def _known_past_terms(known):
    terms = {pt for k in (known or ()) for pt in PAST_TERMS if past_matches(pt, k)}
    for k in (known or ()):
        if k:
            terms.add(k)
    return terms


def _past_contradict_clause(c, known, child_past):
    if child_past and _has_any(c, _ADULT_CHRON_WORDS):
        return True
    if not known:
        return False
    if _has_any(c, _GENERIC_PAST_DENY_WORDS):
        return True
    return any(re.search(r"(?:否认|无)[^，,。；;\n]{0,12}" + re.escape(t), c) for t in _known_past_terms(known))


def _has_past_contradict(plan, text):
    if not text:
        return False
    known = plan.get("known_past")
    child_past = bool(plan.get("child_past"))
    if not known and not child_past:
        return False
    return any(_past_contradict_clause(c, known, child_past) for c in _CLAUSE_SPLIT.split(text) if c.strip())


def copy_safe(plan, sent):
    """0.21：apply_copy 拷入前先过同一份章节禁区，避免把术日未做的已手术句/无切口术式的切口句/英文或元话语拷进下一篇。"""
    if not sent:
        return False
    sub = {k: plan[k] for k in ("role", "nonsurgical", "before_op", "no_incision", "undelivered_preg", "early_preg", "unsettled_dx", "settled_dx", "result_keys") if k in plan}
    sub["target_chars"] = 0
    sub["facts"] = ()
    return not [f for f in check_section(sub, sent) if f != "number_unsourced"]


_SENT_PIECE = re.compile(r"[^。！？!?\n]+[。！？!?\n]?")
_CLAUSE_PIECE = re.compile(r"[^，,；;。！？!?\n]+[，,；;。！？!?\n]?")


def _drop_clauses_if(text, pred, drop_dep=False):
    """按谓词 pred(match) 删掉命中分句；若被删分句带句末标点则保留给前一个分句；drop_dep=True 时连同紧跟的从属分句一并删除。"""
    out, dropping = [], False
    for m in _CLAUSE_PIECE.finditer(text or ""):
        c = m.group(0)
        if pred(m):
            dropping = drop_dep
            if c[-1:] in "。！？!?\n" and out and out[-1][-1:] not in "。！？!?\n":
                out[-1] = out[-1].rstrip("，,；;、") + c[-1]
        elif dropping and _DEPENDENT.match(c.strip()):
            if c[-1:] in "。！？!?\n" and out and out[-1][-1:] not in "。！？!?\n":
                out[-1] = out[-1].rstrip("，,；;、") + c[-1]
        else:
            dropping = False
            out.append(c)
            continue
        if c[-1:] in "。！？!?\n":
            dropping = False
    return "".join(out)


def repair_section(text, fails, plan=None):
    """三次都没过时的修补（0.1.8）：只删不添。"""
    t = text or ""
    if "repeat" in fails:
        bad_p5 = _loop_prefixes(t)
        if bad_p5:
            seen_p5, out_s = {}, []
            for m in _SENT_PIECE.finditer(t):
                s = m.group(0)
                p5 = _cjk5(s)
                if p5 in bad_p5:
                    seen_p5[p5] = seen_p5.get(p5, 0) + 1
                    if seen_p5[p5] > 2:
                        continue
                out_s.append(s)
            t = "".join(out_s)
        bad, seen, out = repeated_clauses(t), set(), []
        for m in _CLAUSE_PIECE.finditer(t):
            c = m.group(0)
            core = _CLAUSE_SPLIT.sub("", c).strip()
            if core in bad and core in seen:
                continue
            seen.add(core)
            out.append(c)
        t = "".join(out)
    if "nonsurg_postop" in fails:
        t = "".join(m.group(0) for m in _SENT_PIECE.finditer(t) if not _POSTOP.search(m.group(0)))
    if "hint_leak" in fails:
        t = "".join(m.group(0) for m in _SENT_PIECE.finditer(t) if not _has_any(m.group(0), _HINT_LEAK_WORDS))
    if "pe_history" in fails:
        t = _drop_clauses_if(t, lambda c: _has_any(c.group(0), _PE_HISTORY_WORDS))
    if "pe_result" in fails:
        t = drop_pe_results(t)
    if "result_unsourced" in fails and plan is not None and plan.get("result_keys") is not None:
        t = drop_results(t, set(plan["result_keys"]))
    if "dx_settled_early" in fails and plan is not None and plan.get("unsettled_dx"):
        t = drop_settled_dx(t, plan["unsettled_dx"], plan.get("settled_dx") or ())
    if "pe_untested" in fails:
        t = _drop_clauses_if(t, lambda c: bool(_PE_UNTESTED.search(c.group(0))))
    if "pe_dup_vitals" in fails:
        t = _drop_clauses_if(t, lambda c: bool(_PE_DUP_VITALS.search(c.group(0))))
    if "senior_in_daily" in fails:
        t = _drop_clauses_if(t, lambda c: _has_any(c.group(0), _SENIOR_WORDS))
    if "before_op_done" in fails:
        t = _drop_clauses_if(t, lambda c: _before_op_clause(c.group(0)))
        if not _clean(t):
            t = "密切观察病情变化。"
    if "no_incision_leak" in fails:
        t = _drop_clauses_if(t, lambda c: _has_any(c.group(0), _INCISION_WORDS))
    if "undelivered_postpartum" in fails:
        t = _drop_clauses_if(t, lambda c: _has_any(c.group(0), _POSTPARTUM_WORDS))
    if "early_preg_fetal" in fails:
        t = _drop_clauses_if(t, lambda c: "胎动" in c.group(0))
    if "english_prose" in fails:
        t = _ENG_PROSE.sub("", t)
        t = re.sub(r"[（(]\s*[）)]", "", t)
        t = re.sub(r"\s{2,}", " ", t)
    if "meta_text" in fails:
        t = "".join(m.group(0) for m in _SENT_PIECE.finditer(t) if not _META_TEXT.search(m.group(0)))
    if "past_contradict" in fails and plan is not None:
        known = plan.get("known_past")
        child_past = bool(plan.get("child_past"))
        t = _drop_clauses_if(t, lambda c: _past_contradict_clause(c.group(0), known, child_past))
        if plan.get("section") == "既往史" and len(_clean(t)) < 4:
            t = "既往体健，否认药物过敏史。" if (child_past or not known) else f"既往有{'、'.join(known)}病史。"
    if "future_date" in fails and plan is not None and plan.get("doc_date"):
        bad_spans = [(s, e) for s, e, _ in future_dates(t, plan["doc_date"], is_pre_op=(plan.get("role") == "PRE"))]
        if bad_spans:
            repaired = _drop_clauses_if(t, lambda c: any(c.start() < e and s < c.end() for s, e in bad_spans))
            if not _clean(repaired):
                for s, e in reversed(bad_spans):
                    t = t[:s] + "近期" + t[e:]
            else:
                t = repaired
    t = re.sub(r"[，,、；;]+(?=[。！？!?])", "", t).strip()
    if truncated(t):
        cut = max(t.rfind(x) for x in _END_OK)
        if cut >= len(t) * 0.5:
            t = t[:cut + 1]
        else:
            comma = max(t.rfind(x) for x in "，,；;、")
            t = (t[:comma] if comma >= len(t) * 0.5 else t.rstrip("，,、；; ")) + "。"
    return t


def admission_leaks(text):
    """"入院情况"里写了住院后才有的事：K4 的触发词，入院前的分句（"3 年前""门诊"）跳过。"""
    return [m for m in K4_RE.finditer(text or "") if not k4_skip(text, m)]


# 0.1.8：触发分句后面这些开头的分句接的是它（"入院后行冠脉造影，示未见明显异常"），一起剔
_DEPENDENT = re.compile(r"^(?:示|提示|显示|结果|报告|可见|未见|考虑|给予|予以?)")


def strip_admission_leak(text):
    """剔掉带触发词的分句，句末补句号。重写三次都没写干净时用，保证交出的"入院情况"只有入院当时的事。
    0.1.8：同一句里紧跟着的从属分句（_DEPENDENT 开头）一起剔，到句号为止。"""
    out = _drop_clauses_if(text, lambda c: bool(admission_leaks(c.group(0))), drop_dep=True).rstrip("，,；; ")
    out = re.sub(r"[，,；;]+(?=[。！？!?])", "", out)
    return out + ("" if not out or out[-1] in "。！？!?" else "。")


def _doc_kind(d):
    if d["doc_type"] == "progress_note":
        return "first_progress" if is_first_progress(d["text"]) else "progress_note"
    return d["doc_type"]


_DONE_V_WORDS = ("已行", "已于", "已经", "已完成", "已实施", "已做", "顺利完成", "行了")


def future_leak(j, docs):
    """结果提前：按统一 (day, hm) 截点检查还没回报的化验数值或尚未回报的检查结果是否提前写进前面的文书。出院记录不查。"""
    out = set()
    post_k = next((x["k"] for x in docs if x.get("role") == "POST"), None)
    for d in docs:
        if d["role"] == "DS":
            continue
        c_ts = (d["day_offset"], _doc_hm(d))
        known = {(x["item"], x["value"]) for e in j["exams"] if exam_ts(e) <= c_ts for x in e["items"]}
        for e in j["exams"]:
            if exam_ts(e) <= c_ts:
                continue
            for x in e["items"]:
                if (x["item"], x["value"]) not in known and re.search(re.escape(x["item"]) + r"[^，,。；;\n]{0,5}?" + re.escape(x["value"]) + r"(?!\d)", d["text"]):
                    out.add(d["k"])
        allowed_keys = set(result_keys([e for e in j["exams"] if exam_ts(e) <= c_ts]))
        future_keys = set(result_keys([e for e in j["exams"] if exam_ts(e) > c_ts])) - allowed_keys
        if future_keys:
            is_hr = (d.get("doc_type") == "hospital_record") or (d.get("role") == "HR")
            segs = [s.text for s in split_sections(d["text"]) if not (is_hr and s.name in PRE_ADMISSION_SECTIONS)]
            for seg_t in segs:
                if any(_ALIAS[m.group(0)] in future_keys for m in result_mentions(seg_t, allowed=allowed_keys)):
                    out.add(d["k"])
                    break
        fut_procs = [
            a for a in (j.get("procedures") or ())
            if a.get("kind") in ("手术", "操作")
            and (d["day_offset"] < a.get("do_day", -1) or (d["day_offset"] == a.get("do_day", -1) and post_k is not None and d["k"] < post_k))
        ]
        if fut_procs and d["k"] not in out:
            is_hr = (d.get("doc_type") == "hospital_record") or (d.get("role") == "HR")
            skip_secs = set(PRE_ADMISSION_SECTIONS) | {"既往史", "个人史", "婚育史", "月经史", "家族史", "手术史", "过敏史"}
            segs = [s.text for s in split_sections(d["text"]) if not (is_hr and s.name in PRE_ADMISSION_SECTIONS) and s.name not in skip_secs]
            cores = set()
            for a in fut_procs:
                nm = a.get("name") or ""
                c1 = re.sub(r"^(?:左|右|双)(?:侧)?", "", nm)
                c2 = re.sub(r"^(?:左|右|双)?(?:侧)?(?:腹腔镜下?|宫腔镜下?|胸腔镜下?|关节镜下?|膀胱镜下?|输尿管镜下?|内镜下|显微镜下|经皮(?:肝)?(?:穿刺)?|经尿道|经阴道|经腹|开放性?|微创|介入)?", "", nm)
                for cc in (c1, c2):
                    if len(cc) >= 4:
                        cores.add(cc[:6])
            for seg_t in segs:
                for cl in _CLAUSE_SPLIT.split(seg_t):
                    if not cl.strip():
                        continue
                    if _before_op_clause(cl):
                        out.add(d["k"])
                        break
                    if not _BEFORE_OP_EXEMPT.search(cl) and _has_any(cl, _DONE_V_WORDS) and any(k in cl for k in cores):
                        out.add(d["k"])
                        break
                if d["k"] in out:
                    break
    return out


def check_admission(j, docs):
    """住院级。返回 [(文书序号或 None, 失败代码)]。"""
    rid = j["journey_id"]
    adm = {"admission_id": rid, "docs": docs}
    fails = []
    res = evaluate(from_admission(adm, rid))
    for h in res["hits"]:
        fails.append((int(h["seg"].split(":", 1)[0]), "g3_" + h["rule_id"]))
    for h in res["n1"]:
        fails.append((int(h["seg"].split(":", 1)[0]), "n1_" + h["type"]))
    for i, d in enumerate(docs):
        for c in evaluate_document(d["text"], rid, i, _doc_kind(d))["conflicts"]:
            if c.pair_id in G1_GATE:
                fails.append((d["k"], c.pair_id))
    rec, _ = care7_from_admission(adm, split_sections, extract_facts)
    if rec is not None:
        for h in check_record(rec):
            if h["rule"] != "AGE-UNPARSED":
                fails.append((None, "b7_" + str(h["rule"])))
    rs = evaluate_admission(build_real(adm))["readings"]
    for k in G2_GATE:
        if rs[k]["adm_conflict"]:
            fails.append((None, k))
    for k in sorted(future_leak(j, docs)):
        fails.append((k, "future_result"))
    s_procs = [a for a in (j.get("procedures") or ()) if a.get("kind") == "手术"]
    dx_name = (j.get("patient") or {}).get("primary_dx", {}).get("name") or ""
    if s_procs:
        op_d0 = s_procs[0]["do_day"]
        post_k = next((x["k"] for x in docs if x.get("role") == "POST"), None)
        all_no_inc = all(no_incision(a) for a in s_procs) and not _has_any(dx_name + "".join(a.get("name") or "" for a in s_procs), _INCISION_WORDS)
        for d in docs:
            role = d.get("role")
            is_pre_op = (d["day_offset"] < op_d0) or (d["day_offset"] == op_d0 and post_k is not None and d["k"] < post_k)
            if role == "POST" or (role in ("SUP", "DAILY", "PRE") and is_pre_op):
                if _POSTOP_DAY_MENTION.search(d["text"]):
                    fails.append((d["k"], "postop_day_mismatch"))
            elif role in ("SUP", "DAILY") and d["day_offset"] > op_d0:
                exp_d = d["day_offset"] - op_d0
                ms = list(_POSTOP_DAY_MENTION.finditer(d["text"]))
                if not ms or any(_cn_int(m.group(1)) != exp_d for m in ms):
                    fails.append((d["k"], "postop_day_mismatch"))
            if all_no_inc and role in ("POST", "SUP", "DAILY", "DS") and _has_any(d["text"], _INCISION_WORDS):
                fails.append((d["k"], "no_incision_leak"))
    elif not _has_any(dx_name, _INCISION_WORDS):
        for d in docs:
            role = d.get("role")
            if role in ("SUP", "DAILY"):
                if _has_any(d["text"], _INCISION_WORDS):
                    fails.append((d["k"], "no_incision_leak"))
            elif role in ("HR", "DS"):
                secs = [s for s in split_sections(d["text"]) if s.name not in _INC_SKIP_SECTIONS]
                if any(_has_any(s.text, _INCISION_WORDS) for s in secs):
                    fails.append((d["k"], "no_incision_leak"))
    if (j.get("patient") or {}).get("pregnant"):
        if not _has_delivery_event(j, s_procs, include_all_comorb=True):
            for d in docs:
                if d.get("role") in ("SUP", "DAILY", "DS") and _has_any(d["text"], _POSTPARTUM_WORDS):
                    fails.append((d["k"], "undelivered_postpartum"))
        if (j["patient"].get("gest_weeks") or 38) < 18:
            for d in docs:
                if "胎动" in d["text"]:
                    fails.append((d["k"], "early_preg_fetal"))
    for d in docs:
        for hc in d.get("_hard_sec_fails") or ():
            fails.append((d["k"], "hard_sec_" + hc))
    return fails
