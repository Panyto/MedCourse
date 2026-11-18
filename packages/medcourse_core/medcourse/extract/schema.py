"""两趟抽取的 schema 和提示词。schema 静态可缓存、不用 maxLength（有 maxLength 时解码可能掉到约 7 tok/s）。

第一趟：候选实体加逐字引文，闭集字段由 XGrammar 约束。
第二趟：同一切片的候选一次判完，reason 排在 assertion 前面，让模型先写依据再定类别。
提示词由总经理编写，伙计逐字使用，要改写进 GAPS。

v0.2（金标 S2G v1 之后，2026-11-06）：v1 成立率 60%，症状只有 20%，主要是我们的定义写漏了，提示词和标注页的说法也对不上：
- 症状定义成"症状和体征"，查体所见全被抽成症状（标注页只写"症状"）；
- 七类是封闭的，护理、饮食、评分、病史条目只能硬塞进最近的一类；
- 状态没有"不是这个病人"，rule_out 把"待排"和"已排除"混在一起，"入院前"和"本次发病以来一直有"分不清。
现在类型和状态的定义都放在 TYPE_DEFS、ASSERT_DEFS 一处，提示词和金标标注页（gold_kit）都从这里读，不会再不一致。

v0.3（金标 S2G3 之后，2026-11-06，老板确认的口径 D1–D4）：
- 已发生包括现在仍患有的慢性病、做了结果正常的检查；既往只指已经结束的病、一次性的事、入院前做过的检查和治疗（D1、D2）；
- 写明治疗目的的用药也算用药，名称写治疗目的；"对症治疗""支持治疗"不抽（D3）；
- 生命体征只有五项，眼压算检查，体重、评分归七类以外；化疗归用药，康复理疗归七类以外（D4 和 S2G3 手术操作的不成立题）。
- 规则 6 写明列表项的引文摘原话、不自己补"否认"：S2G3 既往史每片的七类事件比 v1 少 62%，"否认高血压、糖尿病"里的糖尿病
  被写成引文"否认糖尿病"，不是原文子串，在校验那一步被整条丢掉（verify.ground 另加名称回退）。
"""
from ..contract import ASSERTIONS, CORE_TYPES, EVENT_TYPES

CAP = 40
PASS1_MAX_TOKENS = 3000
PASS2_MAX_TOKENS = 2500

# 类型：(中文名, 定义)。提示词和标注页共用
TYPE_DEFS = {
    "diagnosis": ("诊断", "疾病、综合征、诊断结论，包括既往病（如“高血压病史”里的高血压）"),
    "exam": ("检查", "影像、超声、内镜、心电图、脑电图、肺功能、病理、眼压这类检查；名称写检查手段（如“胸部CT”），不写检查所见"),
    "lab": ("化验", "血、尿、便、体液、分泌物的常规、生化、培养、核酸、抗体、血气等化验项目和指标"),
    "medication": ("用药", "具体的药物，包括化疗药、激素、缩宫素；只写了治疗目的的（如抗感染、利尿、护胃、止血、保肝、化痰、平喘）也算，名称写治疗目的；"
                          "“对症治疗”“支持治疗”这类没有药理指向的不算"),
    "procedure": ("手术操作", "外科手术、介入治疗、内镜下治疗，以及穿刺、活检、插管、置管这类有创操作；化疗、输液归用药，康复训练、理疗、高压氧归七类以外"),
    "vital": ("生命体征", "只有体温、脉搏或心率、呼吸、血压、血氧五项；监测生命体征的医嘱也算（状态是计划要做）。体重、身高、评分不算，眼压算检查"),
    "symptom": ("症状", "患者自己感到的不适，如咳嗽、腹痛、头晕；不包括查体所见"),
    "sign": ("体征", "查体所见，如啰音、压痛、腹软、心律齐"),
    "other": ("七类以外", "护理、饮食、吸氧、心电监护、康复训练、理疗、推拿、高压氧、宣教、会诊、出院手续、评分、体重、身高、治疗目标，"
                       "以及笼统的“完善检查”“入院抽血”“其他不适”"),
}
# 状态：(中文名, 定义)
ASSERT_DEFS = {
    "done": ("已发生／已做", "已经发生、已经做了或已有结果；本次发病以来一直有的症状、正在用的药、现在仍患有的慢性病（如高血压、糖尿病）也算；"
                         "检查、化验做了、结果正常也算"),
    "planned": ("计划要做", "计划、医嘱、建议、拟行，书写时还没做"),
    "history": ("既往", "入院前已经结束的病（已治愈、已切除）和过去一次性的事；入院前在外院、门诊、急诊做过的检查和治疗；上次住院的治疗"),
    "excluded": ("已排除", "文字明确写了已经排除"),
    "negated": ("否认／阴性", "文字明确说没有：症状、体征、疾病没有，或检查没做；检查、化验做了而结果正常不算否认，算已发生"),
    "uncertain": ("疑似／待排", "考虑、疑似、待排、不除外，或给定文字判断不了"),
    "not_patient": ("不是这个病人", "说的不是这个病人：别的病的典型表现、教科书式的描述、手术风险和并发症告知"),
}
assert set(TYPE_DEFS) == set(EVENT_TYPES) and set(ASSERT_DEFS) == set(ASSERTIONS)

_NULLABLE = {"type": ["string", "null"]}

PASS1_SCHEMA = {
    "type": "object",
    "properties": {"events": {"type": "array", "maxItems": CAP, "items": {
        "type": "object",
        "properties": {"event_type": {"type": "string", "enum": list(EVENT_TYPES)}, "name": {"type": "string"},
                       "value": _NULLABLE, "unit": _NULLABLE, "evidence": {"type": "string"}},
        "required": ["event_type", "name", "value", "unit", "evidence"], "additionalProperties": False}}},
    "required": ["events"], "additionalProperties": False}

PASS2_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}, "assertion": {"type": "string", "enum": list(ASSERTIONS)},
                       "time_text": _NULLABLE},
        "required": ["id", "reason", "assertion", "time_text"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False}

SYSTEM = "你是医学信息抽取工具，只依据给定文字抽取，不推断、不补充，只输出 JSON。"


def type_text():
    return "；".join(f"{k} {cn}：{d}" for k, (cn, d) in TYPE_DEFS.items())


def assert_text():
    return "；".join(f"{k} {cn}：{d}" for k, (cn, d) in ASSERT_DEFS.items())


PASS1_RULES = (
    "规则：\n"
    "1. 只抽和这个患者有关的事。病史条目本身（过敏史、家族史、预防接种史、手术外伤史这类标题）不抽，里面写到的具体疾病、手术照抽。\n"
    "2. 查体所见抽成 sign，不抽成 symptom。七类以外的（护理、饮食、宣教、会诊这些）抽成 other。\n"
    "3. name 写事件本身的名称，不带“无”“否认”“未见”这类否定词；检查写检查手段（如“胸部CT”），检查所见不当名称。\n"
    "4. 原文写了具体药名的写药名；只写了治疗目的的（如“抗感染、利尿、护胃”），每个目的抽一条用药，name 写治疗目的；“对症治疗”“支持治疗”不抽。"
    "一串否认的症状、疾病逐个抽；诊断清单、诊断分析、上级医师意见里提到的诊断都要抽。\n"
    "5. value 写数值或结果，没有写 null；unit 写单位，没有写 null。血压写完整的“收缩压/舒张压”（如 130/80）；"
    "药物的 value 只写单次剂量，不写频次和溶媒。\n"
    "6. evidence 从原文逐字摘录包含该事件的最短短句，不改一个字、不补字。列表里的一项（如“否认高血压、糖尿病”里的糖尿病），"
    "evidence 照抄原文里含这一项的那一段（“否认高血压、糖尿病”），不要自己拼成“否认糖尿病”。相同事件不重复。否定的、计划的也要抽，状态留给下一步判断。\n")


def pass1_messages(seg, doc_kind):
    user = (f"下面是一篇《{doc_kind}》里“{seg['section']}”部分的文字。逐条抽取其中明确写到的事件。\n"
            f"event_type 取值：{type_text()}。\n"
            + PASS1_RULES +
            f"\n【文字】\n{seg['text']}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def pass2_messages(seg, doc_kind, cands):
    lines = "\n".join(f"{i}. {c['event_type']}｜{c['name']}｜{c['evidence']}" for i, c in enumerate(cands))
    user = (f"下面是一篇《{doc_kind}》里“{seg['section']}”部分的文字，以及从中抽出的事件。逐条判断。\n"
            f"assertion 取值：{assert_text()}。\n"
            "注意：判断时看这段文字所在的章节和文书类型。出院医嘱、诊疗计划里的事，书写时还没做；入院记录“辅助检查”里列的是入院前的结果。\n"
            "reason 先用一句话写依据；time_text 从原文逐字摘录这条事件的时间说法（如“3天前”“术后第2天”），没有写 null。\n"
            "id 照抄序号，每条都要判。\n\n"
            f"【文字】\n{seg['text']}\n\n【事件】\n{lines}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


CORE = CORE_TYPES
