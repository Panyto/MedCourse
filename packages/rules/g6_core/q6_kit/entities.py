"""关6 的实体抽取尺子（g6-rules-0.1）：从一份 7 字段病历的四段叙述里抽五类实体，
对应 LLM-CARe 论文附录 A.5 的五类知识（症状、辅助检查、实验室检查、手术、药物）。

LLM-CARe 没有公开它从真实病历里抽实体的办法（论文只说"从真实病历里抽取每个诊断相关的医学实体"，
公开代码的参考文件 example-ref.jsonl 里实体是现成的）。这里用规则代替：词表 + 少量开放后缀，全部在本文件和
ruler_g4/terms.py、ruler_g3 药名词表里，没有看任何真实病历，也没有拿 LLM-CARe 的输出来挑词。
两边（真实、合成）用同一把尺子，所以尺子的漏和错对两边是同一个方向；但真实病历写法更杂，漏得更多（README 第 7 节）。

每处实体：类别 cat、原文写法 surface、规范键 key、位置、是否否定 neg。
- surface：知识覆盖（K-COV）用它，和 LLM-CARe 一样做字符串精确匹配；
- key：多样性、长尾用它（药是 ruler_g3 的药名 id，同一个药的不同写法算一个；其余是 NFKC 小写后的写法）。
参考集（配额和 K-COV 的参考）只收**没有被否定**的实体：公开的 example-ref.jsonl 里，"No significant dizziness,
headache" 这类否定的症状不在实体表里。
"""
import re
import unicodedata

from ruler_g3.lexicon import lexicon
from ruler_g3.parse import mention_context
from ruler_g4.terms import DRUG_EXTRA, EXAMS, PROC_CUT, PROC_NAMED, PROC_STOP, SYMPTOMS, TREAT_CLASSES

RULES_VERSION = "g6-rules-0.1"
CATS = ("SYM", "AUX", "LAB", "SURG", "MED")
CAT_CN = {"SYM": "症状体征", "AUX": "辅助检查", "LAB": "实验室检查", "SURG": "手术与操作", "MED": "药物"}
FIELDS4 = ("cc", "hpi", "hc", "di")
OFFICIAL_FIELD = {"cc": "chief_complaint", "hpi": "history_of_present_illness", "hc": "hospital_course",
                  "di": "discharge_instructions"}
CARE7_KEY = {"cc": "Chief Complaint", "hpi": "History of Present Illness", "hc": "Hospital Course",
             "di": "Discharge Instructions"}

# ---------- 词表 ----------
# ruler_g4 的检查词表按"影像、心电、内镜、功能"和"化验"分成两类；生命体征和评分（体温、血压、NYHA、GCS……）不算检查实体。
_NOT_EXAM = set("体温 血压 心率 脉搏 呼吸 血氧饱和度 SpO2 氧饱和度 Killip NYHA GCS NIHSS Child-Pugh 培养 抗体 抗原 核酸 凝血 血气".split())
_AUX_BASE = set("""
CT MRI MRA CTA DSA PET-CT PET/CT B超 彩超 超声 心超 胸片 X线 DR 心电图 动态心电图 Holter 脑电图 肌电图 胃镜 肠镜 结肠镜
支气管镜 喉镜 膀胱镜 造影 冠脉造影 肺功能 骨密度 头颅CT 胸部CT 腹部CT 头颅MRI 颈动脉彩超 心脏彩超 腹部彩超 泌尿系彩超
听力筛查 听力检查 智力测试 发育评估 语言评估 量表 Gesell 骨髓穿刺 骨穿 腰穿
""".split())
AUX_EXTRA = """
超声心动图 心脏超声 胸部X线 胸部正位片 X光 腹部平片 平片 胃肠镜 电子胃镜 电子结肠镜 内镜 MRCP 磁共振 核磁共振 核磁 CT平扫
增强CT 冠状动脉造影 冠脉CTA 颈部血管彩超 下肢血管彩超 血管彩超 TCD 经颅多普勒 动态血压 运动平板 平板试验 视频脑电图 头颅MRA
骨扫描 ECT 甲状腺彩超 乳腺彩超 妇科彩超 子宫附件彩超 B型超声 心脏B超 腹部B超 肝胆胰脾彩超 肝胆彩超 泌尿系B超 X片 肾图
""".split()
LAB_EXTRA = """
生化 生化全套 血生化 肝功 肾功 心肌标志物 尿液分析 隐血 粪隐血 大便隐血 血型 乙肝五项 乙肝两对半 糖耐量 OGTT C肽 胰岛素释放试验
同型半胱氨酸 尿酮体 酮体 痰涂片 抗酸染色 PPD T-SPOT 肺炎支原体抗体 支原体抗体 呼吸道病原体 血常规 尿常规 便常规
""".split()
AUX_TERMS = sorted(_AUX_BASE | set(AUX_EXTRA))
LAB_TERMS = sorted((set(EXAMS) - _AUX_BASE - _NOT_EXAM) | set(LAB_EXTRA))
SYM_TERMS = sorted(set(SYMPTOMS) - {"疼痛", "瞳孔", "肌力", "肌张力", "腱反射", "呼吸音", "心律齐", "肠鸣音", "囟门"})
# 手术与操作：有名字的侵入性操作和非药物治疗（"……术"另用开放后缀抓）
SURG_TERMS = sorted(set(PROC_NAMED) | set("""
胸腔穿刺 腹腔穿刺 心包穿刺 腰椎穿刺 冠脉支架 球囊扩张 射频消融 起搏器植入 内镜下止血 胃管 导尿 留置导尿 中心静脉置管
""".split()))
MED_EXTRA = sorted(set(DRUG_EXTRA))

# 开放后缀
_PREFIX_CUT = set(PROC_CUT) | set("查做示复完善见示拍行予给用服口静滴注射加改换停续及或")
# "……术"往前截得更狠：动词、虚词、"无""有""次""年"一类（"建议患者接受手术""体重由术前""第二次手术"）
_SURG_CUT = _PREFIX_CUT | set("由以受议虑诊始认无有次年成需可要将拟待因者的是在过做做到")
_AUX_SUFFIX = re.compile(r"([一-鿿]{1,6})(CT|CTA|MRI|MRA|彩超|B超|超声|平片|X线|DR|造影|镜检查)")
_LAB_SUFFIX = re.compile(r"([一-鿿A-Za-z]{1,6})(常规|培养|测定|定量|抗体|抗原|核酸|全套|五项|四项)")
_SURG_SUFFIX = re.compile(r"([一-鿿]{1,14})术")
_MED_FORM = re.compile(r"([一-鿿A-Za-z]{2,10})(缓释片|控释片|肠溶片|分散片|咀嚼片|泡腾片|缓释胶囊|肠溶胶囊|软胶囊|胶囊|注射液|注射剂|"
                       r"冻干粉|颗粒|口服液|混悬液|干混悬剂|滴丸|喷雾剂|气雾剂|吸入剂|粉雾剂|软膏|乳膏|滴眼液|片)")
_MED_FORM_STOP = {"胸片", "平片", "X片", "切片", "照片", "图片", "名片", "药片", "唇片", "镜片", "底片", "胶片", "影片", "相片"}
_LAB_STOP = {"血常规", "尿常规", "便常规"}   # 已在词表里，开放后缀不重复
# 地名、机构名、数字一律不当实体（守则 1.2）
_PLACE = re.compile(r"医院|卫生院|诊所|中心|门诊|保健院|省|市|县|区|镇|乡|村|街道|社区|\d")

# 否定：实体前面同一分句 8 字以内有这些字样（"无发热""未诉胸痛""否认……""未行……术"）
_NEG_BEFORE = re.compile(r"(?:无|否认|未诉|不伴|未见|未出现|未再|未有|没有|未|拒绝)[^。；;，,！!？?\n]{0,8}$")


def _term_rx(terms):
    terms = sorted(set(terms), key=len, reverse=True)
    parts = []
    for t in terms:
        p = re.escape(t)
        if re.match(r"^[A-Za-z0-9\-/+.]+$", t):
            p = r"(?<![A-Za-z])" + p + r"(?![A-Za-z])"
        parts.append(p)
    return re.compile("|".join(parts))


_CAT_OF = {}
for _c, _ts in (("MED", MED_EXTRA), ("SURG", SURG_TERMS), ("AUX", AUX_TERMS), ("LAB", LAB_TERMS), ("SYM", SYM_TERMS)):
    for _t in _ts:
        _CAT_OF.setdefault(_t, _c)
_TERMS = _term_rx(_CAT_OF)
# 药名词表最先；其余词表和开放后缀同一级，长的优先（"前列腺彩超"胜过"彩超"，"肺炎支原体抗体"胜过"支原体抗体"）
_PRIORITY = {"lex_drug": 0, "term": 1, "surg_open": 1, "aux_open": 1, "lab_open": 1, "med_open": 2}


def norm_key(s):
    return unicodedata.normalize("NFKC", s or "").lower()


def _cut_prefix(run, cuts=None):
    cuts = _PREFIX_CUT if cuts is None else cuts
    cut = max((i for i, ch in enumerate(run) if ch in cuts), default=-1)
    return run[cut + 1:]


def _negated(text, s):
    return bool(_NEG_BEFORE.search(text[max(0, s - 14):s]))


def _overlaps(a, b, spans):
    return any(a < e and s < b for s, e in spans)


def mentions(text):
    """一段文字里的实体。返回 [{cat, surface, key, start, end, neg, src}]，按位置排序；同一处只算一类（优先级：
    药名词表 > 其余词表和开放后缀（长的优先）> 开放剂型"……片/胶囊"）。"""
    text = text or ""
    cand = []
    lex = lexicon()
    drug_spans = []
    for s, e, alias, did in lex.find(text):
        ctx = mention_context(text, s, e, did)
        if ctx["not_drug"]:
            continue
        drug_spans.append((s, e, did, bool(ctx["negated"])))
    # 药名带剂型（"阿司匹林肠溶片"）：写法取整个产品名，键取药名 id
    forms = []
    for m in _MED_FORM.finditer(text):
        run = _cut_prefix(m.group(1))
        surf = run + m.group(2)
        if len(run) < 2 or surf in _MED_FORM_STOP:
            continue
        forms.append((m.end() - len(surf), m.end(), surf))
    for s, e, did, neg in drug_spans:
        surf, a, b = text[s:e], s, e
        for fs, fe, fsurf in forms:
            if fs <= s and e <= fe:
                surf, a, b = fsurf, fs, fe
                break
        cand.append({"cat": "MED", "surface": surf, "key": "drug:" + did, "start": a, "end": b, "neg": neg, "src": "lex_drug"})
    for m in _TERMS.finditer(text):
        t = m.group(0)
        c = _CAT_OF[t]
        cand.append({"cat": c, "surface": t, "key": norm_key(t), "start": m.start(), "end": m.end(),
                     "neg": _negated(text, m.start()) if c in ("SYM", "SURG") else False, "src": "term"})
    for m in _SURG_SUFFIX.finditer(text):
        name = _cut_prefix(m.group(1), _SURG_CUT) + "术"
        if len(name) < 3 or name in PROC_STOP or (name.endswith("手术") and len(name) <= 4):
            continue
        s = m.end() - len(name)
        cand.append({"cat": "SURG", "surface": name, "key": norm_key(name), "start": s, "end": m.end(),
                     "neg": _negated(text, s), "src": "surg_open"})
    for rx, cat, src in ((_AUX_SUFFIX, "AUX", "aux_open"), (_LAB_SUFFIX, "LAB", "lab_open")):
        for m in rx.finditer(text):
            run = _cut_prefix(m.group(1))
            if not run:
                continue
            surf = run + m.group(2)
            if surf in _LAB_STOP:
                continue
            s = m.end() - len(surf)
            cand.append({"cat": cat, "surface": surf, "key": norm_key(surf), "start": s, "end": m.end(), "neg": False,
                         "src": src})
    for fs, fe, fsurf in forms:
        cand.append({"cat": "MED", "surface": fsurf, "key": norm_key(fsurf), "start": fs, "end": fe,
                     "neg": _negated(text, fs), "src": "med_open"})
    cand.sort(key=lambda x: (_PRIORITY[x["src"]], -(x["end"] - x["start"]), x["start"]))
    out, taken = [], []
    for c in cand:
        if _overlaps(c["start"], c["end"], taken):
            continue
        if _PLACE.search(c["surface"]):
            continue
        taken.append((c["start"], c["end"]))
        c["neg"] = int(bool(c["neg"]))
        out.append(c)
    return sorted(out, key=lambda x: (x["start"], x["end"]))


def field_text(record, f):
    from ruler_g3.adapters import care_get
    return care_get(record, f)


def record_mentions(record):
    """{段: [实体]}，四段是主诉、现病史、住院经过、出院医嘱。"""
    return {f: mentions(field_text(record, f)) for f in FIELDS4}


def ref_entity(record, ms=None):
    """LLM-CARe 参考文件的 entity 格式：{官方段名: {类别: [写法]}}，只收没被否定的，每段内去重、按写法排序。"""
    ms = ms or record_mentions(record)
    out = {}
    for f in FIELDS4:
        by = {}
        for m in ms[f]:
            if not m["neg"]:
                by.setdefault(m["cat"], set()).add(m["surface"])
        out[OFFICIAL_FIELD[f]] = {c: sorted(v) for c, v in sorted(by.items())}
    return out


def surface_set(record, ms=None, affirmed_only=True):
    """四段合起来的（类别, 写法）集合。"""
    ms = ms or record_mentions(record)
    return {(m["cat"], m["surface"]) for f in FIELDS4 for m in ms[f] if not (affirmed_only and m["neg"])}


def key_set(record, ms=None, affirmed_only=True):
    """四段合起来的（类别, 规范键）集合。多样性、长尾用它。"""
    ms = ms or record_mentions(record)
    return {(m["cat"], m["key"]) for f in FIELDS4 for m in ms[f] if not (affirmed_only and m["neg"])}


def counts_by_cat(record, ms=None):
    ms = ms or record_mentions(record)
    out = {c: 0 for c in CATS}
    for c, _ in key_set(record, ms):
        out[c] += 1
    return out
