"""关2 比对器：一次住院按时间轴从前往后看，算 S、K、B、C 四组读数。纯程序，不用模型。

evaluate_admission(adm) 返回 {"version", "readings": {读数号: Reading.record()}, "n_docs"}。
硬冲突读数（type=hard）：n_checkable 可比单位数，n_conflict 冲突单位数，adm_checkable / adm_conflict 住院层指标，
conflicts 每条带两端位置 (doc_idx, start, end)。
带读数（type=band）：value 是这次住院的值（比例或均值），n 是计算它用的单位数。
诊断读数（type=diag）：只作诊断，不进任何判定。
"""
import datetime as dt
import re
from dataclasses import asdict, dataclass, field

from ruler_g1.extractors import (BODY_AGE_RULES, BODY_SEX_RULES, HEAD_AGE, HEAD_SEX, MULTI_LESION_ORGANS,
                                 NON_PATIENT_SECTIONS, age_in_years, extract_laterality)

from . import diagnoses as dxm
from .copyrate import containment, lcs_ratio
from .events import admission_mentions, same_exam
from .text import HIST_RE, clause_span, find_dates, sentence_span, sentences

RULE_VERSION = "g2-rules-0.3"

READINGS = {
    # 读数号: (组, 类型, 一句话)
    "S1": ("S", "hard", "性别：各篇与入院记录抬头"),
    "S2": ("S", "hard", "年龄：各篇与入院记录抬头，差 ≥2 岁"),
    "S3": ("S", "hard", "侧别：入院记录、术前小结、术后首次病程、出院诊断之间同一部位左右相反"),
    "S4a": ("S", "hard", "术式入路：术前拟行与术后、出院记录相反（腔镜/开放），且全文没有中转"),
    "S4b": ("S", "hard", "手术日：出院记录写的手术日期与术后首次病程的日子差 ≥2 天"),
    "S5": ("S", "hard", "过敏：入院否认药物过敏，之后出现过敏史"),
    "K1": ("K", "hard", "术前写术后：手术日以前的文书里出现已发生口吻的术后内容"),
    "K2": ("K", "hard", "病理先于取材：取材以前的文书里出现结果口吻的病理"),
    "K3": ("K", "hard", "结果先于安排：需排期的检查，结果出现在它第一次被安排的那篇文书之前"),
    "K4": ("K", "hard", "出院记录入院情况里写了住院后才有的信息"),
    "K5": ("K", "hard", "未来日期：正文里出现晚于本篇书写日的日期（计划口吻除外）"),
    "B1": ("B", "band", "鉴别诊断闭环率"),
    "B2": ("B", "hard", "无据升级：疑似到确诊之间没有任何新的结果口吻检查"),
    "B3": ("B", "band", "出院诊断此前从未提及的比例"),
    "B4": ("B", "hard", "无交代回退：明确确诊后又写成疑似或排除，且没有修正类交代"),
    "B4b": ("B", "band", "病程里明确确诊的诊断，出院诊断里没有的比例"),
    "B5": ("B", "band", "入院即定论：入院诊断第一条不带疑似修饰且就是出院诊断第一条"),
    "C1a": ("C", "band", "相邻病程 5-gram 复制率（后篇被前篇覆盖的比例，均值）"),
    "C1b": ("C", "band", "病程对入院记录的 5-gram 复制率（均值）"),
    "C1c": ("C", "band", "相邻病程最长公共子串占后篇长度（均值）"),
    "C2": ("C", "band", "可核事实数：结果口吻的检查种类数 + 手术 + 诊断状态变化次数"),
    "K3s": ("K", "diag", "同一篇里同一项需排期检查既有结果又有安排"),
}
HARD = [k for k, v in READINGS.items() if v[1] == "hard"]
K_GROUP = ["K1", "K2", "K3", "K4", "K5"]


@dataclass
class Reading:
    rid: str
    n_checkable: int = 0
    n_conflict: int = 0
    conflicts: list = field(default_factory=list)
    value: object = None
    n: int = 0

    def add(self, ends, detail=""):
        self.n_conflict += 1
        self.conflicts.append({"ends": [list(e) for e in ends], "detail": detail})

    def record(self):
        grp, typ, _ = READINGS[self.rid]
        r = asdict(self)
        r.update({"group": grp, "type": typ,
                  "adm_checkable": self.n_checkable > 0,
                  "adm_conflict": self.n_conflict > 0})
        return r


# ---------- 公用 ----------

FAMILY = re.compile(r"家族|父|母|兄|弟|姐|妹|儿子|女儿|子女|配偶|妻|丈夫|爱人|孙")
PLAN_CLAUSE = re.compile(r"拟|计划|预约|定于|择期|将于|将|待|准备|建议|必要时|复诊|随访|返院|明日|明天|下次|届时|如|若|可能|预防|注意|需|应|防止|避免|告知|交代|风险|并发症")
EXPLAIN_CHANGE = re.compile(r"中转|转为|改为|改行|改开|修正|更正|补充诊断|最终诊断|结合.{0,10}(?:结果|病理|检查)|回报|证实")


def _doc_head(text, n=200):
    return text[:n]


def _demographics(doc):
    """(sexes, ages)：本篇说病人本人是什么性别、多少岁（岁为单位，带 ≥ 的不收）。入院记录不用这个函数。"""
    sexes, ages = [], []
    head = _doc_head(doc.text)
    m = HEAD_SEX.search(head)
    if m:
        sexes.append((m.group(1), m.start(1), m.end(1)))
    m = HEAD_AGE.search(head)
    if m and m.group(3) == "岁" and not m.group(1):
        ages.append((float(m.group(2)), m.start(2), m.end(3)))
    text = doc.text
    for _, rx in BODY_SEX_RULES:
        for m in rx.finditer(text):
            if doc.section_name_at(m.start(1)) in NON_PATIENT_SECTIONS:
                continue
            a, b = sentence_span(text, m.start(1))
            if FAMILY.search(text[a:b]):
                continue
            if any(s == m.start(1) for _, s, _ in sexes):
                continue
            sexes.append((m.group(1), m.start(1), m.end(1)))
    for _, rx in BODY_AGE_RULES:
        for m in rx.finditer(text):
            if m.group(3) != "岁" or m.group(1):
                continue
            if doc.section_name_at(m.start(2)) in NON_PATIENT_SECTIONS:
                continue
            a, b = sentence_span(text, m.start(2))
            if FAMILY.search(text[a:b]):
                continue
            if any(s == m.start(2) for _, s, _ in ages):
                continue
            ages.append((float(m.group(2)), m.start(2), m.end(3)))
    return sexes, ages


def _ref_demographics(hr):
    sex = age = None
    m = HEAD_SEX.search(hr.text)
    if m:
        sex = (m.group(1), m.start(1), m.end(1))
    m = HEAD_AGE.search(hr.text)
    if m and m.group(3) == "岁" and not m.group(1):
        age = (float(m.group(2)), m.start(2), m.end(3))
    return sex, age


def _surgery(adm):
    post, ds = adm.post, adm.ds
    if post is None or ds is None or post.pos > ds.pos:
        return None
    return post


PRE_OP_SAME_DAY = {"入院记录", "首次病程", "术前小结", "术前讨论"}


def _before(adm, anchor):
    """anchor 之前的文书：日子更早，或同一天且是入院记录、首次病程、术前小结、术前讨论。"""
    return [d for d in adm.docs if d.pos < anchor.pos and
            (d.day < anchor.day or (d.day == anchor.day and d.subtype in PRE_OP_SAME_DAY))]


# ---------- S 组 ----------

def s1_s2(adm, r1, r2):
    hr = adm.hr
    if hr is None:
        return
    ref_sex, ref_age = _ref_demographics(hr)
    for d in adm.docs:
        if d is hr:
            continue
        sexes, ages = _demographics(d)
        if ref_sex and sexes:
            vals = {s for s, _, _ in sexes}
            r1.n_checkable += 1
            if len(vals) == 1 and vals != {ref_sex[0]}:
                r1.add([(hr.idx, ref_sex[1], ref_sex[2]), (d.idx, sexes[0][1], sexes[0][2])], f"{ref_sex[0]}→{sexes[0][0]}")
        if ref_age and ref_age[0] >= 1 and ages:
            vals = [a for a, _, _ in ages]
            r2.n_checkable += 1
            if max(vals) - min(vals) < 2 and abs(vals[0] - ref_age[0]) >= 2:
                r2.add([(hr.idx, ref_age[1], ref_age[2]), (d.idx, ages[0][1], ages[0][2])], f"{ref_age[0]:g}→{vals[0]:g}")


def _lat_group(doc, section_names):
    out = []
    for f in extract_laterality(doc.text, (doc.idx, 0, doc.doc_type), doc.sections):
        if section_names is not None and f.section not in section_names:
            continue
        side, organ, *rest = f.value_norm.split("|")
        kv = dict(x.split("=", 1) for x in rest)
        if organ in MULTI_LESION_ORGANS or kv.get("hist") == "1" or kv.get("neg") == "1":
            continue
        out.append((organ, side, doc.idx, f.start, f.end))
    return out


def s3(adm, r):
    groups = {}
    hr, fp, ds, post = adm.hr, adm.fp, adm.ds, _surgery(adm)
    a = []
    if hr:
        a += _lat_group(hr, {"主诉", "现病史", "入院诊断"})
    if fp:
        a += _lat_group(fp, {"入院诊断"})
    groups["A"] = a
    pre = [d for d in adm.docs if d.subtype == "术前小结" and (post is None or d.pos < post.pos)]
    groups["PRE"] = _lat_group(pre[-1], None) if pre else []
    groups["POST"] = _lat_group(post, None) if post else []
    groups["DS"] = _lat_group(ds, {"出院诊断"}) if ds else []
    order = ["A", "PRE", "POST", "DS"]
    for i, gx in enumerate(order):
        for gy in order[i + 1:]:
            organs = {m[0] for m in groups[gx]} & {m[0] for m in groups[gy]}
            for organ in sorted(organs):
                mx = [m for m in groups[gx] if m[0] == organ]
                my = [m for m in groups[gy] if m[0] == organ]
                sx, sy = {m[1] for m in mx}, {m[1] for m in my}
                if "双" in sx or "双" in sy:
                    continue
                r.n_checkable += 1
                if len(sx) == 1 and len(sy) == 1 and sx != sy:
                    r.add([(mx[0][2], mx[0][3], mx[0][4]), (my[0][2], my[0][3], my[0][4])], f"{organ}:{gx}{mx[0][1]}/{gy}{my[0][1]}")


MIN_RE = re.compile(r"腹腔镜|胸腔镜|宫腔镜|关节镜|腔镜")
OPEN_RE = re.compile(r"开腹|开胸|开放手术|开放性手术|开放术式")
CONVERT_RE = re.compile(r"中转|转为开|改为开|改行开|转开腹|改开腹|转开胸|改开胸")


def _approach(doc, span=None):
    a0, a1 = span if span else (0, len(doc.text))
    text = doc.text
    found = []
    for rx, cls in ((MIN_RE, "MIN"), (OPEN_RE, "OPEN")):
        for m in rx.finditer(text, a0, a1):
            a, b = clause_span(text, m.start(), m.end())
            clause = text[a:b]
            if "术" not in clause or HIST_RE.search(clause):
                continue
            found.append((cls, doc.idx, m.start(), m.end()))
    return found


def s4(adm, ra, rb):
    post, ds = _surgery(adm), adm.ds
    if post is None:
        return
    pre = [d for d in adm.docs if d.subtype == "术前小结" and d.pos < post.pos]
    groups = []
    if pre:
        groups.append(("PRE", _approach(pre[-1])))
    groups.append(("POST", _approach(post)))
    dsc = ds.section("诊疗经过") if ds else None
    if dsc:
        groups.append(("DS", _approach(ds, dsc)))
    converted = bool(CONVERT_RE.search(post.text) or (ds and CONVERT_RE.search(ds.text)))
    for i, (gx, fx) in enumerate(groups):
        for gy, fy in groups[i + 1:]:
            if not fx or not fy:
                continue
            ra.n_checkable += 1
            cx, cy = {f[0] for f in fx}, {f[0] for f in fy}
            if len(cx) == 1 and len(cy) == 1 and cx != cy and not converted:
                ra.add([fx[0][1:], fy[0][1:]], f"{gx}{fx[0][0]}/{gy}{fy[0][0]}")
    # S4b 手术日
    if ds is None or adm.admit_date is None or dsc is None:
        return
    text = ds.text
    cands = []
    for s, e, d in find_dates(text[dsc[0]:dsc[1]], adm.admit_date):
        s, e = s + dsc[0], e + dsc[0]
        tail = text[e:min(len(text), e + 40)]
        a, b = sentence_span(text, s, e)
        if re.match(r"[^。；;\n]{0,30}?(?:行|在.{0,15}下行)[^。；;\n]{0,25}?术", tail) and not HIST_RE.search(text[a:b]):
            cands.append((s, e, (d - adm.admit_date).days))
    if not cands:
        return
    rb.n_checkable += 1
    if all(abs(day - post.day) >= 2 for _, _, day in cands):
        s, e, day = cands[0]
        rb.add([(ds.idx, s, e), (post.idx, 0, min(20, len(post.text)))], f"出院记录第{day}天/术后首次病程第{post.day}天")


DENY_ALLERGY = re.compile(r"否认[^。；;\n]{0,12}过敏|无[^。；;\n]{0,6}过敏史|过敏史\s*[:：]?\s*(?:无|否认)")
POS_ALLERGY = re.compile(r"(?:对|有)[^，,。；;\n]{1,12}过敏(?:史)?|过敏史\s*[:：]\s*(?!无|否认|不详)[^，,。；;\n\s]{1,12}")
NEW_REACTION = re.compile(r"出现|发生|考虑|予|给予|停用|停药|皮疹|瘙痒|抗过敏|过敏反应")
NEG_WORD = re.compile(r"否认|无|未|不|没有")


def s5(adm, r):
    hr = adm.hr
    if hr is None:
        return
    m0 = DENY_ALLERGY.search(hr.text)
    if not m0:
        return
    r.n_checkable += 1
    for d in adm.docs:
        if d.pos <= hr.pos:
            continue
        for m in POS_ALLERGY.finditer(d.text):
            a, b = clause_span(d.text, m.start(), m.end())
            sa, sb = sentence_span(d.text, m.start(), m.end())
            if NEG_WORD.search(d.text[a:m.start()]) or NEW_REACTION.search(d.text[sa:sb]):
                continue
            r.add([(hr.idx, m0.start(), m0.end()), (d.idx, m.start(), m.end())], "否认过敏→有过敏")
            return


# ---------- K 组 ----------

K1_RE = re.compile(
    r"术后第\s*[\d一二三四五六七八九十]+\s*[天日]|术后\s*\d+\s*(?:天|日|h|小时)|术中见|术中所见|术中出血|"
    r"手术(?:过程)?顺利|安返病房|返回病房|送返病房|麻醉(?:清醒|苏醒|满意)|术后(?:恢复|切口|伤口)|"
    r"切口(?:愈合|干燥|敷料)|(?:今日|今天|于今日)(?:在[^，,。；;\n]{0,15}下)?行[^，,。；;\n]{1,25}术(?!前)")


def k1(adm, r):
    post = _surgery(adm)
    if post is None:
        return
    before = _before(adm, post)
    if not before:
        return
    r.n_checkable += 1
    for d in before:
        for m in K1_RE.finditer(d.text):
            sa, sb = sentence_span(d.text, m.start(), m.end())
            ca, cb = clause_span(d.text, m.start(), m.end())
            sent = d.text[sa:sb]
            if HIST_RE.search(sent) or re.search(r"史", sent) or PLAN_CLAUSE.search(d.text[ca:cb]):
                continue
            if d.doc_type == "hospital_record" and d.section_name_at(m.start()) in NON_PATIENT_SECTIONS | {"现病史"}:
                continue
            r.add([(d.idx, m.start(), m.end()), (post.idx, 0, min(20, len(post.text)))], m.group(0))
            return


BIOPSY_DONE = re.compile(r"(?:今日|今天|于今日|已于|已行|顺利)[^，,。；;\n]{0,20}(?:活检|穿刺)")
PRE_BIOPSY_WORDS = re.compile(r"穿刺|活检|镜下|内镜|胃镜|肠镜|支气管镜|细胞学")


def _sampling_doc(adm):
    post = _surgery(adm)
    cands = [post] if post else []
    for d in adm.docs:
        if d.doc_type == "progress_note" and (d.subtype == "操作记录" or BIOPSY_DONE.search(d.text)):
            if adm.ds is None or d.pos < adm.ds.pos:
                cands.append(d)
                break
    return min(cands, key=lambda d: d.pos) if cands else None


def k2(adm, r, ments):
    sp = _sampling_doc(adm)
    if sp is None:
        return
    before = _before(adm, sp)
    if not before:
        return
    r.n_checkable += 1
    for d in before:
        for m in ments[d.idx]:
            if m.key != "病理" or m.tone != "R":
                continue
            ca, cb = clause_span(d.text, m.start, m.end)
            if PRE_BIOPSY_WORDS.search(d.text[ca:cb]):
                continue
            r.add([(d.idx, m.start, m.end), (sp.idx, 0, min(20, len(sp.text)))], "病理")
            return


# 0.3：入院记录的辅助检查、现病史、既往史，首次病程的病例特点，按规范写的就是入院前已有的结果，
# 不算"结果先于安排"（fold 0 上 0.2 的 K3 冲突大多是入院记录辅助检查里的门诊结果 + 首次病程里的"完善"）
K3_PRIOR_SECTIONS = {"入院记录": {"辅助检查", "现病史", "既往史"}, "首次病程": {"病例特点"}}


def k3_prior(d, m):
    return m.section in K3_PRIOR_SECTIONS.get(d.subtype, ())


def k3(adm, r, rs, ments):
    by_key = {}
    doc_of = {d.idx: d for d in adm.docs}
    for d in adm.docs:
        for m in ments[d.idx]:
            if m.sched and m.cls not in ("病理", "取材"):
                by_key.setdefault(m.key, []).append((d.pos, m))
    first_pos = adm.docs[0].pos if adm.docs else 0
    for key, lst in by_key.items():
        plans = [(p, m) for p, m in lst if m.tone == "P" and not m.recheck]
        if not plans:
            continue
        p0, m0 = min(plans, key=lambda x: (x[0], x[1].start))
        if p0 <= first_pos:
            continue  # 第一篇里就安排了，之前没有文书，无从"提前"
        r.n_checkable += 1
        early = [(p, m) for p, m in lst if m.tone == "R" and p < p0 and same_exam(m, m0)
                 and not k3_prior(doc_of[m.doc_idx], m)]
        if early:
            _, m1 = early[0]
            r.add([(m1.doc_idx, m1.start, m1.end), (m0.doc_idx, m0.start, m0.end)], key)
    # K3s：同一篇里先有结果、后有安排（只作诊断）
    for d in adm.docs:
        ms = [m for m in ments[d.idx] if m.sched and m.cls not in ("病理", "取材")]
        if not ms:
            continue
        rs.n_checkable += 1
        for m in ms:
            if m.tone != "R":
                continue
            later = [x for x in ms if x.tone == "P" and not x.recheck and x.start > m.end and same_exam(x, m)]
            if later:
                rs.add([(d.idx, m.start, m.end), (d.idx, later[0].start, later[0].end)], m.key)
                break


K4_RE = re.compile(r"入院后|住院期间|经[^，,。；;\n]{0,8}治疗|治疗后|术后|术中|病理(?:示|提示|回报|结果|报告)|复查|出院时|好转出院|行[^，,。；;\n]{1,20}术(?!前)")


# 0.3：fold 0 上 0.2 的 K4 在 25% 的住院报冲突，触发词多在长"入院情况"段的后半段，像是没认出来的
# "诊疗经过"标题把后文并了进来；另有"术后 3 年""门诊复查"之类入院前的事。
K4_COURSE_CUT = re.compile(r"(?:入院后)?(?:住院|诊疗|诊治|治疗)\s*(?:经过|过程)")
K4_CUT_MIN = 20
K4_PRIOR_CLAUSE = re.compile(r"(?<!\d)\d{1,2}\s*[余多+]?\s*(?:年|个月|周)|(?<!\d)\d{1,2}\s*[余多+]?\s*月(?:余|前|来)|"
                             r"外院|门诊|当地|既往|曾经?|病史")


def k4_span(ds):
    """出院记录"入院情况"段落的 (起, 止)，在没认出来的"诊疗经过"类标题处截断；没有这一段返回 None。"""
    if ds is None:
        return None
    sec = ds.section("入院情况")
    if sec is None:
        return None
    m = K4_COURSE_CUT.search(ds.text, sec[0] + K4_CUT_MIN, sec[1])
    return (sec[0], m.start()) if m else sec


def k4_skip(text, m):
    ca, cb = clause_span(text, m.start(), m.end())
    return bool(HIST_RE.search(text[ca:cb]) or K4_PRIOR_CLAUSE.search(text[ca:cb]))


def k4(adm, r):
    ds = adm.ds
    sec = k4_span(ds)
    if sec is None or sec[1] - sec[0] < 10:
        return
    r.n_checkable += 1
    for m in K4_RE.finditer(ds.text, sec[0], sec[1]):
        if k4_skip(ds.text, m):
            continue
        r.add([(ds.idx, m.start(), m.end()), (ds.idx, sec[0], min(sec[1], sec[0] + 10))], m.group(0))
        return


K5_PLAN = re.compile(r"拟|计划|预约|定于|择期|将于|将|待|准备|建议|复诊|随访|返院|明日|明天|下次|届时|"
                     r"预计|预产期|安排|择日|视情况|"
                     r"于[^，,。；;\n]{0,12}(?:复查|返院|入院|化疗|放疗|治疗|手术|拆线|换药|就诊)")
# 0.2：电子病历系统自己打的时间戳（上级医师审签、打印、记录时间）不是正文内容。
# fold 0 上 0.1 的 K5 在 71% 的住院报冲突，多半来自这些行（待 T52 的计数确认）
K5_STAMP_LINE = re.compile(r"签名|签字|签章|审签|审核|审阅|核对|确认|打印|记录时间|记录日期|书写时间|"
                           r"完成时间|修改时间|保存时间|操作时间")
STAMP_LINE_MAX = 60   # 只认短行：一整段正文里顺带出现"确认""核对"不算时间戳行


def _line_of(text, s):
    a = text.rfind("\n", 0, s) + 1
    b = text.find("\n", s)
    return a, (b if b >= 0 else len(text))


def is_stamp_line(text, s):
    a, b = _line_of(text, s)
    line = text[a:b].strip()
    return len(line) <= STAMP_LINE_MAX and bool(K5_STAMP_LINE.search(line))


# 0.3（依据 T52/T52b）：晚于书写日的日期，大半在入院记录的"修正诊断/补充诊断"（按规范事后补写、自带日期）
# 和入院记录抬头；其次是术前小结里的手术日期（"手术时间：…"一类字段，没有"拟"字）；另有文末只有日期的短行。
K5_LATER_SECTIONS = {"修正诊断"}
K5_PRE_OP = {"术前小结", "术前讨论"}
K5_PRE_OP_PLAN = re.compile(r"手术|术|麻醉")
TAIL_LINE_MAX = 30
TAIL_LINES = 3


TAIL_REST_MAX = 6
_TAIL_STRIP = re.compile(r"\[[^\]]{0,8}\]|[\s:：,，.。;；/\\-]|\d|年|月|日|时|分|秒")
_TAIL_CONTENT = re.compile(r"示|提示|回报|结果|检查|复查|手术|术|治疗|予|服|血|尿|CT|MRI|超声")


def is_tail_short_line(text, s):
    """落款日期行：日期所在行是全文最后 3 个非空行之一、不超过 30 字，去掉日期、占位符、标点后最多剩 6 个字，
    且没有检查、治疗一类字眼（"医师：[姓名] 2015-03-04"算，"2015-03-07 血常规示白细胞正常"不算）。"""
    a, b = _line_of(text, s)
    line = text[a:b].strip()
    if len(line) > TAIL_LINE_MAX or _TAIL_CONTENT.search(line):
        return False
    if len(_TAIL_STRIP.sub("", line)) > TAIL_REST_MAX:
        return False
    rest = [x for x in text[b:].split("\n") if x.strip()]
    return len(rest) < TAIL_LINES


def k5_scan(adm):
    """逐篇、逐个正文日期给出去向（诊断计数也用它）：
    header 抬头的书写时间（0.3：入院记录整个抬头段）；orders 出院医嘱；stamp 签名或时间戳行（0.2）、文末短日期行（0.3）；
    later_section 修正/补充诊断（0.3）；plan 计划口吻（0.3：术前小结、术前讨论里带"手术/术/麻醉"的分句）；
    past 不晚于书写日；future 晚于书写日。
    产出 (文书, 起, 止, 日期, 书写日, 去向)。"""
    if adm.admit_date is None:
        return
    for d in adm.docs:
        doc_date = adm.admit_date + dt.timedelta(days=d.day)
        excl = d.section("出院医嘱") if d.doc_type == "discharge_summary" else None
        for i, (s, e, x) in enumerate(find_dates(d.text, doc_date)):
            if i == 0 and s < 80:
                fate = "header"
            elif excl and excl[0] <= s < excl[1]:
                fate = "orders"
            elif d.section_name_at(s) == "签名" or is_stamp_line(d.text, s) or is_tail_short_line(d.text, s):
                fate = "stamp"
            elif d.section_name_at(s) in K5_LATER_SECTIONS:
                fate = "later_section"
            elif d.subtype == "入院记录" and d.section_name_at(s) == "抬头":
                fate = "header"
            elif d.subtype in K5_PRE_OP and K5_PRE_OP_PLAN.search(d.text[slice(*clause_span(d.text, s, e))]):
                fate = "plan"
            elif K5_PLAN.search(d.text[slice(*clause_span(d.text, s, e))]):
                fate = "plan"
            elif (x - doc_date).days >= 1:
                fate = "future"
            else:
                fate = "past"
            yield d, s, e, x, doc_date, fate


def k5(adm, r):
    seen, hit = set(), set()
    for d, s, e, x, doc_date, fate in k5_scan(adm):
        if d.idx not in seen:
            seen.add(d.idx)
            r.n_checkable += 1
        if fate == "future" and d.idx not in hit:
            hit.add(d.idx)
            r.add([(d.idx, s, e), (d.idx, 0, min(20, len(d.text)))], f"+{(x - doc_date).days}天")


# ---------- B 组 ----------

def b_group(adm, ments, rb1, rb2, rb3, rb4, rb4b, rb5):
    inv = dxm.inventory(adm)
    dm = dxm.mentions(adm, inv)
    fp, ds, hr = adm.fp, adm.ds, adm.hr
    by_name = {}
    for x in dm:
        by_name.setdefault(x.norm, []).append(x)

    def related(name):
        out = []
        for n, xs in by_name.items():
            if dxm.dx_match(n, name):
                out += xs
        return sorted(out, key=lambda x: (x.pos, x.start))

    ds_items = dxm.list_items(ds) if ds else []
    ds_out = [it for it in ds_items if it.section == "出院诊断"]

    # B1 鉴别诊断闭环
    if fp is not None:
        diffs = dxm.differential_items(fp)
        if diffs:
            closed = 0
            for it in diffs:
                if it.excluded:
                    closed += 1
                    continue
                later = [x for x in related(it.norm) if x.pos >= fp.pos and not (x.doc_idx == fp.idx and fp.section_name_at(x.start) == dxm.DIFF_SECTION)]
                if any(x.state in ("排除", "确诊") for x in later) or any(dxm.dx_match(it.norm, o.norm) for o in ds_out):
                    closed += 1
            rb1.n = len(diffs)
            rb1.value = closed / len(diffs)

    # B2 无据升级；B4 无交代回退
    result_pos = sorted({d.pos for d in adm.docs for m in ments[d.idx] if m.tone == "R"})
    doc_by_idx = {d.idx: d for d in adm.docs}
    for name, xs in by_name.items():
        xs = sorted(xs, key=lambda x: (x.pos, x.start))
        if any(x.state == "既往" for x in xs):
            continue
        susp = next((x for x in xs if x.state in dxm.SUSPECT_STATES), None)
        if susp is not None:
            conf = next((x for x in xs if x.pos > susp.pos and x.state in dxm.CONFIRM_STATES), None)
            if conf is not None:
                rb2.n_checkable += 1
                evidence = any(susp.pos < p <= conf.pos for p in result_pos)
                d = doc_by_idx[conf.doc_idx]
                sa, sb = sentence_span(d.text, conf.start, conf.end)
                if not evidence and not EXPLAIN_CHANGE.search(d.text[sa:sb]):
                    rb2.add([(susp.doc_idx, susp.start, susp.end), (conf.doc_idx, conf.start, conf.end)], name)
        conf0 = next((x for x in xs if x.state == "确诊"), None)
        if conf0 is not None:
            later = [x for x in xs if x.pos > conf0.pos]
            if later:
                rb4.n_checkable += 1
                back = next((x for x in later if x.state in ("疑似", "排除")), None)
                if back is not None:
                    d = doc_by_idx[back.doc_idx]
                    sa, sb = sentence_span(d.text, back.start, back.end)
                    if not EXPLAIN_CHANGE.search(d.text[sa:sb]):
                        rb4.add([(conf0.doc_idx, conf0.start, conf0.end), (back.doc_idx, back.start, back.end)], name)
            if ds is not None and conf0.doc_idx != ds.idx and ds_out:
                rb4b.n += 1
                if not any(dxm.dx_match(name, o.norm) for o in ds_out):
                    rb4b.n_conflict += 1
    if rb4b.n:
        rb4b.value = rb4b.n_conflict / rb4b.n
        rb4b.n_conflict = 0

    # B3 出院诊断此前从未提及
    if ds_out:
        earlier = [x for x in dm if x.pos < ds.pos and x.state != "否认"]
        unseen = 0
        for it in ds_out:
            if not any(dxm.dx_match(it.norm, x.norm) for x in earlier):
                unseen += 1
        rb3.n = len(ds_out)
        rb3.value = unseen / len(ds_out)

    # B5 入院即定论
    if hr is not None and ds_out:
        hr_items = [it for it in dxm.list_items(hr) if it.section == "入院诊断"]
        if hr_items:
            rb5.n = 1
            rb5.value = 1.0 if (not hr_items[0].hedged and dxm.dx_match(hr_items[0].norm, ds_out[0].norm)) else 0.0
    return dm


# ---------- C 组 ----------

def c_group(adm, ments, dm, rc1a, rc1b, rc1c, rc2):
    progs = [d for d in adm.docs if d.doc_type == "progress_note"]
    hr = adm.hr
    if len(progs) >= 2:
        vals = [containment(b.text, a.text) for a, b in zip(progs, progs[1:])]
        lcs = [lcs_ratio(a.text, b.text) for a, b in zip(progs, progs[1:])]
        rc1a.n, rc1a.value = len(vals), sum(vals) / len(vals)
        rc1c.n, rc1c.value = len(lcs), sum(lcs) / len(lcs)
    if hr is not None and progs:
        vals = [containment(p.text, hr.text) for p in progs]
        rc1b.n, rc1b.value = len(vals), sum(vals) / len(vals)
    kinds = {(m.key, m.region) for d in adm.docs for m in ments[d.idx] if m.tone == "R"}
    n_surg = 1 if _surgery(adm) is not None else 0
    changes = 0
    seq = {}
    for x in dm:
        if x.state in ("疑似", "鉴别", "确诊", "排除", "出院"):
            s = "S" if x.state in dxm.SUSPECT_STATES else ("C" if x.state in dxm.CONFIRM_STATES else "E")
            prev = seq.get(x.norm)
            if prev is not None and prev != s:
                changes += 1
            seq[x.norm] = s
    rc2.n = 1
    rc2.value = float(len(kinds) + n_surg + changes)


# ---------- 入口 ----------

def evaluate_admission(adm):
    R = {k: Reading(k) for k in READINGS}
    ments = admission_mentions(adm)
    s1_s2(adm, R["S1"], R["S2"])
    s3(adm, R["S3"])
    s4(adm, R["S4a"], R["S4b"])
    s5(adm, R["S5"])
    k1(adm, R["K1"])
    k2(adm, R["K2"], ments)
    k3(adm, R["K3"], R["K3s"], ments)
    k4(adm, R["K4"])
    k5(adm, R["K5"])
    dm = b_group(adm, ments, R["B1"], R["B2"], R["B3"], R["B4"], R["B4b"], R["B5"])
    c_group(adm, ments, dm, R["C1a"], R["C1b"], R["C1c"], R["C2"])
    return {"version": RULE_VERSION, "admission_id": adm.admission_id, "source": adm.source,
            "n_docs": len(adm.docs), "readings": {k: v.record() for k, v in R.items()}}


def k_any(result):
    """住院层"K 组任一冲突"：K1–K5 任一可比且冲突。"""
    rs = result["readings"]
    chk = any(rs[k]["adm_checkable"] for k in K_GROUP)
    hit = any(rs[k]["adm_conflict"] for k in K_GROUP)
    return chk, hit


def hit_span(result, rid, doc_idx, start, end, slack=5):
    """埋雷命中：读数 rid 报出冲突，且某一端落在 (doc_idx, start, end) 前后各放宽 slack 字的范围内。"""
    for c in result["readings"][rid]["conflicts"]:
        for di, s, e in c["ends"]:
            if di == doc_idx and s < end + slack and start - slack < e:
                return True
    return False
