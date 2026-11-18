"""关2 的雷（校8）和合法差异（校9）。

所有雷都只用同一次住院自己的句子（找不到时才用固定模板句，并记 mode="template"），
注入位置用本文件自己的简单规则挑，不调用 ruler_g2 的抽取器（B2 的轨迹也用本文件自己的规则找），
这样尺子只能靠"时间和位置不对"抓到雷。例外：S3（0.2）借用关1 v0.5 已定稿的侧别抽取挑要翻的那几处，
因为关2 的 S3 只在这些抽取结果上比；它量的是关2 的比对器，抽取器的召回在关1 量过。

0.2 的改动（依据 G2 CPU 包 1 的 T07、T08）：
- 末尾插入跳过签名、时间戳一类短行，插入的句子前面补句号，不和上一句粘成同一分句；
- K2 只抄"病理结果"的句子，"待病理回报""送病理"一类不抄；
- S3 把出院诊断里该部位同侧的写法全部翻过来（0.1 只翻第一处，另一处没翻，左右并存，尺子不比）；
- B2 删句时把提到任何检查、检验的句子也删掉（0.1 只删带"示""提示"等字的句子，"血常规：WBC 12.3"留着，
  尺子把它当依据）；插入方式下入院记录和首次病程之间的各篇都删；有既往史的诊断不用；
- L3 的"门诊"紧贴检查词插入（0.1 插在句首，检查词在逗号后面时不在同一分句里）。

plant(adm, mine_type, rng) → (新住院, MineRecord) 或 None（这次住院不适合埋这种雷）。
真实病历的埋雷版本只在内存里，不落盘、不打印；只保存 MineRecord（位置、方式）。
"""
import datetime as dt
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ruler_g1.extractors import MULTI_LESION_ORGANS, extract_laterality
from ruler_g2 import diagnoses as dxm

MINE_TYPES = ("S1", "S3", "S4a", "S4b", "K1", "K2", "K3", "K4", "K5", "B2")
# 雷 → 应该报它的读数
MINE_READING = {"S1": "S1", "S3": "S3", "S4a": "S4a", "S4b": "S4b", "K1": "K1", "K2": "K2", "K3": "K3",
                "K4": "K4", "K5": "K5", "B2": "B2"}


@dataclass
class MineRecord:
    mine: str
    doc_idx: int        # 雷所在文书
    start: int
    end: int
    anchor_idx: int     # 与它构成矛盾的另一篇（窗口条件只给这两篇；K4 两端在同一篇）
    mode: str           # own / template / insert
    note: str = ""

    def record(self):
        return asdict(self)


def _insert(adm, doc, at, piece):
    text = doc.text[:at] + piece + doc.text[at:]
    adm.replace_text(doc.idx, text)
    return at, at + len(piece)


def _replace(adm, doc, s, e, new):
    text = doc.text[:s] + new + doc.text[e:]
    adm.replace_text(doc.idx, text)
    return s, s + len(new)


_STAMP = re.compile(r"签名|签字|签章|审签|审核|审阅|打印|记录时间|记录日期|书写时间|完成时间|修改时间|保存时间|操作时间|"
                    r"医师\s*[:：]|医生\s*[:：]")


def _end_of_body(text):
    """正文末尾：去掉末尾空白，再跳过末尾的签名、时间戳一类短行（0.2）。全是这种行时返回去掉空白后的末尾。"""
    end0 = end = len(text.rstrip())
    while end > 0:
        a = text.rfind("\n", 0, end) + 1
        line = text[a:end].strip()
        if line and not (len(line) <= 60 and _STAMP.search(line)):
            return end
        if a == 0:
            break
        end = len(text[:a - 1].rstrip())
    return end0


def _insert_sentence(adm, doc, at, piece):
    """插入一句话；前一个字不是句末标点、换行或分句号时先补一个句号。返回这句话本身的 (起, 止)。"""
    k = at
    while k > 0 and doc.text[k - 1] in " \t　":
        k -= 1
    lead = "" if k == 0 or doc.text[k - 1] in "。；;\n，,：:" else "。"
    s, e = _insert(adm, doc, at, lead + piece)
    return s + len(lead), e


def _section_span(text, title_re, stop_re):
    m = re.search(title_re, text)
    if not m:
        return None
    rest = text[m.end():]
    n = re.search(stop_re, rest)
    return m.end(), m.end() + (n.start() if n else len(rest))


_STOP = r"\n\s*(?:[一二三四五六七八九十]+[、.．]\s*)?[一-鿿]{2,8}\s*[:：]"


def _sent_iter(text):
    a = 0
    for i, ch in enumerate(text):
        if ch in "。；;\n":
            if text[a:i].strip():
                yield a, i + 1
            a = i + 1
    if text[a:].strip():
        yield a, len(text)


def _post(adm):
    ds = adm.ds
    for d in adm.docs:
        if d.subtype == "术后首次病程" and (ds is None or d.pos < ds.pos):
            return d
    return None


def _pre_op_docs(adm, post):
    return [d for d in adm.docs if d.pos < post.pos and d.day < post.day]


# ---------- S1 ----------

SEX_SUBJ = re.compile(r"(患者|病人|患儿)[，,、：:\s]*(男|女)")
HEAD_SEX = re.compile(r"性\s*别\s*[:：]?[\s|]*(男|女)")
HEAD_AGE = re.compile(r"年\s*龄\s*[:：]?[\s|]*(\d{1,3})\s*岁")


def mine_s1(adm, rng):
    hr = adm.hr
    m = HEAD_SEX.search(hr.text) if hr else None
    if not m:
        return None
    sex = m.group(1)
    opp = "女" if sex == "男" else "男"
    progs = [d for d in adm.docs if d.doc_type == "progress_note" and d.subtype != "首次病程"]
    if not progs:
        return None
    target = progs[-1]
    new = adm.clone("mine:S1")
    t = new.doc(target.idx)
    hit = SEX_SUBJ.search(t.text)
    if hit and hit.group(2) == sex:
        s, e = _replace(new, t, hit.start(2), hit.end(2), opp)
        return new, MineRecord("S1", t.idx, s, e, hr.idx, "own")
    age = HEAD_AGE.search(hr.text)
    piece = f"患者，{opp}，{age.group(1)}岁，今日一般情况可。" if age else f"患者，{opp}，今日一般情况可。"
    nl = t.text.find("\n")
    at = nl + 1 if 0 <= nl < 80 else 0
    s, e = _insert(new, t, at, piece)
    return new, MineRecord("S1", t.idx, s, s + 4, hr.idx, "insert")


# ---------- S3 ----------

def _lat(doc, section_names):
    """关1 v0.5 的侧别抽取，去掉多病灶部位、既往口吻、阴性所见（和关2 S3 的口径相同）。
    返回 [(部位, 侧, 侧字的位置, 止)]。"""
    out = []
    for f in extract_laterality(doc.text, (doc.idx, 0, doc.doc_type), doc.sections):
        if section_names is not None and f.section not in section_names:
            continue
        side, organ, *rest = f.value_norm.split("|")
        kv = dict(x.split("=", 1) for x in rest)
        if organ in MULTI_LESION_ORGANS or kv.get("hist") == "1" or kv.get("neg") == "1":
            continue
        out.append((organ, side, f.start, f.end))
    return out


def mine_s3(adm, rng):
    """出院诊断里某个成对部位，入院记录（主诉、现病史、入院诊断）只写了同一侧：把出院诊断里这一部位的
    侧字全部翻过来。术前小结、术后首次病程里该部位也只能是同一侧（否则不选它）。"""
    hr, ds = adm.hr, adm.ds
    if hr is None or ds is None:
        return None
    a_side = {}
    for organ, side, _, _ in _lat(hr, {"主诉", "现病史", "入院诊断"}):
        a_side.setdefault(organ, set()).add(side)
    other = {}
    for d in adm.docs:
        if d.subtype in ("首次病程", "术前小结", "术后首次病程"):
            names = {"入院诊断"} if d.subtype == "首次病程" else None
            for organ, side, _, _ in _lat(d, names):
                other.setdefault(organ, set()).add(side)
    ds_facts = _lat(ds, {"出院诊断"})
    ds_side = {}
    for organ, side, _, _ in ds_facts:
        ds_side.setdefault(organ, set()).add(side)
    for organ in sorted(ds_side):
        sides = ds_side[organ]
        if len(sides) != 1 or "双" in sides or a_side.get(organ) != sides or not other.get(organ, sides) <= sides:
            continue
        side = next(iter(sides))
        flip = "右" if side == "左" else "左"
        targets = [(s, e) for o, sd, s, e in ds_facts if o == organ]
        new = adm.clone("mine:S3")
        t = new.doc(ds.idx)
        text = t.text
        for s, _ in targets:
            if text[s] != side:          # 侧字不是单字（理论上不会），放弃这个部位
                break
            text = text[:s] + flip + text[s + 1:]
        else:
            new.replace_text(t.idx, text)
            s0, e0 = targets[0]
            return new, MineRecord("S3", t.idx, s0, e0, hr.idx, "own", f"{organ}×{len(targets)}")
    return None


# ---------- S4a / S4b ----------

def mine_s4a(adm, rng):
    post = _post(adm)
    if post is None or re.search(r"中转|转为开|改为开|改行开", post.text + (adm.ds.text if adm.ds else "")):
        return None
    pre = [d for d in adm.docs if d.subtype == "术前小结" and d.pos < post.pos]
    if not pre:
        return None
    pre = pre[-1]
    for scope, opened in (("腹腔镜", "开腹"), ("胸腔镜", "开胸")):
        m = re.search(scope + r"[^，,。；;\n]{0,20}术", post.text)
        if m and scope in pre.text:
            new = adm.clone("mine:S4a")
            t = new.doc(post.idx)
            s, e = _replace(new, t, m.start(), m.start() + len(scope), opened)
            return new, MineRecord("S4a", t.idx, s, e, pre.idx, "own", f"{scope}→{opened}")
    for opened, scope in (("开腹", "腹腔镜"), ("开胸", "胸腔镜")):
        m = re.search(opened + r"[^，,。；;\n]{0,20}术", post.text)
        if m and opened in pre.text:
            new = adm.clone("mine:S4a")
            t = new.doc(post.idx)
            s, e = _replace(new, t, m.start(), m.start() + len(opened), scope)
            return new, MineRecord("S4a", t.idx, s, e, pre.idx, "own", f"{opened}→{scope}")
    return None


DATE_SURG = re.compile(r"(\d{4})(\s*[-/.年]\s*)(\d{1,2})(\s*[-/.月]\s*)(\d{1,2})(日?)(?=[^。；;\n]{0,30}?(?:行|在[^。；;\n]{0,15}下行)[^。；;\n]{0,25}?术)")


def mine_s4b(adm, rng):
    post, ds = _post(adm), adm.ds
    if post is None or ds is None:
        return None
    span = _section_span(ds.text, r"诊\s*疗\s*经\s*过\s*[:：]?", _STOP) or _section_span(ds.text, r"住\s*院\s*经\s*过\s*[:：]?", _STOP)
    if span is None:
        return None
    m = DATE_SURG.search(ds.text, span[0], span[1])
    if not m:
        return None
    try:
        d0 = dt.date(int(m.group(1)), int(m.group(3)), int(m.group(5)))
    except ValueError:
        return None
    d1 = d0 + dt.timedelta(days=2)
    if "年" in m.group(2):
        piece = f"{d1.year}{m.group(2)}{d1.month}{m.group(4)}{d1.day}{m.group(6)}"
    else:
        w = len(m.group(3))
        piece = f"{d1.year}{m.group(2)}{d1.month:0{w}d}{m.group(4)}{d1.day:0{len(m.group(5))}d}{m.group(6)}"
    new = adm.clone("mine:S4b")
    t = new.doc(ds.idx)
    s, e = _replace(new, t, m.start(), m.end(), piece)
    return new, MineRecord("S4b", t.idx, s, e, post.idx, "own", "+2天")


# ---------- K1 ----------

POSTOP_SENT = re.compile(r"术中见|安返病房|返回病房|手术顺利|术后第\s*\d|麻醉(?:满意|清醒)")


def mine_k1(adm, rng):
    post = _post(adm)
    if post is None:
        return None
    before = _pre_op_docs(adm, post)
    if not before:
        return None
    progs = [d for d in before if d.doc_type == "progress_note"]
    target = progs[-1] if progs else before[-1]
    piece, mode = None, "own"
    for a, b in _sent_iter(post.text):
        sent = post.text[a:b].strip()
        if POSTOP_SENT.search(sent) and 6 <= len(sent) <= 80:
            piece = sent if sent[-1] in "。；;" else sent + "。"
            break
    if piece is None:
        piece, mode = "患者术后第1天，切口敷料干燥，无渗血渗液。", "template"
    new = adm.clone("mine:K1")
    t = new.doc(target.idx)
    s, e = _insert_sentence(new, t, _end_of_body(t.text), piece)
    return new, MineRecord("K1", t.idx, s, e, post.idx, mode)


# ---------- K2 ----------

# 0.2：只抄已经出了结果的病理句（"病理回报：……""病理示……""病理诊断：……"）；
# 0.1 的写法会抄到术后首次病程里的"标本送病理，待病理结果回报"，那不是结果，尺子不报是对的
PATH_SENT = re.compile(r"病理(?:检查)?(?:结果|报告|诊断)?\s*(?:回报|示|提示|显示|证实|为|[:：])\s*[^\s，,。；;\n]{2}")
PATH_PENDING = re.compile(r"待|未回|未出|尚未|送检|送病理|拟|如|若|必要时|回报后|结果后|进一步|将")
PATH_AVOID = re.compile(r"穿刺|活检|外院|门诊|既往|年前|镜下|内镜|胃镜|肠镜|支气管镜|细胞学")


def mine_k2(adm, rng, dx_name=None):
    post = _post(adm)
    if post is None:
        return None
    hr = adm.hr
    if hr is None or hr.day >= post.day:
        return None
    piece, mode, src = None, "own", post.idx
    for d in adm.docs:
        if d.pos <= post.pos - 1:
            continue
        for a, b in _sent_iter(d.text):
            sent = d.text[a:b].strip()
            if PATH_SENT.search(sent) and not PATH_AVOID.search(sent) and not PATH_PENDING.search(sent) \
                    and 6 <= len(sent) <= 80:
                piece, src = (sent if sent[-1] in "。；;" else sent + "。"), d.idx
                break
        if piece:
            break
    if piece is None:
        if not dx_name:
            return None
        piece, mode = f"病理回报：符合{dx_name}。", "template"
    new = adm.clone("mine:K2")
    t = new.doc(hr.idx)
    span = _section_span(t.text, r"辅\s*助\s*检\s*查\s*[:：]?", _STOP)
    at = span[1] if span else _end_of_body(t.text)
    while at > 0 and t.text[at - 1] == "\n":
        at -= 1
    s, e = _insert_sentence(new, t, at, piece)
    return new, MineRecord("K2", t.idx, s, e, src, mode)


# ---------- K3 ----------

SCHED_M = re.compile(r"MRCP|磁共振|核磁|MRI|胃镜|肠镜|支气管镜|冠脉造影|冠状动脉造影|PET-CT|骨髓穿刺|骨穿|动态心电图|肺功能|血培养|痰培养|尿培养|心脏彩超|超声心动图")
PLAN_M = re.compile(r"(?:拟|完善|预约|申请|择期|建议|进一步)[^，,。；;\n]{0,10}$")
RES_M = re.compile(r"^[^，,。；;\n]{0,8}?(?:示|提示|回报|结果[:：示]|显示)")


def mine_k3(adm, rng):
    docs = adm.docs
    for i, di in enumerate(docs):
        if i == 0:
            continue
        for m in SCHED_M.finditer(di.text):
            a = max(di.text.rfind(ch, 0, m.start()) for ch in "，,。；;\n") + 1
            if not PLAN_M.search(di.text[a:m.start()]) or "复查" in di.text[a:m.start()]:
                continue
            exam = m.group(0)
            if any(exam in d.text for d in docs[:i]):
                continue  # 之前已经提过这项检查，不适合
            for dj in docs[i + 1:]:
                for n in re.finditer(re.escape(exam), dj.text):
                    if not RES_M.search(dj.text[n.end():n.end() + 12]):
                        continue
                    sa = max(dj.text.rfind(ch, 0, n.start()) for ch in "。；;\n") + 1
                    ends = [x for x in (dj.text.find(ch, n.end()) for ch in "。；;\n") if x >= 0]
                    sb = min(ends) if ends else len(dj.text)
                    sent = dj.text[sa:sb].strip()
                    if not (6 <= len(sent) <= 80) or re.search(r"复查|外院|门诊|既往", sent):
                        continue
                    piece = sent + "。"
                    target = docs[i - 1]
                    if target.doc_type == "hospital_record":
                        break   # 0.3：入院记录的辅助检查按规范就是入院前结果，尺子不比，雷也不埋在那里
                    new = adm.clone("mine:K3")
                    t = new.doc(target.idx)
                    at = _end_of_body(t.text)
                    while at > 0 and t.text[at - 1] == "\n":
                        at -= 1
                    s, e = _insert_sentence(new, t, at, piece)
                    return new, MineRecord("K3", t.idx, s, e, di.idx, "own", exam)
    return _mine_k3_template(adm, rng)


# 0.3：真实病历里检查多半在首次病程里安排、前面只有入院记录，"own"方式几乎埋不进去。
# 退而用模板：挑一项这次住院从没提过的需排期检查，较早一篇病程末尾写它的结果，较晚一篇病程末尾写"完善……检查"。
_K3_TEMPLATE_KEYS = ("肺功能", "动态心电图", "骨密度", "脑电图", "肌电图")
_EXAMS = {e["key"]: e for e in json.loads((Path(__file__).resolve().parent.parent / "ruler_g2" / "config" / "exams.json")
                                          .read_text(encoding="utf-8"))["exams"]}


def _mine_k3_template(adm, rng):
    progs = [d for d in adm.docs if d.doc_type == "progress_note"]
    if len(progs) < 2:
        return None
    texts = "".join(d.text for d in adm.docs + adm.off_docs)
    keys = [k for k in _K3_TEMPLATE_KEYS
            if k in _EXAMS and _EXAMS[k]["sched"] and not any(a in texts for a in _EXAMS[k]["aliases"])]
    if not keys:
        return None
    key = rng.choice(keys)
    word = _EXAMS[key]["aliases"][0]
    early, late = progs[0], progs[max(1, len(progs) // 2)]
    new = adm.clone("mine:K3")
    t_late = new.doc(late.idx)
    _insert_sentence(new, t_late, _end_of_body(t_late.text), f"完善{word}检查。")
    t = new.doc(early.idx)
    s, e = _insert_sentence(new, t, _end_of_body(t.text), f"{word}示未见明显异常。")
    return new, MineRecord("K3", t.idx, s, e, late.idx, "template", word)


# ---------- K4 ----------

K4_SRC = re.compile(r"术后|病理[^。；;\n]{0,4}(?:示|回报|结果)|复查[^。；;\n]{0,10}示|入院后[^。；;\n]{0,15}示")


def mine_k4(adm, rng):
    ds = adm.ds
    if ds is None:
        return None
    adm_sec = _section_span(ds.text, r"入\s*院\s*情\s*况\s*[:：]?", _STOP)
    hc = _section_span(ds.text, r"(?:诊\s*疗|住\s*院)\s*经\s*过\s*[:：]?", _STOP)
    if adm_sec is None or hc is None:
        return None
    for a, b in _sent_iter(ds.text[hc[0]:hc[1]]):
        sent = ds.text[hc[0] + a:hc[0] + b].strip()
        if K4_SRC.search(sent) and 6 <= len(sent) <= 80:
            piece = sent if sent[-1] in "。；;" else sent + "。"
            new = adm.clone("mine:K4")
            t = new.doc(ds.idx)
            at = adm_sec[1]
            while at > adm_sec[0] and t.text[at - 1] == "\n":
                at -= 1
            s, e = _insert_sentence(new, t, at, piece)
            return new, MineRecord("K4", t.idx, s, e, ds.idx, "own")
    return None


# ---------- K5 ----------

def mine_k5(adm, rng):
    if adm.admit_date is None:
        return None
    progs = [d for d in adm.docs if d.doc_type == "progress_note"]
    ds = adm.ds
    progs = [d for d in progs if ds is None or d.day + 3 <= ds.day]
    if not progs:
        return None
    t0 = progs[len(progs) // 2]
    d1 = adm.admit_date + dt.timedelta(days=t0.day + 2)
    piece = f"{d1.isoformat()}血常规示白细胞正常。"
    new = adm.clone("mine:K5")
    t = new.doc(t0.idx)
    s, e = _insert_sentence(new, t, _end_of_body(t.text), piece)
    return new, MineRecord("K5", t.idx, s, s + 10, t.idx, "template")


# ---------- B2 ----------

HEDGED_ITEM = re.compile(r"(?:^|[\s；;。，,：:？?）)])(?:\d{1,2}\s*[\.、．)）]\s*)?([^\s；;。，,：:？?\d]{2,20}?)\s*(?:[？?]|待查|可能)")
RESULT_SENT = re.compile(r"示|提示|回报|显示|结果|正常|异常|未见|阳性|阴性|升高|降低|增高|减低")
# 0.2：任何检查、检验的名字（同一份词表，只当词表用，不调用尺子的口吻判断）
_EXAM_WORDS = sorted({a for e in json.loads((Path(__file__).resolve().parent.parent / "ruler_g2" / "config" / "exams.json")
                                              .read_text(encoding="utf-8"))["exams"] for a in e["aliases"]}, key=len, reverse=True)
EXAM_SENT = re.compile("|".join(re.escape(a) for a in _EXAM_WORDS))
DATE_LINE = re.compile(r"\s*\d{4}")


def _has_history(adm, name):
    pat = re.compile(re.escape(name) + r"\s*(?:病)?史|既往[^。；;\n]{0,10}" + re.escape(name) + r"|有\s*" + re.escape(name) + r"\s*(?:病)?史")
    return any(pat.search(d.text) for d in adm.docs)


def _strip_evidence(new, docs, name, keep=None):
    """在这些篇里删去提到检查检验、或带结果字眼的句子（日期抬头行留着）。返回删去的句数。
    0.3：含诊断名的句子不再一律留着（"胸部CT示肺炎"正是依据），只留 keep=(篇号, 位置) 所在的那一句。"""
    removed = 0
    for d in docs:
        t = new.doc(d.idx)
        cursor, out = 0, []
        for a, b in _sent_iter(t.text):
            sent = t.text[a:b]
            if DATE_LINE.match(sent) or (keep and keep[0] == d.idx and a <= keep[1] < b):
                continue
            if RESULT_SENT.search(sent) or EXAM_SENT.search(sent):
                removed += 1
                out.append(t.text[cursor:a])
                cursor = b
        out.append(t.text[cursor:])
        new.replace_text(d.idx, "".join(out))
    return removed


def _b2_confirm_seen(new, name, doc_idx):
    """埋好以后，尺子的诊断词典在那一篇里认不认得出"确诊"状态的这个名字（只管认名字和状态，不管依据）。"""
    return any(x.doc_idx == doc_idx and x.state in dxm.CONFIRM_STATES and dxm.dx_match(x.norm, name)
               for x in dxm.mentions(new))


def mine_b2(adm, rng):
    hr = adm.hr
    if hr is None:
        return None
    nxt = [d for d in adm.docs if d.pos > hr.pos and d.doc_type == "progress_note"]
    if not nxt:
        return None
    # 0.3：候选诊断照尺子的诊断词典取（入院记录里"疑似"状态的名字）。0.2 用自己的正则取名字，
    # fold 0 上 200 个宿主里有 92 个名字尺子不认或认不成"疑似→确诊"
    dm = dxm.mentions(adm)
    cands = []
    for x in dm:
        if x.doc_idx == hr.idx and x.state in dxm.SUSPECT_STATES and x.norm not in cands:
            cands.append(x.norm)
    for name in cands:
        if _has_history(adm, name) or any(x.state == "既往" and dxm.dx_match(x.norm, name) for x in dm):
            continue      # 尺子不比带既往史的诊断
        # 轨迹方式：后面某篇病程写了"name诊断明确/确诊"，删掉入院记录之后到那一篇之间的依据句（确诊那一句留着）
        conf_re = re.compile(re.escape(name) + r"\s*(?:诊断)?(?:明确|确诊|诊断明确)|(?:确诊|证实|诊断为)\s*" + re.escape(name))
        for j, dj in enumerate(nxt):
            c = conf_re.search(dj.text)
            if c is None:
                continue
            new = adm.clone("mine:B2")
            removed = _strip_evidence(new, nxt[:j + 1], name, keep=(dj.idx, c.start()))
            t = new.doc(dj.idx)
            c2 = conf_re.search(t.text)
            if removed and c2 is not None and _b2_confirm_seen(new, name, dj.idx):
                k = t.text.find(name, c2.start())
                return new, MineRecord("B2", t.idx, k, k + len(name), hr.idx, "own", f"删去{removed}句")
            break
        # 插入方式：入院记录之后到首次病程（含）的各篇删去依据句，首次病程开头插入"name诊断明确。"
        fp = adm.fp
        if fp is None:
            return None
        new = adm.clone("mine:B2")
        removed = _strip_evidence(new, [d for d in adm.docs if hr.pos < d.pos <= fp.pos], name)
        t = new.doc(fp.idx)
        text = t.text
        nl = text.find("\n")
        at = nl + 1 if 0 <= nl < 80 else 0
        piece = f"{name}诊断明确。"
        new.replace_text(fp.idx, text[:at] + piece + text[at:])
        if _b2_confirm_seen(new, name, fp.idx):
            return new, MineRecord("B2", fp.idx, at, at + len(name), hr.idx, "insert", f"删去{removed}句")
    return None


_FN = {"S1": mine_s1, "S3": mine_s3, "S4a": mine_s4a, "S4b": mine_s4b, "K1": mine_k1, "K2": mine_k2,
       "K3": mine_k3, "K4": mine_k4, "K5": mine_k5, "B2": mine_b2}


def plant(adm, mine_type, rng, **kw):
    fn = _FN[mine_type]
    return fn(adm, rng, **kw) if kw else fn(adm, rng)


def plant_many(adm, mine_types, rng, **kw):
    """一次住院埋多颗雷（校8 的 2 颗、4 颗）。每颗都在上一颗的结果上埋；任何一颗埋不进去返回 None。"""
    cur, recs = adm, []
    for t in mine_types:
        r = plant(cur, t, rng, **(kw if t == "K2" else {}))
        if r is None:
            return None
        cur, rec = r
        recs.append(rec)
    cur.source = "mine:" + "+".join(mine_types)
    return cur, recs


# ---------- 合法差异（校9） ----------
# 每一种：在干净住院上做一个合法的改动，对应的读数不应该新报冲突

LEGIT_TYPES = ("L1", "L2", "L3", "L5", "L6", "L7", "L8")
LEGIT_READING = {"L1": "S4a", "L2": "K2", "L3": "K3", "L5": "B2", "L6": "S5", "L7": "K1", "L8": "K1"}


def legit(adm, kind, rng):
    if kind == "L1":   # 术式变了，但写明了中转开腹
        r = mine_s4a(adm, rng)
        if r is None:
            return None
        new, rec = r
        t = new.doc(rec.doc_idx)
        _insert(new, t, rec.end, "（术中因粘连严重中转开腹）")
        new.source = "legit:L1"
        return new, rec
    if kind == "L2":   # 外院病理（既往口吻）放进入院记录
        post = _post(adm)
        hr = adm.hr
        if post is None or hr is None:
            return None
        new = adm.clone("legit:L2")
        t = new.doc(hr.idx)
        span = _section_span(t.text, r"辅\s*助\s*检\s*查\s*[:：]?", _STOP)
        at = span[1] if span else _end_of_body(t.text)
        while at > 0 and t.text[at - 1] == "\n":
            at -= 1
        s, e = _insert_sentence(new, t, at, "外院穿刺病理示慢性炎症。")
        return new, MineRecord("L2", t.idx, s, e, post.idx, "insert")
    if kind == "L3":   # 安排之前的结果，写明是门诊做的
        r = mine_k3(adm, rng)
        if r is None:
            return None
        new, rec = r
        t = new.doc(rec.doc_idx)
        k = t.text.find(rec.note, rec.start, rec.end)
        k = rec.start if k < 0 else k
        new.replace_text(t.idx, t.text[:k] + "门诊" + t.text[k:])
        new.source = "legit:L3"
        return new, MineRecord("L3", rec.doc_idx, rec.start, rec.end + 2, rec.anchor_idx, "insert")
    if kind == "L5":   # 疑似变确诊，同一句前面写了新的检查结果（不用病理：确诊可能在取材之前）
        r = mine_b2(adm, rng)
        if r is None:
            return None
        new, rec = r
        t = new.doc(rec.doc_idx)
        sa = rec.start
        while sa > 0 and t.text[sa - 1] not in "。；;\n":
            sa -= 1
        s, e = _insert(new, t, sa, "复查血常规示白细胞较前升高，")
        new.source = "legit:L5"
        return new, MineRecord("L5", rec.doc_idx, s, rec.end + (e - s), rec.anchor_idx, "insert")
    if kind == "L6":   # 住院中新出现的药物过敏反应
        progs = [d for d in adm.docs if d.doc_type == "progress_note"]
        if not progs:
            return None
        new = adm.clone("legit:L6")
        t = new.doc(progs[len(progs) // 2].idx)
        s, e = _insert_sentence(new, t, _end_of_body(t.text), "患者用药后出现皮疹，考虑药物过敏，予停药及抗过敏治疗。")
        return new, MineRecord("L6", t.idx, s, e, adm.hr.idx, "insert")
    if kind == "L7":   # 术前小结里的术后风险告知
        post = _post(adm)
        if post is None:
            return None
        before = [d for d in _pre_op_docs(adm, post) if d.doc_type == "progress_note"]
        if not before:
            return None
        new = adm.clone("legit:L7")
        t = new.doc(before[-1].idx)
        s, e = _insert_sentence(new, t, _end_of_body(t.text), "术后可能出现出血、感染、切口愈合不良等并发症，已向家属交代并签字。")
        return new, MineRecord("L7", t.idx, s, e, post.idx, "insert")
    if kind == "L8":   # 既往手术史里的"术后恢复良好"
        post = _post(adm)
        hr = adm.hr
        if post is None or hr is None:
            return None
        new = adm.clone("legit:L8")
        t = new.doc(hr.idx)
        span = _section_span(t.text, r"既\s*往\s*史\s*[:：]?", _STOP)
        at = span[0] if span else _end_of_body(t.text)
        s, e = _insert(new, t, at, "10年前行阑尾切除术，术后恢复良好。")
        return new, MineRecord("L8", t.idx, s, e, post.idx, "insert")
    raise ValueError(kind)
