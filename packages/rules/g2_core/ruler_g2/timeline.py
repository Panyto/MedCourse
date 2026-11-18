"""把一次住院的文书排成时间轴；病程细类；核心集。

真实病历（导出文件一行）：build_real(record)
合成病历（按槽位骨架逐篇生成）：build_synth(slot, docs)

排序规则（10-20 对导出文件的检查）：
- 一律按 day_offset 排，同一天按 入院记录 → 病程 → 出院记录，再按导出文件里的下标；
- day_offset 为空的病程不进时间轴（pos=None，flags 含 null_day），不参与任何先后判定；
- day_offset 晚于出院记录的病程是出院后记录（flags 含 post_discharge），不进时间轴。
"""
import copy
import datetime as dt
import re
from dataclasses import dataclass, field

from ruler_g1.extractors import HEAD_DATE
from ruler_g1.sections import FIRST_PROGRESS_RE, is_first_progress, split_sections

_SP = r"\s*"


def _t(word):
    return _SP.join(re.escape(ch) for ch in word)


# 病程细类：看开头 TITLE_WINDOW 字里的标题词，取位置最靠前的一个；同一位置按下面的顺序。
# 0.2：窗口从 80 字放到 300 字，和 v0.5 的 is_first_progress 一样。0.1 只看 80 字，fold 0 只认出
# 160 篇术后首次病程、3 篇术前小结；关1 包2 r1 的 A11 在前 300 字里数出 531 篇术后首次病程，
# D0 数出 326 篇术前小结，说明抬头（科室、床号、住院号等）常常超过 80 字。
# 0.1 按规则顺序取第一个命中的；窗口放大以后正文里顺带提到的"术前讨论""会诊意见"也会进窗口，
# 所以改成按位置取最靠前的。
TITLE_WINDOW = 300
SUBTYPE_RULES = [
    ("术后首次病程", re.compile(_t("术后首次病程") + "|" + _t("术后第一次病程") + "|" + _t("术后首程"))),
    ("术前小结", re.compile(_t("术前小结") + "|" + _t("术前总结"))),
    ("术前讨论", re.compile(_t("术前讨论"))),
    ("手术记录", re.compile(_t("手术记录"))),
    ("主任医师查房", re.compile("(?:副\\s*)?" + _t("主任医师查房") + "|" + _t("主任查房"))),
    ("主治医师查房", re.compile(_t("主治医师查房") + "|" + _t("主治查房"))),
    ("上级医师查房", re.compile(_t("上级医师查房"))),
    ("会诊记录", re.compile(_t("会诊记录") + "|" + _t("会诊意见"))),
    ("抢救记录", re.compile(_t("抢救记录"))),
    ("操作记录", re.compile(_t("操作记录"))),
    ("疑难病例讨论", re.compile(_t("疑难病例讨论"))),
    ("交接班记录", re.compile(_t("交班记录") + "|" + _t("接班记录") + "|" + _t("交接班记录"))),
    ("转科记录", re.compile(_t("转出记录") + "|" + _t("转入记录") + "|" + _t("转科记录"))),
    ("阶段小结", re.compile(_t("阶段小结"))),
    ("死亡记录", re.compile(_t("死亡记录") + "|" + _t("死亡讨论"))),
    ("日常查房", re.compile(_t("日常病程") + "|" + _t("日常查房") + "|" + _t("住院医师查房"))),
]
SUPERIOR_ROUNDS = {"主任医师查房", "主治医师查房", "上级医师查房"}
DAILY_LIKE = SUPERIOR_ROUNDS | {"日常查房", "其他病程", "阶段小结"}
TYPE_RANK = {"hospital_record": 0, "progress_note": 1, "discharge_summary": 2}


def classify_progress_detail(text):
    """返回 (细类, 标题词在正文里的位置)；没认出标题时位置为 None。"""
    head = text[:TITLE_WINDOW]
    if is_first_progress(text):
        m = FIRST_PROGRESS_RE.search(head)
        return "首次病程", (m.start() if m else None)
    best = None
    for rank, (name, rx) in enumerate(SUBTYPE_RULES):
        m = rx.search(head)
        if m and (best is None or (m.start(), rank) < best[0]):
            best = ((m.start(), rank), name)
    if best is None:
        return "其他病程", None
    return best[1], best[0][0]


def classify_progress(text):
    return classify_progress_detail(text)[0]


@dataclass
class Doc:
    idx: int                 # 导出文件 docs 列表里的下标（合成病历：骨架序号）
    doc_type: str            # hospital_record / progress_note / discharge_summary
    subtype: str             # 入院记录 / 首次病程 / … / 出院记录
    day: object              # int 或 None
    text: str
    pos: object = None       # 时间轴上的序号；不进时间轴为 None
    flags: set = field(default_factory=set)
    role: str = ""           # 核心集角色：HR FP SUP PRE POST DAILY DS
    _sections: object = None

    @property
    def sections(self):
        if self._sections is None:
            self._sections = split_sections(self.text)
        return self._sections

    def section(self, name):
        """返回 (start, end)；没有该段落返回 None。同名段落取第一个。"""
        for s in self.sections:
            if s.name == name:
                return s.start, s.end
        return None

    def section_name_at(self, pos):
        for s in self.sections:
            if s.start <= pos < s.end:
                return s.name
        return "未知"


@dataclass
class Admission:
    admission_id: str
    source: str              # real / synth:<arm> / floor:<kind> / mine:<type> …
    docs: list               # 时间轴上的文书按 pos 排好；不在时间轴上的放在 off_docs
    off_docs: list = field(default_factory=list)
    admit_date: object = None
    meta: dict = field(default_factory=dict)

    def by_role(self, role):
        return [d for d in self.docs if d.role == role]

    def first(self, pred):
        for d in self.docs:
            if pred(d):
                return d
        return None

    @property
    def hr(self):
        return self.first(lambda d: d.doc_type == "hospital_record")

    @property
    def ds(self):
        for d in reversed(self.docs):
            if d.doc_type == "discharge_summary":
                return d
        return None

    @property
    def fp(self):
        return self.first(lambda d: d.subtype == "首次病程")

    @property
    def post(self):
        """术后首次病程（第一篇，且在出院记录之前）。"""
        return self.first(lambda d: d.subtype == "术后首次病程")

    def doc(self, idx):
        for d in self.docs + self.off_docs:
            if d.idx == idx:
                return d
        raise KeyError(idx)

    def clone(self, source=None):
        new = copy.deepcopy(self)
        for d in new.docs + new.off_docs:
            d._sections = None
        if source:
            new.source = source
        return new

    def replace_text(self, idx, text):
        d = self.doc(idx)
        d.text = text
        d._sections = None


def _subtype(doc_type, text):
    if doc_type == "hospital_record":
        return "入院记录"
    if doc_type == "discharge_summary":
        return "出院记录"
    return classify_progress(text)


def _admit_date_from(text):
    for m in HEAD_DATE.finditer(text):
        if m.group(1) == "入":
            try:
                return dt.date(int(m.group(2)), int(m.group(3)), int(m.group(4)))
            except ValueError:
                return None
    return None


def _order(docs, source, admission_id, admit_date=None, meta=None):
    ds_days = [d.day for d in docs if d.doc_type == "discharge_summary" and d.day is not None]
    discharge_day = max(ds_days) if ds_days else None
    on, off = [], []
    for d in docs:
        if d.day is None:
            d.flags.add("null_day")
            off.append(d)
        elif d.doc_type == "progress_note" and discharge_day is not None and d.day > discharge_day:
            d.flags.add("post_discharge")
            off.append(d)
        else:
            on.append(d)
    on.sort(key=lambda d: (d.day, TYPE_RANK[d.doc_type], d.idx))
    for i, d in enumerate(on):
        d.pos = i
    adm = Admission(admission_id, source, on, off, admit_date, dict(meta or {}))
    if adm.admit_date is None:
        for d in on:
            if d.doc_type in ("hospital_record", "discharge_summary"):
                adm.admit_date = _admit_date_from(d.text)
                if adm.admit_date:
                    break
    adm.meta.setdefault("discharge_day", discharge_day)
    return adm


def build_real(record):
    """record：导出文件的一行（dict）。"""
    docs = []
    for i, x in enumerate(record["docs"]):
        day = x.get("day_offset")
        docs.append(Doc(i, x["doc_type"], _subtype(x["doc_type"], x["text"]),
                        int(day) if day is not None else None, x["text"]))
    meta = {"split_fold": record.get("split_fold"), "admit_no": record.get("admit_no")}
    return _order(docs, "real", str(record["admission_id"]), meta=meta)


SYNTH_TYPE = {"入院记录": ("hospital_record", "入院记录"), "首次病程记录": ("progress_note", "首次病程"),
              "主任医师查房记录": ("progress_note", "主任医师查房"), "主治医师查房记录": ("progress_note", "主治医师查房"),
              "上级医师查房记录": ("progress_note", "上级医师查房"), "日常病程记录": ("progress_note", "日常查房"),
              "术前小结": ("progress_note", "术前小结"), "术后首次病程记录": ("progress_note", "术后首次病程"),
              "出院记录": ("discharge_summary", "出院记录")}


def build_synth(slot, docs, source="synth"):
    """slot：槽位 dict（含 slot_id、admit_date "YYYY-MM-DD"）；
    docs：[{k, doc_type_cn, day, text, role?}]，k 是骨架序号。"""
    out = []
    for x in docs:
        doc_type, subtype = SYNTH_TYPE[x["doc_type_cn"]]
        out.append(Doc(int(x["k"]), doc_type, subtype, int(x["day"]), x["text"], role=x.get("role", "")))
    admit = dt.date.fromisoformat(slot["admit_date"]) if slot.get("admit_date") else None
    return _order(out, source, str(slot["slot_id"]), admit_date=admit, meta={"slot_id": slot["slot_id"]})


# ---------- 核心集 ----------

def core_set(adm, n_daily=3):
    """入院记录、首次病程、第一篇上级医师查房、术前小结和术后首次病程（有手术时）、
    至多 n_daily 篇其他病程（按住院天数的 1/4、2/4、3/4 处挑）、出院记录。
    返回带 role 的新 Admission；缺入院记录、首次病程或出院记录时返回 None。"""
    hr, ds = adm.hr, adm.ds
    fp = adm.fp
    if hr is None or ds is None or fp is None:
        return None
    chosen = {hr.idx: "HR", fp.idx: "FP", ds.idx: "DS"}
    inner = [d for d in adm.docs if d.doc_type == "progress_note" and fp.pos < d.pos < ds.pos]
    sup = next((d for d in inner if d.subtype in SUPERIOR_ROUNDS), None)
    if sup:
        chosen[sup.idx] = "SUP"
    post = next((d for d in adm.docs if d.subtype == "术后首次病程" and d.pos < ds.pos), None)
    if post:
        chosen[post.idx] = "POST"
        pre = [d for d in adm.docs if d.subtype == "术前小结" and d.pos < post.pos]
        if pre:
            chosen[pre[-1].idx] = "PRE"
    pool = [d for d in inner if d.idx not in chosen and d.subtype in DAILY_LIKE]
    los = max(1, ds.day)
    for q in (1, 2, 3)[:n_daily]:
        if not pool:
            break
        target = los * q / 4.0
        best = min(pool, key=lambda d: (abs(d.day - target), d.pos))
        chosen[best.idx] = "DAILY"
        pool.remove(best)
    new = adm.clone()
    new.docs = [d for d in new.docs if d.idx in chosen]
    for i, d in enumerate(new.docs):
        d.pos = i
        d.role = chosen[d.idx]
    new.off_docs = []
    new.meta["core"] = True
    return new


def surgery_day(adm):
    p = adm.post
    return p.day if p is not None else None
