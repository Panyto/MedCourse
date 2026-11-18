"""Q45b（关4、关5 确认轮）新加的工具：新住院抽样、填表病历的宿主、不配对的均值差、等长判别器。

PREREG_Q45b 第 2 节冻结这里的全部设置。0.1 的模块（floors、judge、c2st、demog、stats）一字不改，这里只在它们外面加东西。
"""
import random
import re

from ruler_g3.adapters import care_get
from ruler_g4.ruler import FIELDS, _DATE, _DATE_PH, _RELDAY, nchars

from . import c2st
from .floors import ftpl
from .stats import _pct

NEW_SEED = 20261028          # 新住院抽样的种子
N_NEW = 256                  # 新住院份数（与关1 槽位数相同）
MIN_FIELD_CHARS = 2          # 四段叙述每段至少这么多字

_CJK = re.compile(r"[一-鿿]")
_PLACEHOLDER = re.compile(r"\[[^\[\]\n]{0,30}\]")
_DX_HEAD = re.compile(r"^\s*(?:出院诊断|最后诊断|诊断)\s*[:：]?\s*")
# 诊断之间的分隔：换行、分号、句号；同一行里"1.甲 2.乙"这种，在编号前的空白处切
_DX_SPLIT = re.compile(r"[\n；;。]|\s+(?=\d{1,2}\s*[、.．)）])|[，,](?=\d{1,2}\s*[、.．)）])")
# 每一条开头的编号（"1.""2、""3）"）。"1.2型糖尿病"是第 1 条"2型糖尿病"
_DX_NUM = re.compile(r"^\s*\d{1,2}\s*[、.．)）]\s*")
_TAIL_NOTE = re.compile(r"\s*[（(][^（）()]{0,20}[)）]\s*$")


# ---------- H4-1′：新住院 ----------

def first_dx(text):
    """出院诊断一节 → 第一条诊断的名字（去编号、去占位符、去末尾括号说明，最多 30 字）。认不出返回 ""。
    填表病历只用这一条（关1 的 F-TPL 用的是字典标准名，新住院没有字典匹配，用原文第一条）。"""
    s = _PLACEHOLDER.sub("", text or "")        # 不做 NFKC：填表病历里的诊断要和真实病历里的写法一字不差（"Ⅱ级"不变成"II级"）
    s = _DX_HEAD.sub("", s)
    for part in _DX_SPLIT.split(s):
        p = _DX_NUM.sub("", part).strip(" \t,，:：、")
        p = _TAIL_NOTE.sub("", p).strip(" \t,，:：、")
        if len(_CJK.findall(p)) >= 2:
            return p[:30]
    return ""


def eligible(adm_id, record, reason, used_ids):
    """新住院的入选条件：表头读得出性别和年龄（reason == "ok"）；不在关1 的 256 个槽位里；四段叙述每段至少 2 个字；
    出院诊断第一条认得出。返回 (是否入选, 不入选的原因)。"""
    if reason != "ok":
        return False, reason
    if str(adm_id) in used_ids:
        return False, "used_slot"
    if any(nchars(care_get(record, k)) < MIN_FIELD_CHARS for k in FIELDS):
        return False, "short_field"
    if not first_dx(care_get(record, "dx")):
        return False, "no_dx"
    return True, "ok"


def pick_new(eligible_ids, n=N_NEW, seed=NEW_SEED):
    """从入选的住院号里不放回随机抽 n 个（先排序再抽，结果与输入顺序无关），返回排好序的住院号。"""
    ids = sorted({str(x) for x in eligible_ids})
    rng = random.Random(f"{seed}|q45b|new")
    return sorted(rng.sample(ids, min(n, len(ids))))


def template_for(record):
    """新住院的填表病历：性别、年龄取这次住院表头的，主要诊断取出院诊断第一条，四段是 floors.TPL 原句。"""
    host = dict(record)
    host["Primary Diagnosis"] = first_dx(care_get(record, "dx"))
    return ftpl(host)


# ---------- H4-4：不配对的均值差 ----------

def unpaired_mean_diff(a, b, B=1000, seed=20261014):
    """两组独立样本 a − b 的均值差，两组各自有放回重抽。a、b 是数的列表（None 不用）。
    返回 (差, 下限, 上限, a 的份数, b 的份数, b 的均值)。"""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        nan = float("nan")
        return nan, nan, nan, len(a), len(b), nan
    m = lambda xs: sum(xs) / len(xs)
    est = m(a) - m(b)
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        boots.append(m([rng.choice(a) for _ in a]) - m([rng.choice(b) for _ in b]))
    return est, _pct(boots, 0.025), _pct(boots, 0.975), len(a), len(b), m(b)


# ---------- H5-2′：等长判别器 ----------

def strip_dates(text):
    """去掉日期、日期占位符和"第 N 天"一类相对日（另报"不靠日期还分不分得开"用）。"""
    text = _DATE_PH.sub("", text or "")
    text = _DATE.sub("", text)
    return _RELDAY.sub("", text)


def equal_pair(rec_a, rec_b, fields=FIELDS, no_dates=False):
    """两份 7 字段病历逐段规范化（c2st.normalize），每段都截到两份里较短那份的长度（从段首截），再用"｜"连起来。
    这样每一段两边一样长，长度这条线索没有了。no_dates=True 时先去掉日期类字样再规范化。返回 (a 的输入, b 的输入)。"""
    xa, xb = [], []
    for k in fields:
        ta, tb = care_get(rec_a, k), care_get(rec_b, k)
        if no_dates:
            ta, tb = strip_dates(ta), strip_dates(tb)
        a, b = c2st.normalize(ta), c2st.normalize(tb)
        n = min(len(a), len(b))
        xa.append(a[:n])
        xb.append(b[:n])
    return "｜".join(xa), "｜".join(xb)


def equal_lengths(x):
    """equal_pair 输出的每段长度（只看长度的对照分类器用）。"""
    return [len(p) for p in x.split("｜")]


def null_split_equal(items, strata, reps=c2st.NULL_REPS, seed=c2st.BOOT_SEED, no_dates=False, kind="text"):
    """真实对真实，和 c2st.null_split 同一个结构（同一三位码里两两配对、每对随机一个标 1），
    只是每对先按 equal_pair 截成等长。items=[(槽位, 7 字段病历)]。返回每次的 AUC。"""
    by = {}
    for g, rec in items:
        by.setdefault(strata.get(g, ""), []).append((g, rec))
    out = []
    for r in range(reps):
        rng = random.Random(f"{seed}|nulleq|{r}")
        xs, ys, gs = [], [], []
        for code in sorted(by):
            rows = sorted(by[code], key=lambda t: str(t[0]))
            rng.shuffle(rows)
            for j in range(0, len(rows) - 1, 2):
                (g1, r1), (g2, r2) = rows[j], rows[j + 1]
                x1, x2 = equal_pair(r1, r2, no_dates=no_dates)
                if kind == "length":
                    x1, x2 = equal_lengths(x1), equal_lengths(x2)
                flip = rng.random() < 0.5
                for x, lab in zip((x1, x2), (1, 0) if flip else (0, 1)):
                    xs.append(x)
                    ys.append(lab)
                    gs.append(f"{code}|{j}")
        out.append(c2st.auc(c2st.cv_scores(xs, ys, gs, c2st.SEEDS[0], kind), ys))
    return out
