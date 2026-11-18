"""关5 H5-1：边缘对、联合错。

- LLM-CARe 自己的"人口学典型性"（按病种比性别、年龄分布，性别用 1−TVD，年龄用 1−Wasserstein）：
  这里按论文的说法复现。论文的年龄距离怎么归一化没有公开，我们的口径是 1 − W1(岁)/100，W1 在 5 岁一档的分布上算
  （配额规划本来就按 5 岁一档分配）。按病种（三位码）各算一个，再等权平均。
- 联合：正文和写在年龄栏、性别栏里的值对不对得上。在 fold 0 真实病历上训练"由正文猜年龄段"和"由正文猜性别"的分类器
  （分类器和判别器同一套特征：字符 1–3 元组 TF-IDF，多分类逻辑回归 C=1.0，不调参）。
  配额改写只改年龄栏、不改正文。对每一份被改了年龄段的病历做"二选一"：同一段正文，更像改写前那个年龄段，还是改写后那个？
  比的是"模型给这一段的概率 ÷ 模型在同病种真实病历上给这一段的平均概率"（`prefers_old`）。除以同病种平均，是为了扣掉
  "这个病多见于老人"这类病种层面的信息：正文只透露病种、不透露这个人多大时，两边的比值一样，二选一是五五开；
  正文里有这个人的年龄线索（"退休""患儿""绝经后""高血压 20 年"），才会偏向它原来那个年龄段。
  同一套做法在真实病历上跑一遍（把配额改写原样用在 R7 上），看真实正文有多少能"认出"自己原来的年龄段：这是量具本身灵不灵的门槛。
"""
import re
from collections import Counter

import numpy as np
from ruler_g3.adapters import care_get
from ruler_g3.parse import parse_age_range
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from .c2st import make_vectorizer

AGE_BANDS = ((0, 6, "0-5"), (6, 14, "6-13"), (14, 18, "14-17"), (18, 45, "18-44"), (45, 65, "45-64"), (65, 999, "65+"))
BAND_NAMES = tuple(b[2] for b in AGE_BANDS)
BIN_WIDTH = 5
W1_SCALE = 100.0
MODEL_SEED = 20261027


def band(age):
    """年龄（岁，可以是小数）→ 年龄段名；None → None。"""
    if age is None:
        return None
    for lo, hi, name in AGE_BANDS:
        if lo <= age < hi:
            return name
    return None


def age_bin(age):
    return None if age is None else int(age) // BIN_WIDTH


def stated_age(value):
    """年龄栏 → 岁（下限）。"8个月"是 0.67 岁，"65"按岁算；认不出返回 None。用 ruler_g3 的 parse_age_range（关3 冻结）。
    注意配额规划器自己只取第一个整数（"8个月"当 8 岁），本函数不跟它的错。"""
    return parse_age_range(value)[0]


def stated_band(value):
    return band(stated_age(value))


FAR_YEARS = 15


def changed_far(old_age, new_age, years=FAR_YEARS):
    """H5-1 的主单位：年龄段变了，并且新旧年龄相差 ≥ years 岁。
    配额改写在 5 岁一档里随机取新年龄，"63 改成 66"也算换了年龄段，但从正文上不可能分得出，只会把二选一拉向 0.5；
    所以主读数只看差得远的，全部换了年龄段的另报。任何一边认不出年龄返回 False。"""
    a, b = stated_age(old_age), stated_age(new_age)
    if a is None or b is None:
        return False
    return band(a) != band(b) and abs(a - b) >= years


EXPLICIT_AGE = re.compile(r"\d+\s*(?:岁|月龄|个月大)|[一二两三四五六七八九十]+\s*岁")
NARRATIVE = ("cc", "hpi", "hc", "di")


def explicit_age(record):
    """四段叙述里有没有直接写出"N岁"一类年龄（关1 已经查过这类明写的矛盾；H5-1 另报去掉这些病历的结果）。"""
    return int(any(EXPLICIT_AGE.search(care_get(record, k) or "") for k in NARRATIVE))


def _norm(counter):
    tot = float(sum(counter.values()))
    return {k: v / tot for k, v in counter.items()} if tot else {}


def tvd(a, b):
    """a、b：{类别: 计数}。总变差距离。"""
    p, q = _norm(a), _norm(b)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def w1_bins(a, b, width=BIN_WIDTH):
    """a、b：{档序号: 计数}，档宽 width 岁。一维 Wasserstein-1（岁）。"""
    p, q = _norm(a), _norm(b)
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    cp = cq = 0.0
    total = 0.0
    for i, k in enumerate(keys[:-1]):
        cp += p.get(k, 0.0)
        cq += q.get(k, 0.0)
        total += abs(cp - cq) * (keys[i + 1] - k) * width
    return total


def typicality(synth, ref):
    """synth：[(三位码, 性别'男'/'女'/None, 年龄岁)]；ref：{三位码: {"sex": Counter, "age": Counter(5 岁档)}}，
    就是配额规划用的那张参照分布。返回 {"per_code": {码: {n, t_sex, t_age}}, "t_sex", "t_age", "t"}（病种等权平均）。
    参照里没有的码不算，计数写进 "skipped_codes"。"""
    by = {}
    for code, sex, age in synth:
        d = by.setdefault(code, {"sex": Counter(), "age": Counter(), "n": 0})
        d["n"] += 1
        if sex in ("男", "女"):
            d["sex"][sex] += 1
        b = age_bin(age)
        if b is not None:
            d["age"][b] += 1
    per, skipped = {}, []
    for code in sorted(by):
        r = ref.get(code)
        if not r:
            skipped.append(code)
            continue
        rs = Counter({k: v for k, v in r["sex"].items() if k in ("男", "女")})
        per[code] = {"n": by[code]["n"], "t_sex": 1.0 - tvd(by[code]["sex"], rs),
                     "t_age": max(0.0, 1.0 - w1_bins(by[code]["age"], r["age"]) / W1_SCALE)}
    ts = [v["t_sex"] for v in per.values()]
    ta = [v["t_age"] for v in per.values()]
    out = {"per_code": per, "skipped_codes": skipped,
           "t_sex": float(np.mean(ts)) if ts else float("nan"), "t_age": float(np.mean(ta)) if ta else float("nan")}
    out["t"] = (out["t_sex"] + out["t_age"]) / 2
    return out


# ---------- 正文 → 年龄段 / 性别 ----------

def make_text_model(seed=MODEL_SEED):
    return make_pipeline(make_vectorizer(),
                         LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=seed))


def fit(texts, labels, seed=MODEL_SEED):
    m = make_text_model(seed)
    m.fit(list(texts), list(labels))
    return m


def p_stated(model, texts, stated):
    """每份正文给它自己写的那个标签的概率。标签不在训练类别里的返回 None（单独计数）。"""
    classes = list(model.classes_)
    proba = model.predict_proba(list(texts))
    out = []
    for row, s in zip(proba, stated):
        out.append(float(row[classes.index(s)]) if s in classes else None)
    return out


def crossfit_rows(texts, labels, seed=MODEL_SEED, n_splits=5):
    """真实病历上的折外预测：每份只用没见过它的模型打分。返回 (每份的 {类别: 概率}, 每份给自己标签的先验频率)。
    类别取全部标签；某一折训练集里没有的类别，这一折的概率记 0。先验 = 训练折里该标签的占比。"""
    texts, labels = list(texts), list(labels)
    y = np.asarray(labels)
    classes = sorted(set(labels))
    rows = [None] * len(y)
    prior = [None] * len(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(np.zeros(len(y)), y):
        m = fit([texts[i] for i in tr], [labels[i] for i in tr], seed)
        freq = Counter(labels[i] for i in tr)
        n = float(len(tr))
        for i, row in zip(te, proba_rows(m, [texts[i] for i in te])):
            rows[i] = {c: row.get(c, 0.0) for c in classes}
            prior[i] = freq.get(labels[i], 0) / n
    return rows, prior


def crossfit(texts, labels, seed=MODEL_SEED, n_splits=5):
    """返回 (每份给自己标签的折外概率, 每份给自己标签的先验频率)。"模型比先验多出来的部分"就是正文里真有年龄（性别）信息的证据。"""
    rows, prior = crossfit_rows(texts, labels, seed, n_splits)
    return [r[l] for r, l in zip(rows, labels)], prior


# ---------- 二选一：正文更像改写前还是改写后 ----------

def proba_rows(model, texts):
    """每份正文的 {类别: 概率}。"""
    classes = [str(c) for c in model.classes_]
    return [dict(zip(classes, map(float, row))) for row in model.predict_proba(list(texts))]


def qbar_by_group(rows, groups):
    """rows：每份的 {类别: 概率}；groups：每份的病种（三位码）。返回 {病种: {类别: 平均概率}}。"""
    acc = {}
    for row, g in zip(rows, groups):
        a = acc.setdefault(g, {"n": 0, "sum": {}})
        a["n"] += 1
        for k, v in row.items():
            a["sum"][k] = a["sum"].get(k, 0.0) + v
    return {g: {k: v / a["n"] for k, v in a["sum"].items()} for g, a in acc.items()}


def prefers_old(row, old, new, qbar):
    """row：这段正文的 {类别: 概率}；qbar：同病种真实病历的 {类别: 平均概率}。
    返回 1（更像改写前 old）、0（更像改写后 new）、0.5（一样）；old == new、类别不在模型里、或缺同病种平均时返回 None。"""
    if old == new or old not in row or new not in row or not qbar or old not in qbar or new not in qbar:
        return None
    if qbar[old] <= 0 or qbar[new] <= 0:
        return None
    a, b = row[old] / qbar[old], row[new] / qbar[new]
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def _auc(pos, neg):
    """Mann–Whitney 形式的 AUC：pos 高于 neg 的比例，相等记一半。"""
    if not pos or not neg:
        return None
    gt = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return gt / (len(pos) * len(neg))


def self_auc(rows, own, codes, qbar, min_n=10):
    """没有改写过的病历上，模型读不读得出年龄（或性别）：对每一个类别 b，把"自己就是 b"的病历和"不是 b"的病历
    按 q(b|正文) ÷ 同病种平均 q(b) 排序，算 AUC；两边都至少 min_n 份的类别才算，再等权平均。
    合成病历上这个数接近 0.5，说明模型在合成文本上读不出年龄，H5-1 的二选一就没有意义（任务书写为"未判"）。
    返回 {"per_class": {b: (AUC, 正例数, 反例数)}, "macro": 平均 AUC 或 None}。"""
    per = {}
    classes = sorted({c for r in rows for c in r})
    for b in classes:
        pos, neg = [], []
        for r, o, g in zip(rows, own, codes):
            q = (qbar or {}).get(g) or {}
            if o is None or b not in r or not q.get(b):
                continue
            (pos if o == b else neg).append(r[b] / q[b])
        if len(pos) >= min_n and len(neg) >= min_n:
            per[b] = (_auc(pos, neg), len(pos), len(neg))
    vals = [v[0] for v in per.values()]
    return {"per_class": per, "macro": (sum(vals) / len(vals)) if vals else None}
