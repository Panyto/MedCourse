"""关6 的多样性读数（H6-3 "同病同貌"）。

关卡谱关6 的度量是"分层召回、语义 Vendi、Self-BLEU"。这里全部在**同一个诊断（三位码）之内**算：
同一个病的 30 个合成病人彼此像不像，和同一个病的 30 个真实病人彼此像不像比。

- 内容：两两 Jaccard（每份病历没被否定的实体规范键集合，`entities.key_set`）——H6-3 主读数；
- 写法：Self-BLEU（字符级 BLEU-4，每份对同组其余各份）；
- 整体：Vendi（字符 1–3 元 TF-IDF 的余弦核；Friedman & Dieng 2023），除以份数，在 1/n 到 1 之间；
- 人口学和单项：不同年龄个数、年龄熵、众数占比、最高体温众数占比等（`profile`）。

长度会影响上面每一个读数（写得长的实体多、n 元组多），所以真实病历和合成病历先按槽位配对、逐段截成一样长
（`truncate_pair`，与关5 Q45b 的等长做法同一个意思，只是这里保留原文写法，好抽实体）。
"""
import math
import random
import re
import unicodedata
from collections import Counter

from ruler_g3.adapters import care_get

from .entities import CARE7_KEY, FIELDS4

_PLACEHOLDER = re.compile(r"\[[^\[\]\n]{0,20}\]")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s\W_]+", re.U)
_DIGIT = re.compile(r"\d")
_TEMP = re.compile(r"(\d{2}(?:\.\d)?)\s*(?:℃|°C|度)")


def clean(text):
    """去掉方括号占位符和空白（真实病历有去标识化占位符，合成病历没有；不去掉的话截等长时真实一侧吃亏）。"""
    return _WS.sub("", _PLACEHOLDER.sub("", text or ""))


def truncate_pair(rec_a, rec_b, fields=FIELDS4):
    """两份 7 字段病历，四段叙述各自 clean 后截到两份里较短那份的长度（从段首截）。
    返回 (a′, b′)：只有四段叙述和性别、年龄、主要诊断（后三项原样）。"""
    out_a, out_b = {}, {}
    for r, o in ((rec_a, out_a), (rec_b, out_b)):
        for k in ("Gender", "Age", "Primary Diagnosis"):
            o[k] = care_get(r, {"Gender": "sex", "Age": "age", "Primary Diagnosis": "dx"}[k])
    for f in fields:
        a, b = clean(care_get(rec_a, f)), clean(care_get(rec_b, f))
        n = min(len(a), len(b))
        out_a[CARE7_KEY[f]] = a[:n]
        out_b[CARE7_KEY[f]] = b[:n]
    return out_a, out_b


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise(items, sim):
    """items：[(来源编号, 值)]。来源编号相同的两项（bootstrap 抽到同一份两次）不配对。"""
    tot, n = 0.0, 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][0] == items[j][0]:
                continue
            tot += sim(items[i][1], items[j][1])
            n += 1
    return tot / n if n else float("nan")


def norm_text(text):
    s = unicodedata.normalize("NFKC", text or "")
    s = _PUNCT.sub("", _PLACEHOLDER.sub("", s))
    return _DIGIT.sub("0", s).lower()


def record_text(record, fields=FIELDS4):
    return "".join(norm_text(care_get(record, f)) for f in fields)


def _ngrams(s, n):
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def bleu(hyp, refs, max_n=4):
    """字符级 BLEU-4（均匀权重，n>1 时分子分母各加 1 平滑，Lin & Och 2004；短句惩罚用最接近的参考长度）。"""
    if not hyp or not refs:
        return 0.0
    logs = 0.0
    for n in range(1, max_n + 1):
        h = _ngrams(hyp, n)
        if not h:
            return 0.0
        mx = Counter()
        for r in refs:
            for g, c in _ngrams(r, n).items():
                if c > mx[g]:
                    mx[g] = c
        clip = sum(min(c, mx[g]) for g, c in h.items())
        tot = sum(h.values())
        p = (clip + (1 if n > 1 else 0)) / (tot + (1 if n > 1 else 0))
        if p <= 0:
            return 0.0
        logs += math.log(p) / max_n
    rl = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
    bp = 1.0 if len(hyp) > rl else math.exp(1 - rl / len(hyp))
    return bp * math.exp(logs)


def self_bleu(texts):
    """每份对同组其余各份的 BLEU 的平均（越高越像）。texts：[(来源编号, 规范化文字)]。"""
    vals = []
    for i, (gi, t) in enumerate(texts):
        refs = [u for j, (gj, u) in enumerate(texts) if j != i and gj != gi]
        if refs:
            vals.append(bleu(t, refs))
    return sum(vals) / len(vals) if vals else float("nan")


def vendi(texts, fit_texts=None):
    """Vendi 分数 ÷ 份数。texts：规范化文字列表；fit_texts：拟合 TF-IDF 的文字（同一诊断里要比的各行合起来，
    让两行在同一个特征空间里）。返回 (vendi/n, vendi)。"""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, norm="l2", min_df=1)
    vec.fit(fit_texts if fit_texts is not None else texts)
    x = vec.transform(texts)
    k = (x @ x.T).toarray()
    n = k.shape[0]
    w = np.linalg.eigvalsh(k / n)
    w = w[w > 1e-12]
    v = float(math.exp(-(w * np.log(w)).sum()))
    return v / n, v


def profile(values):
    """一组值（None 不算）：不同值个数、香农熵（比特）、众数占比、众数。"""
    vs = [v for v in values if v is not None]
    if not vs:
        return {"n": 0, "distinct": 0, "entropy": float("nan"), "modal_share": float("nan"), "mode": None}
    c = Counter(vs)
    n = len(vs)
    ent = -sum((k / n) * math.log2(k / n) for k in c.values())
    mode, top = sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]
    return {"n": n, "distinct": len(c), "entropy": ent, "modal_share": top / n, "mode": mode}


def max_temp(record):
    """主诉和现病史里写到的最高体温（35–43℃ 之间的读数），没写返回 None。"""
    vals = []
    for f in ("cc", "hpi"):
        for m in _TEMP.finditer(care_get(record, f)):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if 35.0 <= v <= 43.0:
                vals.append(v)
    return max(vals) if vals else None


def strat_boot(groups, stat, B=1000, seed=20261014):
    """分层 bootstrap：groups = {诊断: [槽位]}，每个诊断内有放回重抽槽位（份数不变），stat({诊断: [槽位]}) → 数。
    返回 (点估计, 下限, 上限)。"""
    est = stat({g: list(v) for g, v in groups.items()})
    rng = random.Random(seed)
    boots = []
    keys = sorted(groups)
    for _ in range(B):
        s = {g: [rng.choice(groups[g]) for _ in groups[g]] for g in keys}
        boots.append(stat(s))
    boots = sorted(b for b in boots if not math.isnan(b))
    if not boots:
        return est, float("nan"), float("nan")

    def pct(q):
        i = (len(boots) - 1) * q
        a, b = int(math.floor(i)), int(math.ceil(i))
        return boots[a] + (boots[b] - boots[a]) * (i - a)
    return est, pct(0.025), pct(0.975)


def within_code_jaccard(sets_by_slot, slots_by_code):
    """每个诊断内两两 Jaccard 的平均，再对诊断等权平均。sets_by_slot：{槽位: 集合}。"""
    vals = []
    for code in sorted(slots_by_code):
        items = [(s, sets_by_slot[s]) for s in slots_by_code[code] if s in sets_by_slot]
        v = mean_pairwise(items, jaccard)
        if not math.isnan(v):
            vals.append(v)
    return sum(vals) / len(vals) if vals else float("nan")
