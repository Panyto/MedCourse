"""相邻病程 tf-idf 余弦（g2-core 0.4 新增）。

照 G2_pkg1 伙计实现的口径原样写进核心代码（pkg1 表 T15），G-E′ 的主要目标就用它：
- 相邻篇：首次病程之后、出院记录之前的病程记录，按顺序两两相邻；
- 字符 2–4 gram，原始词频；平滑 IDF = log((1+N)/(1+df)) + 1，词表和 IDF 只由真实行 R 的这些病程拟合；
- 不在 R 词表里的 gram 丢掉；L2 归一化；每次住院取相邻篇余弦的均值，没有相邻篇的住院不算。
"""
import math
from collections import Counter


def progress_between(adm):
    """一次住院里首次病程之后、出院记录之前的病程正文列表（按顺序）。"""
    fp, ds = adm.fp, adm.ds
    return [d.text for d in adm.docs if d.doc_type == "progress_note"
            and (fp is None or d.pos > fp.pos) and (ds is None or d.pos < ds.pos)]


def _grams(s):
    return Counter(s[i:i + n] for n in (2, 3, 4) for i in range(max(0, len(s) - n + 1)))


class RefTfidf:
    def __init__(self, reference_texts):
        self.df = Counter()
        for t in reference_texts:
            self.df.update(_grams(t).keys())
        self.total = len(reference_texts)

    def vector(self, text):
        v = {k: n * (math.log((1 + self.total) / (1 + self.df[k])) + 1) for k, n in _grams(text).items() if k in self.df}
        norm = math.sqrt(sum(x * x for x in v.values()))
        return {k: x / norm for k, x in v.items()} if norm else {}


def cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0) for k, v in a.items())


def adjacent_tfidf(model, texts):
    """一次住院的相邻篇余弦均值；不足两篇返回 None。"""
    if len(texts) < 2:
        return None
    vs = [model.vector(t) for t in texts]
    vals = [cosine(a, b) for a, b in zip(vs, vs[1:])]
    return sum(vals) / len(vals)
