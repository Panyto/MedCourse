# -*- coding: utf-8 -*-
"""检索：中文按字的一元 + 二元切词，BM25 打分。纯 Python，不依赖向量库，结果确定。
只做"按问题找段落"，不判断段落里有没有答案——那是 8B 的事。"""
import math
import re
from collections import Counter

_KEEP = re.compile(r"[一-鿿A-Za-z0-9]+")


def terms(s):
    out = []
    for run in _KEEP.findall(s or ""):
        if run.isascii():
            out.append(run.lower())
            continue
        out.extend(run)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.tf = [Counter(terms(c["title"] + " " + c["text"])) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(w for t in self.tf for w in t)
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def search(self, query, k=3, docs=None, exclude=(), drop="", form_weight=1.0, exclude_kinds=(), exclude_title=None):
        """drop：要从检索词里去掉的字词（本病种名称）。RB2 第一版 8B 每次都把长病名放进检索词，得分被病名吸走，
        反复印病名的表单段排在前面，"（二）诊断依据"排不上（老板 11-11 查 K80.0 的 trace）。去掉以后检索词空了就用原词。
        form_weight：表单段的得分乘这个数（<1 让正文段优先）。
        exclude_kinds：这些段落类型不检（0.5：症状、体征只看正文段，排除 form）。
        exclude_title：标题匹配这个正则的段落不检（0.5：不做手术的病种问抗菌药时，排除指导原则里围手术期/介入预防用药章节）。"""
        q = Counter(terms(query))
        if drop:
            dropped = set(terms(drop))
            kept = Counter({w: n for w, n in q.items() if w not in dropped})
            q = kept or q
        scored = []
        for i, c in enumerate(self.chunks):
            if (docs is not None and c["doc"] not in docs) or c["id"] in exclude or c.get("kind") in exclude_kinds:
                continue
            if exclude_title is not None and exclude_title.search(c.get("title") or ""):
                continue
            tf, s = self.tf[i], 0.0
            for w, qn in q.items():
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if c.get("kind") == "form":
                s *= form_weight
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.chunks[i] for _, i in scored[:k]]
