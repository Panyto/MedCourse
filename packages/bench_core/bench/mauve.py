"""MAUVE（Pillutla et al., NeurIPS 2021）的标准算法：联合 k-means 量化后算散度前沿下的面积，带 Krichevsky–Trofimov 平滑。

自己写一份是为了不依赖 mauve-text 的接口；装得上官方包时可以再交叉核对。特征两种：
- 默认字符 n-gram TF-IDF + SVD（只要 scikit-learn，确定性）；
- --embed-model 路径：本机 sentence-transformers 模型（如 bge-small-zh-v1.5），不联网、不送外部服务。

python -B -m bench.mauve --rows rows.json --out 输出目录 [--embed-model 路径] [--clusters-per 10]
结果写 raw/all.mauve.row.jsonl，记分牌自动读。
"""
import argparse
import json
import math
import os
import sys

from . import canon, paths
from .safe import guarded, log

SEED = 20261101
GRID = 1000


def frontier_area(p, q):
    """p、q 是同一批簇上的概率。前沿下面积 AUC 在 [1/2, 1]（不相交 1/2，同分布 1），MAUVE = 2·AUC − 1，落在 [0, 1]。"""
    xs, ys = [], []
    for i in range(1, GRID):
        lam = i / GRID
        r = [lam * a + (1 - lam) * b for a, b in zip(p, q)]
        kl_qr = sum(b * math.log(b / c) for b, c in zip(q, r) if b > 0)
        kl_pr = sum(a * math.log(a / c) for a, c in zip(p, r) if a > 0)
        xs.append(math.exp(-kl_qr))
        ys.append(math.exp(-kl_pr))
    pts = sorted(zip(xs, ys))
    pts = [(0.0, 1.0)] + pts + [(1.0, 0.0)]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return 2 * area - 1


def kt_hist(labels, k):
    counts = [0] * k
    for c in labels:
        counts[c] += 1
    n = len(labels)
    return [(c + 0.5) / (n + 0.5 * k) for c in counts]


def mauve_star(labels_p, labels_q, k):
    return frontier_area(kt_hist(labels_p, k), kt_hist(labels_q, k))


def features(texts, embed_model=None):
    import numpy as np
    if embed_model:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(embed_model, device="cuda" if os.environ.get("BENCH_EMBED_CUDA") else "cpu")
        return np.asarray(m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False))
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    x = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, max_features=200000).fit_transform(texts)
    n = min(100, min(x.shape) - 1)
    return TruncatedSVD(n_components=n, random_state=SEED).fit_transform(x)


def compare(p_texts, q_texts, embed_model=None, per=10):
    """k = (两边总份数)/per。k-means 用同一个种子，只跑一次。"""
    from sklearn.cluster import KMeans
    feats = features(p_texts + q_texts, embed_model)
    k = max(2, (len(p_texts) + len(q_texts)) // per)
    lab = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(feats)
    return mauve_star(list(lab[:len(p_texts)]), list(lab[len(p_texts):]), k), k


def narrative(lines):
    """四段叙述（主诉、现病史、住院经过、出院医嘱），规范化同判别器。投影失败的住院不计。"""
    from q45_kit import c2st
    from q45_kit.realtext import care7_from_admission
    from ruler_g1 import extract_facts, split_sections
    out = []
    for x in lines:
        rec = x["record"] if x["kind"] == "care7" else care7_from_admission(
            {"admission_id": x["id"], "docs": x["docs"]}, split_sections, extract_facts)[0]
        if rec is not None:
            out.append(c2st.care7_text(rec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-model")
    ap.add_argument("--clusters-per", type=int, default=10)
    a = ap.parse_args()
    for name in ("g1", "g45"):
        sys.path.insert(0, paths.core_path(name))
    with open(a.rows, encoding="utf-8") as f:
        cfg = json.load(f)
    specs = {s["row"]: s for s in cfg["rows"]}
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    out = []
    for row, ref in sorted((cfg.get("pairs") or {}).items()):
        if row not in specs or ref not in specs:
            continue
        p = narrative(canon.load_row(specs[ref]))
        q = narrative(canon.load_row(specs[row]))
        v, k = compare(p, q, a.embed_model, a.clusters_per)
        out.append({"row": row, "metric": "mauve", "value": v, "n": len(q), "clusters": k})
        log("mauve", row=row, value=round(v, 4), n=len(q), clusters=k)
    with open(os.path.join(a.out, "raw", "all.mauve.row.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    guarded(main)
