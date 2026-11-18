"""关5 判别器：两样本分类检验（C2ST，Lopez-Paz & Oquab 2017）。

把同一批槽位上的合成病历和真实病历混在一起，训练一个分类器分真假，看它在没见过的槽位上能分多准（留出 AUC）。
AUC 接近 0.5 = 分不出来；接近 1 = 一眼认得出。

固定设置（PREREG_Q45 第 5 节，不调参）：
- 文本先规范化：NFKC、去掉去标识化占位符 [..]、数字一律变 0、去空白、去表格分隔符"|"、小写。这样分类器不能靠"有没有占位符"
  "日期长什么样""有没有表格线"分真假；
- 特征：字符 1–3 元组 TF-IDF（sublinear_tf，训练折里 min_df=2，最多 50,000 个）；
- 分类器：L2 逻辑回归，C=1.0，liblinear；
- 交叉验证：按槽位分组的分层 5 折（同一槽位的真、假两份永远在同一折），5 个种子各跑一遍，主读数是 5 遍的平均 AUC；
  区间：第一遍的折外得分按槽位整簇 bootstrap 1,000 次（种子 20261014）；
- 对照：只看长度的分类器（每段字数取 log1p）；真实对真实（同一三位码里两两配对、随机标真假，20 次，见 null_split）；
  地板（空泛病历、填表病历）。
依赖 numpy、scikit-learn（版本写进 ENV.md）。
"""
import math
import random
import re
import unicodedata

import numpy as np
from ruler_g3.adapters import care_get
from ruler_g4.ruler import FIELDS, doc_role, nchars
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEEDS = (20261027, 20261028, 20261029, 20261030, 20261031)
N_SPLITS = 5
BOOT_B = 1000
BOOT_SEED = 20261014
NULL_REPS = 20
REAL_DF_MIN = 20

_PLACEHOLDER = re.compile(r"\[[^\[\]\n]{0,30}\]")
_DIGIT = re.compile(r"\d")
_WS = re.compile(r"\s+")


def normalize(text):
    s = unicodedata.normalize("NFKC", text or "")
    s = _PLACEHOLDER.sub("", s)
    s = _DIGIT.sub("0", s)
    s = _WS.sub("", s)
    s = s.replace("|", "")          # 导出文件里表格的分隔符（NFKC 后全角｜也变成|）：排版痕迹，不是内容
    return s.lower()


def care7_text(record, fields=FIELDS):
    """7 字段病历 → 判别器输入：四段叙述（主诉、现病史、住院经过、出院医嘱）各自规范化后用"｜"连起来。
    性别、年龄、主要诊断是给定属性，不进判别器（单段判别另算，见任务书）。"""
    return "｜".join(normalize(care_get(record, k)) for k in fields)


def care7_lengths(record, fields=FIELDS):
    return [nchars(care_get(record, k)) for k in fields]


ADM_ROLES = ("hr", "fp", "course", "discharge")


def admission_text(adm, roles=ADM_ROLES):
    """整次住院 → 判别器输入：指定类型的文书按原顺序规范化后用"｜"连起来。"""
    return "｜".join(normalize(d.get("text")) for d in (adm.get("docs") or []) if doc_role(d) in roles)


def admission_lengths(adm):
    """整次住院的长度特征：入院记录、首次病程、其余病程合计、出院记录的字数，和文书篇数。"""
    tot = {r: 0 for r in ADM_ROLES}
    docs = adm.get("docs") or []
    for d in docs:
        tot[doc_role(d)] += nchars(d.get("text"))
    return [tot[r] for r in ADM_ROLES] + [len(docs)]


def make_vectorizer():
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=50000,
                           lowercase=False, norm="l2", dtype=np.float64)


def make_classifier(seed):
    return LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=seed)


def _len_features(rows):
    return np.log1p(np.asarray(rows, dtype=np.float64))


def folds(labels, groups, seed, n_splits=N_SPLITS):
    """按组分层的 5 折：[(训练下标, 测试下标)]。同一组（槽位）的几份永远在同一折。"""
    y = np.asarray(labels, dtype=int)
    g = np.asarray([str(x) for x in groups])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, g))


def cv_scores(xs, labels, groups, seed, kind="text", n_splits=N_SPLITS):
    """xs：kind="text" 时是规范化后的字符串列表；kind="length" 时是每份的字数向量列表。
    返回折外得分（判为 1 的概率）。"""
    y = np.asarray(labels, dtype=int)
    oof = np.full(len(y), np.nan)
    for tr, te in folds(labels, groups, seed, n_splits):
        if kind == "text":
            vec = make_vectorizer()
            xtr = vec.fit_transform([xs[i] for i in tr])
            xte = vec.transform([xs[i] for i in te])
        else:
            arr = _len_features(xs)
            mu, sd = arr[tr].mean(axis=0), arr[tr].std(axis=0)
            sd[sd == 0] = 1.0
            xtr, xte = (arr[tr] - mu) / sd, (arr[te] - mu) / sd
        clf = make_classifier(seed)
        clf.fit(xtr, y[tr])
        oof[te] = clf.predict_proba(xte)[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("C2ST_OOF_INCOMPLETE")
    return oof


def auc(scores, labels):
    return float(roc_auc_score(np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)))


def cluster_auc_ci(scores, labels, groups, B=BOOT_B, seed=BOOT_SEED):
    """按槽位整簇重抽（同一槽位的真、假两份一起进出），返回 (下限, 上限)。只有一类的重抽样本跳过并计数。"""
    by = {}
    for s, y, g in zip(scores, labels, groups):
        by.setdefault(str(g), []).append((s, y))
    keys = sorted(by)
    rng = random.Random(seed)
    vals, skipped = [], 0
    for _ in range(B):
        rows = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        ys = [y for _, y in rows]
        if len(set(ys)) < 2:
            skipped += 1
            continue
        vals.append(auc([s for s, _ in rows], ys))
    vals.sort()
    if not vals:
        return float("nan"), float("nan"), skipped

    def pct(q):
        i = (len(vals) - 1) * q
        a, b = int(math.floor(i)), int(math.ceil(i))
        return vals[a] + (vals[b] - vals[a]) * (i - a)
    return pct(0.025), pct(0.975), skipped


def c2st(pos, neg, kind="text", seeds=SEEDS):
    """pos：[(槽位, x)] 合成（标签 1）；neg：[(槽位, x)] 真实（标签 0）。x 是规范化文本或字数向量。
    返回 {auc_mean, auc_runs, auc0, ci_lo, ci_hi, boot_skipped, n_pos, n_neg, n_groups, oof0}。"""
    xs = [x for _, x in pos] + [x for _, x in neg]
    ys = [1] * len(pos) + [0] * len(neg)
    gs = [g for g, _ in pos] + [g for g, _ in neg]
    runs, oof0 = [], None
    for i, sd in enumerate(seeds):
        oof = cv_scores(xs, ys, gs, sd, kind)
        runs.append(auc(oof, ys))
        if i == 0:
            oof0 = oof
    lo, hi, skipped = cluster_auc_ci(oof0, ys, gs)
    return {"auc_mean": float(np.mean(runs)), "auc_runs": runs, "auc0": runs[0], "ci_lo": lo, "ci_hi": hi,
            "boot_skipped": skipped, "n_pos": len(pos), "n_neg": len(neg), "n_groups": len(set(map(str, gs))),
            "oof0": [float(v) for v in oof0], "labels": ys, "groups": [str(g) for g in gs]}


def null_split(items, strata, reps=NULL_REPS, seed=BOOT_SEED, kind="text"):
    """真实对真实：items=[(槽位, x)]，strata={槽位: 三位码}。和正式检验同一个结构：
    每次在同一个三位码里把槽位随机两两配成对（单出来的那个不用），每对随机一个标 1、一个标 0，按"对"分组交叉验证
    （正式检验里是同一槽位的合成和真实一对）。两边来自同一个分布，AUC 应在 0.5 附近；离得远说明流程本身有漏洞。
    注意不要改成"整体随机分两半"：那样同一诊断的近似病历会被分到两边，留出 AUC 会系统性地低于 0.5。返回每次的 AUC。"""
    by = {}
    for g, x in items:
        by.setdefault(strata.get(g, ""), []).append((g, x))
    out = []
    for r in range(reps):
        rng = random.Random(f"{seed}|null|{r}")
        xs, ys, gs = [], [], []
        for code in sorted(by):
            rows = sorted(by[code], key=lambda t: str(t[0]))
            rng.shuffle(rows)
            for j in range(0, len(rows) - 1, 2):
                flip = rng.random() < 0.5
                for (g, x), lab in zip(rows[j:j + 2], (1, 0) if flip else (0, 1)):
                    xs.append(x)
                    ys.append(lab)
                    gs.append(f"{code}|{j}")
        out.append(auc(cv_scores(xs, ys, gs, SEEDS[0], kind), ys))
    return out


# ---------- 哪些字串在出卖它（只出满足守则 1.2 的短串） ----------
_INST = re.compile(r"[医院]|诊所|卫生|中心|门诊")
_PLACE = re.compile(r"[省市县区镇乡村]|街道")
_DATEISH = re.compile(r"0[年月日号]|[年月]0")
SYNTH_DF_MIN = 5


def feature_mask_reason(ngram, real_df, synth_df):
    """返回 None 表示可以显示；否则返回屏蔽原因。
    - 含占位符、含"医""院"、诊所、卫生、中心、门诊，含省市县区镇乡村、街道，或像日期的，一律不显示（守则 1.2）；
    - 只出现在合成病历里的（real_df=0）可以显示（合成文本，E1）；
    - 真实病历里出现在 1–19 份里的不显示（可能是个人内容）；
    - 真实病历里常见、但合成病历里不到 5 份的也不显示：只在真实一侧常见的字串可能是没去干净的机构名碎片，
      宁可不看，只报条数。"""
    if "[" in ngram or "]" in ngram:
        return "占位符"
    if _INST.search(ngram) or _PLACE.search(ngram):
        return "机构或地名字样"
    if _DATEISH.search(ngram):
        return "像日期"
    if real_df == 0:
        return None
    if real_df < REAL_DF_MIN:
        return "真实病历里不到 20 份"
    if synth_df < SYNTH_DF_MIN:
        return "只在真实病历里常见"
    return None


def top_features(pos_texts, neg_texts, k=30, seed=SEEDS[0]):
    """在全部数据上拟合一次，取最偏向合成（系数最大）和最偏向真实（系数最小）的各 k 个字串，
    附上它在真实、合成两边各出现在几份里，以及能不能显示。"""
    vec = make_vectorizer()
    x = vec.fit_transform(list(pos_texts) + list(neg_texts))
    y = np.asarray([1] * len(pos_texts) + [0] * len(neg_texts))
    clf = make_classifier(seed)
    clf.fit(x, y)
    names = vec.get_feature_names_out()
    coef = clf.coef_[0]
    xb = (x > 0).astype(np.int64)
    pos_df = np.asarray(xb[:len(pos_texts)].sum(axis=0)).ravel()
    neg_df = np.asarray(xb[len(pos_texts):].sum(axis=0)).ravel()
    order = np.argsort(coef)
    picks = [("synthetic", i) for i in order[::-1][:k]] + [("real", i) for i in order[:k]]
    out = []
    for side, i in picks:
        reason = feature_mask_reason(names[i], int(neg_df[i]), int(pos_df[i]))
        out.append({"side": side, "ngram": names[i] if reason is None else None, "coef": float(coef[i]),
                    "synth_df": int(pos_df[i]), "real_df": int(neg_df[i]), "masked": reason})
    return out
