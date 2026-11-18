"""知识覆盖 K-COV（LLM-CARe 论文 4 节、附录 A.5 的口径）：参考集里每个诊断的实体，有多少比例在同诊断的
合成病历里出现，**字符串精确匹配**（论文原话："we use exact string-level matching"）。

论文没写是"整批合成病历里出现过就算"还是"逐份算再平均"，也没写加不加权。这里：
- 主口径 `kcov(pool, vocab)`：参考词表里的（类别, 写法），在 pool 任何一份病历的四段叙述里作为子串出现过，就算覆盖；
  覆盖条数 ÷ 词表条数（不加权）。五类合起来是 "all"，每类另给；
- `weighted=True`：每条按它在参考集里出现的病历份数加权（常见词分量重）；
- `kcov_per_record`：逐份算再平均（另报）。
K-COV 依赖池的份数和长短，所以只在**份数相同**的池之间比（任务书写明）。
"""
from .entities import CATS, FIELDS4, field_text, surface_set


def ref_vocab(ref_records, ms_list=None):
    """参考集 → {(类别, 写法): 出现的病历份数}（只收没被否定的，同 entities.ref_entity）。"""
    vocab = {}
    for i, r in enumerate(ref_records):
        ss = surface_set(r, ms_list[i] if ms_list else None, affirmed_only=True)
        for item in ss:
            vocab[item] = vocab.get(item, 0) + 1
    return vocab


def pool_text(records):
    return ["\n".join(field_text(r, f) for f in FIELDS4) for r in records]


def covered_items(texts, vocab):
    """pool（已经连成文字）里出现过的词表条目集合。"""
    out = set()
    for item in vocab:
        w = item[1]
        for t in texts:
            if w in t:
                out.add(item)
                break
    return out


def kcov(records, vocab, weighted=False, texts=None):
    """返回 {"all": 比例, 类别: 比例, "n_items": {…}}。词表某一类为空时该类记 None。"""
    texts = texts if texts is not None else pool_text(records)
    cov = covered_items(texts, vocab)
    out, n_items = {}, {}
    for c in ("all",) + CATS:
        items = [it for it in vocab if c == "all" or it[0] == c]
        n_items[c] = len(items)
        if not items:
            out[c] = None
            continue
        if weighted:
            den = sum(vocab[it] for it in items)
            out[c] = sum(vocab[it] for it in items if it in cov) / den
        else:
            out[c] = sum(1 for it in items if it in cov) / len(items)
    out["n_items"] = n_items
    return out


def kcov_per_record(records, vocab, texts=None):
    """另一种读法（论文没说清是哪种）：逐份算"词表里有多少比例出现在这一份里"，再对份数平均。返回 {"all": …, 类别: …}。"""
    texts = texts if texts is not None else pool_text(records)
    out = {}
    for c in ("all",) + CATS:
        items = [it for it in vocab if c == "all" or it[0] == c]
        if not items or not texts:
            out[c] = None
            continue
        out[c] = sum(sum(1 for it in items if it[1] in t) / len(items) for t in texts) / len(texts)
    return out


def gain_attribution(before_texts, after_texts, vocab, requested):
    """语料层前后新覆盖的条目里，有多少是 Critic 点名要加的（requested：{写法}，这一批病历所有轮次的 add 词并集）。
    返回 (新覆盖条数, 其中被点名的条数)。"""
    b = covered_items(before_texts, vocab)
    a = covered_items(after_texts, vocab)
    new = a - b
    return len(new), sum(1 for it in new if it[1] in requested)
