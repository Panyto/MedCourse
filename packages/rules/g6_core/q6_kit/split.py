"""关6 的抽样和分组，全部事先定死（字符串种子，与哈希种子无关；输入先排序，结果与输入顺序无关）。

- `split_hold_ref`：头部诊断的真实病人分成留出组 HOLD（15 人，H6-1 当"真人对照"）和参考组 REF（其余，配额和 K-COV 的参考）；
- `halves`：同一诊断的合成病历分成两半（每半 15 份，和 HOLD 份数相同）；
- `pick_tail`：长尾诊断。门槛 10 份 = LLM-CARe 自己的剔除线按比例换到我们的数据上：它在 71 万份里剔除不足 500 份的诊断，
  500 / 710,000 × 15,192 ≈ 10.7；
- `tail_alias`：长尾诊断在一切进包的表里只用代号 TAIL01…（诊断本身不足 20 份，按守则 1.2 不出原名）。
"""
import random

SPLIT_SEED = "20261028|g6split"
HALF_SEED = "20261028|g6half"
TAIL_SEED = "20261028|g6tail"
N_HOLD = 15
MIN_SPLIT = 30
TAIL_MAX_ALL = 10
TAIL_K = 40
TAIL_N = 10          # 每个长尾诊断生成几份


def split_hold_ref(ids_by_code, n_hold=N_HOLD, min_n=MIN_SPLIT, seed=SPLIT_SEED):
    """ids_by_code：{三位码: [住院号]}（fold 0、7 字段截得出来的）。返回 {三位码: (HOLD 列表, REF 列表)}，两个列表都排好序。
    份数不足 min_n 的诊断不分：HOLD 为空，REF 是全部。"""
    out = {}
    for code in sorted(ids_by_code):
        ids = sorted(set(ids_by_code[code]))
        if len(ids) < min_n:
            out[code] = ([], ids)
            continue
        rng = random.Random(f"{seed}|{code}")
        hold = sorted(rng.sample(ids, n_hold))
        hs = set(hold)
        out[code] = (hold, [i for i in ids if i not in hs])
    return out


def halves(slot_ids, code, seed=HALF_SEED):
    """同一诊断的槽位分成两半（先排序再洗）。奇数份时第一半多一份。"""
    ids = sorted(slot_ids)
    rng = random.Random(f"{seed}|{code}")
    rng.shuffle(ids)
    k = (len(ids) + 1) // 2
    return sorted(ids[:k]), sorted(ids[k:])


def pick_tail(all_counts, fold0_ok_counts, max_all=TAIL_MAX_ALL, k=TAIL_K, seed=TAIL_SEED, exclude=()):
    """all_counts：{三位码: 全部 fold 的住院数}（D0 的边际统计，只有编码）；fold0_ok_counts：{三位码: fold 0 里 7 字段截得出来的住院数}。
    入选：全部 fold ≤ max_all 份，且 fold 0 截得出来的 ≥ 1 份，且不在 exclude（关1 的 10 个头部诊断）。多于 k 个时随机取 k 个。"""
    elig = sorted(c for c, n in all_counts.items()
                  if n <= max_all and fold0_ok_counts.get(c, 0) >= 1 and c not in set(exclude))
    if len(elig) <= k:
        return elig
    rng = random.Random(seed)
    return sorted(rng.sample(elig, k))


def tail_alias(codes):
    return {c: f"TAIL{i + 1:02d}" for i, c in enumerate(sorted(codes))}


def tail_seed(alias, j, attempt=0):
    """长尾生成第 j 份的种子（与关1 生成 CARe-S1 的种子无关）。"""
    import hashlib
    return int(hashlib.sha256(f"Q6|gen|{alias}|{j}|{attempt}".encode()).hexdigest()[:8], 16)
