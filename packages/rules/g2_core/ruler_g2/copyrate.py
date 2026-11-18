"""篇间复制率：字符 5-gram 覆盖率、最长公共子串（后缀自动机）；跨病人相似度。"""
import re

from .text import char_ngrams


def _clean(s):
    return re.sub(r"\s+", "", s)


def containment(b, a, n=5):
    """b 的 5-gram 有多少比例出现在 a 里。b 太短（不足 n 字）返回 0。"""
    gb = char_ngrams(b, n)
    if not gb:
        return 0.0
    return len(gb & char_ngrams(a, n)) / len(gb)


def jaccard(a, b, n=5):
    ga, gb = char_ngrams(a, n), char_ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def longest_common_substring(a, b):
    """后缀自动机，O(len(a)+len(b))。"""
    a, b = _clean(a), _clean(b)
    if not a or not b:
        return 0
    link, length, nxt = [-1], [0], [{}]
    last = 0
    for ch in a:
        cur = len(length)
        length.append(length[last] + 1)
        link.append(-1)
        nxt.append({})
        p = last
        while p != -1 and ch not in nxt[p]:
            nxt[p][ch] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = nxt[p][ch]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(length)
                length.append(length[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                while p != -1 and nxt[p].get(ch) == q:
                    nxt[p][ch] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
        last = cur
    best = cur_len = 0
    v = 0
    for ch in b:
        while v and ch not in nxt[v]:
            v = link[v]
            cur_len = length[v]
        if ch in nxt[v]:
            v = nxt[v][ch]
            cur_len += 1
        best = max(best, cur_len)
    return best


def lcs_ratio(a, b):
    """后篇 b 与前篇 a 的最长公共子串占 b 长度的比例。"""
    nb = len(_clean(b))
    return longest_common_substring(a, b) / nb if nb else 0.0


def cross_patient_similarity(pool, n=5):
    """pool：[(admission_id, group_key, text)]，text 是这次住院全部病程拼起来的正文。
    返回 {admission_id: 与同组其他住院的最大 5-gram Jaccard}。group_key 通常是科室。"""
    grams = [(aid, g, char_ngrams(t, n)) for aid, g, t in pool]
    out = {}
    for i, (aid, g, ga) in enumerate(grams):
        best = 0.0
        for j, (bid, h, gb) in enumerate(grams):
            if i == j or g != h or not ga or not gb:
                continue
            best = max(best, len(ga & gb) / len(ga | gb))
        out[aid] = best
    return out
