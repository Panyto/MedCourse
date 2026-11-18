"""进包前的打码。

语料层之后的合成病历（S6、T6、F-QOK）里有从真实病历抄来的词，Critic 的反馈句里也有。进包的这类文字一律过这里：
- 一个词在 fold 0 真实病历里出现的份数 < 20、或长于 10 个字、或像地名机构名（守则 1.2），换成【词:类别:n字】；
- 长尾诊断的病名（及其三位码类目名）在每一段里都换成【病名:n字】，代号见 split.tail_alias。
份数够、够短的常见词（"发热""血常规""阿司匹林"）照原样留，总经理要读它们插在哪里、插得对不对。
`leak_count` 是打码后的核对：受保护的词还剩几处，必须是 0。
"""
import re

from .entities import CAT_CN

MIN_DF = 20
MAX_LEN = 10
_PLACE = re.compile(r"医院|卫生院|诊所|中心|门诊|保健院|省|市|县|区|镇|乡|村|街道|社区")
TEXT_KEYS = ("Primary Diagnosis", "Chief Complaint", "History of Present Illness", "Hospital Course",
             "Discharge Instructions")


def protected(words, df, min_df=MIN_DF, max_len=MAX_LEN):
    """words：{写法: 类别}；df：{写法: fold 0 份数}。返回要打码的 {写法: 类别}。"""
    return {w: c for w, c in words.items() if df.get(w, 0) < min_df or len(w) > max_len or _PLACE.search(w)}


def _sub_all(text, table):
    """table：{原词: 替换}，长的先换；换过的地方不再参与后面的匹配。"""
    if not text or not table:
        return text
    rx = re.compile("|".join(re.escape(w) for w in sorted(table, key=lambda x: (-len(x), x))))
    return rx.sub(lambda m: table[m.group(0)], text)


def word_table(prot):
    return {w: f"【词:{CAT_CN.get(c, c)}:{len(w)}字】" for w, c in prot.items()}


def dx_table(names):
    return {n: f"【病名:{len(n)}字】" for n in names if n}


def mask_text(text, prot=None, dx_names=()):
    table = {}
    table.update(dx_table(dx_names))
    if prot:
        table.update(word_table(prot))
    return _sub_all(text, table)


def mask_record(record, prot=None, dx_names=()):
    out = dict(record)
    for k in TEXT_KEYS:
        if k in out and isinstance(out[k], str):
            out[k] = mask_text(out[k], prot, dx_names)
    return out


def mask_feedback(feedback, prot=None, dx_names=()):
    """{段: [反馈句]} → 同样打码。"""
    return {f: [mask_text(s, prot, dx_names) for s in ss] for f, ss in feedback.items()}


def leak_count(text, prot=None, dx_names=()):
    words = list(prot or {}) + [n for n in dx_names if n]
    return sum(text.count(w) for w in words) if text else 0
