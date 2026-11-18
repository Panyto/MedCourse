# -*- coding: utf-8 -*-
"""把公开文本（临床路径、抗菌药物指导原则）切成检索用的片段。只切，不抽任何字段。

按行累积，遇到标题行（第X部分、一、（一））另起一段；一段超过 max_chars 就在行边界断开。
每段带最近的标题和文档名，检索结果给模型看时用。目录里带点线的行（"……… 12"）丢掉。
"""
import re

from .sections import kind_of

HEAD_L1 = re.compile(r"^\s*(第[一二三四五六七八九十]+部分|附录\s*[0-9一二三四五六七八九十]+)\s*\S")
HEAD_L2 = re.compile(r"^\s*([一二三四五六七八九十]+、)\s*\S")
HEAD_L3 = re.compile(r"^\s*([（(][一二三四五六七八九十]{1,3}[）)])\s*\S")
TOC_RE = re.compile(r"[.．…·]{6,}\s*\d*\s*$")
FORM_RE = re.compile(r"临床路径表单")


def make_chunks(doc_id, text, max_chars=500):
    chunks, cur, h1, h2, h3, leaf, in_form = [], [], "", "", "", "", False

    def flush():
        body = "\n".join(cur).strip()
        if body:
            full = leaf if in_form else " / ".join(x for x in (h1, h2, h3) if x)
            chunks.append({"id": f"{doc_id}#{len(chunks)}", "doc": doc_id, "title": full[:60],
                           "kind": "form" if in_form else kind_of(leaf), "text": body})
        cur.clear()

    for line in text.splitlines():
        s = line.strip()
        if not s or TOC_RE.search(s):
            continue
        if FORM_RE.search(s):
            flush()
            in_form, leaf = True, s
        elif HEAD_L1.match(s):
            flush()
            h1, h2, h3, leaf = s, "", "", s
        elif HEAD_L2.match(s):
            flush()
            h2, h3, leaf = s, "", s
        elif HEAD_L3.match(s):
            flush()
            h3, leaf = s, s
        if cur and sum(len(x) for x in cur) + len(s) > max_chars:
            flush()
        cur.append(s)
    flush()
    return chunks
