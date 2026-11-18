"""合成病历抽样带回本地（守则 v1.6 P5，2026-11-05 老板批准）。只读流程产物里的文书正文，不读计划；真实语料只用来查重，只出计数。

python -B -m medcourse.synth_export --run 目录 --source 真实fold0.jsonl --label W8B_armA --out export/SYNTH_W8B_armA_20261105
python -B -m medcourse.synth_export --run 模板目录 --source 真实fold0.jsonl --label T_ctrl --out export/CHECK_T_ctrl --check-only

能带回的只有 S1 自由模式的合成文书：统计表只留 k≥11 的格子，写手是没用真实病历训练过的模型（W-8B 对话版、模板）。下面这些不出 ys：
配对模式的产物（槽位来自真实住院）；journeys.jsonl、drafts.jsonl（逐住院的计划，守则 v1.5 第二节）；统计表、S2 抽取产物、真实原文；
用 fold 0 LoRA 写的文本（要那一轮任务书逐条授权）。

抽之前查两样，命中的那份换下一份：
1. 罕见片段：合成文书的同一行里任何连续 GRAM 个字（去掉空白、数字一律换成 0），在真实 fold 0 里（同样去空白、数字换 0）
   出现在 1–(k-1) 次住院里，就算命中。出现在 0 次（自己写的）或至少 k 次住院（常见写法）不算。只数次数，不出片段。
2. 号码样式：手机号、身份证号、9 位以上连续数字、"住院号""病案号""床号""电话"后面跟数字。

0.1.7（S1S2 第六次：三个包都是 0/20）：0.1.6 用 20 字、不换数字、整篇连起来比，每包 7.5 万个片段里约 170 个落进"1–10 次"，
每份住院平均 8 个，20 份全被换掉。20 个字在中文病历里太短：程序写的诊断清单跨行拼出"两个诊断连在一起"、化验数值碰巧相同、
k≥11 的写法在原文里写法略有不同，都会落进去。现在数字换 0、片段不跨行、判定用 30 字；20 字、30 字两种长度都按位置分类计数
（程序写的行、写手写的正文；原文带不带数字），--check-only 只出计数不打包，给模板写手的产出做对照。

产物：docs/NN.txt（每份住院一份，文书按顺序）、index.csv、check.csv（只有计数）、README.md、MANIFEST.sha256，再打成同名 .tar.gz。
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import tarfile

from .safe import SafeError, guarded, log

MAX_N = 20
GRAM = 30
DIAG_GRAMS = (20, 30)
K = 11
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")
PATTERNS = (("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")), ("id_number", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
            ("long_digits", re.compile(r"\d{9,}")), ("record_number", re.compile(r"(?:住院号|病案号|床号|门诊号|电话|手机|身份证)[^\d\n]{0,6}\d")))
# 程序写的节（compiler.prog）和诊断清单；其余带标题的节、没有标题的病程正文是写手写的
PROGRAM_SECTIONS = ("主诉", "个人史", "婚育史", "月经史", "家族史", "辅助检查", "入院诊断", "初步诊断", "术前诊断", "拟施手术名称和方式", "拟施麻醉方式",
                    "手术时间", "术中诊断", "手术方式", "麻醉方式", "出院诊断")
LLM_SECTIONS = ("现病史", "既往史", "体格检查", "专科情况", "病例特点", "拟诊讨论", "鉴别诊断", "诊疗计划", "简要病情", "手术指征", "注意事项", "手术简要经过",
                "术后处理措施", "术后应当特别注意观察的事项", "入院情况", "诊疗经过", "出院情况", "出院医嘱")
_TITLE = re.compile(r"^(" + "|".join(sorted(PROGRAM_SECTIONS + LLM_SECTIONS, key=len, reverse=True)) + r")：")
_HEADER = re.compile(r"姓名：\[患者\]|^入院时间：|^\d{4}-\d{2}-\d{2} \d{2}:\d{2}|^(?:入院记录|首次病程记录|出院记录)$")
_LIST_ITEM = re.compile(r"^\d+\.")


def norm(s):
    return _DIGIT.sub("0", _WS.sub("", s or ""))


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def lines_of(adm):
    """每份文书按行切，给每行标位置：program（程序写的节、诊断清单、抬头）或 writer（写手写的）。返回 [(类别, 去空白的原行)]。"""
    out = []
    for d in adm["docs"]:
        cur, in_list = "writer", False
        for i, raw in enumerate((d.get("text") or "").split("\n")):
            line = raw.strip()
            if not line:
                continue
            m = _TITLE.match(line)
            if m:
                cur = "program" if m.group(1) in PROGRAM_SECTIONS else "writer"
                in_list = m.group(1).endswith("诊断")
            elif in_list and _LIST_ITEM.match(line):
                cur = "program"
            elif in_list:
                in_list, cur = False, "writer"
            kind = "program" if (i == 0 or _HEADER.search(line)) else cur
            out.append((kind, _WS.sub("", line)))
    return out


def _gram_index(adm, g):
    """{归一后的片段: {(类别, 原文带数字), ...}}，片段不跨行。"""
    idx = {}
    for kind, line in lines_of(adm):
        n = norm(line)
        for i in range(len(n) - g + 1):
            idx.setdefault(n[i:i + g], set()).add((kind, bool(_DIGIT.search(line[i:i + g]))))
    return idx


def rare_span_counts(cands, source_lines, grams=DIAG_GRAMS, k=K):
    """每份候选、每种长度：在真实语料里出现于 1..k-1 次住院的片段有几个。返回 ({长度: {住院号: 命中数}}, 计数)。"""
    per = {g: {a["admission_id"]: _gram_index(a, g) for a in cands} for g in grams}
    pools = {g: set().union(*(set(x) for x in per[g].values())) if per[g] else set() for g in grams}
    seen = {g: {} for g in grams}
    n_real = 0
    for obj in source_lines:
        n_real += 1
        hits = {g: set() for g in grams}
        for d in obj.get("docs") or ():
            t = norm(d.get("text"))
            for g in grams:
                pool, h = pools[g], hits[g]
                for i in range(len(t) - g + 1):
                    x = t[i:i + g]
                    if x in pool:
                        h.add(x)
        for g in grams:
            sg = seen[g]
            for x in hits[g]:
                sg[x] = min(k, sg.get(x, 0) + 1)
    stats = {"real_admissions_scanned": n_real}
    out = {}
    for g in grams:
        rare = {x for x, n in seen[g].items() if n < k}
        stats.update({f"g{g}_synthetic_grams": len(pools[g]), f"g{g}_seen_in_real": len(seen[g]), f"g{g}_rare_in_real": len(rare)})
        cat = {}
        for idx in per[g].values():
            for x in rare & set(idx):
                for kind, digit in idx[x]:
                    key = f"g{g}_rare_{kind}_{'digit' if digit else 'nodigit'}"
                    cat[key] = cat.get(key, 0) + 1
        stats.update(cat)
        out[g] = {aid: len(rare & set(idx)) for aid, idx in per[g].items()}
        stats[f"g{g}_candidates_with_rare"] = sum(1 for v in out[g].values() if v)
    return out, stats


def pattern_hits(adm):
    text = "\n".join(d.get("text") or "" for d in adm["docs"])
    return {name: len(rx.findall(text)) for name, rx in PATTERNS}


def _iter_source(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def render(adm, nn, meta):
    head = f"# {nn:02d}  {adm['admission_id']}  合成病历（虚构；S1 自由模式；{meta}）\n\n"
    body = "".join(f"===== {d.get('k', i):02d} {d.get('doc_type_cn') or d.get('doc_type')}　住院第{d.get('day_offset')}天 =====\n{(d.get('text') or '').rstrip()}\n\n"
                   for i, d in enumerate(adm["docs"]))
    return head + body


def _guard_out(out):
    norm_path = os.path.abspath(out).replace("\\", "/")
    if "/derived/" in norm_path + "/" or "/data/" in norm_path + "/":
        raise SafeError("out_under_derived_or_data")
    if os.path.exists(out) and os.listdir(out):
        raise SafeError("out_not_empty")


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export(run, source, out, label, n=MAX_N, seed=20261105, g=GRAM, k=K, source_lines=None, check_only=False):
    if not 1 <= n <= MAX_N:
        raise SafeError("n_out_of_range", max=MAX_N)
    _guard_out(out)
    with open(os.path.join(run, "counts.json"), encoding="utf-8") as f:
        counts = json.load(f)
    if counts.get("mode") != "free":
        raise SafeError("only_free_mode")                          # 配对模式的槽位来自真实住院
    if counts.get("writer") not in ("llm", "template"):
        raise SafeError("writer_unknown")
    if "lora" in str(counts.get("model") or "").lower():
        raise SafeError("lora_writer_needs_task_authorization")
    adms = _read_jsonl(os.path.join(run, "admissions.jsonl"))
    ids = sorted(a["admission_id"] for a in adms)
    rng = random.Random(f"{seed}|{label}")
    rng.shuffle(ids)
    by_id = {a["admission_id"]: a for a in adms}
    cands = [by_id[x] for x in ids[:min(len(ids), 3 * n)]]
    grams = tuple(sorted(set(DIAG_GRAMS) | {g}))
    rare, rs = rare_span_counts(cands, source_lines if source_lines is not None else _iter_source(source), grams, k)
    c = {"run_admissions": len(adms), "candidates": len(cands), "n_target": n, "gram_chars": g, "k": k, "digits_to_zero": 1, "cross_line": 0,
         "check_only": int(check_only), **rs}
    chosen = []
    for a in cands:
        ph = pattern_hits(a)
        for name, v in ph.items():
            c[f"pattern_{name}"] = c.get(f"pattern_{name}", 0) + v
        if rare[g][a["admission_id"]]:
            c["dropped_rare_span"] = c.get("dropped_rare_span", 0) + 1
            continue
        if any(ph.values()):
            c["dropped_pattern"] = c.get("dropped_pattern", 0) + 1
            continue
        if len(chosen) < n:
            chosen.append(a)
    c["exported"] = 0 if check_only else len(chosen)
    c["would_export"] = len(chosen)
    os.makedirs(out, exist_ok=True)
    _write_csv(os.path.join(out, "check.csv"), ["metric", "value"], sorted(c.items()))
    meta = f"{counts.get('writer')}{'·上一篇对照臂' if counts.get('prev_note') else ''}，{counts.get('version')}，种子 {counts.get('seed')}"
    if check_only:
        return c, None, None
    os.makedirs(os.path.join(out, "docs"), exist_ok=True)
    rows = []
    for nn, a in enumerate(chosen, 1):
        with open(os.path.join(out, "docs", f"{nn:02d}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(render(a, nn, meta))
        notes = sum(1 for d in a["docs"] if d.get("role") in ("SUP", "DAILY"))
        rows.append((f"{nn:02d}", a["admission_id"], len(a["docs"]), notes, sum(len(d.get("text") or "") for d in a["docs"])))
    _write_csv(os.path.join(out, "index.csv"), ["nn", "admission_id", "docs", "course_notes", "chars"], rows)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# 合成病历抽样 {label}\n\n来源：`{os.path.basename(os.path.normpath(run))}`（{meta}，模型 {counts.get('model') or '无'}），"
                f"从 {len(adms)} 份里随机抽 {len(chosen)} 份（种子 `{seed}|{label}`）。\n\n"
                "全部是 S1 自由模式按 k≥11 统计表规划、由没用真实病历训练过的写手写出的虚构文书（守则 v1.6 P5）。"
                "不含计划（journeys、drafts）、统计表、S2 抽取产物和任何真实原文。\n\n"
                f"带回前的检查见 `check.csv`：同一行里连续 {g} 字（去空白、数字换 0）在真实 fold 0 里只出现于 1–{k - 1} 次住院的，整份换掉"
                "（`dropped_rare_span`）；号码样式命中的整份换掉（`dropped_pattern`）。\n")
    files = sorted(os.path.relpath(os.path.join(dp, x), out).replace("\\", "/") for dp, _, fs in os.walk(out) for x in fs)
    with open(os.path.join(out, "MANIFEST.sha256"), "w", encoding="utf-8", newline="\n") as f:
        for x in files:
            with open(os.path.join(out, x), "rb") as fb:
                f.write(f"{hashlib.sha256(fb.read()).hexdigest()}  ./{x}\n")
    tar = os.path.normpath(out) + ".tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=os.path.basename(os.path.normpath(out)))
    with open(tar, "rb") as fb:
        sha = hashlib.sha256(fb.read()).hexdigest()
    return c, tar, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", required=True, help="真实 fold 0 语料（只用来查罕见片段，只出计数）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=int, default=MAX_N)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--check-only", action="store_true", help="只出 check.csv，不写正文、不打包（对照用）")
    a = ap.parse_args()
    os.umask(0o077)
    c, tar, sha = export(a.run, a.source, a.out, a.label, a.n, a.seed, check_only=a.check_only)
    log("synth_export_done", would_export=c["would_export"], exported=c["exported"], candidates=c["candidates"],
        dropped_rare_span=c.get("dropped_rare_span", 0), dropped_pattern=c.get("dropped_pattern", 0),
        tar=os.path.basename(tar) if tar else None, sha256=sha)
    if not a.check_only and c["exported"] < a.n:
        raise SafeError("exported_short", exported=c["exported"])


if __name__ == "__main__":
    guarded(main)
