"""CPU 小实验 1：R 对 R 的判别下限。回答"AUC 能压到多低才算到头"。

 a. null_7field     7 字段逐段等长，同码配对随机分两边（Q45b 已有 0.49，这里是复现）
 b. null_admission  整次住院（入院、首程、出院）等长，同码配对随机分两边
 c. null_<类型>     只看入院记录 / 首次病程 / 其他病程 / 出院记录
 d. dept_<代号>     同疾病章（三位码首字母）内，两个大科配对，每对等长：这是"院内真实差异"的 AUC
 e. *_length        只看长度的对照，应接近 0.5

都用 q45_kit.c2st 的同一个分类器（字符 n-gram、分组 5 折）。真实文本只在内存；科室名只作分组，表里用代号。
需要 scikit-learn，在 ys 上跑。

python -B -m bench.rvr --path 导出文件 --dx-map dx_map.csv --out 目录 [--dept-only] [--dept-map work/对照表.csv]
dx_map.csv 列：admission_id、code3（D0 v1.1 匹配结果；没匹配上的留空）。科室从入院记录抬头读。

科室名按三层看（S0_rvr_dept_diag.csv，只有计数）：raw 抬头原样；stripped 去掉病区、院区、括号；major 归到大科。
AUC 只算 major 层。MIN_PAIRS 不降：这个 AUC 决定 0.82 还算不算冲刺线，30 对以下区间太宽，凑不够就等全量。
"""
import argparse
import csv
import os
import random
import re
import sys

from . import canon, paths
from .safe import guarded, log

_DEPT = re.compile(r"科\s*室\s*[：:]\s*([^\s|｜，,；;:：]{2,12})")
REPS = 20
TOP_DEPT_PAIRS = 3
MIN_PAIRS = 30
SEED = 20261101
_DEPT_PAREN = re.compile(r"[（(][^）)]*[）)]")
_DEPT_SITE = re.compile(r"^(?:东|西|南|北|中|本部|总|新|老)?(?:院区|院|部)(?=.)|^(?:东|西|南|北)区")
_DEPT_WARD = re.compile(r"(?:第?[一二三四五六七八九十\d]+|[A-Za-z]|[东西南北中])?(?:病区|病房|区|组)$")
_DEPT_NUM = re.compile(r"第?[一二三四五六七八九十\d]+(?=科$)")
# 顺序有讲究：外科写在内科前面，"心胸""心血管外"写在"心"前面。只看关键词，认不出的留 stripped 那层的名字
DEPT_MAJOR = (("肿瘤|放疗|化疗", "肿瘤科"), ("新生儿|儿", "儿科"), ("妇|产|生殖", "妇产科"), ("神经外|神外|脑外", "神经外科"),
              ("神经内|神内|卒中", "神经内科"), ("胸外|心胸", "胸外科"), ("心外|心血管外|血管外", "心血管外科"), ("心内|心血管|心脏", "心内科"),
              ("泌尿", "泌尿外科"), ("骨|脊柱|关节|创伤|手外", "骨科"), ("普外|普通外|胃肠外|消化外|肝胆|甲状腺|乳腺|结直肠|肛肠|疝|微创外|^外科$", "普外科"),
              ("呼吸|肺", "呼吸内科"), ("消化|胃肠|肝病", "消化内科"), ("肾", "肾内科"), ("内分泌|糖尿病", "内分泌科"), ("血液", "血液科"),
              ("风湿|免疫", "风湿免疫科"), ("感染|传染", "感染科"), ("老年", "老年科"), ("全科|综合内|普内|^内科$", "全科"), ("耳鼻|头颈", "耳鼻喉科"),
              ("眼", "眼科"), ("口腔|颌面", "口腔科"), ("皮肤", "皮肤科"), ("重症|ICU|监护", "重症医学科"), ("急诊", "急诊科"),
              ("康复", "康复科"), ("中医", "中医科"), ("介入", "介入科"), ("疼痛", "疼痛科"))


def read_codes(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("code3"):
                out[str(r["admission_id"])] = r["code3"]
    return out


def dept_of(line):
    for d in line["docs"]:
        if d["doc_type"] == "hospital_record":
            m = _DEPT.search(d["text"][:400])
            return m.group(1) if m else None
    return None


def dept_stripped(name):
    """去掉括号、院区和病区、科室序号："东院区心内一科二病区" → "心内科"。"""
    if not name:
        return None
    s = _DEPT_PAREN.sub("", name)
    s = _DEPT_SITE.sub("", s)
    s = _DEPT_WARD.sub("", s)
    s = _DEPT_NUM.sub("", s)
    return s or name


def dept_major(name):
    s = dept_stripped(name)
    if not s:
        return None
    for pat, major in DEPT_MAJOR:
        if re.search(pat, s):
            return major
    return s


def role_text(line, role, is_fp):
    parts = []
    for d in line["docs"]:
        fp = d["doc_type"] == "progress_note" and is_fp(d["text"])
        got = {"hr": d["doc_type"] == "hospital_record", "fp": fp, "course": d["doc_type"] == "progress_note" and not fp,
               "ds": d["doc_type"] == "discharge_summary", "adm": d["doc_type"] != "progress_note" or fp}[role]
        if got:
            parts.append(d["text"])
    return "\n".join(parts)


def null_equal(texts, strata, c2st, reps=REPS, kind="text"):
    """同码两两配对、截等长、随机分两边，和 q45b.null_split_equal 同结构，但对象是一整段文本。"""
    by = {}
    for g, t in texts.items():
        if g in strata and t:
            by.setdefault(strata[g], []).append(g)
    out = []
    for r in range(reps):
        rng = random.Random(f"{SEED}|rvr|{r}")
        xs, ys, gs = [], [], []
        for code in sorted(by):
            ids = sorted(by[code])
            rng.shuffle(ids)
            for j in range(0, len(ids) - 1, 2):
                a, b = c2st.normalize(texts[ids[j]]), c2st.normalize(texts[ids[j + 1]])
                n = min(len(a), len(b))
                if n == 0:
                    continue
                x1, x2 = a[:n], b[:n]
                if kind == "length":
                    x1, x2 = [len(x1)], [len(x2)]
                flip = rng.random() < 0.5
                for x, lab in zip((x1, x2), (1, 0) if flip else (0, 1)):
                    xs.append(x)
                    ys.append(lab)
                    gs.append(f"{code}|{j}")
        out.append(c2st.auc(c2st.cv_scores(xs, ys, gs, c2st.SEEDS[0], kind), ys))
    return out


def _pair_ids(depts, codes):
    """{(甲, 乙): [(住院甲, 住院乙)…]}：每个疾病章内两两配对，同一对科室跨章累计。"""
    groups = {}
    for i, d in depts.items():
        c = codes.get(i)
        if d and c:
            groups.setdefault((c[0], d), []).append(i)
    pair_ids = {}
    chapters = sorted({k[0] for k in groups})
    for ch in chapters:
        ds = sorted(d for c, d in groups if c == ch)
        for x in range(len(ds)):
            for y in range(x + 1, len(ds)):
                a, b = sorted(groups[(ch, ds[x])]), sorted(groups[(ch, ds[y])])
                rng = random.Random(f"{SEED}|dept|{ch}|{x}|{y}")
                rng.shuffle(a)
                rng.shuffle(b)
                pair_ids.setdefault((ds[x], ds[y]), []).extend(zip(a, b))
    return pair_ids


def dept_pairs(depts, codes):
    """配对数最多的几组科室：[(甲, 乙, [(住院甲, 住院乙)…])]。选过的住院不再用，几组之间不重叠。"""
    top = sorted(_pair_ids(depts, codes).items(), key=lambda t: (-len(t[1]), t[0]))
    used, out = set(), []
    for (p, q), pairs in top:
        one = [(a, b) for a, b in pairs if a not in used and b not in used]
        if len(one) < MIN_PAIRS or len(out) >= TOP_DEPT_PAIRS:
            continue
        used.update(i for pair in one for i in pair)
        out.append((p, q, one))
    return out


def dept_diag(depts, codes):
    """三层科室名各有多少种、配对够不够。只有计数，不出科室名。"""
    out = []
    for level, fn in (("raw", lambda s: s), ("stripped", dept_stripped), ("major", dept_major)):
        one = {i: fn(d) for i, d in depts.items()}
        sizes = sorted((len(v) for v in _pair_ids(one, codes).values()), reverse=True)
        top = (sizes + [0, 0, 0])[:3]
        out.append({"level": level, "admissions": len(one), "with_dept": sum(v is not None for v in one.values()),
                    "with_dept_and_code": sum(1 for i, v in one.items() if v and codes.get(i)), "distinct": len({v for v in one.values() if v}),
                    "dept_pairs_ge1": len(sizes), "dept_pairs_ge15": sum(n >= 15 for n in sizes), "dept_pairs_ge30": sum(n >= MIN_PAIRS for n in sizes),
                    "top1_pairs": top[0], "top2_pairs": top[1], "top3_pairs": top[2], "qualifying_groups": len(dept_pairs(one, codes))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--dx-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--dept-only", action="store_true", help="只跑科室部分，写 S0_rvr_dept.csv，不动已有的空对照")
    ap.add_argument("--dept-map", help="代号和科室名的对照表，只能写到 work/，不进包")
    a = ap.parse_args()
    for name in ("g1", "g45"):
        sys.path.insert(0, paths.core_path(name))
    from q45_kit import c2st, q45b
    from q45_kit.realtext import care7_from_admission
    from ruler_g1 import extract_facts, is_first_progress, split_sections
    os.makedirs(a.out, exist_ok=True)
    codes = read_codes(a.dx_map)
    lines = [canon.admission_line(x, "R") for x in canon.read_jsonl(a.path)]
    rows = []

    def record(name, vals, n, kind="text"):
        rows.append({"experiment": name, "kind": kind, "auc_mean": sum(vals) / len(vals), "auc_min": min(vals), "auc_max": max(vals),
                     "reps": len(vals), "units": n})
        log("rvr", experiment=name, kind=kind, auc=round(sum(vals) / len(vals), 4), units=n)

    if not a.dept_only:
        recs = {}
        for x in lines:
            r, _ = care7_from_admission({"admission_id": x["id"], "docs": x["docs"]}, split_sections, extract_facts)
            if r is not None:
                recs[x["id"]] = r
        items = [(i, recs[i]) for i in sorted(recs) if i in codes]
        record("null_7field", q45b.null_split_equal(items, codes, reps=a.reps), len(items))
        record("null_7field", q45b.null_split_equal(items, codes, reps=min(5, a.reps), kind="length"), len(items), "length")
        for role, name in (("adm", "null_admission"), ("hr", "null_hr"), ("fp", "null_fp"), ("course", "null_course"), ("ds", "null_ds")):
            texts = {x["id"]: role_text(x, role, is_first_progress) for x in lines}
            record(name, null_equal(texts, codes, c2st, reps=a.reps), sum(1 for t in texts.values() if t))
    texts = {x["id"]: role_text(x, "adm", is_first_progress) for x in lines}
    if not a.dept_only:
        record("null_admission", null_equal(texts, codes, c2st, reps=min(5, a.reps), kind="length"), len(texts), "length")

    depts = {x["id"]: dept_of(x) for x in lines}
    known = sum(1 for d in depts.values() if d)
    log("rvr_dept", admissions=len(lines), with_dept=known)
    diag = dept_diag(depts, codes)
    with open(os.path.join(a.out, "S0_rvr_dept_diag.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag[0]))
        w.writeheader()
        w.writerows(diag)
    chosen = dept_pairs({i: dept_major(d) for i, d in depts.items()}, codes)
    if a.dept_map:
        with open(a.dept_map, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["code", "dept_a", "dept_b", "pairs"])
            w.writerows((f"D{n + 1}", p, q, len(pairs)) for n, (p, q, pairs) in enumerate(chosen))
    for n, (p, q, pairs) in enumerate(chosen):
        pos, neg = [], []
        for j, (i1, i2) in enumerate(pairs):
            t1, t2 = c2st.normalize(texts[i1]), c2st.normalize(texts[i2])
            m = min(len(t1), len(t2))
            pos.append((f"{n}|{j}", t1[:m]))
            neg.append((f"{n}|{j}", t2[:m]))
        r = c2st.c2st(pos, neg, "text")
        rows.append({"experiment": f"dept_D{n + 1}", "kind": "text", "auc_mean": r["auc_mean"], "auc_min": r["ci_lo"], "auc_max": r["ci_hi"],
                     "reps": len(r["auc_runs"]), "units": len(pairs)})
    with open(os.path.join(a.out, "S0_rvr_dept.csv" if a.dept_only else "S0_rvr.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "kind", "auc_mean", "auc_min", "auc_max", "reps", "units"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    guarded(main)
