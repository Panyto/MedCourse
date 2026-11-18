"""从公开的《疾病分类与代码国家临床版 2.0》生成两张随包的表（只用标准库）：

  medcourse/data/icd_names.csv     归一后的诊断名 → 三位码。同一个名字对应不同三位码的不收。
  medcourse/data/icd_category.csv  三位码 → 一个标准名，诊断名凑不够 k 次时规划器用它兜底。

python -B tools/build_icd_tables.py 疾病分类与代码国家临床版2.0.xlsx medcourse/data

全部来自公开词典，不读任何病历。归一同 D0 的 normalize_name：去掉标点和空白、转小写。
"""
import csv
import hashlib
import os
import re
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CODE3 = re.compile(r"^([A-Z]\d{2})\.", re.I)
PUNCT_SPACE = re.compile(r"[\W_]+", flags=re.UNICODE)
VAGUE = re.compile(r"其他|未特指|未分类|NOS")
# 按规则挑出来的名字不像病历写法的，换成临床常用名（公开知识）
OVERRIDE = {"K80": "胆囊结石", "O34": "瘢痕子宫"}


def normalize_name(value):
    return PUNCT_SPACE.sub("", value or "").lower()


def _col(ref):
    v = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        v = v * 26 + ord(ch) - 64
    return v - 1


def xlsx_rows(path):
    with ZipFile(path) as z:
        shared = ["".join(n.text or "" for n in si.iter(NS + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si")]
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
            vals = {}
            for c in row.findall(NS + "c"):
                v = c.find(NS + "v")
                x = "" if v is None else v.text or ""
                if c.attrib.get("t") == "s" and x:
                    x = shared[int(x)]
                vals[_col(c.attrib["r"])] = x
            if vals:
                yield [vals.get(i, "") for i in range(max(vals) + 1)]


def category_name(entries):
    """三位码下挑一个标准名：.900 是干净的名字就用它；否则挑被本组其他名字包含次数最多的（通常是主干病名）；再不行取编码最小的。
    干净 = 不带逗号、不带"其他""未特指"。"""
    clean = [(c, n) for c, n in entries if "，" not in n and "," not in n and not VAGUE.search(n)]
    for c, n in clean:
        if c.endswith(".900"):
            return n
    if clean:
        names = [n for _, n in entries]
        return max(clean, key=lambda cn: (sum(cn[1] in x for x in names if x != cn[1]), -len(cn[1]), [-ord(ch) for ch in cn[0]]))[1]
    return re.sub(r"[，,].*$", "", sorted(entries)[0][1])


def main(xlsx, out_dir):
    by_name, by_code3 = {}, {}
    for i, row in enumerate(xlsx_rows(xlsx), start=1):
        if i <= 3 or len(row) < 3:
            continue
        code = (row[0] or "").strip()
        name = (row[2] or "").strip()
        m = CODE3.match(code)
        if not m or not name:
            continue
        c3 = m.group(1).upper()
        by_name.setdefault(normalize_name(name), set()).add(c3)
        by_code3.setdefault(c3, []).append((code, name))
    os.makedirs(out_dir, exist_ok=True)
    names = sorted((k, next(iter(v))) for k, v in by_name.items() if k and len(v) == 1)
    with open(os.path.join(out_dir, "icd_names.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["name", "code3"])
        w.writerows(names)
    cats = sorted((c3, OVERRIDE.get(c3) or category_name(sorted(e))) for c3, e in by_code3.items())
    with open(os.path.join(out_dir, "icd_category.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["code3", "name"])
        w.writerows(cats)
    sha = hashlib.sha256(open(xlsx, "rb").read()).hexdigest()
    print(f"source_sha256 {sha} names {len(names)} ambiguous {sum(len(v) > 1 for v in by_name.values())} code3 {len(cats)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
