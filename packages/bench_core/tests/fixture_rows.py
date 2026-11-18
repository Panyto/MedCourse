"""把 g2_core 的虚构病历做成 bench 的输入行。全部虚构，不含真实内容。"""
import json
import os


def admissions():
    """每次住院多复制几份、改一点数字，凑够跑判别器的量。"""
    import fixtures
    base = [fixtures.adm_gb(), fixtures.adm_knee(), fixtures.adm_pna()]
    out = []
    for i in range(40):
        one = json.loads(json.dumps(base[i % 3]))
        one["admission_id"] = f"F{i:03d}"
        for d in one["docs"]:
            d["text"] = d["text"].replace("80次/分", f"{70 + i % 25}次/分")
        out.append(one)
    return out


def care7(i):
    return {"Gender": "男" if i % 2 else "女", "Age": str(20 + i), "Primary Diagnosis": "社区获得性肺炎",
            "Chief Complaint": f"咳嗽{i % 9 + 1}天", "History of Present Illness": f"患者{i % 9 + 1}天前受凉后咳嗽，体温38.{i % 9}℃。",
            "Hospital Course": f"入院后予头孢曲松2g qd静滴{i % 5 + 3}天，复查白细胞{6 + i % 5}.{i % 9}×10^9/L。",
            "Discharge Instructions": "门诊随诊，出院带药：阿莫西林0.5g tid口服。"}


def write_rows(tmp):
    r, s = os.path.join(tmp, "r.jsonl"), os.path.join(tmp, "s.jsonl")
    c, codes = os.path.join(tmp, "c.jsonl"), os.path.join(tmp, "codes.csv")
    adms = admissions()
    with open(r, "w", encoding="utf-8") as f:
        for a in adms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with open(s, "w", encoding="utf-8") as f:
        for a in adms:
            b = {"admission_id": a["admission_id"], "docs": [{**d, "text": d["text"].replace("患者", "病人")} for d in a["docs"]]}
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(c, "w", encoding="utf-8") as f:
        for i in range(40):
            f.write(json.dumps({"slot_id": f"F{i:03d}", "record": care7(i)}, ensure_ascii=False) + "\n")
    with open(codes, "w", encoding="utf-8", newline="") as f:
        f.write("admission_id,code3\n")
        for i in range(40):
            f.write(f"F{i:03d},{'JKM'[i % 3]}{i % 3}\n")
    cfg = {"reference": "R", "codes": codes, "pairs": {"S": "R", "C": "R"},
           "efficiency": {"S": {"calls": 80, "input_tokens": 1000, "output_tokens": 2000, "seconds": 40}},
           "rows": [{"row": "R", "path": r, "kind": "admission", "view": "core", "n_expected": 40},
                    {"row": "S", "path": s, "kind": "admission", "view": "core"},
                    {"row": "C", "path": c, "kind": "care7"}]}
    p = os.path.join(tmp, "rows.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    return p
