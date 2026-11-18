"""S3 探针（0.1.7）：用公开临床知识写的虚构分侧槽位，配对模式跑，专门量侧别前后一致（评测台 S3）。不来自病历，不代表自然分布。

python -B -m medcourse.probe --out probe_slots.jsonl
python -B -m medcourse.pipeline --stats stats_v07.json --mode paired --slots probe_slots.jsonl --writer template --out 目录

fold 0 只有约 1,600 次住院，分侧术式没有一种写法凑够 k=11 次，自由模式 500 次规划里分侧手术是 0，S3 不可测。探针只回答"真有分侧手术时，
我们的规划、编译、写手、闸门能不能让入院诊断、术前小结、术后首程、出院诊断的侧别一致"。槽位里写明术式（procedure），规划器照用，
不从统计表取；其余（检查、用药、症状）照常从统计表按诊断和 code3 取。每个槽位左右各一份。
"""
import argparse
import json
import os

from .safe import guarded, log

# (性别, 年龄, 科室, 诊断（侧别写 {S}）, code3, 住院天数, 手术日, 术式（侧别写 {S}）)。公开临床常识
BASE = (
    ("男", 58, "普外科", "{S}侧腹股沟斜疝", "K40", 4, 1, "{S}侧腹股沟疝无张力修补术"),
    ("男", 6, "小儿外科", "{S}侧腹股沟斜疝", "K40", 3, 1, "腹腔镜下{S}侧腹股沟斜疝疝囊高位结扎术"),
    ("女", 78, "骨科", "{S}侧股骨颈骨折", "S72", 10, 3, "{S}侧人工股骨头置换术"),
    ("男", 67, "骨科", "{S}侧股骨颈骨折", "S72", 12, 4, "{S}侧全髋关节置换术"),
    ("女", 66, "骨科", "{S}膝关节骨性关节炎", "M17", 9, 2, "{S}侧全膝关节置换术"),
    ("女", 72, "眼科", "{S}眼年龄相关性白内障", "H25", 3, 1, "{S}眼白内障超声乳化摘除联合人工晶体植入术"),
    ("女", 45, "甲状腺外科", "{S}侧甲状腺结节", "E04", 5, 2, "{S}侧甲状腺腺叶切除术"),
    ("女", 52, "乳腺外科", "{S}侧乳腺癌", "C50", 8, 2, "{S}侧乳腺癌改良根治术"),
    ("男", 63, "胸外科", "{S}肺上叶恶性肿瘤", "C34", 10, 3, "胸腔镜{S}肺上叶切除术"),
    ("男", 45, "泌尿外科", "{S}侧输尿管结石", "N20", 4, 1, "{S}侧输尿管镜钬激光碎石术"),
    ("女", 34, "妇科", "{S}侧卵巢囊肿", "N83", 5, 1, "腹腔镜{S}侧卵巢囊肿剥除术"),
    ("男", 55, "血管外科", "{S}下肢静脉曲张", "I83", 5, 1, "{S}侧大隐静脉高位结扎剥脱术"),
    ("女", 61, "骨科", "{S}侧桡骨远端骨折", "S52", 6, 2, "{S}侧桡骨远端骨折切开复位内固定术"),
    ("男", 31, "骨科", "{S}侧锁骨骨折", "S42", 5, 1, "{S}侧锁骨骨折切开复位内固定术"),
    ("男", 49, "骨科", "{S}侧胫腓骨骨折", "S82", 9, 2, "{S}侧胫腓骨骨折切开复位内固定术"),
)


def slots():
    out = []
    for i, (sex, age, dept, dx, code, los, sd, proc) in enumerate(BASE):
        for side in ("左", "右"):
            out.append({"slot_id": f"PROBE{i + 1:02d}{'L' if side == '左' else 'R'}", "sex": sex, "age": age, "dept": dept, "dx_name": dx.replace("{S}", side),
                        "dx_code3": code, "los": los, "surgical": True, "surgery_day": sd, "procedure": proc.replace("{S}", side), "probe": True})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.umask(0o077)
    rows = slots()
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    log("probe_slots_done", n=len(rows))


if __name__ == "__main__":
    guarded(main)
