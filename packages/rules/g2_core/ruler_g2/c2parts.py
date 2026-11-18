"""C2 拆成三个分量（g2-core 0.4 新增，不改 readings.py 的 C2 本身）。

C2 = 结果口吻的检查种类数 + 有无手术（0/1） + 诊断状态变化次数（readings.c_group）。
G-E 的 C2 比真实低 37%，但第三个分量用的是会把"初步诊断为 X"读成确诊的诊断词典，而 G-E 不给判断层，
正文里状态词少，这一项会被压低。拆开看，才知道是真变薄还是计数口径。
"""
from . import diagnoses as dxm
from .events import admission_mentions
from .readings import Reading, _surgery, b_group


def c2_components(adm):
    """返回 {"exam_kinds", "surgery", "dx_changes", "C2"}，C2 与 evaluate_admission 里的 C2 相同。"""
    ments = admission_mentions(adm)
    dummies = [Reading(k) for k in ("B1", "B2", "B3", "B4", "B4b", "B5")]
    dm = b_group(adm, ments, *dummies)
    kinds = {(m.key, m.region) for d in adm.docs for m in ments[d.idx] if m.tone == "R"}
    n_surg = 1 if _surgery(adm) is not None else 0
    changes, seq = 0, {}
    for x in dm:
        if x.state in ("疑似", "鉴别", "确诊", "排除", "出院"):
            s = "S" if x.state in dxm.SUSPECT_STATES else ("C" if x.state in dxm.CONFIRM_STATES else "E")
            prev = seq.get(x.norm)
            if prev is not None and prev != s:
                changes += 1
            seq[x.norm] = s
    return {"exam_kinds": len(kinds), "surgery": n_surg, "dx_changes": changes,
            "C2": float(len(kinds) + n_surg + changes)}
