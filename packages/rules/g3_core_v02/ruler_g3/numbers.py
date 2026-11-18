"""N1：生理上不可能、量级写错的数值；N2：合成病历里在底稿和槽位找不到出处的数值（只作诊断）。

N1 的范围故意取得很宽，只抓写错量级、写错单位一类的错。值为 0 的不算（死亡记录里"心率 0 次/分"是真的）。
"""
import re

_T = re.compile(r"(?:体温|(?<![A-Za-z])T)[^\d，,。；;\n]{0,6}?(\d{2}(?:\.\d+)?)\s*(?:℃|°C|度)")
_RATE_CN = re.compile(r"(脉搏|心率|呼吸)[^\d，,。；;\n]{0,4}?(\d{1,3})\s*次\s*[/／]\s*分")
_RATE_EN = re.compile(r"(?<![A-Za-z])(P|HR|R)\s*[:：]?\s*(\d{1,3})\s*次\s*[/／]\s*分")
_BP = re.compile(r"(?:血压|(?<![A-Za-z])BP)\s*[:：]?\s*(\d{2,3})\s*[/／]\s*(\d{2,3})\s*mmHg", re.I)
_K = re.compile(r"(?:血钾|血清钾|钾离子|(?<![A-Za-z])K\+?(?=\s*[:：]?\s*\d))\s*[:：]?\s*(\d+(?:\.\d+)?)\s*mmol/L", re.I)
_NA = re.compile(r"(?:血钠|血清钠|钠离子|(?<![A-Za-z])Na\+?(?=\s*[:：]?\s*\d))\s*[:：]?\s*(\d+(?:\.\d+)?)\s*mmol/L", re.I)
_HB = re.compile(r"(?:血红蛋白|(?<![A-Za-z])Hb|(?<![A-Za-z])HGB)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(g/L|g/dL)", re.I)
_WBC = re.compile(r"(?:白细胞(?:计数)?|(?<![A-Za-z])WBC)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(?:[×xX*]\s*)?10\^?9\s*[/／]\s*L", re.I)
_GLU = re.compile(r"(?:血糖|(?<![A-Za-z])GLU)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*mmol/L", re.I)
_CR = re.compile(r"(?:肌酐|(?<![A-Za-z])S?Cr|CREA)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(?:μmol/L|umol/L|µmol/L)", re.I)


def _bad(kind, v):
    if v == 0:
        return False
    if kind == "T":
        return v < 25 or v > 45
    if kind in ("P", "HR"):
        return v < 10 or v > 300
    if kind == "R":
        return v < 3 or v > 100
    if kind == "SBP":
        return v < 30 or v > 300
    if kind == "DBP":
        return v > 200
    if kind == "K":
        return v < 1.0 or v > 12
    if kind == "NA":
        return v < 90 or v > 200
    if kind == "HB_gL":
        return v < 15 or v > 260
    if kind == "HB_gdL":
        return v < 1.5 or v > 26
    if kind == "WBC":
        return v > 1000
    if kind == "GLU":
        return v < 0.5 or v > 150
    if kind == "CR":
        return v > 5000
    return False


def n1_hits(text):
    out = []
    text = text or ""

    def add(kind, m, v, grp=1):
        if _bad(kind, v):
            out.append({"type": kind, "value": v, "start": m.start(grp), "end": m.end(grp), "text": m.group(0)})

    for m in _T.finditer(text):
        add("T", m, float(m.group(1)))
    for m in _RATE_CN.finditer(text):
        kind = {"脉搏": "P", "心率": "HR", "呼吸": "R"}[m.group(1)]
        add(kind, m, float(m.group(2)), 2)
    for m in _RATE_EN.finditer(text):
        kind = {"P": "P", "HR": "HR", "R": "R"}[m.group(1)]
        add(kind, m, float(m.group(2)), 2)
    for m in _BP.finditer(text):
        s, d = float(m.group(1)), float(m.group(2))
        add("SBP", m, s, 1)
        add("DBP", m, d, 2)
        if s > 0 and d > 0 and s <= d:
            out.append({"type": "SBP<=DBP", "value": s, "start": m.start(1), "end": m.end(2), "text": m.group(0)})
    for m in _K.finditer(text):
        add("K", m, float(m.group(1)))
    for m in _NA.finditer(text):
        add("NA", m, float(m.group(1)))
    for m in _HB.finditer(text):
        add("HB_gL" if m.group(2).lower() == "g/l" else "HB_gdL", m, float(m.group(1)))
    for m in _WBC.finditer(text):
        add("WBC", m, float(m.group(1)))
    for m in _GLU.finditer(text):
        add("GLU", m, float(m.group(1)))
    for m in _CR.finditer(text):
        add("CR", m, float(m.group(1)))
    return out


# ---------- N2（只用于有底稿的关2 各组，只作诊断） ----------

_N2 = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(mmol/L|μmol/L|umol/L|g/L|×10\^?9/L|10\^?9/L|mmHg|℃|次/分|U/L|mg/L|ng/mL|pg/mL|%|cm|mm)", re.I)


def n2_numbers(text, source_text):
    """正文里带单位的数值，在 source_text（该槽位底稿和属性的 JSON 串）里找不到同一个数字串的，算"无出处"。
    返回 (总数, 无出处数, 无出处的样本列表)。"""
    src = source_text or ""
    total, unsourced, samples = 0, 0, []
    for m in _N2.finditer(text or ""):
        total += 1
        if m.group(1) not in src:
            unsourced += 1
            if len(samples) < 20:
                samples.append(m.group(0))
    return total, unsourced, samples
