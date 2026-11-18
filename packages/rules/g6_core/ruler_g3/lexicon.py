"""药名词表、规则表、剂量表的读取。数据文件在 data/ 下，由总经理维护。"""
import csv
import os
import re

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _rows(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Lexicon:
    def __init__(self):
        self.alias_to_id = {}
        self.classes = {}
        for r in _rows("drug_lexicon.csv"):
            did = r["drug_id"]
            self.classes[did] = set(c for c in r["classes"].split("|") if c)
            for a in r["aliases"].split("|"):
                a = a.strip()
                if a:
                    self.alias_to_id[a] = did
        # 最长优先：同一位置先试长的别名（"阿莫西林克拉维酸"先于"阿莫西林"）
        aliases = sorted(self.alias_to_id, key=len, reverse=True)
        self.rx = re.compile("|".join(re.escape(a) for a in aliases))

    def find(self, text):
        for m in self.rx.finditer(text):
            yield m.start(), m.end(), m.group(0), self.alias_to_id[m.group(0)]

    def in_target(self, drug_id, target):
        """target：drug:<id> / class:<cls> / class:<cls>-<排除的 id>。"""
        kind, _, val = target.partition(":")
        if kind == "drug":
            return drug_id == val
        if kind == "class":
            cls, _, excl = val.partition("-")
            return cls in self.classes.get(drug_id, ()) and drug_id != excl
        return False


def _num(x):
    x = (x or "").strip()
    return float(x) if x else None


class Rules:
    def __init__(self):
        self.rules = _rows("drug_rules.csv")
        self.dose = {}
        for r in _rows("dose_table.csv"):
            self.dose[r["drug_id"]] = {
                "adult_max": _num(r["adult_max_daily_mg"]),
                "ped_max_mgkg": _num(r["ped_max_mgkg_day"]),
                "ped_cap": _num(r["ped_cap_daily_mg"]),
                "ped_adult_threshold": _num(r["ped_adultdose_threshold_daily_mg"]),
                "sources": r["source_ids"],
            }
        self.sources = {r["source_id"]: r for r in _rows("sources.csv")}


# WHO 0–5 岁男童体重中位数（S14），月龄 → kg；中间线性插值，60 月以后取 60 月的值
WHO_BOYS_P50 = [(0, 3.3), (1, 4.5), (2, 5.6), (3, 6.4), (6, 7.9), (9, 8.9), (12, 9.6),
                (18, 10.9), (24, 12.2), (36, 14.3), (48, 16.3), (60, 18.3)]


def weight_p50(age_years):
    m = max(0.0, age_years * 12.0)
    pts = WHO_BOYS_P50
    if m >= pts[-1][0]:
        return pts[-1][1]
    for (m0, w0), (m1, w1) in zip(pts, pts[1:]):
        if m0 <= m <= m1:
            return w0 + (w1 - w0) * (m - m0) / (m1 - m0)
    return pts[0][1]


_LEX = None
_RULES = None


def lexicon():
    global _LEX
    if _LEX is None:
        _LEX = Lexicon()
    return _LEX


def rules():
    global _RULES
    if _RULES is None:
        _RULES = Rules()
    return _RULES
