# -*- coding: utf-8 -*-
"""出处核对：一条事实有出处，要同时满足
1. 引文 quote 在路径原文里（去空白、统一全半角后逐字找得到）；
2. 取值 value 在引文里（文字型：归一后是引文的子串；数值型：取值里的每个数都在引文里出现）。
做不到的标 grounded=False；source 写 model_prior 的不核，直接算无出处。全部确定性，不用模型。"""
import re

# 0.4（老板 11-11 批）：各种横线、波浪线统一成 "-"；引文、取值首尾的标点去掉。原文、引文、取值走同一个归一，对 v0、v1 一样
_FW = str.maketrans("０１２３４５６７８９．－～~–—（）：，；", "0123456789." + "-----" + "():,;")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_EDGE = "。．.，,；;：:、 "


def norm(s):
    return _WS.sub("", str(s or "")).translate(_FW)


def quote_in_source(quote, source_text):
    q = norm(quote).strip(_EDGE)
    return bool(q) and q in norm(source_text)


def value_in_quote(value, quote):
    q = norm(quote)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _num_str(value) in _NUM.findall(q)
    if isinstance(value, (list, tuple)):
        return bool(value) and all(value_in_quote(v, quote) for v in value)
    v = norm(value).strip(_EDGE)
    if not v:
        return False
    if v in q:
        return True
    nums = _NUM.findall(v)
    # 数值型文字（"7-8天""72-96小时"）：数都在引文里，且去掉数以后的单位字也在引文里
    if nums and all(n in _NUM.findall(q) for n in nums):
        rest = _NUM.sub("", v).strip("-~")
        return all(ch in q for ch in rest)
    return False


def _num_str(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def check_fact(fact, source_text):
    """就地写 fact["grounded"]，返回它。fact 至少有 value、source；有出处的还要 quote。"""
    if fact.get("source") == "model_prior":
        fact["grounded"] = False
    else:
        fact["grounded"] = quote_in_source(fact.get("quote"), source_text) and value_in_quote(fact.get("value"), fact.get("quote"))
    return fact["grounded"]


def grounded_rate(facts_by_field):
    """{字段: [fact, ...]} → {字段: (有出处数, 总数)}，按字段类型报"有出处率"。"""
    return {k: (sum(1 for f in v if f.get("grounded")), len(v)) for k, v in facts_by_field.items()}
