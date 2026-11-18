"""关4 的两种退化解（关卡谱关4"退化解"一栏），都不调模型：

- F-TPL 填表病历（"逐条列出规范要素的空模板"）：每个要素写成一句模板句，填的是对谁都一样的常数。
  除了性别、年龄、主要诊断三个给定属性，每个病人的病历一字不差。它是 H4-1 的地板。
- F-KW 关键词堆砌（"命中清单，内容空泛"）：宿主病历原样保留，在现病史、住院经过、出院医嘱末尾各加一句
  只报要素名、不给任何内容的话。它看评委会不会因为多了几个要素名就给更高的完整性分。

模板的句子是总经理照《病历书写基本规范》第十八条、第二十四条的要素和 LLM-CARe 9 条完整性清单的字面意思写的，
写完就冻结，没有拿评委试过、也不允许按评委结果改（PREREG_Q45）。
"""
from ruler_g3.adapters import CARE_KEYS, care_get

TPL = {
    "Chief Complaint": "{dx}相关症状1周。",
    "History of Present Illness": "患者于入院前1周无明显诱因急性起病，出现{dx}相关症状及体征，症状持续存在。"
                                  "病程中精神、饮食、睡眠、大小便均正常，体重无明显变化。",
    "Hospital Course": "入院后完善血常规、尿常规、肝肾功能、电解质及影像学等相关检查，予药物治疗及对症支持治疗，病情好转。",
    "Discharge Instructions": "出院后继续口服药物治疗，每种药物每次1片，每日1次。定期门诊复查。",
}

KW = {
    "History of Present Illness": "起病时间、起病缓急、诱因、主要症状均已问及；病程中精神、饮食、睡眠、大小便及体重变化情况均已记录。",
    "Hospital Course": "已完善相关检查及化验，已给予相应治疗。",
    "Discharge Instructions": "各药物的剂量和用法均遵医嘱。",
}

_OUT_KEYS = ("Gender", "Age", "Primary Diagnosis", "Chief Complaint", "History of Present Illness",
             "Hospital Course", "Discharge Instructions")
_FIELD = {"Gender": "sex", "Age": "age", "Primary Diagnosis": "dx", "Chief Complaint": "cc",
          "History of Present Illness": "hpi", "Hospital Course": "hc", "Discharge Instructions": "di"}


def _canonical(record):
    """不论宿主用英文键还是中文键，都读成 LLM-CARe 的 7 个英文键。"""
    return {k: care_get(record, _FIELD[k]) for k in _OUT_KEYS}


def ftpl(record):
    """填表病历：性别、年龄、主要诊断取宿主的，其余四段是模板。"""
    base = _canonical(record)
    dx = base["Primary Diagnosis"]
    out = {k: base[k] for k in ("Gender", "Age", "Primary Diagnosis")}
    for k, t in TPL.items():
        out[k] = t.format(dx=dx)
    return out


def _join(a, b):
    a = (a or "").rstrip()
    if a and a[-1] not in "。；;！!？?":
        a += "。"
    return a + b


def fkw(record):
    """关键词堆砌：宿主原样，三段末尾各加一句只报要素名的话。"""
    out = _canonical(record)
    for k, t in KW.items():
        out[k] = _join(out[k], t)
    return out


def vague(sex, age, dx):
    """G1-B 配套 P5 的空泛病历 V（关1 已用过，这里只用于核对 ys 上 V 文件的来源）。"""
    return {"Gender": sex, "Age": age, "Primary Diagnosis": dx,
            "Chief Complaint": f"因{dx}入院。",
            "History of Present Illness": f"患者因{dx}入院，入院后完善相关检查。",
            "Hospital Course": "予对症支持治疗，病情好转。",
            "Discharge Instructions": "出院后注意休息，门诊随诊。"}


assert set(_FIELD.values()) == set(CARE_KEYS)
