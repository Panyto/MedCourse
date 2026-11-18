"""关3 的雷：在病历里埋一个已知的用药硬错误，配一个只差这一处的对照版。

做法是"插一句"：雷版和对照版插进同一位置、句式相同的一句出院带药，只有药或剂量不同；
需要改年龄、加妊娠诊断的雷，两版都改。真值由模板写死（每条模板的期望规则和严重度，
由 tests 用剂量表的公式核过），不靠尺子自己判。

宿主条件：对照版在尺子上没有任何命中（硬错误、警示、性别、儿科成人量、N1 都没有）。
改年龄的雷另要求正文里没有"N岁"和成人特有词，免得评委因为年龄和病史对不上而判不通过。

每个病例对造好以后过一遍注入自检（check_care7、check_admission）：句子确实只插了一次、插在该插的地方，
年龄和妊娠诊断确实改到了尺子读得出的样子，雷版和对照版只差那一句。不过的宿主不进病例，记进 report。
雷版的"检出"要求命中落在插进去的那一句里（case["span"]），不认别处的命中。
"""
import copy
import random
import re

from ruler_g3.adapters import from_care7, from_admission, care_get, doc_kind
from ruler_g3.engine import evaluate
from ruler_g3.parse import AGE_HDR, PREG_END, SEX_HDR, parse_age, parse_sex

# 每条：(雷版句子, 对照版句子, 期望规则, 期望严重度, 结构化字段供测试核对)
# 结构化字段：(drug_id, 单次 mg, 每日次数)；年龄、妊娠类的雷不需要剂量
TEMPLATES = {
    "M-QUIN": {
        "age": "8岁",
        "history_lt": 8,       # 7 字段宿主正文里"N年"的 N 都要小于它，免得改了年龄以后病史年数不可能
        "rows": [
            ("出院后口服左氧氟沙星片0.5g，每日一次，共7天。", "出院后口服阿莫西林胶囊0.5g，每日三次，共7天。", "D-AGE-LEVO", "severe", None),
            ("出院带药：盐酸左氧氟沙星胶囊 0.2g bid。", "出院带药：头孢克肟分散片 0.1g bid。", "D-AGE-LEVO", "severe", None),
            ("继续予可乐必妥500mg口服，1次/日。", "继续予阿奇霉素片250mg口服，1次/日。", "D-AGE-LEVO", "severe", None),
        ],
    },
    "M-PEDDOSE": {
        "age": "2岁",          # 尺子按年龄上限 3 岁取体重 14.3 kg（宁可漏）；每行比值在这个体重下都 ≥ 2
        "history_lt": 2,
        "rows": [
            ("出院后口服阿莫西林胶囊1.0g，每日三次，共7天。", "出院后口服阿莫西林颗粒0.25g，每日三次，共7天。", "D-DOSE-PED", "severe", ("amoxicillin", 1000, 3)),
            ("出院带药：对乙酰氨基酚片0.5g，每4小时一次。", "出院带药：对乙酰氨基酚混悬滴剂0.15g，每6小时一次。", "D-DOSE-PED", "severe", ("acetaminophen", 500, 6)),
            ("继续口服阿奇霉素片0.5g，每日两次。", "继续口服阿奇霉素干混悬剂0.1g，每日一次。", "D-DOSE-PED", "severe", ("azithromycin", 500, 2)),
            ("头孢曲松钠4g静脉滴注，每日一次。", "头孢曲松钠0.5g静脉滴注，每日一次。", "D-DOSE-PED", "severe", ("ceftriaxone", 4000, 1)),
        ],
    },
    "M-PREG": {
        # 关1 的 256 个槽位里 18–45 岁女性只有 4 个，所以 7 字段宿主两版都改成 28 岁女性（同 M-QUIN 改年龄的做法）；
        # 整次住院只用本来就是 18–45 岁女性的宿主，改年龄只是把年龄统一成 28 岁
        "sex": "女",
        "age": "28岁",
        "history_lt": 15,
        "dx_add": "；宫内妊娠20周",
        "rows": [
            ("继续口服缬沙坦胶囊80mg，每日一次。", "继续口服拉贝洛尔片100mg，每日两次。", "D-PREG-RAS", "severe", None),
            ("出院后予卡托普利片12.5mg tid降压。", "出院后予硝苯地平控释片30mg qd降压。", "D-PREG-RAS", "severe", None),
            ("继续服用厄贝沙坦片150mg qd。", "继续服用甲基多巴片250mg bid。", "D-PREG-RAS", "severe", None),
        ],
    },
    "M-DOSE10": {
        "rows": [
            ("继续口服氨氯地平片50mg，每日一次。", "继续口服氨氯地平片5mg，每日一次。", "D-DOSE-ADULT", "severe", ("amlodipine", 50, 1)),
            ("出院后服用阿托伐他汀钙片200mg qn。", "出院后服用阿托伐他汀钙片20mg qn。", "D-DOSE-ADULT", "severe", ("atorvastatin", 200, 1)),
            ("二甲双胍片2.5g，每日三次。", "二甲双胍片0.5g，每日三次。", "D-DOSE-ADULT", "severe", ("metformin", 2500, 3)),
            ("继续口服美托洛尔片500mg bid。", "继续口服美托洛尔片50mg bid。", "D-DOSE-ADULT", "severe", ("metoprolol", 500, 2)),
            ("左氧氟沙星片2.0g，每日一次。", "左氧氟沙星片0.5g，每日一次。", "D-DOSE-ADULT", "severe", ("levofloxacin", 2000, 1)),
        ],
    },
    # 下面两类只测尺子召回，不进评委
    "M-SEX": {
        "rows": [
            ("予缩宫素10U肌内注射。", "予维生素K1 10mg肌内注射。", "D-SEX-OBST", "mild", None),
            ("出院带药：米非司酮片25mg bid。", "出院带药：甲钴胺片0.5mg tid。", "D-SEX-OBST", "mild", None),
        ],
    },
    "M-VITAL": {
        "rows": [
            ("复查体温47.6℃。", "复查体温36.6℃。", "N1", "mild", None),
            ("复查血钾45.3mmol/L。", "复查血钾4.5mmol/L。", "N1", "mild", None),
            ("复测血压320/190mmHg。", "复测血压132/80mmHg。", "N1", "mild", None),
        ],
    },
}

JUDGE_MINES = ("M-QUIN", "M-PEDDOSE", "M-PREG", "M-DOSE10")
ADULT_WORDS = re.compile(r"高血压|冠心病|糖尿病|吸烟|饮酒|绝经|退休|心肌梗死|心梗|脑梗|慢性阻塞|慢阻肺|前列腺|老年|月经|妊娠|孕|配偶|丈夫|妻子")
AGE_IN_TEXT = re.compile(r"\d+\s*(?:岁|周岁)")
YEARS_IN_TEXT = re.compile(r"(?<![\d.])(\d{1,2})\s*余?\s*年(?![级代度])")
MALE_OLD_WORDS = re.compile(r"前列腺|睾丸|阴囊|阴茎|包皮|精索|精液|勃起|男性|男孩|妻子|绝经|退休|老年|患儿|小儿|新生儿|婴儿|幼儿|家长")


def _join(text, sent):
    text = (text or "").rstrip()
    if text and text[-1] not in "。；;！!？?":
        text += "。"
    return text + sent


def _clean(res):
    return not res["hits"] and not res["n1"]


# ---------- 7 字段病历 ----------

def _care_body(rec):
    return "\n".join(care_get(rec, k) for k in ("cc", "hpi", "hc", "di"))


def care7_eligible(rec, mine):
    sex = care_get(rec, "sex")
    dx = care_get(rec, "dx")
    body = _care_body(rec)
    lim = TEMPLATES[mine].get("history_lt")
    if lim is not None and any(int(m.group(1)) >= lim for m in YEARS_IN_TEXT.finditer(body + "\n" + dx)):
        return False
    if mine in ("M-QUIN", "M-PEDDOSE"):
        if AGE_IN_TEXT.search(body) or ADULT_WORDS.search(body) or ADULT_WORDS.search(dx):
            return False
    if mine == "M-PREG":
        if AGE_IN_TEXT.search(body) or MALE_OLD_WORDS.search(body) or MALE_OLD_WORDS.search(dx):
            return False
        if PREG_END.search(body) or PREG_END.search(dx):
            return False
    if mine == "M-DOSE10":
        from ruler_g3.parse import parse_age
        age = parse_age(care_get(rec, "age"))
        if age is None or age < 18:
            return False
    if mine == "M-SEX":
        from ruler_g3.parse import parse_sex
        if parse_sex(sex) != "M":
            return False
    return True


def _key(rec, field, default):
    from ruler_g3.adapters import CARE_KEYS
    for k in CARE_KEYS[field]:
        if k in rec:
            return k
    return default


def apply_care7(rec, mine, idx, variant):
    t = TEMPLATES[mine]
    mined, control = t["rows"][idx][:2]
    r = copy.deepcopy(rec)
    if "sex" in t:
        r[_key(r, "sex", "Gender")] = t["sex"]
    if "age" in t:
        r[_key(r, "age", "Age")] = t["age"]
    if "dx_add" in t:
        k = _key(r, "dx", "Primary Diagnosis")
        r[k] = (r.get(k) or "") + t["dx_add"]
    k = _key(r, "di", "Discharge Instructions")
    r[k] = _join(r.get(k), mined if variant == "mined" else control)
    return r


def _sentences(mine, idx):
    return TEMPLATES[mine]["rows"][idx][:2]


def _locate(rec, sent):
    """sent 在 Record 各段里恰好出现一次时返回 (段名, 起, 止)，否则 None。"""
    found = []
    for name, text, _ in rec.segs:
        i = (text or "").find(sent)
        while i >= 0:
            found.append((name, i, i + len(sent)))
            i = text.find(sent, i + 1)
    return found[0] if len(found) == 1 else None


def check_care7(host, mine, idx, mined, control):
    """注入自检（7 字段）。返回没过的检查名列表，空列表就是过了。"""
    t = TEMPLATES[mine]
    ms, cs = _sentences(mine, idx)
    bad = []
    dk = _key(host, "di", "Discharge Instructions")
    if mined.get(dk) != _join(host.get(dk), ms) or control.get(dk) != _join(host.get(dk), cs):
        bad.append("di_suffix")
    if _locate(from_care7(mined), ms) is None or _locate(from_care7(control), cs) is None:
        bad.append("sentence_once")
    if {k for k in set(mined) | set(control) if mined.get(k) != control.get(k)} - {dk}:
        bad.append("only_sentence")
    changed = {k for k in set(host) | set(mined) if host.get(k) != mined.get(k)}
    allowed = {dk} | ({_key(host, "age", "Age")} if "age" in t else set()) | ({_key(host, "dx", "Primary Diagnosis")} if "dx_add" in t else set()) \
        | ({_key(host, "sex", "Gender")} if "sex" in t else set())
    if changed - allowed:
        bad.append("rest_same")
    if "age" in t:
        want = parse_age(t["age"])
        if from_care7(mined).age != want or from_care7(control).age != want:
            bad.append("age")
    if "sex" in t and not (from_care7(mined).sex == from_care7(control).sex == parse_sex(t["sex"])):
        bad.append("sex")
    if "dx_add" in t:
        k = _key(host, "dx", "Primary Diagnosis")
        if mined.get(k) != (host.get(k) or "") + t["dx_add"] or not evaluate(from_care7(mined))["pregnant"] \
                or not evaluate(from_care7(control))["pregnant"]:
            bad.append("preg")
    return bad


def _note(report, key, item):
    if report is not None:
        report.setdefault(key, []).append(item)


def build_care7_cases(pool_name, records, mine, n, seed="20261026", report=None):
    """records：[(host_id, 7 字段 dict)]。返回成对的 case 列表（每个宿主一对），最多 n 对。
    report：传一个 dict 进来，就把跳过的宿主按原因记进去（host_dirty、control_dirty、inject_fail）。"""
    rng = random.Random(f"{seed}|{pool_name}|{mine}")
    hosts = [(h, r) for h, r in records if care7_eligible(r, mine)]
    rng.shuffle(hosts)
    out = []
    for host_id, rec in hosts:
        if len(out) >= 2 * n:
            break
        if not _clean(evaluate(from_care7(rec, host_id))):
            _note(report, "host_dirty", host_id)
            continue
        idx = rng.randrange(len(TEMPLATES[mine]["rows"]))
        ctrl = apply_care7(rec, mine, idx, "control")
        if not _clean(evaluate(from_care7(ctrl, host_id))):
            _note(report, "control_dirty", host_id)
            continue
        mined = apply_care7(rec, mine, idx, "mined")
        bad = check_care7(rec, mine, idx, mined, ctrl)
        if bad:
            _note(report, "inject_fail", (host_id, bad))
            continue
        row = TEMPLATES[mine]["rows"][idx]
        for variant, r, sent in (("mined", mined, row[0]), ("control", ctrl, row[1])):
            out.append({"case_id": f"{pool_name}|{host_id}|{mine}|{variant}", "pool": pool_name, "host_id": host_id,
                        "mine": mine, "variant": variant, "template_idx": idx,
                        "expected_rule": row[2] if variant == "mined" else None,
                        "expected_severity": row[3] if variant == "mined" else None,
                        "span": _locate(from_care7(r), sent), "record": r})
    return out


# ---------- 整次住院（只测尺子召回） ----------

def apply_admission(adm, mine, idx, variant):
    t = TEMPLATES[mine]
    mined, control = t["rows"][idx][:2]
    a = copy.deepcopy(adm)
    docs = a.get("docs") or []
    if "age" in t or "sex" in t:
        for d in docs:
            if doc_kind(d) == "hr":
                if "age" in t:
                    d["text"] = AGE_HDR.sub("年龄：" + t["age"], d.get("text") or "", count=1)
                if "sex" in t:
                    d["text"] = SEX_HDR.sub("性别：" + t["sex"], d.get("text") or "", count=1)
                break
    target = None
    for d in docs:
        if doc_kind(d) == "discharge":
            target = d
    if target is None and docs:
        target = docs[-1]
    if target is None:
        return a
    add = ""
    if "dx_add" in t:
        add += "\n补充诊断：" + t["dx_add"].lstrip("；") + "\n"
    target["text"] = _join(target.get("text"), add + (mined if variant == "mined" else control))
    return a


def admission_eligible(adm, mine):
    rec = from_admission(adm)
    if mine == "M-PREG":
        end_scope = "\n".join(t for _, t, k in rec.segs if k in ("course", "discharge"))
        if rec.sex != "F" or rec.age is None or not (18 <= rec.age <= 45) or PREG_END.search(end_scope):
            return False
    if mine == "M-DOSE10" and (rec.age is None or rec.age < 18):
        return False
    if mine == "M-SEX" and rec.sex != "M":
        return False
    if mine in ("M-QUIN", "M-PEDDOSE") and rec.age is None:
        return False
    return True


def check_admission(host, mine, idx, mined, control):
    """注入自检（整次住院）。返回没过的检查名列表。"""
    t = TEMPLATES[mine]
    ms, cs = _sentences(mine, idx)
    bad = []
    hd, md, cd = host.get("docs") or [], mined.get("docs") or [], control.get("docs") or []
    if not (len(hd) == len(md) == len(cd)):
        return ["doc_count"]
    hrec, mrec, crec = from_admission(host), from_admission(mined), from_admission(control)
    if _locate(hrec, ms) is not None or _locate(hrec, cs) is not None or _locate(mrec, ms) is None or _locate(crec, cs) is None:
        bad.append("sentence_once")
    diff = [i for i in range(len(md)) if md[i].get("text") != cd[i].get("text")]
    if len(diff) != 1 or md[diff[0]]["text"].replace(ms, cs, 1) != cd[diff[0]]["text"]:
        bad.append("only_sentence")
    if "age" in t:
        want = parse_age(t["age"])
        if mrec.age != want or crec.age != want:
            bad.append("age")
    if "sex" in t and not (mrec.sex == crec.sex == parse_sex(t["sex"])):
        bad.append("sex")
    if "dx_add" in t and (not evaluate(mrec)["pregnant"] or not evaluate(crec)["pregnant"]):
        bad.append("preg")
    return bad


def build_admission_cases(pool_name, adms, mine, n, seed="20261026", report=None):
    """adms：[(host_id, adm)]。只在内存里用；返回的 case 含改过的 adm，真实病历不得落盘。"""
    rng = random.Random(f"{seed}|{pool_name}|{mine}")
    hosts = [(h, a) for h, a in adms if admission_eligible(a, mine)]
    rng.shuffle(hosts)
    out = []
    for host_id, adm in hosts:
        if len(out) >= 2 * n:
            break
        if not _clean(evaluate(from_admission(adm, host_id))):
            _note(report, "host_dirty", host_id)
            continue
        idx = rng.randrange(len(TEMPLATES[mine]["rows"]))
        ctrl = apply_admission(adm, mine, idx, "control")
        if not _clean(evaluate(from_admission(ctrl, host_id))):
            _note(report, "control_dirty", host_id)
            continue
        mined = apply_admission(adm, mine, idx, "mined")
        bad = check_admission(adm, mine, idx, mined, ctrl)
        if bad:
            _note(report, "inject_fail", (host_id, bad))
            continue
        row = TEMPLATES[mine]["rows"][idx]
        out.append({"case_id": f"{pool_name}|{host_id}|{mine}|mined", "pool": pool_name, "host_id": host_id, "mine": mine,
                    "variant": "mined", "template_idx": idx, "expected_rule": row[2], "expected_severity": row[3],
                    "span": _locate(from_admission(mined), row[0]), "adm": mined})
        out.append({"case_id": f"{pool_name}|{host_id}|{mine}|control", "pool": pool_name, "host_id": host_id, "mine": mine,
                    "variant": "control", "template_idx": idx, "expected_rule": None, "expected_severity": None,
                    "span": _locate(from_admission(ctrl), row[1]), "adm": ctrl})
    return out


def _in_span(h, span):
    return span is None or (h["seg"] == span[0] and span[1] <= h["start"] < span[2])


def detected(res, case):
    """雷版：尺子在插进去的那一句里报了期望规则（剂量类要求严重度一致）；对照版：尺子没有任何命中。"""
    if case["variant"] == "control":
        return _clean(res)
    rule = case["expected_rule"]
    span = case.get("span")
    if rule == "N1":
        return any(_in_span(h, span) for h in res["n1"])
    for h in res["hits"]:
        if h["rule_id"] == rule and _in_span(h, span) and \
                (case["expected_severity"] in (None, h["severity"]) or h["severity"] == "severe"):
            return True
    return False
