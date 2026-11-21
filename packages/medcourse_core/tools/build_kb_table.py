# -*- coding: utf-8 -*-
"""Build compact public clinical-pathway KB (medcourse/data/kb_v22.json) from Route B kb_v2.2_agentic JSON files."""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = (
    r"C:\Users\91548\.gemini\antigravity\brain\e5b949cd-11ae-41fb-b120-e17cb9ab1730"
    r"\scratch\kb_v22_unpack\kb_v2.2_agentic"
)
DEFAULT_OUT = os.path.join(ROOT, "medcourse", "data", "kb_v22.json")

_VALID_EXAM_KEYS = {
    "血常规", "尿常规", "粪常规", "肝功能", "肾功能", "电解质", "血糖", "血脂", "凝血",
    "D二聚体", "血气", "利钠肽", "心肌肌钙蛋白", "肌钙蛋白", "心肌酶", "C反应蛋白",
    "降钙素原", "糖化血红蛋白", "淀粉酶", "血沉", "肿瘤标志物", "甲状腺功能",
    "CT", "磁共振", "超声", "心脏超声", "X线", "MRCP", "PET",
    "心电图", "动态心电图", "动态血压", "肺功能", "脑电图", "肌电图", "骨密度", "睡眠监测",
    "胃镜", "肠镜", "支气管镜", "喉镜", "膀胱镜", "超声内镜", "ERCP", "胶囊内镜",
    "冠脉造影", "脑血管造影", "骨髓穿刺", "腰椎穿刺",
    "血培养", "痰培养", "尿培养", "脓液培养", "分泌物培养", "病理", "活检",
}

_SYM_NOISE = re.compile(
    r"血常规|尿常规|大便常规|便常规|粪便常规|肝肾功能|肝功能|肾功能|电解质|血型|凝血功能|"
    r"胸片|腹部平片|心电图|CT|超声|KUB|X线|MRI|查体|体格检查|体力状态|评分|症状加重$|"
    r"急症|紧急|合并症|发症$|血液透析|起病|肠型|蠕动波|叩诊|鼓音|肠鸣音|气过水声|金属音|振水音|"
    r"肺实变|湿性啰音|湿罗音|哮鸣音|三凹征|鼻翼扇动|通常无症状|CCS|2－3天|下呼吸道症状|"
    r"^急性$|^压痛$|^可闻$|胸廓饱满|呼吸加速|放射|加剧|局灶症状"
)
_SIGN_NOISE = re.compile(
    r"体格检查$|腹部检查$|体力状态评分|是否|ST段|T波|心电图|肌钙蛋白|肾小球滤过率|eGFR|"
    r"终末期肾病|慢性肾脏病|监测神经系统|急性起病|通常无症状|通常正常|临床发作"
)
_PAREN = re.compile(r"(?:（[^）]*）|\([^)]*\)|（[^）]*$|\([^)]*$)")


def _region_from_name(key, name):
    if key not in ("CT", "磁共振", "超声", "X线"):
        return ""
    s = name or ""
    if "心脏" in s or "心动图" in s or "心超" in s:
        return "心"
    if re.search(r"腹|肝|胆|胰|脾|KUB", s):
        return "腹"
    if re.search(r"头颅|颅脑|头部|脑", s):
        return "头"
    if re.search(r"胸|肺|正侧位片|胸片", s):
        return "胸"
    if re.search(r"泌尿|双肾|输尿管|膀胱", s):
        return "泌尿"
    if re.search(r"盆腔|子宫|附件", s):
        return "盆"
    if re.search(r"颈|甲状腺", s):
        return "颈"
    return ""


def _exam_key_consistent(key, name):
    s = name or ""
    if key == "动态血压" and "血压" not in s:
        return False
    if key == "超声" and not re.search(r"超|B超|彩超|EUS", s, re.I):
        return False
    if key == "心脏超声" and not re.search(r"心超|心脏.{0,3}超|超声心动图", s):
        return False
    return key in _VALID_EXAM_KEYS


def _clean_symptoms(items):
    out, seen = [], set()
    for raw in items or ():
        s = _PAREN.sub("", str(raw or "")).strip("。；;，, ")
        # Preserve anatomical disjunctions when left half lacks symptom noun (e.g., 胸骨后或心前区...疼痛)
        if "或" in s:
            left = s.split("或", 1)[0]
            if not re.search(r"痛|热|闷|喘|吐|泻|晕|肿|咳|血|便|黄疸", left):
                s = s.split("或", 1)[1]
        for part in re.split(r"[、，,;；和]|(?:或(?=.{2,}(?:痛|热|闷|喘|吐|泻|晕|肿|咳|血|便|炎|心绞痛)))|(?:并(?=[肛呕腹发胸]))", s):
            t = re.sub(r"^(?:发病初期即出现|不明原因的|偶尔有|偶有|可有|常有|出现|表现为|发作的|伴|伴有|明显|持续性|发作性|反复|突发)+", "", part)
            t = re.sub(r"等消化不良症状$", "", t).replace("烦燥", "烦躁").strip("。；;，, 等")
            if not t or len(t) < 2 or len(t) > 12 or _SYM_NOISE.search(t):
                continue
            if "心前区剧烈的压榨性疼痛" in t:
                t = "心前区疼痛"
            if t == "脓血便" and "便血" in seen:
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _clean_signs(items):
    out, seen = [], set()
    for raw in items or ():
        s = str(raw or "").strip("。；;，, ")
        for part in re.split(r"[，,;；]", s):
            t = part.strip("。；;，, ")
            t = t.replace("（+）", "阳性").replace("(+)", "阳性").replace("湿罗音", "湿啰音")
            t = re.sub(r"^(?:腹部查体可见|临床表现|常见体征[：:])", "", t).strip()
            if not t or len(t) < 2 or len(t) > 22 or _SIGN_NOISE.search(t):
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _first_val(facts, field, default=None):
    rows = facts.get(field) or ()
    if not rows:
        return default, False, ""
    r = rows[0]
    return r.get("value", default), bool(r.get("grounded")), r.get("quote") or ""


def _parse_exams(rows, allow_quote_match=True):
    out, seen = [], set()
    for r in rows or ():
        quote = r.get("quote") or ""
        grounded = bool(r.get("grounded"))
        for it in r.get("value") or ():
            key = it.get("key") or ""
            name = (it.get("name") or "").strip()
            if not key or key == "无对应" or not _exam_key_consistent(key, name):
                continue
            if not grounded and allow_quote_match and quote:
                stem = re.split(r"[（(、，/+]", name)[0][:3]
                if len(stem) >= 2 and stem not in quote and key[:2] not in quote:
                    continue
            if name == "动态血糖监测":
                continue
            reg = _region_from_name(key, name)
            if key == "动态血压" and reg:
                reg = ""
            # Handle compound lab mention like 肝肾功能 -> ensure both 肝功能 and 肾功能
            keys_to_add = [key]
            if "肝肾功能" in name and key == "肝功能":
                keys_to_add = ["肝功能", "肾功能"]
            for k_add in keys_to_add:
                r_add = reg if k_add == key else ""
                pair = (k_add, r_add)
                if pair not in seen:
                    seen.add(pair)
                    out.append({"key": k_add, "region": r_add, "name": f"{k_add}" if k_add != key else name})
    return out


def convert_one(path):
    obj = json.load(open(path, encoding="utf-8"))
    meta = obj.get("meta") or {}
    facts = obj.get("facts") or {}
    icd10 = meta.get("icd10") or ""
    code3 = icd10.split(".", 1)[0].split("_", 1)[0][:3]
    los_raw, _, _ = _first_val(facts, "los_max", {})
    los_max = int(los_raw.get("max")) if isinstance(los_raw, dict) and los_raw.get("max") else 7
    sg_val, _, _ = _first_val(facts, "surgery_gate", False)
    surgical = bool(sg_val)
    proc_val, proc_g, _ = _first_val(facts, "procedure", "")
    proc_str = (proc_val or "").strip() if (surgical and proc_g and isinstance(proc_val, str)) else ""
    cs_raw, _, _ = _first_val(facts, "chief_symptoms", [])
    sg_items, _, _ = _first_val(facts, "signs", [])
    pc_raw, _, _ = _first_val(facts, "postop_care", {})
    watch = []
    incision, drain = False, False
    if surgical and isinstance(pc_raw, dict):
        incision = bool(pc_raw.get("incision"))
        drain = bool(pc_raw.get("drain"))
        for w in pc_raw.get("watch") or ():
            ws = str(w).strip("。；;，, ")
            if ws and ws not in ("观察病情", "血常规", "尿常规", "大便常规+隐血", "肝功能", "肾功能", "血电解质"):
                watch.append(ws)
    dc_raw, _, _ = _first_val(facts, "discharge_criteria", [])
    discharge = [str(x).strip("。；;，, ") for x in (dc_raw or ()) if len(str(x).strip()) >= 4][:6]
    return {
        "icd10": icd10,
        "code3": code3,
        "pathway": meta.get("pathway") or "",
        "version": meta.get("version") or "",
        "issuer": meta.get("issuer") or "",
        "los_max": los_max,
        "surgical": surgical,
        "procedure": proc_str,
        "chief_symptoms": _clean_symptoms(cs_raw),
        "signs": _clean_signs(sg_items),
        "exams_required": _parse_exams(facts.get("exams_required")),
        "exams_optional": _parse_exams(facts.get("exams_optional")),
        "postop_incision": incision,
        "postop_drain": drain,
        "postop_watch": watch[:6],
        "discharge_criteria": discharge,
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    files = sorted(glob.glob(os.path.join(src, "kb_*.json")))
    if not files:
        raise SystemExit(f"no kb_*.json found in {src}")
    entries = {e["icd10"]: e for e in (convert_one(f) for f in files)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"version": "kb-v2.2-clean", "n_pathways": len(entries), "entries": entries}, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(entries)} pathways to {out}")


if __name__ == "__main__":
    main()
