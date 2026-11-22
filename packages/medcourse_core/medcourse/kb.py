# -*- coding: utf-8 -*-
"""Route B public clinical-pathway knowledge base (kb_v22) loader & pure-public clinical priors (v0.3.2)."""
import json
import os
import re

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DEFAULT_KB_PATH = os.path.join(_DATA_DIR, "kb_v22.json")
_KB_CACHE = {}

def _has_any(text, words):
    t = text or ""
    return any(w in t for w in words)

_ORGAN_NEG_SYMS = {
    "肺": ("咯血", "胸痛", "呼吸困难", "盗汗"),
    "咽喉": ("呼吸困难", "声音嘶哑", "咯血", "吞咽困难"),
    "心": ("晕厥", "端坐呼吸", "夜间阵发性呼吸困难", "双下肢水肿"),
    "腹": ("呕血", "黑便", "巩膜黄染", "停止排气排便"),
    "脑": ("意识障碍", "四肢抽搐", "大小便失禁", "饮水呛咳"),
    "神经": ("意识障碍", "四肢抽搐", "感觉丧失", "饮水呛咳"),
    "泌尿": ("肉眼血尿", "排尿中断", "少尿", "畏寒寒战"),
    "盆": ("异常阴道流液", "晕厥", "高热", "肛门坠胀"),
    "乳腺": ("乳头溢液", "皮肤破溃", "畏寒发热", "腋窝肿痛"),
    "骨": ("肢端麻木", "感觉异常", "高热", "局部破溃"),
    "血管": ("胸痛", "晕厥", "呼吸困难", "皮肤破溃"),
    "全身": ("意识障碍", "高热", "抽搐", "呼吸困难"),
}
_DEFAULT_NEG_SYMS = ("畏寒寒战", "意识障碍", "呼吸困难")


def load_kb(path=None):
    p = path or _DEFAULT_KB_PATH
    if p not in _KB_CACHE:
        with open(p, encoding="utf-8") as f:
            _KB_CACHE[p] = json.load(f)
    return _KB_CACHE[p]


def load_default_kb():
    return load_kb(_DEFAULT_KB_PATH) if os.path.exists(_DEFAULT_KB_PATH) else None


def match_kb(kb, code3, dx_name="", tkind=None, age=None):
    """Match a pathway entry by ICD-10 code3 / diagnosis / treatment kind."""
    if not kb:
        return None
    entries = kb.get("entries") or {}
    if not entries:
        return None
    name = dx_name or ""
    c3 = (code3 or "")[:3]
    if tkind == "化疗" and (c3 == "C18" or "结肠" in name):
        return entries.get("C18_chemo")
    if c3 == "K80":
        if "急性" in name or "急" in name:
            return entries.get("K80.0_acute") or entries.get("K80.1_chronic")
        return entries.get("K80.1_chronic") or entries.get("K80.0_acute")
    if c3 in ("J18", "J21"):
        for k, ent in entries.items():
            if ent.get("code3") == c3:
                return ent
    if c3 == "I20":
        return entries.get("I20.0")
    if c3 in entries:
        return entries[c3]
    for ent in entries.values():
        if c3 and ent.get("code3") == c3:
            return ent
    return None


def organ_neg_symptoms(organs, exclude_syms=()):
    """Return 2-3 organ-appropriate differential negative symptoms without repeating present ones."""
    excl = set(exclude_syms or ())
    pool = []
    for o in organs or ():
        for s in _ORGAN_NEG_SYMS.get(o, ()):
            if s not in pool:
                pool.append(s)
    if not pool:
        pool = list(_DEFAULT_NEG_SYMS)
    out = [s for s in pool if not any((s in e or e in s) for e in excl)]
    return out[:3]


def _side_label(name):
    s = name or ""
    return "双侧" if "双" in s else ("左侧" if "左" in s else ("右侧" if "右" in s else ""))


def organ_pe_facts(j, d):
    """Return anatomically consistent (pe_fact, spec_fact) for HR.体格检查 & HR.专科情况."""
    p = j["patient"]
    raw_dx = p["primary_dx"]["name"]
    safe_dx = re.sub(
        r"(?:入院后|住院期间|经[^，,。；;\n]{0,8}治疗|治疗后|术后|术中|病理(?:示|提示|回报|结果|报告)|复查|出院时|好转出院|维持性|辅助性?|化学治疗|化疗|放射治疗|放疗|随访检查)+",
        "",
        p["primary_dx"].get("underlying") or raw_dx,
    ).strip("，,、 ") or "原发病"
    organs = list(p["primary_dx"].get("organs") or ())
    syms = d.get("symptoms") or []
    sym_str = "、".join(syms)
    kb_signs = [s for s in ((j.get("kb_facts") or {}).get("signs") or ()) if not _has_any(s, ("术后", "复查", "治疗后", "出院"))]
    side = _side_label(raw_dx)

    if p.get("pregnant"):
        gw = p.get("gest_weeks") or 38
        if gw < 14 or _has_any(raw_dx, ("异位妊娠", "宫外孕", "葡萄胎", "胚胎停", "稽留流产")):
            abd = f"下腹部轻压痛，无反跳痛，伴{sym_str}表现" if syms else "腹平软，下腹部无明显压痛"
            pe = f"心肺听诊未见明显异常，{abd}，双下肢无明显水肿"
            spec = f"妇产科专科查体与停经{gw}周及{safe_dx}相符，无大量活动性阴道流血"
        elif gw < 20:
            pe = f"心肺听诊未见明显异常，下腹微隆，宫底高度与停经{gw}周妊娠相符，双下肢无明显水肿"
            spec = f"宫底高度符合孕{gw}周，专科检查与{safe_dx}相符"
        else:
            pe = (
                f"心肺听诊未见明显异常，腹隆，宫底高度与停经{gw}周妊娠相符，"
                + (f"可及宫缩，伴{sym_str}表现，" if syms else "胎心音规整，")
                + "双下肢无明显水肿"
            )
            spec = f"宫高腹围符合孕{gw}周，胎心音正常，专科检查与{safe_dx}相符"
        return pe, spec

    if "阑尾" in raw_dx:
        sev = "伴反跳痛及肌紧张" if _has_any(raw_dx, ("腹膜炎", "化脓", "穿孔", "坏疽")) else "无明显反跳痛"
        pe = f"心肺听诊未见明显异常，腹平坦，右下腹麦氏点压痛阳性，{sev}，肠鸣音减弱"
        spec = f"右下腹麦氏点固定压痛（+），{sev}，符合{safe_dx}腹部专科体征"
        return pe, spec

    if _has_any(raw_dx, ("腹膜炎", "穿孔", "肠梗阻", "胰腺炎", "胆囊炎", "胆石", "胆总管", "消化道")):
        loc = "右上腹" if "胆" in raw_dx else ("中上腹" if "胰" in raw_dx else "腹部")
        sev = "压痛、反跳痛及肌紧张" if _has_any(raw_dx, ("腹膜炎", "穿孔")) else "压痛"
        extra = "，".join(x for x in kb_signs if "腹" in x or "墨菲" in x or "肠鸣音" in x)[:24]
        sig_part = extra or f"{loc}{sev}"
        pe = f"双肺呼吸音清，心律齐，{sig_part}" + ("，Murphy征阳性" if "胆" in raw_dx and "墨菲" not in sig_part else "")
        spec = f"腹部专科查体见{sig_part}，与{safe_dx}表现相符"
        return pe, spec

    if _has_any(raw_dx, ("脑梗", "脑出血", "偏瘫", "卒中", "帕金森")) or "脑" in organs or "神经" in organs:
        focal = "病侧肢体肌力减退" if _has_any(sym_str, ("乏力", "偏瘫", "无力", "麻木")) else "神经系统定位体征与病情相符"
        pe = f"神志清楚，双侧瞳孔等大等圆，对光反射灵敏，{focal}，心肺腹查体未见明显异常"
        spec = f"神志清楚，颅神经及肢体肌力、肌张力、病理征检查与{safe_dx}定位表现一致"
        return pe, spec

    if _has_any(raw_dx, ("乳腺", "乳房", "疝", "骨折", "肿物", "肿块", "结节", "痔", "结石")) and not p["primary_dx"].get("treatment"):
        target = f"{side}{safe_dx.lstrip('左右双侧')}" if side and not safe_dx.startswith(("左", "右", "双")) else safe_dx
        if "痔" in raw_dx:
            pe = "心肺腹查体未见明显异常，肛周专科可见局部病变体征"
            spec = "胸膝位肛门视诊及指检可见肛缘病变体征，无明显活动性出血"
        elif "乳腺" in raw_dx or "乳房" in raw_dx:
            loc = side or "患侧"
            pe = f"心肺腹查体未见明显异常，{loc}乳腺可触及界清肿物，活动度可"
            spec = f"{loc}乳房局部可及肿物，表面皮肤无橘皮样变及乳头溢液，腋窝淋巴结未及肿大"
        elif "疝" in raw_dx:
            loc = side or "患侧"
            pe = f"心肺听诊未见明显异常，腹软，{loc}腹股沟区可见可复性肿物"
            spec = f"{loc}腹股沟区肿物站立位明显、平卧后可还纳，外环口可及冲击感"
        elif "结石" in raw_dx and "泌尿" in organs:
            loc = side or "患侧"
            pe = f"心肺听诊未见明显异常，腹平软，{loc}肾区叩击痛阳性，输尿管走行区轻压痛"
            spec = f"{loc}脊肋角及肾区叩击痛阳性，下腹部及会阴部无明显异常"
        else:
            pe = f"心肺腹查体未见明显异常，{target}局部查体存在相应阳性体征"
            spec = f"{p['dept']}专科查体见{target}局部相应体征"
        return pe, spec

    if "肺" in organs or "咽喉" in organs:
        lung_sign = "、".join(x for x in kb_signs if _has_any(x, ("啰音", "哮鸣音", "实变", "三凹")))[:24]
        if not lung_sign:
            lung_sign = "双肺呼吸音粗，可闻及湿啰音" if syms else "双肺呼吸音清"
        pe = f"{lung_sign}，心率规整，心律齐，腹平软无压痛"
        spec = f"{p['dept']}专科查体：{lung_sign}，胸廓对称无畸形"
        return pe, spec

    if "心" in organs:
        card_sign = "、".join(x for x in kb_signs if _has_any(x, ("啰音", "奔马律", "水肿", "颈静脉", "杂音")))[:24]
        if not card_sign:
            card_sign = "心律齐，心音可，双下肢无明显水肿" if not syms else "心界及心肺听诊与病情相符"
        pe = f"神志清楚，{card_sign}，腹平软无压痛"
        spec = f"心内科专科查体：{card_sign}"
        return pe, spec

    if syms:
        pe = f"神志清楚，查体合作，全身浅表淋巴结未触及肿大，查体见与{sym_str}相符的阳性体征"
    else:
        pe = "浅表淋巴结未触及肿大，心肺腹查体未见明显异常"
    if p["primary_dx"].get("treatment") and not syms:
        spec = f"{p['dept']}专科查体未见新发局部异常体征"
    else:
        spec = f"{p['dept']}专科查体与{'主要症状' if syms else safe_dx}相符"
    return pe, spec


def build_b_only_stats(kb=None):
    """Build a standalone 100% public mc-stats-0.21 dict from Route B KB + ICD/guideline priors (zero hospital data)."""
    kb = kb or load_default_kb() or {"entries": {}}
    entries = kb.get("entries") or {}

    dx_spec = {
        "K80": ("胆囊结石合并急性胆囊炎", "K", "肝胆外科", 6, True, "腹腔镜胆囊切除术", ["胆绞痛", "发热"], "K80.0_acute"),
        "K35": ("急性化脓性阑尾炎伴腹膜炎", "K", "普外科", 6, True, "腹腔镜阑尾切除术", ["转移性右下腹痛", "发热"], None),
        "K56": ("肠梗阻", "K", "普外科", 8, True, "肠粘连松解术", ["腹痛", "腹胀", "呕吐"], "K56"),
        "K85": ("轻症急性胰腺炎", "K", "消化内科", 8, False, None, ["腹痛", "恶心"], "K85"),
        "I20": ("不稳定型心绞痛", "I", "心内科", 6, True, "冠状动脉造影及支架植入术", ["胸痛", "胸闷"], "I20.0"),
        "I21": ("急性心肌梗死", "I", "心内科", 9, True, "经皮冠状动脉介入术", ["胸骨后压榨性疼痛", "大汗"], "I21"),
        "I50": ("心力衰竭", "I", "心内科", 8, False, None, ["气短", "乏力", "下肢水肿"], "I50"),
        "I63": ("脑梗死", "I", "神经内科", 9, False, None, ["肢体麻木", "言语不清"], "I63"),
        "I84": ("血栓性外痔", "I", "肛肠外科", 3, True, "血栓性外痔切除术", ["肛门剧痛"], "I84"),
        "J15": ("社区获得性肺炎", "J", "呼吸内科", 7, False, None, ["咳嗽", "咳痰", "发热"], "J15"),
        "J18": ("支气管肺炎", "J", "儿科", 6, False, None, ["发热", "咳嗽"], "J18.0"),
        "E11": ("2型糖尿病", "E", "内分泌科", 7, False, None, ["口渴", "多饮", "多尿"], "E11"),
        "N20": ("右侧输尿管结石", "N", "泌尿外科", 5, True, "经尿道右侧输尿管软镜钬激光碎石取石术", ["腰痛", "血尿"], "N20"),
        "Z51": ("结肠恶性肿瘤术后辅助化学治疗", "Z", "肿瘤内科", 4, False, None, [], "C18_chemo"),
        "O80": ("孕39周待产", "O", "产科", 4, True, "子宫下段剖宫产术", ["停经", "下腹阵痛"], None),
        "D24": ("右侧乳腺肿物", "D", "乳腺外科", 5, True, "右侧乳腺肿物切除术", ["右侧乳腺肿物"], None),
    }

    joint_items = {
        "男|40-65|K": 120, "女|40-65|K": 80, "男|18-40|K": 60,
        "男|65+|I": 120, "女|65+|I": 100, "男|40-65|I": 80,
        "男|40-65|J": 70, "女|40-65|J": 50, "男|1-6|J": 50, "女|1-6|J": 40,
        "女|40-65|E": 60, "男|40-65|N": 60, "女|40-65|Z": 50,
        "女|18-40|O": 60, "女|40-65|D": 50,
    }
    code3_by_ch = {}
    dx_names, dx_names_ch, dept, los_t, surg_t, procs, pfam = {}, {}, {}, {}, {}, {}, {}
    exams_t, syms_t, csyms_t, comorbid_t = {}, {}, {}, {}

    med_by_code = {
        "K80": [("抗菌药", "cefuroxime"), ("抑酸药", "omeprazole")],
        "K35": [("抗菌药", "ceftriaxone"), ("抗菌药", "metronidazole")],
        "K56": [("抗菌药", "cefuroxime"), ("抑酸药", "omeprazole")],
        "K85": [("抑酸药", "omeprazole"), ("解热镇痛抗炎药", "acetaminophen")],
        "I20": [("抗血小板药", "aspirin"), ("调脂药", "atorvastatin"), ("β受体阻滞剂", "metoprolol")],
        "I21": [("抗血小板药", "aspirin"), ("抗血小板药", "clopidogrel"), ("调脂药", "atorvastatin")],
        "I50": [("利尿剂", "furosemide"), ("RAS抑制剂", "valsartan"), ("β受体阻滞剂", "metoprolol")],
        "I63": [("抗血小板药", "aspirin"), ("调脂药", "atorvastatin")],
        "I84": [("解热镇痛抗炎药", "ibuprofen")],
        "J15": [("抗菌药", "ceftriaxone"), ("祛痰镇咳药", "ambroxol")],
        "J18": [("抗菌药", "azithromycin"), ("祛痰镇咳药", "ambroxol"), ("平喘药", "budesonide")],
        "E11": [("降糖药", "metformin"), ("降糖药", "acarbose")],
        "N20": [("α受体阻滞剂", "tamsulosin"), ("抗菌药", "cefuroxime")],
        "Z51": [("抗肿瘤药", "oxaliplatin"), ("抗肿瘤药", "capecitabine"), ("胃肠动力和止吐药", "ondansetron")],
        "O80": [("子宫收缩药", "oxytocin"), ("抗菌药", "cefuroxime")],
        "D24": [("抗菌药", "cefuroxime"), ("解热镇痛抗炎药", "ibuprofen")],
    }
    meds_t, med_cls_t = {}, {}

    for c3, (dname, ch, dp, dl, is_sg, proc, s_list, kb_id) in dx_spec.items():
        code3_by_ch.setdefault(ch, {"n": 0, "items": {}})
        code3_by_ch[ch]["items"][c3] = 50
        code3_by_ch[ch]["n"] += 50
        dx_names[c3] = {"n": 50, "items": {dname: 50}}
        dx_names_ch.setdefault(ch, {"n": 0, "items": {}})
        dx_names_ch[ch]["items"][dname] = 50
        dx_names_ch[ch]["n"] += 50
        dept[c3] = {"n": 50, "items": {dp: 50}}
        los_val = entries[kb_id]["los_max"] if (kb_id and kb_id in entries) else dl
        los_t[c3] = {"n": 50, "items": {str(max(3, min(los_val, dl))): 30, str(max(3, dl - 1)): 20}}
        surg_t[c3] = {"n": 50, "items": {"yes": 45 if is_sg else 0, "no": 5 if is_sg else 50}}
        if proc:
            procs[c3] = {"n": 50, "items": {proc: 50}}

        kb_ent = entries.get(kb_id) if kb_id else None
        ex_items = {}
        if kb_ent:
            for ex in kb_ent.get("exams_required") or ():
                ex_items[f"{ex['key']}|{ex.get('region', '')}"] = 48
            for ex in (kb_ent.get("exams_optional") or ())[:4]:
                ex_items[f"{ex['key']}|{ex.get('region', '')}"] = 22
        else:
            base_ex = {
                "K35": {"血常规|": 50, "C反应蛋白|": 45, "超声|腹": 42, "CT|腹": 40, "肝功能|": 35, "肾功能|": 35, "凝血|": 35, "心电图|": 35},
                "O80": {"血常规|": 50, "尿常规|": 45, "凝血|": 48, "肝功能|": 40, "超声|盆": 45, "心电图|": 40},
                "D24": {"血常规|": 50, "凝血|": 45, "肝功能|": 40, "超声|乳腺": 48, "心电图|": 40},
            }
            ex_items = base_ex.get(c3, {"血常规|": 50, "心电图|": 35})
        exams_t[c3] = {"n": 50, "items": ex_items}

        k_syms = list((kb_ent.get("chief_symptoms") or ()) if kb_ent else ()) or list(s_list)
        if s_list:
            cs_items = {s: 45 for s in s_list}
            all_s = {s: 45 for s in s_list}
            for s in k_syms:
                if s not in all_s and len(s) <= 8:
                    all_s[s] = 28
            csyms_t[c3] = {"n": 50, "items": cs_items}
            syms_t[c3] = {"n": 50, "items": all_s}
        else:
            syms_t[c3] = {"n": 50, "items": {"恶心": 20, "乏力": 20}}

        m_list = med_by_code.get(c3, [("抑酸药", "omeprazole")])
        meds_t[c3] = {"n": 50, "items": {did: 45 for _, did in m_list}}
        cls_counts = {}
        for cls_name, _ in m_list:
            cls_counts[cls_name] = 45
        med_cls_t[c3] = {"n": 50, "items": cls_counts}

    meds_t["kid:*"] = {"n": 100, "items": {"azithromycin": 60, "ambroxol": 70, "budesonide": 60, "ceftriaxone": 40, "ibuprofen": 30}}
    med_cls_t["kid:*"] = {"n": 100, "items": {"抗菌药": 80, "祛痰镇咳药": 70, "平喘药": 60}}

    drug_meta = {
        "ceftriaxone": ("头孢曲松", "2g", "qd", "静滴", "抗菌药"),
        "cefuroxime": ("头孢呋辛", "1.5g", "bid", "静滴", "抗菌药"),
        "metronidazole": ("甲硝唑", "0.5g", "bid", "静滴", "抗菌药"),
        "azithromycin": ("阿奇霉素", "0.1g", "qd", "静滴", "抗菌药"),
        "ambroxol": ("氨溴索", "15mg", "bid", "口服", "祛痰镇咳药"),
        "budesonide": ("布地奈德", "0.5mg", "bid", "雾化吸入", "平喘药"),
        "omeprazole": ("奥美拉唑", "40mg", "qd", "静滴", "抑酸药"),
        "aspirin": ("阿司匹林", "0.1g", "qd", "口服", "抗血小板药"),
        "clopidogrel": ("氯吡格雷", "75mg", "qd", "口服", "抗血小板药"),
        "atorvastatin": ("阿托伐他汀", "20mg", "qn", "口服", "调脂药"),
        "metoprolol": ("美托洛尔", "25mg", "bid", "口服", "β受体阻滞剂"),
        "furosemide": ("呋塞米", "20mg", "qd", "口服", "利尿剂"),
        "valsartan": ("缬沙坦", "80mg", "qd", "口服", "RAS抑制剂"),
        "metformin": ("二甲双胍", "0.5g", "bid", "口服", "降糖药"),
        "acarbose": ("阿卡波糖", "50mg", "tid", "口服", "降糖药"),
        "tamsulosin": ("坦索罗辛", "0.2mg", "qn", "口服", "α受体阻滞剂"),
        "oxaliplatin": ("奥沙利铂", "130mg", "qd", "静滴", "抗肿瘤药"),
        "capecitabine": ("卡培他滨", "1.5g", "bid", "口服", "抗肿瘤药"),
        "ondansetron": ("昂丹司琼", "8mg", "qd", "静滴", "胃肠动力和止吐药"),
        "oxytocin": ("缩宫素", "10U", "qd", "静滴", "子宫收缩药"),
        "ibuprofen": ("布洛芬", "0.2g", "prn", "口服", "解热镇痛抗炎药"),
        "acetaminophen": ("对乙酰氨基酚", "0.5g", "prn", "口服", "解热镇痛抗炎药"),
    }
    med_alias, med_dose, med_freq, med_route, med_class_drug, med_class_multi = {}, {}, {}, {}, {}, {}
    for did, (cn, dose, freq, route, cls_name) in drug_meta.items():
        med_alias[did] = {"n": 100, "items": {cn: 100}}
        med_dose[did] = {"n": 100, "items": {dose: 100}}
        med_freq[did] = {"n": 100, "items": {freq: 100}}
        med_route[did] = {"n": 100, "items": {route: 100}}
        med_class_drug.setdefault(cls_name, {"n": 0, "items": {}})
        med_class_drug[cls_name]["items"][did] = 50
        med_class_drug[cls_name]["n"] += 50
        med_class_multi.setdefault(cls_name, {"n": 100, "items": {"2+": 25 if cls_name in ("抗菌药", "抗血小板药", "降糖药", "抗肿瘤药") else 0}})

    comorbid_t = {
        "I": {"n": 200, "items": {"高血压病": 80, "2型糖尿病": 50, "高脂血症": 45, "低钾血症": 35}},
        "K": {"n": 200, "items": {"高血压病": 40, "脂肪肝": 45, "电解质紊乱": 35, "低蛋白血症": 30}},
        "J": {"n": 200, "items": {"高血压病": 35, "电解质紊乱": 30}},
        "Z": {"n": 100, "items": {"结肠恶性肿瘤术后": 90, "轻度贫血": 30}},
        "*": {"n": 1000, "items": {"高血压病": 150, "电解质紊乱": 80}},
    }

    return {
        "version": "mc-stats-0.21",
        "k": 11,
        "n_admissions": 1000,
        "_kb": kb,
        "joint": {"n": sum(joint_items.values()), "items": joint_items},
        "code3": dict(code3_by_ch, **{"*": {"n": 1000, "items": {c: 50 for c in dx_spec}}}),
        "dx_names": dx_names,
        "dx_names_ch": dx_names_ch,
        "dx_stems": {},
        "dept": dict(dept, **{"*": {"n": 1000, "items": {"内科": 400, "外科": 400, "妇产科": 100, "儿科": 100}}}),
        "los": dict(los_t, **{"*": {"n": 1000, "items": {"5": 300, "6": 400, "7": 300}}}),
        "surgery": dict(surg_t, **{"*": {"n": 1000, "items": {"yes": 400, "no": 600}}}),
        "surgery_day": {"n": 400, "items": {"1": 220, "2": 130, "3": 50}},
        "procedures": dict(procs, **{"*": {"n": 400, "items": {"腹腔镜胆囊切除术": 100}}}),
        "proc_family": pfam,
        "exams": dict(exams_t, **{"*": {"n": 1000, "items": {
            "血常规|": 950, "尿常规|": 700, "肝功能|": 800, "肾功能|": 800, "电解质|": 800, "心电图|": 750,
            "X线|胸": 350, "CT|胸": 220, "CT|腹": 150, "CT|头": 140, "CT|泌尿": 60, "X线|腹": 80,
            "超声|腹": 240, "超声|心": 180, "心脏超声|": 180, "超声|盆": 80, "超声|泌尿": 80, "超声|颈": 70, "超声|乳腺": 50,
            "磁共振|头": 90, "磁共振|腹": 40, "痰培养|": 100, "血培养|": 80, "冠脉造影|": 60, "动态血压|": 80, "动态心电图|": 80, "肺功能|": 70,
        }}}),
        "exam_kinds": {"*": {"n": 1000, "q": [5, 7, 8, 10, 13]}},
        "exam_days": {"血常规|": {"0": 800, "1": 150}, "心电图|": {"0": 800, "1": 100}, "CT|头": {"0": 700, "1": 200}, "CT|胸": {"0": 500, "1": 400}, "超声|腹": {"0": 600, "1": 300}},
        "exam_lags": {"kind:检验": {"0": 900, "1": 100}, "kind:功能": {"0": 950, "1": 50}, "kind:影像": {"0": 400, "1": 500, "2": 100}, "kind:培养": {"2": 300, "3": 600}, "kind:病理": {"3": 500, "4": 400}},
        "meds": dict(meds_t, **{"*": {"n": 1000, "items": {"omeprazole": 400, "ceftriaxone": 300}}}),
        "med_class": dict(med_cls_t, **{"*": {"n": 1000, "items": {"抑酸药": 400, "抗菌药": 400}}}),
        "med_class_drug": med_class_drug,
        "med_class_multi": med_class_multi,
        "med_alias": med_alias,
        "med_dose": med_dose,
        "med_freq": med_freq,
        "med_route": med_route,
        "symptoms": dict(syms_t, **{"*": {"n": 1000, "items": {"腹痛": 300, "发热": 300, "咳嗽": 200}}}),
        "chief_symptoms": csyms_t,
        "durations": {"*": {"n": 1000, "items": {"2天": 200, "3天": 350, "5天": 200, "1周": 150, "1个月": 100}}},
        "lab_state": {
            "I": {"n": 200, "items": {"TNI|abn": 140, "TNI|norm": 60, "BNP|abn": 130, "BNP|norm": 70, "CRP|abn": 90, "CRP|norm": 110}},
            "K35": {"n": 50, "items": {"WBC|abn": 42, "WBC|norm": 8, "NEUT|abn": 42, "NEUT|norm": 8, "CRP|abn": 44, "CRP|norm": 6}},
            "K85": {"n": 50, "items": {"AMY|abn": 42, "AMY|norm": 8, "WBC|abn": 35, "WBC|norm": 15}},
            "J": {"n": 200, "items": {"WBC|abn": 110, "WBC|norm": 90, "NEUT|abn": 100, "NEUT|norm": 100, "CRP|abn": 130, "CRP|norm": 70}},
            "K": {"n": 200, "items": {"WBC|abn": 95, "WBC|norm": 105, "ALT|abn": 70, "ALT|norm": 130, "TBIL|abn": 60, "TBIL|norm": 140, "AMY|abn": 70, "AMY|norm": 130}},
            "*": {"n": 1000, "items": {"WBC|abn": 320, "WBC|norm": 680, "NEUT|abn": 300, "NEUT|norm": 700, "HGB|abn": 150, "HGB|norm": 850, "PLT|abn": 120, "PLT|norm": 880, "K|abn": 100, "K|norm": 900, "NA|abn": 80, "NA|norm": 920}},
        },
        "labs": {
            "WBC": {"q": [3.8, 5.8, 7.9, 11.4, 16.5], "n": 900, "unit": "×10^9/L"},
            "NEUT": {"q": [45.0, 60.0, 72.0, 82.0, 91.0], "n": 900, "unit": "%"},
            "HGB": {"q": [92, 115, 128, 142, 160], "n": 900, "unit": "g/L"},
            "PLT": {"q": [110, 160, 215, 275, 345], "n": 900, "unit": "×10^9/L"},
            "CRP": {"q": [2.0, 5.0, 12.0, 35.0, 85.0], "n": 500, "unit": "mg/L"},
            "K": {"q": [3.3, 3.8, 4.1, 4.5, 5.0], "n": 800, "unit": "mmol/L"},
            "CREA": {"q": [52, 66, 80, 102, 168], "n": 600, "unit": "μmol/L"},
            "BUN": {"q": [3.4, 4.4, 5.4, 6.9, 10.5], "n": 600, "unit": "mmol/L"},
            "ALT": {"q": [12, 20, 28, 46, 115], "n": 600, "unit": "U/L"},
            "TBIL": {"q": [6.5, 11.0, 16.0, 26.0, 55.0], "n": 600, "unit": "μmol/L"},
            "TNI": {"q": [0.005, 0.012, 0.025, 0.08, 0.85], "n": 300, "unit": "ng/mL"},
            "BNP": {"q": [22, 48, 95, 240, 880], "n": 300, "unit": "pg/mL"},
            "AMY": {"q": [42, 58, 76, 138, 340], "n": 300, "unit": "U/L"},
        },
        "labs_by": {},
        "vitals": {
            "adult": {
                "T": {"q": [36.3, 36.5, 36.7, 37.0, 38.0], "n": 500},
                "P": {"q": [66, 74, 80, 88, 102], "n": 500},
                "R": {"q": [16, 18, 19, 20, 22], "n": 500},
                "SBP": {"q": [108, 118, 128, 140, 162], "n": 500},
                "DBP": {"q": [64, 72, 78, 84, 96], "n": 500},
            },
        },
        "past": {"40-65": {"n": 400, "items": {"高血压": 120, "糖尿病": 60}}, "65+": {"n": 300, "items": {"高血压": 130, "糖尿病": 70}}, "*": {"n": 1000, "items": {"高血压": 250}}},
        "comorbid": comorbid_t,
        "note_general": {"n": 500, "精神睡眠饮食": 0.25, "二便": 0.15},
        "note_vitals": {"rate": 0.65},
        "section_chars": {
            "入院记录|现病史": {"q": [120, 160, 200, 240, 280], "n": 200},
            "入院记录|体格检查": {"q": [90, 110, 130, 160, 200], "n": 200},
            "出院记录|诊疗经过": {"q": [160, 240, 320, 420, 550], "n": 200},
        },
        "notes_per_day": {"q": [0.4, 0.6, 0.8, 1.0, 1.1], "n": 300},
        "mentions_per_adm": {"q": [4, 6, 9, 13, 18], "n": 300},
        "p_dose_written": 0.45,
        "copy": {
            "全部": {"全部": {"reuse": 0.22, "exact": 0.15, "pairs": 500}},
            "文书细类": {
                "主治医师查房": {"reuse": 0.24, "exact": 0.16, "pairs": 150},
                "主任医师查房": {"reuse": 0.22, "exact": 0.15, "pairs": 150},
                "上级医师查房": {"reuse": 0.22, "exact": 0.15, "pairs": 100},
                "其他病程": {"reuse": 0.26, "exact": 0.18, "pairs": 200},
            },
            "天数间隔": {"0": {"reuse": 0.22, "exact": 0.15, "pairs": 150}, "1": {"reuse": 0.22, "exact": 0.15, "pairs": 200}, "2": {"reuse": 0.22, "exact": 0.15, "pairs": 80}, "3+": {"reuse": 0.25, "exact": 0.17, "pairs": 70}},
        },
    }
