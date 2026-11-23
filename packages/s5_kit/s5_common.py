# -*- coding: utf-8 -*-
"""Shared utilities for S5 Stage-1 planning, generation, and benchmarking.

Does not modify the frozen medcourse_core package in any way.
"""
import copy
import json
import os
import re
import sys


def setup_paths():
    """Register local, repo-sibling, or server paths for frozen medcourse_core and gate kits."""
    here = os.path.dirname(os.path.abspath(__file__))
    pkgs_dir = os.path.dirname(here)
    rules_dir = os.path.join(pkgs_dir, "rules")

    env_candidates = {
        "BENCH_G1": [os.path.join(rules_dir, "ruler_g1_core")],
        "BENCH_G3": [
            os.path.join(rules_dir, "g3_core"),
            "/home/weify/pyt/gates/S0/g3_core",
            r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\g3_core",
        ],
        "BENCH_G2": [
            os.path.join(rules_dir, "g2_core"),
            "/home/weify/pyt/gates/G3/g2_core",
            r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\g2_core",
        ],
        "BENCH_B7": [
            os.path.join(rules_dir, "b7_rules"),
            "/home/weify/pyt/gates/S0/b7_rules",
            r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\b7_rules",
        ],
        "BENCH_G45": [
            os.path.join(rules_dir, "g45_core"),
            "/home/weify/pyt/gates/Q45b/g45_core",
            r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\g45_core",
        ],
        "BENCH_G6": [
            os.path.join(rules_dir, "g6_core"),
            "/home/weify/pyt/gates/Q6/g6_core",
            r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\g6_core",
        ],
    }
    for key, paths in env_candidates.items():
        if not os.path.isdir(os.environ.get(key, "")):
            for p in paths:
                if os.path.isdir(p):
                    os.environ[key] = p
                    break
    candidates = [
        os.environ.get("MEDCOURSE_CORE_DIR", ""),
        os.environ.get("MC_CORE", ""),
        "/home/weify/pyt/product/v037_s5r2/medcourse_core",
        "/home/weify/pyt/product/v036_s5r2/medcourse_core",
        "/home/weify/pyt/product/v035_s5r2/medcourse_core",
        "/home/weify/pyt/product/v031_s3/medcourse_core",
        "/home/weify/pyt/gates/S0/bench_core",
        "/home/weify/pyt/gates/S0",
        os.path.join(pkgs_dir, "medcourse_core"),
        os.path.join(pkgs_dir, "bench_core"),
        r"D:\Panyto\Project\关卡测试归档\产品开发\S1\medcourse_core",
        r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\bench_core",
        os.path.join(rules_dir, "ruler_g1_core"),
        os.path.join(rules_dir, "g2_core"),
        os.path.join(rules_dir, "g3_core"),
        os.path.join(rules_dir, "b7_rules"),
        os.path.join(rules_dir, "g45_core"),
        r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\g2_core",
        r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\g3_core",
        r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\b7_rules",
        r"D:\Panyto\Project\关卡测试归档\MedCourse_repo\packages\rules\g45_core",
    ]
    for p in reversed(candidates):
        if p and os.path.isdir(p):
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)


setup_paths()

from medcourse import kb, planner  # noqa: E402


# All 17 pathways in kb_v22.json with Route A / Route B restriction specs.
# (kb_key, target_code3_for_A, name_filter_kws, fallback_ch, route_b_meta)
# route_b_meta = (dx_name, ch, dept, los, surg, proc, syms, meds, age_bands)
PATHWAY_SPECS = [
    ("C18_chemo", "Z51", ("维持性化学治疗", "手术后恶性肿瘤化学治疗", "化疗"), "Z",
     ("结肠恶性肿瘤术后辅助化学治疗", "Z", "肿瘤内科", 4, False, None, [],
      [("抗肿瘤药", "oxaliplatin"), ("抗肿瘤药", "capecitabine"), ("胃肠动力和止吐药", "ondansetron")],
      ("40-65", "65+"))),
    ("E11", "E11", ("2型糖尿病",), "E",
     ("2型糖尿病", "E", "内分泌科", 7, False, None, ["口渴", "多饮", "多尿"],
      [("降糖药", "metformin"), ("降糖药", "acarbose")],
      ("40-65", "65+"))),
    ("I20.0", "I20", ("不稳定型心绞痛",), "I",
     ("不稳定型心绞痛", "I", "心内科", 6, True, "冠状动脉造影及支架植入术", ["胸痛", "胸闷"],
      [("抗血小板药", "aspirin"), ("调脂药", "atorvastatin"), ("β受体阻滞剂", "metoprolol")],
      ("40-65", "65+"))),
    ("I21", "I21", ("心肌梗死",), "I",
     ("急性ST段抬高型心肌梗死", "I", "心内科", 9, True, "经皮冠状动脉介入术", ["胸骨后压榨性疼痛", "大汗"],
      [("抗血小板药", "aspirin"), ("抗血小板药", "clopidogrel"), ("调脂药", "atorvastatin")],
      ("40-65", "65+"))),
    ("I25", "I25", ("冠状动脉粥样硬化",), "I",
     ("冠状动脉粥样硬化性心脏病", "I", "心内科", 4, True, "冠状动脉造影术", ["胸闷", "胸痛"],
      [("抗血小板药", "aspirin"), ("调脂药", "atorvastatin"), ("β受体阻滞剂", "metoprolol")],
      ("40-65", "65+"))),
    ("I50", "I50", ("心力衰竭", "心功能不全"), "I",
     ("心力衰竭", "I", "心内科", 8, False, None, ["气短", "乏力", "下肢水肿"],
      [("利尿剂", "furosemide"), ("RAS抑制剂", "valsartan"), ("β受体阻滞剂", "metoprolol")],
      ("40-65", "65+"))),
    ("I63", "I63", ("脑梗死",), "I",
     ("脑梗死", "I", "神经内科", 9, False, None, ["肢体麻木", "言语不清"],
      [("抗血小板药", "aspirin"), ("调脂药", "atorvastatin")],
      ("40-65", "65+"))),
    ("I84", "I84", ("内痔", "外痔", "混合痔"), "I",
     ("血栓性外痔", "I", "肛肠外科", 3, True, "血栓性外痔切除术", ["肛门剧痛"],
      [("解热镇痛抗炎药", "ibuprofen")],
      ("18-40", "40-65"))),
    ("J15", "J15", ("肺炎",), "J",
     ("社区获得性肺炎", "J", "呼吸内科", 7, False, None, ["咳嗽", "咳痰", "发热"],
      [("抗菌药", "ceftriaxone"), ("祛痰镇咳药", "ambroxol")],
      ("40-65", "65+"))),
    ("J18.0", "J18", ("支气管肺炎", "肺炎"), "J",
     ("支气管肺炎", "J", "儿科", 6, False, None, ["发热", "咳嗽"],
      [("抗菌药", "azithromycin"), ("祛痰镇咳药", "ambroxol"), ("平喘药", "budesonide")],
      ("0-1", "1-6"))),
    ("J21", "J21", ("毛细支气管炎",), "J",
     ("急性毛细支气管炎", "J", "儿科", 6, False, None, ["喘憋", "咳嗽", "低热"],
      [("平喘药", "budesonide"), ("祛痰镇咳药", "ambroxol")],
      ("0-1", "1-6"))),
    ("K56", "K56", ("肠梗阻",), "K",
     ("肠梗阻", "K", "普外科", 8, True, "肠粘连松解术", ["腹痛", "腹胀", "呕吐"],
      [("抗菌药", "cefuroxime"), ("抑酸药", "omeprazole")],
      ("40-65", "65+"))),
    ("K80.0_acute", "K80", ("胆囊结石伴有急性胆囊炎", "胆囊结石伴急性胆囊炎", "急性胆囊炎"), "K",
     ("胆囊结石合并急性胆囊炎", "K", "肝胆外科", 6, True, "腹腔镜胆囊切除术", ["胆绞痛", "发热"],
      [("抗菌药", "cefuroxime"), ("抑酸药", "omeprazole")],
      ("40-65", "65+"))),
    ("K80.1_chronic", "K80", ("胆囊结石伴胆囊炎", "胆总管结石", "慢性胆囊炎"), "K",
     ("慢性胆囊炎", "K", "肝胆外科", 5, True, "腹腔镜胆囊切除术", ["右上腹隐痛", "腹胀"],
      [("抗菌药", "cefuroxime"), ("抑酸药", "omeprazole")],
      ("40-65", "65+"))),
    ("K85", "K85", ("急性胰腺炎",), "K",
     ("轻症急性胰腺炎", "K", "消化内科", 8, False, None, ["腹痛", "恶心"],
      [("抑酸药", "omeprazole"), ("解热镇痛抗炎药", "acetaminophen")],
      ("18-40", "40-65"))),
    ("N18", "N18", ("慢性肾脏病", "终末期肾病"), "N",
     ("慢性肾脏病5期", "N", "肾内科", 8, False, None, ["乏力", "水肿", "纳差"],
      [("利尿剂", "furosemide"), ("RAS抑制剂", "valsartan")],
      ("40-65", "65+"))),
    ("N20", "N20", ("肾结石", "输尿管结石"), "N",
     ("肾结石", "N", "泌尿外科", 5, True, "经皮肾镜碎石取石术", ["腰痛", "血尿"],
      [("α受体阻滞剂", "tamsulosin"), ("抗菌药", "cefuroxime")],
      ("18-40", "40-65"))),
]

SPEC_BY_KEY = {row[0]: row for row in PATHWAY_SPECS}

T2_EXCLUDE_PREFIXES = ("Z51", "O04", "O80")

ACUTE_SEVERITY_WORDS = (
    "呼吸衰竭", "心力衰竭", "心功能不全", "休克", "脓毒", "败血症",
    "弥漫性血管内凝血", "骨髓抑制", "急性肾", "消化道出血", "肺栓塞",
    "脑疝", "酸中毒", "腹膜炎", "低血容量",
)


def _strip_proc_name(name):
    return re.sub(r"（[^）]*）", "", name or "").strip()


def build_pathway_stats(arm, kb_key, stats_a=None, kb_data=None):
    """Build an in-memory mc-stats-0.21 dict restricted to a single kb_v22 pathway."""
    if kb_key not in SPEC_BY_KEY:
        raise KeyError(f"Unknown pathway key: {kb_key}")
    _, c3, name_kws, ch, rb = SPEC_BY_KEY[kb_key]
    dx_name, _, dept, dl, is_sg, proc, s_list, m_list, age_bands = rb
    kb_data = kb_data or kb.load_default_kb()
    entries = (kb_data or {}).get("entries") or {}
    kb_ent = entries.get(kb_key)

    if arm == "route_b":
        st = copy.deepcopy(kb.build_b_only_stats(kb_data))
        st["joint"] = {
            "n": len(age_bands) * 100,
            "items": {f"{sx}|{bd}|{ch}": 50 for sx in ("男", "女") for bd in age_bands},
        }
        st["code3"] = {ch: {"n": 100, "items": {c3: 100}}, "*": {"n": 100, "items": {c3: 100}}}
        st["dx_names"][c3] = {"n": 100, "items": {dx_name: 100}}
        st["dx_names_ch"][ch] = {"n": 100, "items": {dx_name: 100}}
        st["dept"][c3] = {"n": 100, "items": {dept: 100}}
        los_max = int((kb_ent or {}).get("los_max") or dl)
        d_lo = max(3, min(los_max, dl))
        d_hi = max(3, min(los_max, max(dl, d_lo + 2)))
        st["los"][c3] = {"n": 100, "items": {str(d_lo): 60, str(d_hi): 40}}
        st["surgery"][c3] = {"n": 100, "items": {"yes": 90 if is_sg else 0, "no": 10 if is_sg else 100}}
        if proc:
            st["procedures"][c3] = {"n": 100, "items": {_strip_proc_name(proc): 100}}
            if kb_ent and kb_ent.get("procedure"):
                st["procedures"][c3]["items"][_strip_proc_name(kb_ent["procedure"])] = 60

        ex_items = {}
        if kb_ent:
            for ex in kb_ent.get("exams_required") or ():
                ex_items[f"{ex['key']}|{ex.get('region', '')}"] = 96
            for ex in (kb_ent.get("exams_optional") or ())[:5]:
                ex_items[f"{ex['key']}|{ex.get('region', '')}"] = 44
        if not ex_items:
            ex_items = {"血常规|": 100, "心电图|": 70}
        st["exams"][c3] = {"n": 100, "items": ex_items}

        k_syms = list((kb_ent or {}).get("chief_symptoms") or ()) or list(s_list)
        if s_list:
            all_s = {s: 90 for s in s_list}
            for s in k_syms:
                if s not in all_s and len(s) <= 8:
                    all_s[s] = 56
            st["chief_symptoms"][c3] = {"n": 100, "items": {s: 90 for s in s_list}}
            st["symptoms"][c3] = {"n": 100, "items": all_s}
        else:
            st["symptoms"][c3] = {"n": 100, "items": {"恶心": 40, "乏力": 40}}

        st["meds"][c3] = {"n": 100, "items": {did: 90 for _, did in m_list}}
        cls_cnt = {}
        for cls_name, _ in m_list:
            cls_cnt[cls_name] = 90
        st["med_class"][c3] = {"n": 100, "items": cls_cnt}
        if kb_key == "C18_chemo":
            st["comorbid"]["Z"] = {"n": 100, "items": {"结肠恶性肿瘤术后": 95, "轻度贫血": 25}}
        elif age_bands == ("0-1", "1-6"):
            st["past"] = {"0-1": {"n": 50, "items": {}}, "1-6": {"n": 50, "items": {}}, "*": {"n": 100, "items": {}}}
        return st

    if stats_a is None:
        raise ValueError("stats_a is required for route_a_only / route_ab")
    st = copy.deepcopy(stats_a)
    if arm == "route_a_only":
        st.pop("_kb", None)
    elif arm == "route_ab":
        st["_kb"] = kb_data
    else:
        raise ValueError(f"Unsupported arm: {arm}")

    joint_filtered = {
        k: v for k, v in (st.get("joint") or {}).get("items", {}).items()
        if k.split("|")[2] == ch and (age_bands != ("0-1", "1-6") or k.split("|")[1] in ("0-1", "1-6", "6-18"))
    }
    if not joint_filtered:
        joint_filtered = {f"{sx}|{bd}|{ch}": 50 for sx in ("男", "女") for bd in age_bands}
    st["joint"] = {"n": sum(joint_filtered.values()), "items": joint_filtered}
    st["code3"][ch] = {"n": 100, "items": {c3: 100}}
    c3_pool = (((st.get("dx_names") or {}).get(c3) or {}).get("items") or {})
    if c3_pool:
        matched = {k: v for k, v in c3_pool.items() if any(w in k for w in name_kws)}
        if not matched and kb_key == "K80.1_chronic":
            matched = {k: v for k, v in c3_pool.items() if "急性" not in k}
        if matched:
            st["dx_names"][c3] = {"n": sum(matched.values()), "items": matched}
    elif arm == "route_ab":
        st["dx_names"][c3] = {"n": 100, "items": {dx_name: 100}}
    if kb_key == "C18_chemo":
        # Anchor Z51 comorbidity prior on colon surgery postoperative status when sampling C18_chemo.
        st.setdefault("comorbid", {})["Z"] = {"n": 100, "items": {"结肠恶性肿瘤术后": 90, "轻度贫血": 25}}
    return st


def build_oracle_prompt(j, spec, section_name, target_chars):
    """Build unmasked full-stay prompt exposing future labs/exams/surgery/diagnoses for Arm 5 (OracleLeak)."""
    p = j["patient"]
    draft = planner.journey_to_draft(j)
    obs_lines = []
    for o in draft["observations"]:
        obs_lines.append(f"- 第{o['order_day']}天开{o['name']}（第{o['result_day']}天回报）：{o['result']}")
    act_lines = []
    for a in j.get("procedures") or ():
        act_lines.append(f"- 第{a['decide_day']}天决定、第{a['do_day']}天行{a['name']}（{a.get('anesthesia', '')}，侧别：{a.get('side', '无')}）")
    med_lines = [
        f"- {m['name']} {m.get('dose', '')} {m.get('freq', '')} {m.get('route', '')}（第{m['start_day']}–{m['stop_day']}天）".strip()
        for m in (j.get("meds") or ())
    ]
    dx_lines = [f"- 第{x['day']}天 {x['name']} -> {x['state']}" for x in (j.get("dx_events") or ())]
    final_dx = "、".join(draft["final_diagnoses"])
    chief = draft["presentation"]["chief_complaint"]
    return (
        f"请为一位虚构患者写《{spec['doc_type_cn']}》的“{section_name}”一节，标称时间是住院第{spec['day']}天。\n"
        f"【患者】{p['sex']}，{p['age_text']}，{p['dept']}；住院总天数 {j['los']} 天\n"
        f"【全周期完整主诉与既往史】主诉：{chief}；既往史：{draft['presentation']['past_history'] or '无'}\n"
        f"【出院终局诊断】{final_dx}\n"
        "【全周期诊断演进记录】\n" + ("\n".join(dx_lines) or "- 无") + "\n"
        "【全周期所有检查及已出结果】\n" + ("\n".join(obs_lines) or "- 无") + "\n"
        "【全周期手术与操作安排】\n" + ("\n".join(act_lines) or "- 无手术") + "\n"
        "【全周期用药清单】\n" + ("\n".join(med_lines) or "- 无") + "\n"
        "【要求】\n"
        f"1. 约 {target_chars} 字，写成连贯流畅的中文临床病历正文，不要分条、不要输出小标题或方括号占位符。\n"
        "2. 可自由结合患者本次住院全部诊疗信息、检查结论与最终诊断及手术结局进行全面分析。"
    )


def compute_smoke10_fingerprint(core_dir, plan_journeys_path, served_model, cfg_tag):
    """Compute SHA256 binding medcourse_core archive/manifest, shard_0 planned stays, model, and vLLM cfg."""
    import hashlib

    h = hashlib.sha256()
    if core_dir:
        cdir = os.path.abspath(core_dir)
        for p in (
            os.path.join(os.path.dirname(cdir), "medcourse_core_v0.3.6.tar.gz"),
            os.path.join(cdir, "MANIFEST.sha256"),
            os.path.join(cdir, "VERSION"),
        ):
            if os.path.isfile(p):
                h.update(os.path.basename(p).encode("utf-8"))
                h.update(open(p, "rb").read())
    if plan_journeys_path and os.path.isfile(plan_journeys_path):
        h.update(b"|journeys|")
        h.update(open(plan_journeys_path, "rb").read())
    h.update(f"|model={served_model}|cfg={cfg_tag}".encode("utf-8"))
    return h.hexdigest()


def smoke10_stamp_valid(stamp_path, expected_fingerprint):
    """Return True only when an atomic smoke10 pass stamp exists, matches inputs, and passed all gates."""
    if not stamp_path or not os.path.isfile(stamp_path):
        return False
    try:
        d = json.load(open(stamp_path, encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(d, dict) or d.get("passed") is not True:
        return False
    if d.get("input_fingerprint") != expected_fingerprint:
        return False
    try:
        return (
            int(d.get("llm_errors", 1)) == 0
            and float(d.get("discard_rate", 1.0)) <= 0.25
            and int(d.get("delivered_adm_gate_fails", 1)) == 0
            and int(d.get("delivered_oracle_any_hits", 1)) == 0
        )
    except Exception:
        return False


def record_smoke10_attempt_and_stamp(attempt_path, stamp_path, counts_dict, fingerprint):
    """Write diagnostic attempt JSON first; only atomically write stamp after all gates pass."""
    import datetime as dt

    rec = dict(counts_dict)
    rec["input_fingerprint"] = fingerprint
    rec["recorded_at_utc"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(os.path.dirname(os.path.abspath(attempt_path)), exist_ok=True)
    with open(attempt_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    llm_err = int(rec.get("llm_errors", 0))
    disc = float(rec.get("discard_rate", 1.0))
    adm_fails = int(rec.get("delivered_adm_gate_fails", 0))
    ora_hits = int(rec.get("delivered_oracle_any_hits", 0))
    ok = (llm_err == 0 and disc <= 0.25 and adm_fails == 0 and ora_hits == 0)
    if not ok:
        if stamp_path and os.path.exists(stamp_path):
            os.unlink(stamp_path)
        assert llm_err == 0, f"smoke10_llm_errors={llm_err}"
        assert disc <= 0.25, f"smoke10_discard_rate={disc}"
        assert adm_fails == 0, f"smoke10_adm_fails={adm_fails}"
        assert ora_hits == 0, f"smoke10_oracle_hits={ora_hits}"

    stamp = dict(rec, passed=True)
    tmp = f"{stamp_path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stamp, f, ensure_ascii=False, indent=2)
    os.replace(tmp, stamp_path)
    return stamp
