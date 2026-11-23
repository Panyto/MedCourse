# -*- coding: utf-8 -*-
"""Zero-prior same-backbone baseline generator (DIRECT-32B).

Strictly isolated from hospital statistics (stats_v021.json), KB priors (kb_v22.json),
planner, compiler, and epistemic/clinical gates. Uses only public clinical archetype
prompts and the same Qwen3-32B-AWQ model with standardized multi-document formatting.
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from s5_common import setup_paths

setup_paths()

from medcourse import writer  # noqa: E402
from medcourse.contract import admission_line  # noqa: E402
from medcourse.llm import Client, LLMError, WRITE  # noqa: E402


PUBLIC_ARCHETYPES = [
    ("普通外科一区", "急性腹痛伴发热，疑似急性胆囊炎或胆石症，需完善超声/CT评估是否需腹腔镜手术", (28, 72), "any", (4, 8)),
    ("普通外科二区", "转移性右下腹痛，待排急性阑尾炎与回盲部炎症，视影像及血象评估是否手术", (18, 65), "any", (3, 7)),
    ("普通外科三区", "肛周肿痛或便血加重，疑似血栓性外痔/肛周脓肿，评估保守与手术时机", (22, 62), "any", (3, 5)),
    ("胃肠外科一区", "腹痛腹胀伴停止排气排便，疑似肠梗阻，需鉴别粘连性与占位性并评估减压/手术", (35, 78), "any", (5, 11)),
    ("肝胆外科一区", "右上腹绞痛伴黄疸待查，需行MRCP/彩超鉴别胆总管结石并择期取石评估", (35, 75), "any", (5, 10)),
    ("肝胆外科二区", "上腹剧痛伴血淀粉酶升高待查，需鉴别急性胰腺炎及胆源性因素", (28, 68), "any", (6, 12)),
    ("泌尿外科一区", "突发腰痛伴肉眼血尿，疑似上尿路结石，需完善CTU及评估体外碎石或镜下取石", (24, 65), "any", (4, 9)),
    ("泌尿外科二区", "排尿困难伴前列腺增生或膀胱占位待查，入院行影像及膀胱镜评估手术指征", (55, 82), "男", (4, 9)),
    ("胸外科一区", "肺部结节或胸闷查因，需完善增强CT及肺功能评估微创胸腔镜手术可行性", (42, 74), "any", (5, 9)),
    ("骨科一区", "外伤后髋部或四肢疼痛活动受限，待排骨折与软组织损伤，评估内固定手术方案", (25, 80), "any", (5, 10)),
    ("心内科一区", "反复发作性胸痛胸闷，待排不稳定型心绞痛或非ST段抬高型心梗，行心肌酶及冠脉评估", (45, 78), "any", (4, 8)),
    ("心内科二区", "突发胸骨后压榨样疼痛，疑诊急性ST段抬高型心肌梗死，急诊评估再灌注治疗", (45, 80), "any", (5, 10)),
    ("心内科三区", "活动后气促、双下肢水肿加重，心力衰竭急性失代偿，明确诱因并调整抗心衰方案", (52, 84), "any", (6, 12)),
    ("心内科四区", "高血压控制不佳伴头晕心悸，排查继发性高血压及靶器官损害", (40, 76), "any", (4, 8)),
    ("心内科五区", "阵发性心悸胸闷，心房颤动或室上速待查，评估抗凝与射频消融指征", (48, 78), "any", (4, 8)),
    ("神经内科一区", "突发一侧肢体无力伴言语含糊，急性脑梗死与TIA/出血鉴别及二级预防评估", (50, 82), "any", (6, 13)),
    ("神经内科二区", "反复头晕行走不稳待查，排查后循环缺血与前庭外周性眩晕", (46, 78), "any", (4, 8)),
    ("呼吸内科一区", "咳嗽咳痰伴发热气促，社区获得性肺炎排查非典型病原体及重症倾向", (30, 80), "any", (5, 11)),
    ("呼吸内科二区", "慢性咳嗽气短急性加重，慢性阻塞性肺疾病急性加重期抗感染及呼吸支持", (58, 84), "any", (6, 12)),
    ("消化内科一区", "黑便或呕血待查，疑似上消化道出血，行急诊胃镜与抑酸止血治疗评估", (35, 76), "any", (4, 9)),
    ("消化内科二区", "反复中上腹痛伴消瘦纳差，鉴别慢性胃炎/消化性溃疡与消化道肿瘤", (42, 76), "any", (4, 8)),
    ("内分泌科一区", "口渴多饮消瘦伴血糖显著升高，2型糖尿病评估急慢性并发症及胰岛素方案调整", (36, 74), "any", (5, 10)),
    ("内分泌科二区", "糖尿病足合并局部感染或酮症酸中毒倾向，多学科创面评估与强化降糖", (48, 78), "any", (6, 12)),
    ("肾内科一区", "乏力水肿伴血肌酐升高，评估慢性肾脏病分期、急性加重因素及透析导入时机", (38, 78), "any", (5, 11)),
    ("肾内科二区", "蛋白尿、镜下血尿查因，排除继发性肾病并评估肾活检指征", (22, 58), "any", (5, 9)),
    ("肿瘤内科一区", "结肠恶性肿瘤术后规范辅助化疗，评估化疗前骨髓储备与不良反应监测", (44, 74), "any", (3, 6)),
    ("肿瘤内科二区", "肺恶性肿瘤常规维持化疗或免疫治疗周期，排查感染与免疫相关不良事件", (48, 76), "any", (3, 6)),
    ("肿瘤内科三区", "乳腺或胃部恶性肿瘤术后周期性复查及辅助化疗入院", (40, 72), "any", (3, 6)),
    ("妇科一区", "下腹痛或异常子宫出血待查，鉴别子宫肌瘤、卵巢囊肿扭转或盆腔炎并评估手术", (24, 56), "女", (4, 8)),
    ("产科一区", "妊娠晚期血压升高或先兆临产，评估妊娠期高血压疾病及分娩方式", (23, 39), "女", (4, 7)),
    ("儿科一区", "小儿发热咳嗽喘息待查，鉴别支气管肺炎与毛细支气管炎并动态评估呼吸状态", (1, 10), "any", (4, 9)),
    ("儿科二区", "儿童反复腹痛呕吐待查，鉴别急性胃肠炎、肠套叠或过敏性紫癜", (3, 14), "any", (3, 7)),
    ("甲乳外科一区", "甲状腺占位或乳腺结节拟行进一步超声穿刺及外科切除指征评估", (26, 66), "any", (3, 6)),
    ("神经外科一区", "头痛呕吐伴影像学占位/硬膜下积液待查，评估手术减压指征", (35, 76), "any", (5, 10)),
    ("血管外科一区", "单侧下肢肿胀疼痛待查，急诊超声排查深静脉血栓形成及抗凝管理", (40, 78), "any", (4, 8)),
    ("血液内科一区", "重度贫血或血小板减少查因，完善骨髓形态学排查良恶性血液病", (25, 72), "any", (5, 10)),
]

_DOC_HDR_RE = re.compile(r"={2,}\s*DOC\|([^\r\n]+?)\s*={2,}")
_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")

_ROLE_ORDER = {"HR": 0, "FP": 1, "PRE": 2, "POST": 3, "SUP": 4, "DAILY": 5, "DS": 6}


def _age_text(age_val):
    return f"{age_val}岁"


def sample_direct_cases(n, seed, shard_idx=0):
    """Deterministically build zero-prior clinical case descriptors from PUBLIC_ARCHETYPES."""
    rng = random.Random(int(seed) + int(shard_idx) * 10007)
    cases = []
    order = list(range(len(PUBLIC_ARCHETYPES)))
    for i in range(n):
        if i % len(order) == 0:
            rng.shuffle(order)
        dept, ctx, (alo, ahi), sex_rule, (llo, lhi) = PUBLIC_ARCHETYPES[order[i % len(order)]]
        sex = sex_rule if sex_rule in ("男", "女") else rng.choice(("男", "女"))
        age = rng.randint(alo, ahi)
        los = rng.randint(llo, lhi)
        aid = f"D32S{int(shard_idx)}-{seed}-{i:05d}"
        cases.append({
            "admission_id": aid,
            "idx": i,
            "dept": dept,
            "clinical_context": ctx,
            "sex": sex,
            "age": age,
            "age_text": _age_text(age),
            "los": los,
        })
    return cases


def build_direct_prompt(case):
    """Build single zero-prior whole-stay generation prompt obeying NHC chart norms."""
    n_daily = max(2, min(4, case["los"] - 2))
    return (
        f"请为一位虚构住院患者撰写完整住院病历序列，严格遵守《病历书写基本规范》。\n"
        f"【患者基本信息】科室：{case['dept']}；性别：{case['sex']}；年龄：{case['age_text']}；"
        f"入院日期：2024年；预计住院总天数：{case['los']}天（第0天入院，第{case['los']}天出院）。\n"
        f"【入院临床情境】{case['clinical_context']}。\n"
        f"【输出格式要求（必须严格遵守分隔标记）】\n"
        f"请按住院时间先后顺序输出共 {n_daily + 3} 篇文书，每篇文书必须以单独一行标记开头：\n"
        f"1) ===DOC|doc_type=hospital_record|role=HR|day=0|title=入院记录===\n"
        f"   须含小节：主诉、现病史、既往史、个人史与家族史、体格检查、辅助检查、入院诊断（疑似诊断可加？）。\n"
        f"2) ===DOC|doc_type=progress_note|role=FP|day=0|title=首次病程记录===\n"
        f"   须含小节：病例特点、拟诊讨论（列出鉴别诊断）、诊疗计划。\n"
        f"3) 按天数递增输出 {n_daily} 篇查房/术前/术后病程记录（day 从 1 到 {case['los'] - 1}，例如：\n"
        f"   ===DOC|doc_type=progress_note|role=DAILY|day=1|title=病程记录===\n"
        f"   如涉及上级医师查房可用 role=SUP，术前小结可用 role=PRE，术后首次病程可用 role=POST）。\n"
        f"4) ===DOC|doc_type=discharge_summary|role=DS|day={case['los']}|title=出院记录===\n"
        f"   须含小节：入院情况、诊疗经过、出院情况、出院诊断、出院医嘱。\n"
        f"要求中文临床表达自然规范，体征、检查结果与药物剂量合理，时间线符合日常住院逻辑，切勿输出解释或多余前缀。"
    )


def _infer_doc_fields(meta, target_los):
    role = (meta.get("role") or "").strip().upper()
    title = (meta.get("title") or "").strip()
    doc_type = (meta.get("doc_type") or "").strip().lower()
    if doc_type in ("hospital_record", "hr") or role == "HR" or "入院记录" in title:
        doc_type = "hospital_record"
        role = "HR"
    elif doc_type in ("discharge_summary", "ds") or role == "DS" or "出院记录" in title:
        doc_type = "discharge_summary"
        role = "DS"
    else:
        doc_type = "progress_note"
        if role not in ("FP", "SUP", "DAILY", "PRE", "POST"):
            if "首次" in title:
                role = "FP"
            elif "术前" in title:
                role = "PRE"
            elif "术后" in title:
                role = "POST"
            elif any(w in title for w in ("查房", "主任", "主治")):
                role = "SUP"
            else:
                role = "DAILY"
    try:
        day = int(str(meta.get("day", "0")).strip())
    except ValueError:
        day = target_los if doc_type == "discharge_summary" else 0
    day = max(0, min(max(30, target_los), day))
    if doc_type == "hospital_record":
        day = 0
    doc_type_cn = title or (
        "入院记录" if doc_type == "hospital_record" else ("出院记录" if doc_type == "discharge_summary" else "病程记录")
    )
    return doc_type, role, day, doc_type_cn


def parse_direct_stay(raw_text, target_los):
    """Parse ===DOC|...=== blocks into contract-compatible chronological docs. Returns (docs, reason)."""
    clean_raw = _FENCE_RE.sub("", (raw_text or "").strip())
    matches = list(_DOC_HDR_RE.finditer(clean_raw))
    if len(matches) < 3:
        return None, "too_few_doc_headers"

    docs = []
    for idx, m in enumerate(matches):
        hdr = m.group(1)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(clean_raw)
        body = clean_raw[start:end].strip()
        body = re.sub(r"^\s*===+\s*", "", body).strip()
        if len(body) < 40:
            continue
        meta = {}
        for part in hdr.split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k.strip().lower()] = v.strip()
        doc_type, role, day, doc_type_cn = _infer_doc_fields(meta, target_los)
        docs.append({
            "doc_type": doc_type,
            "day_offset": day,
            "text": body,
            "doc_type_cn": doc_type_cn,
            "role": role,
        })

    if len(docs) < 3:
        return None, "too_few_nonempty_docs"
    has_hr = any(d["doc_type"] == "hospital_record" and d["day_offset"] == 0 for d in docs)
    has_pn = any(d["doc_type"] == "progress_note" for d in docs)
    has_end = any(d["doc_type"] == "discharge_summary" or d["day_offset"] >= 2 for d in docs)
    if not (has_hr and has_pn and has_end):
        return None, "missing_required_doc_roles"
    total_chars = sum(len(d["text"]) for d in docs)
    if total_chars < 400:
        return None, "too_short_total"

    docs.sort(key=lambda d: (d["day_offset"], _ROLE_ORDER.get(d["role"], 5)))
    for k, d in enumerate(docs):
        d["k"] = k
    return docs, None


def _mock_generate_text(case):
    los = case["los"]
    mid = max(1, los // 2)
    is_surg_ctx = any(w in case["clinical_context"] for w in ("胆囊", "阑尾", "肠梗阻", "结石", "痔", "骨折", "手术"))
    if is_surg_ctx:
        day_mid = (
            f"===DOC|doc_type=progress_note|role=SUP|day={mid}|title=主治医师查房记录===\n"
            f"主治医师查房，患者腹部CT提示病灶明确，保守观察症状未完全缓解，考虑诊断明确，今日拟行手术评估，拟于明日在全麻下行微创切除术或内镜处理，已告知手术风险。"
        )
        day_post = (
            f"===DOC|doc_type=progress_note|role=POST|day={max(2, los - 1)}|title=术后首次病程记录===\n"
            f"患者今日在全麻下行腹腔镜下病灶切除术，术程顺利，术后予头孢呋辛钠 1.5g ivgtt q12h 预防感染及奥美拉唑 40mg ivgtt qd 护胃支持治疗。"
        )
    else:
        day_mid = (
            f"===DOC|doc_type=progress_note|role=SUP|day={mid}|title=主治医师查房记录===\n"
            f"今日主治医师查房，实验室与影像回报支持原发病诊断，已基本排除急诊外科急腹症可能，继续予规范内科治疗。予莫西沙星 0.4g ivgtt qd 或阿司匹林 100mg po qd 对症支持。"
        )
        day_post = (
            f"===DOC|doc_type=progress_note|role=DAILY|day={max(2, los - 1)}|title=病程记录===\n"
            f"患者生命体征平稳，症状明显改善，复查炎症及脏器功能指标较前好转，拟进一步巩固后评估出院。"
        )
    return "\n".join([
        "===DOC|doc_type=hospital_record|role=HR|day=0|title=入院记录===",
        f"主诉：因{case['clinical_context'][:16]}入院。",
        f"现病史：患者{case['sex']}性，{case['age_text']}，入院前1天出现明显不适来诊，门诊拟诊收入{case['dept']}。",
        "既往史：否认传染病史，既往慢性病史不详，无药物过敏史。",
        "体格检查：T 36.8℃，P 84次/分，R 20次/分，BP 126/78mmHg，神清，专科查体见相应体征。",
        "辅助检查：急诊血常规及心电图初步检查未见危急值。",
        "入院诊断：待明确诊断？",
        "===DOC|doc_type=progress_note|role=FP|day=0|title=首次病程记录===",
        f"病例特点：1. 老年/成年{case['sex']}性；2. 急性或亚急性起病；3. 初步考虑{case['clinical_context'][:20]}。",
        "拟诊讨论：综合症状体征初步怀疑原发病可能，同时需与急诊外科危重疾患相鉴别，目前暂不除外其他合并症。",
        "诊疗计划：1. 完善血常规、肝肾功能、CT等辅助检查；2. 暂予对症支持，监测生命体征，必要时调整方案。",
        "===DOC|doc_type=progress_note|role=DAILY|day=1|title=病程记录===",
        "患者夜间休息可，诉不适略缓解。血常规及生化指标回报中，继续原方案观察，待影像回报后决定下一步治疗。",
        day_mid,
        day_post,
        f"===DOC|doc_type=discharge_summary|role=DS|day={los}|title=出院记录===",
        f"入院情况：患者因{case['clinical_context'][:16]}入院，生命体征尚平稳。",
        "诊疗经过：入院后完善检查明确诊断，予规范抗感染/护胃或手术对症治疗后病情好转。",
        "出院情况：一般情况可，无明显不适，准予出院。",
        f"出院诊断：{case['dept']}主诊断已明确。",
        "出院医嘱：门诊定期随访，不适随诊，注意休息及规范用药。",
    ])


def _seed_of(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def make_admission_record(case, docs):
    """Wrap parsed docs into a contract-compatible admission dict with zero prior leakage."""
    rec = admission_line({"journey_id": case["admission_id"], "admit_date": "2024-06-15"}, docs)
    rec["meta"]["s5_arm"] = "direct_32b"
    rec["meta"]["dept"] = case["dept"]
    rec["meta"]["sex"] = case["sex"]
    rec["meta"]["age"] = case["age"]
    rec["meta"]["target_los"] = case["los"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261230)
    ap.add_argument("--writer", choices=("llm", "mock"), default="llm")
    ap.add_argument("--base-url", default="http://127.0.0.1:8105/v1")
    ap.add_argument("--model", default="Qwen3-32B-AWQ")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=3600)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = sample_direct_cases(args.n, args.seed, shard_idx=args.shard_idx)
    tl = threading.local()
    lock = threading.Lock()
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "llm_errors": 0}
    t0 = time.perf_counter()

    def _get_client():
        c = getattr(tl, "client", None)
        if c is None:
            c = Client(args.base_url, args.model, timeout=360, retries=2)
            tl.client = c
        return c

    def _run_case(case):
        prompt = build_direct_prompt(case)
        docs = None
        last_reason = "init"
        for attempt in range(2):
            seed = _seed_of(args.seed, case["admission_id"], attempt) % (2 ** 31)
            if args.writer == "mock":
                raw_txt = _mock_generate_text(case)
                with lock:
                    stats["calls"] += 1
            else:
                use_prompt = prompt if attempt == 0 else (
                    prompt + "\n注意：上一次格式解析失败，请务必严格以 ===DOC|doc_type=...|role=...|day=...|title=...=== 独立成行分隔每篇文书。"
                )
                try:
                    r = _get_client().generate(
                        messages=[
                            {"role": "system", "content": writer.SYSTEM},
                            {"role": "user", "content": use_prompt},
                        ],
                        max_tokens=args.max_tokens,
                        sampling=WRITE,
                        seed=seed,
                    )
                except LLMError as e:
                    with lock:
                        stats["llm_errors"] += 1
                    return None, f"llm_{e}"
                with lock:
                    stats["calls"] += 1
                    stats["prompt_tokens"] += int(r.get("prompt_tokens", 0))
                    stats["completion_tokens"] += int(r.get("completion_tokens", 0))
                raw_txt = r.get("text", "")

            parsed, reason = parse_direct_stay(raw_txt, case["los"])
            if parsed is not None:
                docs = parsed
                break
            last_reason = reason or "parse_failed"

        if docs is None:
            return None, last_reason

        return make_admission_record(case, docs), None

    if args.writer == "mock" or args.concurrency <= 1:
        outcomes = [_run_case(c) for c in cases]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            outcomes = list(ex.map(_run_case, cases))

    kept_rows = []
    discards = []
    for case, (rec, err) in zip(cases, outcomes):
        if rec is not None:
            kept_rows.append(rec)
        else:
            discards.append((case["admission_id"], err))

    with open(os.path.join(args.out, "admissions.jsonl"), "w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "discard.csv"), "w", encoding="utf-8") as f:
        f.write("admission_id,reason\n")
        for aid, rsn in discards:
            f.write(f"{aid},{rsn}\n")

    wall_s = round(time.perf_counter() - t0, 2)
    counts = {
        "version": "direct_32b_v1",
        "s5_arm": "direct_32b",
        "shard_idx": args.shard_idx,
        "seed": args.seed,
        "writer": args.writer,
        "model": args.model,
        "planned": len(cases),
        "kept": len(kept_rows),
        "discarded": len(discards),
        "llm_errors": stats["llm_errors"],
        "calls": stats["calls"],
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "wall_s": wall_s,
    }
    with open(os.path.join(args.out, "counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    print(json.dumps({"tag": "direct_32b_shard_done", **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
