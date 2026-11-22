"""S1 主流程：抽病人 → 排逐日事件 → 切当天可见信息 → 分节写 → 闸门，不过就重写。

python -B -m medcourse.pipeline --stats stats_v0.json --mode free --n 500 --writer template --out 目录
python -B -m medcourse.pipeline --stats stats_v0.json --mode paired --slots slots.jsonl --ids ids.txt --writer llm \
    --base-url http://127.0.0.1:8101/v1 --model Qwen3-8B-Base --raw --out 目录

产物：admissions.jsonl（AdmissionJSONL，bench 直接读）、journeys.jsonl、drafts.jsonl（oracle2 量泄露用）、counts.json、discard.csv。
配对模式用了真实槽位，产物不出 ys。自由模式的文书（admissions.jsonl 里的正文）从 2026-11-05 起按守则 v1.6 可以经 synth_export
每次抽不超过 20 份带回本地；journeys.jsonl、drafts.jsonl 是逐住院的计划，仍不出 ys。counts.json、discard.csv 只有计数。

0.5（v0.1.6）：--prev-note 对照臂；按病程"今日重点"的类别计数（focus_*）。
0.6（v0.1.7）：--no-filler 对照臂；首次过住院级闸门的失败码计数（first_fail_*）。
0.7（v0.1.8）：--no-filler2 对照臂（不点任何套话的名）；节级三次都没过的 truncated、repeat、nonsurg_postop 由 gate.repair_section
修补再交出（sec_*_repaired 计数）。
0.8（v0.1.9）：节级修补多 pe_history、hint_leak 两项；拷贝不拷症状词（compiler.copy_banned）；入院记录写完把体格检查的摘录
（去掉入院以后的话）交给出院记录"入院情况"当事实（ctx）；入院记录在住院级重写时，出院记录跟着重写，两边的查体对得上。
审查员第八轮补充：专科情况拿同一篇刚写好的体格检查摘录当事实（pe_ref）。
"""
import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import compiler, gate, kb, planner, writer
from .contract import admission_line
from .llm import LLMError
from .safe import SafeError, guarded, log

MAX_SECTION_TRIES = 3
MAX_DOC_REWRITES = 2
VERSION = "s1-pipeline-0.3.4"  # 0.34：临床防穿帮微封口（非手术切口/未分娩产褥/器官映像与部位补全/Z51跨科肿瘤过滤/儿童与新生儿约束）
PE_EXCERPT, SPEC_EXCERPT = 300, 60   # 0.1.11：出院记录"入院情况"的摘录 120 → 300 字（第十次合成病历 M6，阳性体征在 120 字以后）
PE_REF = 400                         # 0.1.11：专科情况提示里给的体格检查全文上限
_SENT_END = "。！？!?"


def _excerpt(text, n):
    """前 n 字以内、截在句号上的一段；第一句就超过 n 字时截到 n 字补句号。"""
    t = (text or "").strip()
    if len(t) <= n:
        return t
    cut = max(t.rfind(x, 0, n + 1) for x in _SENT_END)
    return t[:cut + 1] if cut > 0 else t[:n].rstrip("，,、；; ") + "。"


def hr_context(texts):
    """入院记录写出的体格检查摘录（不到 40 字时补专科情况的），去掉带入院以后说法的分句（gate.strip_admission_leak），
    给出院记录"入院情况"当事实。专科情况是照着体格检查写的（pe_ref），一般不重复摘。"""
    parts = [_excerpt(texts.get("体格检查"), PE_EXCERPT)]
    if len(parts[0]) < 40:
        parts.append(_excerpt(texts.get("专科情况"), SPEC_EXCERPT))
    pe = "".join(gate.strip_admission_leak(x) for x in parts if x)
    return {"hr_pe": pe} if pe else {}


def write_doc(j, spec, specs, stats, w, rng, tally, prev_note, round_=0, ctx=None, first_doc_out=None):
    """写一篇。每节最多采 3 次，仍不过就取失败最少的那次，交给住院级闸门。"""
    epistemic = not getattr(w, "ablate_epistemic", False)
    plans = compiler.compile_doc(j, spec, specs, stats, rng, ctx, epistemic=epistemic)
    texts, first_texts = {}, {}
    for pl in plans:
        if pl["mode"] == "program":
            continue
        if spec["role"] != "DS":
            vday = 0 if pl.get("admission_only") else spec["day"]
            vis = compiler.visible(j, vday)
            # 0.1.12.1（老板 11-10 批准）：这一节里结果口吻的检查只能是截至当天已回报的（"入院情况"这类只写入院当时的节到第 0 天）
            pl["result_keys"] = gate.result_keys(vis["results"])
            if epistemic:
                # 0.2.0（老板 11-11 批准）：当天仍处于"疑似"状态的诊断不许写成确诊或排除定论（只看截至当天可见状态，不看未来）
                pl["unsettled_dx"] = sorted(n for n, st in vis["dx"].items() if st == "疑似" and len(n) >= 2)
                pl["settled_dx"] = sorted(
                    {n for n, st in vis["dx"].items() if st != "疑似" and len(n) >= 2}
                    | {c["name"] for c in j["patient"]["comorbidities"] if c["at_admission"] and len(c["name"]) >= 2}
                )
        if getattr(w, "no_filler", False) and spec["role"] in compiler.COURSE:
            # 0.1.7 套话约束臂：病程提示词多一条，不写没给的生命体征、不每篇都写笼统套话
            pl["no_filler"] = True
        if getattr(w, "no_filler2", False) and spec["role"] in compiler.COURSE:
            # 0.1.8 对照臂 D：同上，但不点"密切观察""无特殊不适"这些词的名
            pl["no_filler2"] = True
        if getattr(w, "prev_note", False) and spec["role"] in compiler.COURSE and prev_note:
            # 0.1.6 对照臂：写手看到上一篇病程最终交出的正文（含拷贝进来的句子），要求不照搬
            pl["prev_text"] = prev_note["text"]
        if pl.get("pe_ref") and texts.get("体格检查"):
            # 0.1.9（审查员第八轮补充）：专科情况要和刚写好的体格检查一致。0.1.11：不再当事实给（写手照抄成专科情况，第十次 M6），
            # 整段（≤ PE_REF 字）放进提示只作对照；数字不进 allowed，照抄体征数字会被闸门挡回重写
            pe = _excerpt(texts["体格检查"], PE_REF)
            pl["hint"] = (pl.get("hint") or "") + "同一篇已写好的体格检查（只作对照，不要照抄）：" + pe
            pl["pe_ref_chars"] = len(pe)
        for f in pl.get("focus") or ():
            tally["focus_" + f] = tally.get("focus_" + f, 0) + 1
        tally["sections"] = tally.get("sections", 0) + 1
        splice = getattr(w, "splice_copy", True)
        if not splice and "copy" in pl and prev_note:
            carry = compiler.pick_carry_fact(
                prev_note["text"], j, spec, prev_note["day"], stats, rng,
                banned=compiler.copy_banned(j, spec["day"]) + list(prev_note.get("overused") or ()),
                safe_fn=lambda s, _p=pl: gate.copy_safe(_p, s),
            )
            if carry:
                pl["facts"] = list(pl["facts"]) + [compiler._fact("carry", carry)]
        tries = 4 if not splice else MAX_SECTION_TRIES
        best, best_f, last_fails = None, None, None
        for attempt in range(tries):
            if not splice and attempt and last_fails:
                pl["retry_fails"] = last_fails
            else:
                pl.pop("retry_fails", None)
            text = w.write(j, spec, pl, attempt + 10 * round_)
            if attempt == 0 and first_doc_out is not None:
                first_texts[pl["section"]] = text
            fails = gate.check_section(pl, text)
            last_fails = fails
            if attempt:
                tally["section_retries"] = tally.get("section_retries", 0) + 1
            if best is None or len(fails) < len(best_f):
                best, best_f = text, fails
            if not fails:
                break
        pl.pop("retry_fails", None)
        for code in best_f:
            tally["sec_" + code] = tally.get("sec_" + code, 0) + 1
        if "admission_leak" in best_f:
            # 0.1.5："入院情况"三次都带住院后的事，剔掉带触发词的分句再交出（0.1.4 带着交出，W-8B 的 D_K4 6/20 都是这样来的）
            best = gate.strip_admission_leak(best)
            tally["sec_admission_leak_stripped"] = tally.get("sec_admission_leak_stripped", 0) + 1
        fix_pool = gate.FIXABLE if splice else gate.PIPELINE_SAFE_FIX
        fixable = [c for c in best_f if c in fix_pool]
        if fixable:
            # 0.1.8 / 0.30：TemplateWriter 对照保持 FIXABLE，LLMWriter 仅保留纯格式安全修补（truncated/repeat 等）
            best = gate.repair_section(best, fixable, pl)
            for code in fixable:
                tally[f"sec_{code}_repaired"] = tally.get(f"sec_{code}_repaired", 0) + 1
        if not best_f:
            tally["sections_first_try"] = tally.get("sections_first_try", 0) + (attempt == 0)
        if splice and "copy" in pl and prev_note:
            gap_day = prev_note["day"]
            budget = compiler.copy_budget(stats, spec, gap_day, prev_role=prev_note.get("role"))
            pl["copy"]["budget"] = round(budget, 4)
            banned = compiler.copy_banned(j, spec["day"]) + list(prev_note.get("overused") or ())
            best, added = compiler.apply_copy(
                best, prev_note["text"], budget, rng, banned, diag=tally,
                banned_meds=compiler.inactive_meds(j, spec["day"]),
                allowed_nums=compiler.active_med_nums(j, spec["day"]),
                safe_fn=lambda s, _p=pl: gate.copy_safe(_p, s),
                max_sents=compiler.copy_max_sents(spec, prev_note.get("role")),
            )
            tally["copied_chars"] = tally.get("copied_chars", 0) + added
            tally["note_chars"] = tally.get("note_chars", 0) + len(best)
        texts[pl["section"]] = best
    if first_doc_out is not None:
        first_doc_out.append(compiler.assemble(j, spec, specs, plans, first_texts))
    return compiler.assemble(j, spec, specs, plans, texts), texts


def _note_copy_source(role, texts):
    """提取每类可作下一篇拷贝源的文书正文（FP 病例特点+拟诊+计划 / SUP·DAILY 正文 / PRE 简要病情 / POST 术后处理措施）。"""
    if role == "FP":
        return "".join(texts.get(x, "") for x in ("病例特点", "拟诊讨论", "诊疗计划"))
    if role in ("SUP", "DAILY"):
        return texts.get("正文", "")
    if role == "PRE":
        return texts.get("简要病情", "")
    if role == "POST":
        return texts.get("术后处理措施", "")
    return ""


def _overused_sents(note_bodies, specs, before_k):
    """0.21：同一住院里已在 ≥2 篇 SUP/DAILY 出现的 ≥20 字长句不再继续拷入，防止触发 cross_repeat。"""
    cnt = {}
    for idx, body in note_bodies.items():
        if idx < before_k and specs[idx].get("role") in compiler.COURSE:
            for part in re.split(r"[。！？!?\n]", body):
                p = part.strip()
                if len(p) >= 20:
                    cnt[p] = cnt.get(p, 0) + 1
    return [p for p, n in cnt.items() if n >= 2]


def prev_note_for(k, specs, note_bodies):
    """第 k 篇之前最近一篇可拷贝文书的正文；第 k 篇不参与拷贝或前面没有源文书时返回 None。"""
    if specs[k]["role"] not in ("SUP", "PRE", "POST", "DAILY"):
        return None
    ks = [x for x in note_bodies if x < k]
    return {"day": specs[max(ks)]["day"], "text": note_bodies[max(ks)]} if ks else None


def _prev_ctx(k, specs, note_bodies):
    base = prev_note_for(k, specs, note_bodies)
    if not base:
        return None
    ks = [x for x in note_bodies if x < k]
    base["role"] = specs[max(ks)]["role"]
    base["overused"] = _overused_sents(note_bodies, specs, k)
    return base


def generate_one(jid, stats, w, seed, slot, tally, first_docs_out=None):
    """返回 (journey, docs, 失败码列表)。docs 为 None 表示丢弃。"""
    rng = random.Random(planner.seed_of(seed, jid))
    j = planner.plan_journey(stats, rng, jid, slot)
    if j is None:
        return None, None, ["no_patient"]
    specs = planner.schedule(j, stats, rng)
    errs = planner.check_plan(j, specs)
    if errs:
        return j, None, ["plan_" + e for e in sorted(set(errs))]
    docs, note_bodies, ctx = [], {}, {}
    for spec in specs:
        fbox = [] if first_docs_out is not None else None
        text, texts = write_doc(j, spec, specs, stats, w, rng, tally, _prev_ctx(spec["k"], specs, note_bodies), ctx=ctx, first_doc_out=fbox)
        docs.append({"doc_type": spec["doc_type"], "day_offset": spec["day"], "text": text, "doc_type_cn": spec["doc_type_cn"], "k": spec["k"], "role": spec["role"]})
        if first_docs_out is not None:
            first_docs_out.append({"doc_type": spec["doc_type"], "day_offset": spec["day"], "text": fbox[0], "doc_type_cn": spec["doc_type_cn"], "k": spec["k"], "role": spec["role"]})
        if spec["role"] == "HR":
            ctx = hr_context(texts)
        src = _note_copy_source(spec["role"], texts)
        if src:
            note_bodies[spec["k"]] = src
    fails = gate.check_admission(j, docs)
    tally["first_pass"] = tally.get("first_pass", 0) + (not fails)
    # 0.1.7：首次检查的失败码（每次住院每个码记一次），S3 探针看写手写反侧别有多频繁
    for code in sorted({c for _, c in fails}):
        tally["first_fail_" + code] = tally.get("first_fail_" + code, 0) + 1
    for r in range(1, MAX_DOC_REWRITES + 1):
        if not fails:
            break
        tally["adm_rewrites"] = tally.get("adm_rewrites", 0) + 1
        ks = {k for k, _ in fails if k is not None} or {d["k"] for d in docs}
        hr_k = next(s["k"] for s in specs if s["role"] == "HR")
        if hr_k in ks:
            # 0.1.9：入院记录重写了，出院记录"入院情况"跟着重写，两边的查体对得上
            ks |= {s["k"] for s in specs if s["role"] == "DS"}
        for k in sorted(ks):
            spec = specs[k]
            # 重写病程也要按上一篇病程拷贝（0.1 这里传 None，重写后的病程丢了拷贝）
            text, texts = write_doc(j, spec, specs, stats, w, rng, tally, _prev_ctx(k, specs, note_bodies), r, ctx=ctx)
            docs[k] = dict(docs[k], text=text)
            if spec["role"] == "HR":
                ctx = hr_context(texts)
            src = _note_copy_source(spec["role"], texts)
            if src:
                note_bodies[k] = src
        fails = gate.check_admission(j, docs)
    if fails:
        return j, None, sorted({c for _, c in fails})
    return j, docs, []


def _load_slots(path, ids):
    keep = None
    if ids:
        with open(ids, encoding="utf-8") as f:
            keep = {x.strip() for x in f if x.strip()}
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                if keep is None or str(s["slot_id"]) in keep:
                    out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats")
    ap.add_argument("--kb")
    ap.add_argument("--no-kb", action="store_true")
    ap.add_argument("--kb-only", action="store_true", help="纯路线 B 公开先验模式（不加载医院统计表，0.30）")
    ap.add_argument("--mode", required=True, choices=("free", "paired"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--slots")
    ap.add_argument("--ids")
    ap.add_argument("--writer", default="llm", choices=("llm", "template"))
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prev-note", action="store_true", help="病程提示词带上一篇病程，要求不照搬（0.1.6 W-8B 对照臂）")
    ap.add_argument("--no-filler", action="store_true", help="病程提示词加一条不写笼统套话（0.1.7 W-8B 对照臂）")
    ap.add_argument("--no-filler2", action="store_true", help="同上，但不点套话的名（0.1.8 W-8B 对照臂 D）")
    ap.add_argument("--num-placeholder", action="store_true", help="将事实中的日期/数值替换为 <D1>/<N1> 占位符并由程序确定性回填（0.21.2 / S3）")
    ap.add_argument("--ablate-epistemic", action="store_true", help="认识论对称消融：保持可见客观历史不变，剥离显式诊断状态编译与 unsettled_dx 断言门控（0.33）")
    ap.add_argument("--save-first-try", action="store_true", help="同步导出每节首轮 attempt=0 未经门控重试前的文书至 first_try_admissions.jsonl（0.33）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.stats and not a.kb_only:
        raise SafeError("needs_stats_or_kb_only")
    os.umask(0o077)
    os.makedirs(a.out, exist_ok=True)
    kb_data = None if a.no_kb else (kb.load_kb(a.kb) if a.kb else kb.load_default_kb())
    if a.kb_only:
        stats = kb.build_b_only_stats(kb_data)
    else:
        with open(a.stats, encoding="utf-8") as f:
            stats = json.load(f)
        if kb_data is not None:
            stats["_kb"] = kb_data
    try:
        planner.require_stats(stats)
    except planner.StatsVersionError:
        raise SafeError("stats_version_rebuild_with_" + planner.STATS_VERSION)
    w = writer.TemplateWriter() if a.writer == "template" else writer.LLMWriter(a.base_url, a.model, a.raw, num_placeholder=a.num_placeholder)
    w.prev_note = a.prev_note
    if a.no_filler and a.no_filler2:
        raise SafeError("no_filler_both")
    w.no_filler = a.no_filler
    w.no_filler2 = a.no_filler2
    w.ablate_epistemic = a.ablate_epistemic
    if a.mode == "paired":
        if not a.slots:
            raise SafeError("paired_needs_slots")
        work = [(str(s["slot_id"]), s) for s in _load_slots(a.slots, a.ids)]
    else:
        work = [(f"MC{a.seed}-{i:05d}", None) for i in range(a.n)]
    tally, discards, lock = {"planned": len(work), "kept": 0, "discarded": 0, "llm_errors": 0}, {}, threading.Lock()
    t0 = time.perf_counter()
    out_names = ("admissions", "journeys", "drafts") + (("first_try_admissions",) if a.save_first_try else ())
    paths = {x: open(os.path.join(a.out, f"{x}.jsonl"), "w", encoding="utf-8") for x in out_names}

    def one(item):
        jid, slot = item
        local = {}
        local_first = [] if a.save_first_try else None
        try:
            j, docs, fails = generate_one(jid, stats, w, a.seed, slot, local, first_docs_out=local_first)
        except LLMError as e:
            with lock:
                tally["llm_errors"] += 1
            log("llm_error", code=str(e), id=jid)
            return
        with lock:
            for k, v in local.items():
                tally[k] = tally.get(k, 0) + v
            if docs is None:
                tally["discarded"] += 1
                stratum = (j["patient"]["primary_dx"]["code3"][:1] if j else "?") or "?"
                for code in fails:
                    discards[(stratum, code)] = discards.get((stratum, code), 0) + 1
                return
            tally["kept"] += 1
            tally["chars"] = tally.get("chars", 0) + sum(len(d["text"]) for d in docs)
            tally["docs"] = tally.get("docs", 0) + len(docs)
            paths["admissions"].write(json.dumps(admission_line(j, docs), ensure_ascii=False) + "\n")
            paths["journeys"].write(json.dumps(j, ensure_ascii=False) + "\n")
            paths["drafts"].write(json.dumps({"slot_id": jid, "draft": planner.journey_to_draft(j)}, ensure_ascii=False) + "\n")
            if local_first is not None:
                paths["first_try_admissions"].write(json.dumps(admission_line(j, local_first), ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(a.concurrency) as pool:
        list(pool.map(one, work))
    for f in paths.values():
        f.close()
    wall = time.perf_counter() - t0
    tally.update({"wall_s": round(wall, 1), "calls": w.calls, "prompt_tokens": w.prompt_tokens, "completion_tokens": w.completion_tokens,
                  "calls_per_kchar": round(1000 * w.calls / tally["chars"], 3) if tally.get("chars") else None,
                  "calls_per_adm": round(w.calls / max(1, tally["kept"]), 2), "first_pass_rate": round(tally.get("first_pass", 0) / max(1, tally["planned"]), 4),
                  "discard_rate": round(tally["discarded"] / max(1, tally["planned"]), 4), "version": VERSION, "seed": a.seed, "mode": a.mode, "writer": a.writer,
                  "prev_note": a.prev_note, "no_filler": a.no_filler, "no_filler2": a.no_filler2, "num_placeholder": a.num_placeholder,
                  "ablate_epistemic": a.ablate_epistemic, "model": a.model if a.writer == "llm" else None})
    with open(os.path.join(a.out, "counts.json"), "w", encoding="utf-8") as f:
        json.dump(tally, f, ensure_ascii=False, indent=1, sort_keys=True)
    with open(os.path.join(a.out, "discard.csv"), "w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["chapter", "code", "n"])
        cw.writerows(sorted((s, c, n) for (s, c), n in discards.items()))
    log("pipeline_done", kept=tally["kept"], discarded=tally["discarded"], calls=w.calls)
    if tally["llm_errors"]:
        raise SafeError("llm_errors", n=tally["llm_errors"])


if __name__ == "__main__":
    guarded(main)
