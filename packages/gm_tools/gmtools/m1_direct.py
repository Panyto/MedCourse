# -*- coding: utf-8 -*-
"""M1 第三组（老板 11-08 批准）：不用真实统计，同一写手直接生成整套文书。m1-direct-0.1。

和第一组（MC-8B，v0.1.13.1）只差一样：写手手里没有规划给的任何临床事实。其余全部照第一组：
- 同一批病人、同一组文书：读第一组的 admissions.jsonl，篇数、文书类型、住院日、抬头（程序写的姓名占位、性别、年龄、科室、时间）原样照搬；
- 同一份节目录、同样的目标字数：节标题从第一组那篇的正文里认，目标字数取第一组那一节写出来的字数；
- 同一个写手：同一个模型、同一句 SYSTEM、同一组采样参数（llm.WRITE）、同样的 max_tokens 算法、同样的 writer.clean。

写手多拿到两样第一组的节级提示里没有的东西（保守，偏向第三组）：
- 这次住院的主要诊断（出院诊断名）和住院天数（第一组的规划里有，按天给）；
- 本次住院已经写好的病历（最近 CONTEXT_CHARS 字），让它前后接得上。

没有规划，也就没有闸门、没有重写、没有拷贝；真未来泄露没有时间线可对，不适用。

用法（ys，PYTHONPATH 指 v0.1.13.1 的 medcourse_core 和本包）：
python -B -m gmtools.m1_direct --run 第一组输出目录 --base-url http://127.0.0.1:8101/v1 --model Qwen3-8B --out 输出目录 [--limit N] [--concurrency 8]

产物：admissions.jsonl（AdmissionJSONL，bench 直接读；住院号前加 D）、counts.json（只有计数）。
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from medcourse.llm import WRITE, Client, LLMError
from medcourse.planner import seed_of
from medcourse.safe import SafeError, guarded, log
from medcourse.writer import SYSTEM, clean

VERSION = "m1-direct-0.1"
CONTEXT_CHARS = 3000
HEAD_LINES = {"入院记录": 3, "首次病程记录": 3, "出院记录": 3}   # compiler.header：这三种抬头 3 行，其余 1 行
TITLES = ("主诉", "现病史", "既往史", "个人史", "婚育史", "月经史", "家族史", "体格检查", "专科情况", "辅助检查", "入院诊断",
          "病例特点", "初步诊断", "拟诊讨论", "鉴别诊断", "诊疗计划",
          "简要病情", "术前诊断", "手术指征", "拟施手术名称和方式", "拟施麻醉方式", "注意事项",
          "手术时间", "术中诊断", "手术方式", "麻醉方式", "手术简要经过", "术后处理措施", "术后应当特别注意观察的事项",
          "入院情况", "诊疗经过", "出院诊断", "出院情况", "出院医嘱")
LIST_SECTIONS = {"入院诊断", "初步诊断", "出院诊断", "术前诊断", "术中诊断"}
_TITLE = re.compile("^(" + "|".join(sorted(map(re.escape, TITLES), key=len, reverse=True)) + ")：")


def split_doc(text, doc_type_cn):
    """第一组的一篇 → (抬头, [(节标题, 正文)])。标题为空的是病程正文。"""
    lines = (text or "").split("\n")
    n = HEAD_LINES.get(doc_type_cn, 1)
    head = "\n".join(lines[:n]) + "\n"
    secs = []
    for line in lines[n:]:
        m = _TITLE.match(line)
        if m:
            secs.append([m.group(1), line[m.end():]])
        elif secs:
            secs[-1][1] += "\n" + line
        elif line.strip():
            secs.append(["", line])
        # 抬头之后、第一节之前的空行不要
    return head, [(t, b.strip()) for t, b in secs]


def section_prompt(head_info, doc_type_cn, day, section, target, context):
    sec = section or "正文"
    if section in LIST_SECTIONS:
        form = f"3. 按病历格式逐条列出诊断（1.××），不写别的。\n"
    else:
        form = f"3. 约 {target} 字，写连贯的病历文字，不要分条，不要写小标题。\n"
    return (f"请为一位虚构患者写《{doc_type_cn}》的“{sec}”一节，书写时间是住院第{day}天。\n"
            f"【患者】{head_info['sex']}，{head_info['age_text']}，{head_info['dept']}；这次住院的主要诊断：{head_info['dx']}；共住院{head_info['los']}天。\n"
            f"【本次住院已经写好的病历】\n{context or '无'}\n"
            "【要求】\n"
            "1. 症状、查体、检查和结果、用药和剂量、数值，都按你的临床知识自己决定，要符合这位患者和这一天。\n"
            "2. 和已经写好的病历前后一致；站在书写当天的时间点写，不写以后才知道的事。\n"
            + form
            + "只输出这一节的正文。")


_WHO = re.compile(r"性别：(\S+)\s+年龄：(\S+)\s+科室：(\S+)")


def head_info(adm, journey):
    """性别、年龄、科室从第一组入院记录的抬头读（和规划同源）；主要诊断、住院天数从规划读。"""
    hr = next(d for d in adm["docs"] if d.get("doc_type_cn") == "入院记录")
    m = _WHO.search(hr["text"])
    if not m:
        raise SafeError("hr_header_unreadable", id=adm["admission_id"])
    return {"sex": m.group(1), "age_text": m.group(2), "dept": m.group(3),
            "dx": journey["patient"]["primary_dx"]["name"], "los": journey["los"]}


class Direct:
    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.t = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "finish_length": 0, "empty_sections": 0, "sections": 0}

    def _add(self, **kw):
        with self.lock:
            for k, v in kw.items():
                self.t[k] = self.t.get(k, 0) + v

    def write_section(self, aid, k, info, doc_type_cn, day, section, target, context):
        target = max(10, target)
        r = self.client.generate(messages=[{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": section_prompt(info, doc_type_cn, day, section, target, context)}],
                                 max_tokens=min(1600, int(target * 1.6) + 120), sampling=WRITE,
                                 seed=seed_of(aid, k, section or "正文", "m1-direct") % (2 ** 31))
        body = clean(r["text"], section or "正文")
        self._add(calls=1, sections=1, prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
                  finish_length=int(r["finish"] == "length"), empty_sections=int(not body))
        return body

    def admission(self, adm, journey):
        info = head_info(adm, journey)
        aid = adm["admission_id"]
        written, docs = [], []
        for doc in adm["docs"]:
            head, secs = split_doc(doc["text"], doc.get("doc_type_cn"))
            parts = [head]
            for title, body in secs:
                context = ("\n".join(written) + "\n" + "".join(parts)).strip()[-CONTEXT_CHARS:]
                text = self.write_section(aid, doc.get("k"), info, doc.get("doc_type_cn"), doc.get("day_offset"), title, len(body), context)
                if not title:
                    parts.append(text + "\n")
                elif title in LIST_SECTIONS:
                    parts.append(f"{title}：\n{text}\n")
                else:
                    parts.append(f"{title}：{text}\n")
            full = "".join(parts)
            written.append(full)
            docs.append(dict(doc, text=full))
        return {"admission_id": "D" + aid, "docs": docs, "meta": dict(adm.get("meta") or {}, source=VERSION)}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="第一组（MC-8B）的输出目录：admissions.jsonl、journeys.jsonl、counts.json")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    os.umask(0o077)
    with open(os.path.join(a.run, "counts.json"), encoding="utf-8") as f:
        src = json.load(f)
    if src.get("mode") != "free" or src.get("writer") != "llm":
        raise SafeError("source_not_free_llm")
    adms = sorted(_read_jsonl(os.path.join(a.run, "admissions.jsonl")), key=lambda x: x["admission_id"])
    if a.limit:
        adms = adms[:a.limit]
    journeys = {j["journey_id"]: j for j in _read_jsonl(os.path.join(a.run, "journeys.jsonl"))}
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "admissions.jsonl")
    if os.path.exists(out_path):
        raise SafeError("out_exists")
    d = Direct(Client(a.base_url, a.model))
    lock, tally = threading.Lock(), {"planned": len(adms), "kept": 0, "llm_errors": 0, "chars": 0, "docs": 0}
    t0 = time.perf_counter()
    f_out = open(out_path, "w", encoding="utf-8")

    def one(adm):
        try:
            line = d.admission(adm, journeys[adm["admission_id"]])
        except LLMError as e:
            with lock:
                tally["llm_errors"] += 1
            log("llm_error", code=str(e), id=adm["admission_id"])
            return
        with lock:
            tally["kept"] += 1
            tally["docs"] += len(line["docs"])
            tally["chars"] += sum(len(x["text"]) for x in line["docs"])
            f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(a.concurrency) as pool:
        list(pool.map(one, adms))
    f_out.close()
    tally.update(d.t)
    tally.update({"wall_s": round(time.perf_counter() - t0, 1), "version": VERSION, "mode": "free", "writer": "llm", "model": a.model,
                  "source_seed": src.get("seed"), "source_version": src.get("version"), "context_chars": CONTEXT_CHARS,
                  "discarded": tally["planned"] - tally["kept"],
                  "discard_rate": round((tally["planned"] - tally["kept"]) / max(1, tally["planned"]), 4),
                  "calls_per_kchar": round(1000 * d.t["calls"] / tally["chars"], 3) if tally["chars"] else None,
                  "calls_per_adm": round(d.t["calls"] / max(1, tally["kept"]), 2)})
    with open(os.path.join(a.out, "counts.json"), "w", encoding="utf-8") as f:
        json.dump(tally, f, ensure_ascii=False, indent=1, sort_keys=True)
    log("m1_direct_done", kept=tally["kept"], calls=d.t["calls"], llm_errors=tally["llm_errors"])
    if tally["llm_errors"]:
        raise SafeError("llm_errors", n=tally["llm_errors"])


if __name__ == "__main__":
    guarded(main)
