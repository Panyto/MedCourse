# -*- coding: utf-8 -*-
"""M1 第三组的病人层：不给任何统计，让同一写手自己"出病人"，看模型先验下的病人长什么样。m1-patients-0.1。

第三组的整套文书（m1_direct）沿用第一组的病人，病人层的先验在那里比不出来；这里单独量。
对照是 NHS 那句"GPT-4o 几乎给所有病人写青霉素过敏"和 LLM-CARe 每 256 份只有 9 种年龄。

python -B -m gmtools.m1_patients gen --base-url http://127.0.0.1:8101/v1 --model Qwen3-8B --n 256 --out 目录
python -B -m gmtools.m1_patients summary --patients 目录/patients.jsonl --journeys 第一组/journeys.jsonl --out 表.csv

产物：patients.jsonl（模型虚构的病人头，合成材料）、counts.json；summary 只出计数。
"""
import argparse
import csv
import json
import os
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from medcourse.llm import WRITE, Client, LLMError
from medcourse.planner import seed_of
from medcourse.safe import SafeError, guarded, log
from medcourse.writer import SYSTEM

VERSION = "m1-patients-0.1"
PROMPT = ("请虚构一位正在中国某三级甲等医院住院的患者，给出：科室（dept）、性别（sex，男或女）、年龄（age，整数岁）、"
          "这次住院的主要诊断（dx）、住院天数（los，整数）。只输出 JSON。")
SCHEMA = {"type": "object", "properties": {"dept": {"type": "string"}, "sex": {"type": "string", "enum": ["男", "女"]},
                                           "age": {"type": "integer"}, "dx": {"type": "string"}, "los": {"type": "integer"}},
          "required": ["dept", "sex", "age", "dx", "los"], "additionalProperties": False}
N_AGE = 256


def gen(client, n, concurrency=8):
    out, t, lock = [None] * n, Counter(), threading.Lock()

    def one(i):
        try:
            r = client.generate(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}],
                                max_tokens=200, sampling=WRITE, seed=seed_of("m1-patients", i) % (2 ** 31), schema=SCHEMA)
        except LLMError:
            with lock:
                t["llm_errors"] += 1
            return
        try:
            x = json.loads(r["text"])
            ok = isinstance(x, dict) and all(k in x for k in SCHEMA["required"])
        except ValueError:
            ok = False
        with lock:
            t["calls"] += 1
            if ok:
                out[i] = {"i": i, **{k: x[k] for k in SCHEMA["required"]}}
            else:
                t["parse_fail"] += 1

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(one, range(n)))
    return [x for x in out if x], dict(t)


def _q(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def header_stats(rows):
    """rows：[{dept, sex, age, dx, los}]。只出计数和分位数。"""
    n = len(rows)
    ages = [r["age"] for r in rows if isinstance(r.get("age"), int)]
    sample = rows if n <= N_AGE else random.Random("20261111|m1|age").sample(rows, N_AGE)
    dx = Counter(r["dx"] for r in rows)
    los = [r["los"] for r in rows if isinstance(r.get("los"), int)]
    return {"n": n, "distinct_age_256": len({r["age"] for r in sample}), "distinct_age": len(set(ages)),
            "age_p10": _q(ages, .1), "age_p50": _q(ages, .5), "age_p90": _q(ages, .9),
            "female_share": round(sum(r["sex"] == "女" for r in rows) / n, 4) if n else None,
            "distinct_dept": len({r["dept"] for r in rows}), "distinct_dx": len(dx),
            "top1_dx_share": round(dx.most_common(1)[0][1] / n, 4) if n else None,
            "top5_dx_share": round(sum(c for _, c in dx.most_common(5)) / n, 4) if n else None,
            "los_p10": _q(los, .1), "los_p50": _q(los, .5), "los_p90": _q(los, .9)}


def journey_rows(path):
    """第一组的病人头（规划按 k≥11 统计表抽的），口径同上。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                p = j["patient"]
                out.append({"dept": p["dept"], "sex": p["sex"], "age": p["age"], "dx": p["primary_dx"]["name"], "los": j["los"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--base-url", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--n", type=int, default=256)
    g.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--patients", required=True)
    s.add_argument("--journeys", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    os.umask(0o077)
    if a.cmd == "gen":
        os.makedirs(a.out, exist_ok=True)
        rows, t = gen(Client(a.base_url, a.model), a.n)
        with open(os.path.join(a.out, "patients.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        t.update({"planned": a.n, "valid": len(rows), "version": VERSION, "model": a.model})
        with open(os.path.join(a.out, "counts.json"), "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=1, sort_keys=True)
        log("m1_patients_done", valid=len(rows), parse_fail=t.get("parse_fail", 0))
        if t.get("llm_errors"):
            raise SafeError("llm_errors", n=t["llm_errors"])
        return
    with open(a.patients, encoding="utf-8") as f:
        llm_rows = [json.loads(x) for x in f if x.strip()]
    cols = {"LLM-patients": header_stats(llm_rows), "MC-8B": header_stats(journey_rows(a.journeys))}
    keys = list(cols["MC-8B"])
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + list(cols))
        for k in keys:
            w.writerow([k] + [cols[c][k] for c in cols])


if __name__ == "__main__":
    guarded(main)
