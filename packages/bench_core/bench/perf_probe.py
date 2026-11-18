"""算力短测探针：并发打一个 OpenAI 兼容服务，量吞吐、延迟、结构化输出的开销。只用标准库，提示词是固定虚构文本。

python -B perf_probe.py --base-url http://127.0.0.1:8101/v1 --model Qwen3-8B-Base --concurrency 1,16,64,128 \
    --requests 200 --input-chars 2000 --max-tokens 512 --out perf_8b.csv [--json-schema schema.json] [--label awq]

带 --json-schema 时默认走抽取提示词、输出上限 2048。v0.1 这里用改写提示词加 512 token，256 条全被截断（S0_pkg2），
只能算开销，算不出能用的结构化产能。finish_stop、finish_length、json_ok_per_s 要和吞吐一起看。

API key 从环境变量 VLLM_API_KEY 读，不打印、不写文件。
"""
import argparse
import csv
import json
import os
import random
import statistics
import threading
import time
import urllib.request

SEED_TEXT = ("虚构患者张三，男，58岁，因右上腹痛3天入甲县人民医院。入院后查血常规示白细胞12.1×10^9/L，腹部CT示胆囊结石。"
             "予头孢曲松2g qd静滴抗感染，择期行腹腔镜胆囊切除术，术后恢复可，复查血常规正常后出院。")
# 抽取用：句子各不相同，免得模型对着同一句反复抽。事件数只在提示词里限制，schema 上限另外比
EXTRACT_SENTENCES = ("虚构患者李四，女，{age}岁，因反复咳嗽{d}天入甲县人民医院。", "查体：体温{t}℃，脉搏{p}次/分，血压{bp}/78mmHg，双肺可闻及湿啰音。",
                     "血常规示白细胞{w}×10^9/L，中性粒细胞比例{n}%。", "胸部CT示右下肺斑片状渗出影，考虑社区获得性肺炎。", "予头孢曲松2g qd静滴抗感染，氨溴索30mg tid口服化痰。",
                     "第{d}日复查C反应蛋白{c}mg/L，较前下降。", "既往高血压病史{y}年，长期口服氨氯地平5mg qd，否认糖尿病病史。", "腹部彩超未见明显异常，心电图示窦性心律。",
                     "患者诉胸闷{d}小时，予吸氧后缓解，复查血气分析pH 7.{g}。", "行纤维支气管镜检查，术中未见新生物，留取灌洗液送检。", "灌洗液培养示肺炎链球菌，排除结核感染。",
                     "体温恢复正常{d}天，咳嗽减轻，今日办理出院，嘱门诊随诊。")


def _extract_body(chars, rng):
    body, n = [], 0
    while n < chars:
        order = list(EXTRACT_SENTENCES)
        rng.shuffle(order)
        for s in order:
            one = s.format(age=rng.randint(18, 90), d=rng.randint(1, 9), t=f"{rng.uniform(36, 39.5):.1f}", p=rng.randint(60, 110), bp=rng.randint(100, 160),
                           w=f"{rng.uniform(4, 20):.1f}", n=rng.randint(50, 92), c=rng.randint(3, 120), y=rng.randint(2, 20), g=rng.randint(30, 45))
            body.append(one)
            n += len(one)
    return "".join(body)[:chars]


def prompt(chars, seed, task="rewrite"):
    rng = random.Random(seed)
    if task == "extract":
        head = (f"第{seed}例。请从下面的虚构病历片段抽取结构化事件：患者性别、年龄，以及明确写出的诊断、检查、用药、操作、生命体征和症状。"
                "evidence 逐字摘录原文短句，不超过 30 字；相同事件不重复；事件不超过 20 条。只输出 JSON。\n")
        return head + _extract_body(chars, rng)
    head = f"第{seed}例。请把下面的虚构病历摘要改写成一篇日常病程记录，只输出正文。\n"
    body = []
    n = 0
    while n < chars:
        s = SEED_TEXT.replace("3天", f"{rng.randint(1, 9)}天").replace("12.1", f"{rng.uniform(4, 20):.1f}")
        body.append(s)
        n += len(s)
    return head + "".join(body)[:chars]


def one_call(base_url, key, model, text, max_tokens, schema, timeout, raw=False):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.95, "top_k": 50, "seed": 20261101}
    if raw:
        body["prompt"] = text
    else:
        body["messages"] = [{"role": "user", "content": text}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if max_tokens and schema is None:
        body["min_tokens"] = max_tokens  # 固定输出长度，吞吐才可比
        body["ignore_eos"] = True
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "probe", "schema": schema}}
    path = "/completions" if raw else "/chat/completions"
    req = urllib.request.Request(base_url.rstrip("/") + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    dt_ = time.perf_counter() - t0
    usage = data.get("usage") or {}
    ok_json, n_events = None, None
    if schema is not None:
        try:
            parsed = json.loads(data["choices"][0]["message"]["content"])
            ok_json = True
            if isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
                n_events = len(parsed["events"])
        except (ValueError, KeyError, IndexError, TypeError):
            ok_json = False
    finish = (data.get("choices") or [{}])[0].get("finish_reason")
    return dt_, usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0, ok_json, finish, n_events


def run_level(args, key, schema, conc):
    jobs = list(range(args.requests))
    lock = threading.Lock()
    res, errors = [], {}

    def work():
        while True:
            with lock:
                if not jobs:
                    return
                seed = jobs.pop()
            try:
                out = one_call(args.base_url, key, args.model, prompt(args.input_chars, seed, args.task), args.max_tokens, schema, args.timeout, args.raw)
                with lock:
                    res.append(out)
            except Exception as e:
                with lock:
                    errors[type(e).__name__] = errors.get(type(e).__name__, 0) + 1

    t0 = time.perf_counter()
    ts = [threading.Thread(target=work, daemon=True) for _ in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    lat = sorted(r[0] for r in res)
    comp, pr = sum(r[2] for r in res), sum(r[1] for r in res)
    js = [r[3] for r in res if r[3] is not None]
    events = [r[5] for r in res if r[5] is not None]
    return {"label": args.label, "task": args.task, "input_chars": args.input_chars, "max_tokens": args.max_tokens,
            "model": args.model, "concurrency": conc, "requests": args.requests, "ok": len(res),
            "errors": sum(errors.values()), "error_types": "|".join(sorted(errors)), "wall_s": round(wall, 2),
            "req_per_s": round(len(res) / wall, 3) if wall else None, "completion_tok_per_s": round(comp / wall, 1) if wall else None,
            "prompt_tok_per_s": round(pr / wall, 1) if wall else None, "latency_p50": round(statistics.median(lat), 3) if lat else None,
            "latency_p95": round(lat[int(0.95 * (len(lat) - 1))], 3) if lat else None, "mean_prompt_tokens": round(pr / len(res), 1) if res else None,
            "mean_completion_tokens": round(comp / len(res), 1) if res else None, "json_ok": sum(js) if js else None,
            "json_ok_per_s": round(sum(js) / wall, 3) if js and wall else None, "finish_stop": sum(r[4] == "stop" for r in res),
            "finish_length": sum(r[4] == "length" for r in res), "mean_events": round(sum(events) / len(events), 2) if events else None,
            "structured": schema is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="1,16,64,128")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--input-chars", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, help="默认 512，带 --json-schema 时默认 2048")
    ap.add_argument("--task", choices=("rewrite", "extract"), help="默认 rewrite，带 --json-schema 时默认 extract")
    ap.add_argument("--json-schema")
    ap.add_argument("--label", default="")
    ap.add_argument("--raw", action="store_true", help="走 /v1/completions，给没有 chat template 的基座模型用；不能和 --json-schema 同用")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.raw and a.json_schema:
        ap.error("--raw 不能和 --json-schema 同用")
    a.max_tokens = a.max_tokens or (2048 if a.json_schema else 512)
    a.task = a.task or ("extract" if a.json_schema else "rewrite")
    key = os.environ.get("VLLM_API_KEY", "EMPTY")
    schema = None
    if a.json_schema:
        with open(a.json_schema, encoding="utf-8") as f:
            schema = json.load(f)
    rows = []
    for c in (int(x) for x in a.concurrency.split(",")):
        rows.append(run_level(a, key, schema, c))
        print(json.dumps({k: rows[-1][k] for k in ("concurrency", "ok", "errors", "req_per_s", "completion_tok_per_s", "latency_p95")}), flush=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
