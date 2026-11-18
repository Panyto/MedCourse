# -*- coding: utf-8 -*-
"""规则层单挑对比（老板 11-10 批）：手写规则 vs 8B 作答，逐条比一致率。

比三样：
1. 麻醉、切口、引流：A256 规划里出现过的每个术式名。规则是 planner.anesthesia_for，切口和引流由 compiler.no_incision 推出
   （局部麻醉、静脉麻醉 → 没有切口和引流）。
2. 开单到出报告的间隔：评测台全部检查键（公开词表）。规则是 planner.SCHED[检查大类]，间隔在 0 到这个数之间随机，
   不在 SCHED 里的大类（检验、功能）是 0 天。8B 的天数落在 [0, 上限] 里算一致。
3. 停药理由：drug_class.STOP_REASON 的全部药物大类，加上 A256 规划里出现过的非长期用药大类。8B 从同一组理由里选一个。

8B 这里只凭模型自己的知识回答（不检索路径），结果一律记为模型先验。
采样用 llm.EXTRACT（温度 0），种子固定，输出用 JSON schema 限定选项。

在 ys 上由伙计跑，读 product/derived 下的 journeys.jsonl（规划，不出 ys）。出两份：
- 汇总（--summary）：只有计数和一致率，没有名称，可以带回；
- 逐条（--items）：术式名、规则答案、8B 答案，留在 ys 的 derived 目录。带不带回由老板定。检查键、药物大类是公开词表。
"""
import argparse
import csv
import json
import os

ANES = ("全身麻醉", "局部麻醉", "静脉麻醉", "腰硬联合麻醉", "其他椎管内麻醉", "神经阻滞麻醉", "不需要麻醉")
NEURAXIAL = {"腰硬联合麻醉", "其他椎管内麻醉"}
LAG_MAX = 14
VERSION = "rule-compare-0.1"


def load_rules():
    """从核心包取手写规则（v0.1.13.1）。单独成函数，自检时换成假的。"""
    from medcourse import compiler, drug_class, planner
    return {"anesthesia": planner.anesthesia_for,
            "no_incision": lambda anes: compiler.no_incision({"anesthesia": anes}),
            "sched": dict(planner.SCHED), "kind_of": dict(planner.KIND_OF),
            "stop_reason": drug_class.stop_reason,
            "stop_classes": tuple(drug_class.STOP_REASON),
            "stop_options": tuple(sorted(set(drug_class.STOP_REASON.values()) | {drug_class.DEFAULT_STOP}))}


def items_from_journeys(lines):
    """规划 → (术式名集合, 非长期用药大类集合, 出现过的检查键集合)。只取名称，不留病人信息。"""
    procs, classes, keys = set(), set(), set()
    for line in lines:
        j = json.loads(line)
        for a in j.get("procedures", []):
            if a.get("name"):
                procs.add(a["name"])
        for m in j.get("meds", []):
            if m.get("class") and not m.get("chronic"):
                classes.add(m["class"])
        for e in j.get("exams", []):
            if e.get("key"):
                keys.add(e["key"])
    return procs, classes, keys


def _extract():
    try:
        from medcourse.llm import EXTRACT
        return EXTRACT
    except ImportError:   # 自检没有核心包时用同样的取值
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}


def _ask(client, prompt, schema, seed):
    r = client.generate(sampling=_extract(), seed=seed, prompt=prompt, max_tokens=128, schema=schema)
    try:
        return json.loads(r["text"])
    except (ValueError, TypeError):
        return None


def ask_anesthesia(client, name, seed):
    p = (f"住院手术或操作：{name}。按国内常规，这项手术或操作最常用的麻醉方式是什么？"
         f"另外，它有没有需要缝合的体表切口（穿刺点不算），术后通常会不会留置引流管？只回答 JSON。")
    s = {"type": "object", "properties": {"anesthesia": {"enum": list(ANES)}, "incision": {"type": "boolean"}, "drain": {"type": "boolean"}},
         "required": ["anesthesia", "incision", "drain"]}
    return _ask(client, p, s, seed)


def ask_lag(client, key, seed):
    p = (f"住院患者开了{key}检查。从开单当天算起，通常第几天能拿到正式报告？当天出报告填 0。"
         f"只回答 JSON，天数是 0 到 {LAG_MAX} 的整数。")
    s = {"type": "object", "properties": {"days": {"type": "integer", "minimum": 0, "maximum": LAG_MAX}}, "required": ["days"]}
    return _ask(client, p, s, seed)


def ask_stop(client, cls, options, seed):
    p = f"住院患者使用{cls}，住院期间停用时，病程记录里最常写的停药理由是哪一个？只回答 JSON，从给定选项里选。"
    s = {"type": "object", "properties": {"reason": {"enum": list(options)}}, "required": ["reason"]}
    return _ask(client, p, s, seed)


def compare(client, rules, procs, classes, keys, seed=20261110):
    """返回 (逐条行, 汇总)。汇总只有计数。"""
    rows, summ = [], {"version": VERSION, "seed": seed}
    # 1 麻醉、切口、引流
    n = agree = agree_relax = inc_agree = drain_agree = fail = 0
    for name in sorted(procs):
        rule = rules["anesthesia"](name)
        rule_inc = not rules["no_incision"](rule)
        ans = ask_anesthesia(client, name, seed)
        n += 1
        if not ans or ans.get("anesthesia") not in ANES:
            fail += 1
            rows.append(("anesthesia", name, rule, "", "parse_fail"))
            continue
        got = ans["anesthesia"]
        agree += got == rule
        agree_relax += got == rule or (got in NEURAXIAL and rule in NEURAXIAL)
        inc_agree += bool(ans.get("incision")) == rule_inc
        drain_agree += bool(ans.get("drain")) == rule_inc
        rows.append(("anesthesia", name, f"{rule}|切口{int(rule_inc)}", f"{got}|切口{int(bool(ans.get('incision')))}|引流{int(bool(ans.get('drain')))}",
                     "agree" if got == rule else "disagree"))
    summ["anesthesia"] = {"n": n, "agree": agree, "agree_neuraxial_merged": agree_relax, "incision_agree": inc_agree,
                          "drain_agree": drain_agree, "parse_fail": fail}
    # 2 回报间隔：全部评测台检查键，另记 A256 里出现过的子集
    by_kind, fail = {}, 0
    for key in sorted(rules["kind_of"]):
        kind = rules["kind_of"][key]
        hi = rules["sched"].get(kind, 0)
        ans = ask_lag(client, key, seed)
        d = by_kind.setdefault(kind, {"n": 0, "agree": 0, "over": 0, "in_a256": 0, "agree_in_a256": 0, "days_8b": [], "rule_max": hi})
        if not ans or not isinstance(ans.get("days"), int):
            fail += 1
            rows.append(("lag", key, f"0-{hi}", "", "parse_fail"))
            continue
        days = ans["days"]
        ok = 0 <= days <= hi
        d["n"] += 1
        d["agree"] += ok
        d["over"] += days > hi
        d["days_8b"].append(days)
        if key in keys:
            d["in_a256"] += 1
            d["agree_in_a256"] += ok
        rows.append(("lag", key, f"0-{hi}", str(days), "agree" if ok else "disagree"))
    summ["lag"] = {"by_kind": by_kind, "parse_fail": fail}
    # 3 停药理由
    n = agree = fail = 0
    for cls in sorted(set(rules["stop_classes"]) | set(classes)):
        rule = rules["stop_reason"](cls)
        ans = ask_stop(client, cls, rules["stop_options"], seed)
        n += 1
        if not ans or ans.get("reason") not in rules["stop_options"]:
            fail += 1
            rows.append(("stop", cls, rule, "", "parse_fail"))
            continue
        agree += ans["reason"] == rule
        rows.append(("stop", cls, rule, ans["reason"], "agree" if ans["reason"] == rule else "disagree"))
    summ["stop"] = {"n": n, "agree": agree, "parse_fail": fail}
    return rows, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--summary", required=True, help="只有计数，可带回")
    ap.add_argument("--items", required=True, help="逐条名称和答案，留在 ys")
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    from medcourse.llm import Client
    with open(a.journeys, encoding="utf-8") as f:
        procs, classes, keys = items_from_journeys(f)
    rows, summ = compare(Client(a.base_url, a.model), load_rules(), procs, classes, keys, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.items)), exist_ok=True)
    with open(a.items, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("part", "item", "rule", "model_8b", "result"))
        w.writerows(rows)
    with open(a.summary, "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print("rule_compare_done", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_kind"} if isinstance(v, dict) else v
                                           for k, v in summ.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
