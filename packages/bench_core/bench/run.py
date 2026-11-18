"""统一跑分入口。

python -B -m bench.run --rows rows.json --out 输出目录 [--readers extra,g1,g2,g3,g3old,g4,g4old,g6,c2] [--neg 基线行,基线行]

rows.json：
{
 "reference": "R",                         负对照裁决用的真实行
 "tfidf_reference": {"path":"…","kind":"admission","view":"core","filter_ids":"…","n_expected":200},
                                           （可选）相邻 tf-idf 拟合词表用的行，只拟合不出读数；没配就用 reference
 "codes": "dx_map.csv",                    （可选）列 admission_id 或 slot_id、code3；rows 里单行也可写自己的 "codes"
 "pairs": {"G-E2": "R", "CARe-S1": "R7"},  判别器和同病重合的配对
 "efficiency": {"G-E2": {"calls":1521,"admissions":187,"chars":742776,"input_tokens":..,"output_tokens":..,"seconds":..}},
 "rows": [{"row":"R","path":"…","kind":"admission","view":"core","filter_ids":"…","n_expected":187}, …]
}

所有行在本进程读入，负对照在内存里生成，然后按读数分别起子进程、经 stdin 传过去；读数结果只有数字。
"""
import argparse
import json
import os
import subprocess
import sys

from . import VERSION, aggregate, canon, negcontrols, paths
from .safe import SafeError, guarded, log

ALL_READERS = ("extra", "g1", "g2", "g3", "g3old", "g4", "g4old", "g6", "c2")
TFIDF_REF_ROW = "__tfidf_ref__"


def load_codes(path):
    import csv
    if not path:
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            key = r.get("slot_id") or r.get("admission_id")
            if key and r.get("code3"):
                out[str(key)] = r["code3"]
            if r.get("admission_id") and r.get("code3"):
                out[str(r["admission_id"])] = r["code3"]
    return out


def efficiency(cfg, totals):
    out = {}
    for row, e in (cfg.get("efficiency") or {}).items():
        adm = e.get("admissions") or totals.get(row, {}).get("admissions")
        chars = e.get("chars") or totals.get(row, {}).get("chars")
        tok = None
        if e.get("input_tokens") is not None and e.get("output_tokens") is not None:
            tok = e["input_tokens"] + e["output_tokens"]
        calls = e.get("calls")
        out[row] = {"admissions": adm, "calls_per_adm": calls / adm if calls and adm else None,
                    "calls_per_kchar": 1000.0 * calls / chars if calls and chars else None,
                    "tokens_per_kchar": 1000.0 * tok / chars if tok and chars else None,
                    "seconds_per_kadm": 1000.0 * e["seconds"] / adm if e.get("seconds") and adm else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--readers", default=",".join(ALL_READERS))
    ap.add_argument("--neg", default="", help="逗号分隔的基线行，在这些行上生成负对照")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    with open(a.rows, encoding="utf-8") as f:
        cfg = json.load(f)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)

    sys.path.insert(0, paths.core_path("g2"))
    lines, kinds, sizes = [], {}, []
    for spec in cfg["rows"]:
        one = canon.load_row(spec)
        one, dropped = canon.apply_core_view(one)
        kinds[spec["row"]] = spec.get("kind", "admission")
        sizes.append({"row": spec["row"], "kind": kinds[spec["row"]], "lines": len(one), "core_dropped": dropped})
        log("row_loaded", row=spec["row"], lines=len(one), core_dropped=dropped)
        lines.extend(one)
    ref_lines = []
    if cfg.get("tfidf_reference"):
        ref_lines, dropped = canon.apply_core_view(canon.load_row({**cfg["tfidf_reference"], "row": TFIDF_REF_ROW}))
        sizes.append({"row": TFIDF_REF_ROW, "kind": "admission", "lines": len(ref_lines), "core_dropped": dropped})
        log("row_loaded", row=TFIDF_REF_ROW, lines=len(ref_lines), core_dropped=dropped)

    pairs = dict(cfg.get("pairs") or {})
    bases = [x for x in a.neg.split(",") if x]
    nc_lines = []
    for base in bases:
        part = [x for x in lines if x["row"] == base]
        if not part:
            raise SafeError("neg_base_missing", row=base)
        made = negcontrols.build(part)
        nc_lines.extend(made)
        for name in negcontrols.NAMES:
            row = f"{name}@{base}"
            kinds[row] = "admission"
            if base in pairs:
                pairs[row] = pairs[base]
            elif base == cfg.get("reference"):
                pairs[row] = base
    all_lines = lines + nc_lines
    totals = {}
    for x in lines:
        t = totals.setdefault(x["row"], {"admissions": 0, "chars": 0})
        t["admissions"] += 1
        if x["kind"] == "admission":
            t["chars"] += sum(len(d["text"]) for d in x["docs"])
        else:
            t["chars"] += sum(len(str(v)) for v in x["record"].values() if v is not None)

    row_codes = {}
    for spec in cfg["rows"]:
        one = load_codes(spec.get("codes") or cfg.get("codes"))
        row_codes[spec["row"]] = one
        for base in bases:
            if base == spec["row"]:
                for name in negcontrols.NAMES:
                    row_codes[f"{name}@{base}"] = one
    config = {"reference": cfg.get("reference"), "pairs": pairs, "codes": row_codes,
              "tfidf_reference_row": TFIDF_REF_ROW if ref_lines else None}

    def encode(rows):
        return "\n".join(json.dumps(x, ensure_ascii=False) for x in rows).encode("utf-8")

    payload = encode(all_lines)
    prefix = os.path.join(raw, "all")
    status = []
    for reader in a.readers.split(","):
        env = dict(os.environ)
        env["PYTHONPATH"] = paths.pythonpath_for(reader, root)
        env["PYTHONIOENCODING"] = "utf-8"
        cmd = [sys.executable, "-B", "-m", "bench.worker", reader, prefix, json.dumps(config, ensure_ascii=False)]
        proc = subprocess.run(cmd, input=encode(all_lines + ref_lines) if reader == "g2" and ref_lines else payload,
                              env=env, cwd=root, stderr=subprocess.PIPE)
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] if proc.stderr else []
        status.append({"reader": reader, "returncode": proc.returncode, "last_log": tail[0][:200] if tail else ""})
        log("reader_done", reader=reader, returncode=proc.returncode, last_log=tail[0][:200] if tail else "")

    eff = efficiency(cfg, totals)
    board = aggregate.scoreboard(raw, kinds, eff)
    tables = os.path.join(a.out, "tables")
    aggregate.write_csv(os.path.join(tables, "S0_scoreboard.csv"), board)
    aggregate.write_csv(os.path.join(tables, "S0_rows.csv"), sizes, ["row", "kind", "lines", "core_dropped"])
    aggregate.write_csv(os.path.join(tables, "S0_readers.csv"), status, ["reader", "returncode", "last_log"])
    if bases:
        matrix, summ = aggregate.negcontrol_matrix(board, bases, negcontrols.NAMES, cfg.get("reference"))
        aggregate.write_csv(os.path.join(tables, "S0_neg_matrix.csv"), matrix,
                            ["negative_control", "base_row", "binding", "group", "metric", "nc_value", "base_value", "base_in_floor", "nc_in_floor",
                             "rejected", "needs_paired_measure"])
        aggregate.write_csv(os.path.join(tables, "S0_neg_summary.csv"), summ, ["kind", "name", "binding", "count", "pass"])
        aggregate.write_csv(os.path.join(tables, "S0_neg_paired.csv"), aggregate.paired_table(matrix),
                            ["group", "metric", "count", "negative_controls"])
    with open(os.path.join(a.out, "tables", "S0_version.json"), "w", encoding="utf-8") as f:
        json.dump({"bench": VERSION, "readers": a.readers.split(","), "neg_bases": bases}, f, ensure_ascii=False)
    if any(s["returncode"] for s in status):
        raise SafeError("reader_failed", failed=[s["reader"] for s in status if s["returncode"]])


if __name__ == "__main__":
    guarded(main)
