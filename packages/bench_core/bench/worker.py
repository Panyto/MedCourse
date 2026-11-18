"""子进程入口：python -B -m bench.worker 读数名 输出前缀 配置json。行从 stdin 进来（不落盘），读数写到输出前缀下。"""
import importlib
import json
import sys

from .safe import guarded, numeric_only


def main():
    reader, prefix, config = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    stdin = open(sys.stdin.fileno(), encoding="utf-8", closefd=False)
    rows = [json.loads(line) for line in stdin if line.strip()]
    mod = importlib.import_module("bench.readers." + ("g3" if reader == "g3old" else reader))
    per, whole = mod.run(rows, config)
    with open(f"{prefix}.{reader}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in per:
            f.write(json.dumps(numeric_only(r), ensure_ascii=False, sort_keys=True) + "\n")
    with open(f"{prefix}.{reader}.row.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in whole:
            f.write(json.dumps(numeric_only(r), ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    guarded(main)
