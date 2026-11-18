"""规则包的位置都从环境变量读，和 bench_core 同名。g3_core 必须排在 g45_core 前面：g45_core 里带一份 ruler_g3 0.2，闸门要用 0.3。"""
import os
import sys

_ORDER = ("BENCH_G3", "BENCH_G2", "BENCH_B7", "BENCH_G45")


def setup(need=_ORDER):
    missing = [k for k in need if not os.path.isdir(os.environ.get(k, ""))]
    if missing:
        raise RuntimeError("missing_core:" + ",".join(missing))
    for k in reversed(_ORDER):
        if k in need and os.environ[k] not in sys.path:
            sys.path.insert(0, os.environ[k])
