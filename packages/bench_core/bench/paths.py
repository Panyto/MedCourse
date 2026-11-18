"""各规则包的位置都从环境变量读，子进程只把本读数需要的包放上 sys.path，避免 ruler_g3 0.2、0.3 撞车。"""
import os

ENV = {"g1": "BENCH_G1", "b7": "BENCH_B7", "g2": "BENCH_G2", "g3": "BENCH_G3", "g3old": "BENCH_G3_OLD",
       "g45": "BENCH_G45", "g45old": "BENCH_G45_OLD", "g6": "BENCH_G6"}

# 读数 → 依赖的包，顺序就是 sys.path 顺序
READER_CORES = {
    "extra": [],
    "g1": ["g1", "b7", "g45"],
    "g2": ["g2"],
    "g3": ["g3"],
    "g3old": ["g3old"],
    "g4": ["g45", "g1"],
    "g4old": ["g45old", "g1"],
    "c2": ["g45", "g1"],
    "g6": ["g6", "g1"],
}


def core_path(name):
    p = os.environ.get(ENV[name])
    if not p or not os.path.isdir(p):
        raise RuntimeError(f"missing_core:{ENV[name]}")
    return p


def pythonpath_for(reader, bench_root):
    return os.pathsep.join([bench_root] + [core_path(c) for c in READER_CORES[reader]])
