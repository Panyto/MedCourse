#!/usr/bin/env python3
"""Cross-platform test runner for all active MedCourse packages and MANIFEST checks."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKGS = REPO_ROOT / "packages"
RULES = PKGS / "rules"


def base_env() -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "BENCH_G1": str(RULES / "ruler_g1_core"),
        "BENCH_B7": str(RULES / "b7_rules"),
        "BENCH_G2": str(RULES / "g2_core"),
        "BENCH_G3": str(RULES / "g3_core"),
        "BENCH_G3_OLD": str(RULES / "g3_core_v02"),
        "BENCH_G45": str(RULES / "g45_core"),
        "BENCH_G6": str(RULES / "g6_core"),
        "MC_CORE": str(PKGS / "medcourse_core"),
    })
    return env


def run_step(name: str, cmd: list, cwd: Path, extra_py_paths: list = None, unset_vars: list = None):
    print(f"\n=== [{name}] ===", flush=True)
    env = base_env()
    if extra_py_paths:
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in extra_py_paths)
    else:
        env.pop("PYTHONPATH", None)
    for k in unset_vars or ():
        env.pop(k, None)
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"Step failed ({r.returncode}): {name}")


def main():
    run_step(
        "0. MANIFEST & Public Safety Gate",
        [sys.executable, "-B", str(REPO_ROOT / "scripts/sync_from_workspace.py"), "--verify-only"],
        REPO_ROOT,
    )
    run_step(
        "1. medcourse_core",
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
        PKGS / "medcourse_core",
    )
    run_step(
        "2. route_b_kit",
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
        PKGS / "route_b_kit",
        extra_py_paths=[PKGS / "route_b_kit"],
    )
    run_step(
        "3. gold_kit",
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
        PKGS / "gold_kit",
        extra_py_paths=[PKGS / "gold_kit", PKGS / "medcourse_core"],
    )
    run_step(
        "4. gm_tools",
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
        PKGS / "gm_tools",
        extra_py_paths=[PKGS / "gm_tools", PKGS / "medcourse_core"],
    )
    has_sklearn = importlib.util.find_spec("sklearn") is not None
    bench_unset = []
    if not has_sklearn:
        print("\n[Note] scikit-learn not found in current Python; skipping bench_core TestEndToEnd (requires sklearn for g6/c2).", flush=True)
        bench_unset = ["BENCH_G2"]
    run_step(
        "5. bench_core",
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
        PKGS / "bench_core",
        unset_vars=bench_unset,
    )
    run_step(
        "6. s5_kit",
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
        PKGS / "s5_kit",
        extra_py_paths=[
            PKGS / "s5_kit",
            PKGS / "medcourse_core",
            PKGS / "bench_core",
            RULES / "ruler_g1_core",
            RULES / "b7_rules",
            RULES / "g2_core",
            RULES / "g3_core",
            RULES / "g45_core",
            RULES / "g6_core",
        ],
    )
    print("\nALL REPOSITORY TESTS AND MANIFEST CHECKS PASSED!", flush=True)


if __name__ == "__main__":
    main()
