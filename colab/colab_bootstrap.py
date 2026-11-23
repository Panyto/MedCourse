#!/usr/bin/env python3
"""Colab Ultra bootstrap & safe runner for non-sensitive public/synthetic MedCourse tasks."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIVACY_GUARD_PAT = re.compile(r"(deid_run9|stats_v0\d|events_norm|slots_real)", re.I)


class PrivacyGuardError(RuntimeError):
    """Raised when restricted hospital/patient-derived filenames are passed on cloud Colab."""


def _assert_public_or_synthetic(value: str, label: str = "path"):
    if value and PRIVACY_GUARD_PAT.search(str(value)):
        raise PrivacyGuardError(
            f"Blocked {label}={value!r}: patient/hospital-derived data must stay on controlled institutional servers, never Google Colab."
        )


def setup_env(repo_root=None) -> dict:
    """Configure environment variables and sys.path for all MedCourse packages."""
    root = Path(repo_root or REPO_ROOT).resolve()
    pkgs = root / "packages"
    rules = pkgs / "rules"
    env_map = {
        "PYTHONIOENCODING": "utf-8",
        "BENCH_G1": str(rules / "ruler_g1_core"),
        "BENCH_B7": str(rules / "b7_rules"),
        "BENCH_G2": str(rules / "g2_core"),
        "BENCH_G3": str(rules / "g3_core"),
        "BENCH_G3_OLD": str(rules / "g3_core_v02"),
        "BENCH_G45": str(rules / "g45_core"),
        "BENCH_G6": str(rules / "g6_core"),
        "MC_CORE": str(pkgs / "medcourse_core"),
    }
    for k, v in env_map.items():
        os.environ[k] = v

    py_roots = [
        pkgs / "medcourse_core",
        pkgs / "route_b_kit",
        pkgs / "bench_core",
        pkgs / "gm_tools",
        pkgs / "gold_kit",
        pkgs / "s5_kit",
    ]
    for p in reversed(py_roots):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    os.environ["PYTHONPATH"] = os.pathsep.join(str(p) for p in py_roots)
    return {
        "repo_root": str(root),
        "public_sources": str(root / "public_data" / "route_b_sources"),
        "diseases_cfg": str(pkgs / "route_b_kit" / "config" / "diseases_v0.json"),
        "exams_json": str(rules / "g2_core" / "ruler_g2" / "config" / "exams.json"),
        **env_map,
    }


def inspect_gpu() -> dict:
    """Inspect attached GPU hardware and verify VRAM suitability for Colab Ultra."""
    if not shutil.which("nvidia-smi"):
        print("[GPU] nvidia-smi not found (CPU runtime).")
        return {"available": False}
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    gpus = []
    for line in r.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": int(parts[0]), "name": parts[1], "total_mib": int(parts[2]), "free_mib": int(parts[3])})
    for g in gpus:
        vram_gb = g["total_mib"] / 1024.0
        note = "OK for Qwen/Qwen3-32B-AWQ (fits A100/L4) & Qwen/Qwen3-8B bf16" if vram_gb >= 22.0 else "WARNING: <22 GiB (switch Colab Ultra runtime to A100 or L4)"
        print(f"[GPU {g['index']}] {g['name']} | Total: {vram_gb:.1f} GiB | Free: {g['free_mib']/1024:.1f} GiB -> {note}")
    return {"available": bool(gpus), "gpus": gpus}


def wait_vllm(port: int = 8101, timeout_s: int = 420, proc=None, log_path: str = "") -> list:
    """Wait until loopback vLLM /v1/models responds HTTP 200."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = Path(log_path).read_text(encoding="utf-8", errors="ignore")[-2500:] if log_path and Path(log_path).exists() else ""
            raise RuntimeError(f"vLLM exited early with code {proc.returncode}.\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("id") for m in data.get("data", [])]
                    print(f"[vLLM] Ready on {url} -> models={models}")
                    return models
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(3)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for {url}")


def start_vllm(
    model: str = "Qwen/Qwen3-8B",
    port: int = 8101,
    served_name: str = None,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.88,
    dtype: str = "auto",
    download_dir: str = None,
    log_path: str = "/tmp/medcourse_vllm.log",
    timeout_s: int = 420,
):
    """Launch loopback OpenAI-compatible vLLM server compatible with medcourse.llm.Client."""
    _assert_public_or_synthetic(model, "model")
    if download_dir:
        _assert_public_or_synthetic(download_dir, "download_dir")
        os.makedirs(download_dir, exist_ok=True)
    if dtype == "bfloat16" and any(k in (model or "").lower() for k in ("awq", "gptq")):
        dtype = "auto"
    try:
        return {"proc": None, "models": wait_vllm(port=port, timeout_s=2)}
    except (TimeoutError, urllib.error.URLError):
        pass

    served = served_name or ("Qwen3-8B" if model.endswith("Qwen3-8B") else model.split("/")[-1])
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", served,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--no-enable-log-requests",
    ]
    if download_dir:
        cmd.extend(["--download-dir", str(download_dir)])

    env = dict(os.environ)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.makedirs(Path(log_path).parent, exist_ok=True)
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    print(f"[vLLM] Starting pid={proc.pid} model={model} as {served!r} on 127.0.0.1:{port} (log={log_path})...")
    models = wait_vllm(port=port, timeout_s=timeout_s, proc=proc, log_path=log_path)
    return {"proc": proc, "port": port, "served_name": served, "models": models, "log_path": log_path}


def stop_vllm(handle=None):
    """Stop vLLM process launched by start_vllm."""
    proc = handle.get("proc") if isinstance(handle, dict) else handle
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("[vLLM] Stopped.")


def run_route_b(
    out_dir: str,
    only: str = "",
    concurrency: int = 6,
    rounds: int = 3,
    k: int = 3,
    extra: int = 3,
    port: int = 8101,
    model: str = "Qwen3-8B",
):
    """Execute public clinical-pathway Route B extraction via route_b.build_kb_agentic."""
    _assert_public_or_synthetic(out_dir, "out_dir")
    paths = setup_env()
    out_p = Path(out_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-B", "-m", "route_b.build_kb_agentic",
        "--src", paths["public_sources"],
        "--config", paths["diseases_cfg"],
        "--exams", paths["exams_json"],
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--model", model,
        "--out", str(out_p),
        "--rounds", str(rounds),
        "--k", str(k),
        "--extra", str(extra),
        "--concurrency", str(concurrency),
    ]
    if only:
        cmd.extend(["--only", only])
    r = subprocess.run(cmd, env=dict(os.environ))
    if r.returncode != 0:
        raise RuntimeError(f"route_b.build_kb_agentic failed with exit code {r.returncode}")
    return out_p


def run_synthetic_fixture_demo(
    out_dir: str,
    n: int = 4,
    use_llm: bool = False,
    port: int = 8101,
    model: str = "Qwen3-8B",
    seed: int = 20261118,
) -> dict:
    """Run an end-to-end synthetic-fixture MedCourse generation pass (zero real patient data)."""
    _assert_public_or_synthetic(out_dir, "out_dir")
    paths = setup_env()
    mc_tests = str(Path(paths["MC_CORE"]) / "tests")
    if mc_tests not in sys.path:
        sys.path.insert(0, mc_tests)
    import fixtures_mc  # noqa: E402
    from medcourse import pipeline, writer  # noqa: E402
    from medcourse.contract import admission_line  # noqa: E402

    st = fixtures_mc.stats()
    w = writer.LLMWriter(f"http://127.0.0.1:{port}/v1", model) if use_llm else writer.TemplateWriter()
    out_p = Path(out_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    tally = {"planned": n, "kept": 0, "discarded": 0}
    adms = []
    for i in range(n):
        jid = f"SYN-COLAB-{i:04d}"
        j, docs, fails = pipeline.generate_one(jid, st, w, seed + i, None, tally)
        if docs is None:
            tally["discarded"] += 1
        else:
            tally["kept"] += 1
            adms.append(admission_line(j, docs))
    (out_p / "admissions_synth_demo.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in adms) + ("\n" if adms else ""),
        encoding="utf-8",
    )
    summary = {
        "mode": "llm" if use_llm else "template",
        "kept": tally["kept"],
        "discarded": tally["discarded"],
        "total_docs": sum(len(a["docs"]) for a in adms),
        "out_file": str(out_p / "admissions_synth_demo.jsonl"),
    }
    (out_p / "demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Demo]", summary)
    return summary


def run_kb_only_demo(
    out_dir: str,
    n: int = 4,
    use_llm: bool = False,
    port: int = 8101,
    model: str = "Qwen3-32B-AWQ",
    seed: int = 20261122,
) -> dict:
    """Run 100% public Route-B (--kb-only, kb_v22.json) NHC-standard synthesis without hospital stats."""
    _assert_public_or_synthetic(out_dir, "out_dir")
    setup_env()
    from medcourse import kb, pipeline, writer  # noqa: E402
    from medcourse.contract import admission_line  # noqa: E402

    st = kb.build_b_only_stats()
    if use_llm:
        w = writer.LLMWriter(f"http://127.0.0.1:{port}/v1", model)
        w.no_filler2 = True
    else:
        w = writer.TemplateWriter()
    out_p = Path(out_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    tally = {"planned": n, "kept": 0, "discarded": 0}
    adms = []
    for i in range(n):
        jid = f"KBONLY-COLAB-{i:04d}"
        j, docs, fails = pipeline.generate_one(jid, st, w, seed + i, None, tally)
        if docs is None:
            tally["discarded"] += 1
        else:
            tally["kept"] += 1
            adms.append(admission_line(j, docs))
    out_file = out_p / "admissions_kb_only_demo.jsonl"
    out_file.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in adms) + ("\n" if adms else ""), encoding="utf-8")
    summary = {
        "mode": "kb_only_" + ("llm" if use_llm else "template"),
        "kept": tally["kept"],
        "discarded": tally["discarded"],
        "total_docs": sum(len(a["docs"]) for a in adms),
        "out_file": str(out_file),
    }
    (out_p / "kb_only_demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[KB-Only Demo]", summary)
    return summary
