#!/usr/bin/env python3
"""Sync active MedCourse packages and public Route B source texts from workspace, verify MANIFESTs and public safety."""
import argparse
import hashlib
import os
import re
import shutil
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = REPO_ROOT.parent

PACKAGE_MAP = [
    ("产品开发/S1/medcourse_core", "packages/medcourse_core"),
    ("产品开发/路线B/route_b_kit", "packages/route_b_kit"),
    ("产品开发/S0/bench_core", "packages/bench_core"),
    ("产品开发/S1/gm_tools", "packages/gm_tools"),
    ("产品开发/S1/gold_kit", "packages/gold_kit"),
    ("产品开发/S5/s5_kit", "packages/s5_kit"),
    ("关1_单篇事实一致/2_代码/ruler_g1_core", "packages/rules/ruler_g1_core"),
    ("关1_单篇事实一致/2_代码/b7_rules", "packages/rules/b7_rules"),
    ("关2_跨文书时序一致/2_代码/g2_core", "packages/rules/g2_core"),
    ("产品开发/S0/g3_core", "packages/rules/g3_core"),
    ("关3_临床正确与安全/2_代码/g3_core", "packages/rules/g3_core_v02"),
    ("关4关5_写全不注水与真实感/2_代码/g45_core", "packages/rules/g45_core"),
    ("关6_多样性与长尾/2_代码/g6_core", "packages/rules/g6_core"),
]

PUBLIC_SOURCE_SUBDIRS = [
    ("产品开发/路线B/原文/nhc2019_text", "public_data/route_b_sources/nhc2019_text"),
    ("产品开发/路线B/原文/cma_county", "public_data/route_b_sources/cma_county"),
    ("产品开发/路线B/原文/abx_guideline_2015", "public_data/route_b_sources/abx_guideline_2015"),
]

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

FORBIDDEN_EXTS = {".tar.gz", ".tgz", ".zip", ".pdf", ".pem", ".key", ".env"}
FORBIDDEN_NAME_PATS = [
    re.compile(r"^stats_v", re.I),
    re.compile(r"^events.*\.(jsonl|json|csv|tsv)$", re.I),
    re.compile(r"deid_run9", re.I),
    re.compile(r"^SYNTH_"),
]
ALLOWED_JSONL = {
    "packages/rules/g6_core/q6_kit/official/example-ref.jsonl",
}
FORBIDDEN_TEXT_PATS = [
    (re.compile(r"\b122\.207\.\d+\.\d+\b"), "public_server_ip"),
    (re.compile(r"\b100\.79\.\d+\.\d+\b"), "internal_server_ip"),
    (re.compile(r"(?:sk-|hf_|ghp_|gho_)[A-Za-z0-9_]{16,}"), "secret_token"),
]
MAX_SINGLE_FILE_BYTES = 2 * 1024 * 1024  # 2 MB cap (public icd_names.csv is ~1.0 MB)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def copy_tree_clean(src: Path, dst: Path) -> int:
    if not src.is_dir():
        raise FileNotFoundError(f"Missing source package directory: {src}")
    if dst.exists():
        shutil.rmtree(dst, onerror=_remove_readonly)
    copied = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for fn in files:
            sfile = Path(root) / fn
            if sfile.suffix in SKIP_SUFFIXES:
                continue
            dfile = target_dir / fn
            shutil.copy2(sfile, dfile)
            os.chmod(dfile, stat.S_IWRITE | stat.S_IREAD)
            copied += 1
    return copied


def copy_public_texts(src: Path, dst: Path) -> int:
    if not src.is_dir():
        raise FileNotFoundError(f"Missing public source directory: {src}")
    if dst.exists():
        shutil.rmtree(dst, onerror=_remove_readonly)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and (f.suffix.lower() == ".txt" or f.name == "SOURCES.json"):
            dfile = dst / f.name
            shutil.copy2(f, dfile)
            os.chmod(dfile, stat.S_IWRITE | stat.S_IREAD)
            copied += 1
    return copied


def verify_manifests() -> list:
    checked = []
    for manifest_path in sorted((REPO_ROOT / "packages").rglob("MANIFEST.sha256")):
        pkg_dir = manifest_path.parent
        count = 0
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            digest, rel = re.split(r"\s+\*?", line, maxsplit=1)
            target = pkg_dir / rel
            if not target.is_file() and rel.startswith(pkg_dir.name + "/"):
                target = pkg_dir.parent / rel
            if not target.is_file():
                raise AssertionError(f"MANIFEST missing file: {target}")
            actual = sha256_file(target)
            if actual != digest:
                raise AssertionError(f"MANIFEST sha256 mismatch in {pkg_dir.name}/{rel}: {actual} != {digest}")
            count += 1
        checked.append((str(pkg_dir.relative_to(REPO_ROOT)), count))
    return checked


def run_public_safety_gate() -> int:
    total_files = 0
    for f in sorted(REPO_ROOT.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(REPO_ROOT)
        if ".git" in rel.parts:
            continue
        total_files += 1
        low_name = f.name.lower()
        for ext in FORBIDDEN_EXTS:
            if low_name.endswith(ext):
                raise AssertionError(f"Forbidden extension in public repo: {rel}")
        if f.suffix.lower() == ".jsonl" and rel.as_posix() not in ALLOWED_JSONL:
            raise AssertionError(f"Unallowlisted jsonl in public repo: {rel}")
        for pat in FORBIDDEN_NAME_PATS:
            if pat.search(f.name):
                raise AssertionError(f"Forbidden filename pattern in public repo: {rel}")
        size = f.stat().st_size
        if size > MAX_SINGLE_FILE_BYTES:
            raise AssertionError(f"File exceeds {MAX_SINGLE_FILE_BYTES} bytes ({size}): {rel}")
        if f.suffix.lower() in {".py", ".md", ".json", ".jsonl", ".csv", ".txt", ".sh", ".ipynb", ".toml", ".yml", ".yaml"}:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            for pat, label in FORBIDDEN_TEXT_PATS:
                m = pat.search(txt)
                if m:
                    raise AssertionError(f"Forbidden text ({label}) in {rel}: {m.group(0)}")
    return total_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Source workspace root directory")
    ap.add_argument("--verify-only", action="store_true", help="Only verify manifests and safety gate without copying")
    args = ap.parse_args()
    ws = Path(args.workspace).resolve()

    if not args.verify_only:
        total_pkg = 0
        for src_rel, dst_rel in PACKAGE_MAP:
            n = copy_tree_clean(ws / src_rel, REPO_ROOT / dst_rel)
            total_pkg += n
            print(f"Synced package {dst_rel}: {n} files")
        total_pub = 0
        for src_rel, dst_rel in PUBLIC_SOURCE_SUBDIRS:
            n = copy_public_texts(ws / src_rel, REPO_ROOT / dst_rel)
            total_pub += n
            print(f"Synced public texts {dst_rel}: {n} files")
        print(f"Total synced: {total_pkg} package files, {total_pub} public text files")

    manifests = verify_manifests()
    for pkg_rel, n in manifests:
        print(f"MANIFEST verified OK: {pkg_rel} ({n} entries)")
    total_files = run_public_safety_gate()
    print(f"PUBLIC SAFETY GATE PASSED: {total_files} files audited, 0 violations.")


if __name__ == "__main__":
    main()
