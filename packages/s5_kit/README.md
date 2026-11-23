# `s5_kit` (S5/S6 Downstream Utility, Mechanism Attribution & Privacy Audit Suite)

`s5_kit` implements the Stage-1 / Stage-2 offline planning, counterfactual generation orchestration, zero-trust semantic ledger auditing, fixed-rule mechanism attribution, downstream `TSTR` benchmarking (`TF-IDF + LR` & `BAAI/bge-small-zh-v1.5`), and calibrated privacy evaluation (`S6-1` MIA & `S6-2` clinician blinding) on top of frozen [`medcourse_core`](../medcourse_core/README.md) `v0.3.7`.

## Module Overview

- **Planning & Generation**:
  - `s5_common.py`: Shared pathway configuration (`PATHWAY_SPECS` across 17 representative diseases), `build_pathway_stats`, oracle prompt builder, and atomic `SMOKE10` input-fingerprint stamp helpers.
  - `plan_s5_corpora.py`: Pre-plans counterfactual `S5-1` trajectories (shared across `Full`, `VisibilityOnly`, `OracleLeak`) and 3-prior `S5-2` trajectories (`Route-B` `--kb-only`, `Route-A-Only`, `Route-AB`).
  - `gen_s5_shard.py`, `gen_direct_32b.py`, `run_s5_r2_master.sh`, `run_*.sh`: Controlled multi-arm generation drivers with `--save-first-try` (`attempt=0` snapshot) support and `fold 9` write guards.
- **Zero-Trust Auditing & Mechanism Attribution**:
  - `build_v037_ledgers.py`: Re-verifies planned trajectories without trusting pre-stamped `can_confirm`/`incidental_findings` flags, audits comorbidity drop rates across chapters/diseases, and runs `gate.audit_item_numbers` across all 7 document roles.
  - `analyze_mechanism_contribution.py`: Isolates core compiler (`First-Try`) vs. `gate.py` rewriting (`Post-Gate`) and temporal visibility (`OracleLeak` vs `VisOnly` vs `Full`) under fixed `v0.3.7` rules and shared planned trajectories.
  - `audit_s5_intrinsic.py` & `audit_s5_2_17d.py`: Intrinsic clinical safety/oracle leakage audit and unified 17-pathway D0-1 non-past text probe.
- **Downstream TSTR & Privacy Calibration**:
  - `t123_bench.py` & `t12_neural_bench.py`: Two-stage `--fit-and-freeze` -> read-only `--eval-frozen` benchmarks for `T1a/T1b` (late surgery escalation), `T2a/T2b` (diagnostic assertion tracking with `2x2` `{Full, VisOnly} x {planner, text_rule}` ablation), and `T3` (drug-group & dose extraction), with strict `fold 9` Stage-1 guardrails.
  - `build_synth_exported_release.py`, `s6_mia_audit.py`, `build_s6_blind_pack.py`: Full-reference 30-gram export filtering, positive-control calibrated `S6-1` MIA suite, and `S6-2` paired blind clinical review pack builder.
- **Unit Tests**:
  - `test_s5_kit.py`: 18 unit tests covering all 17 Route-B pathways, parity checks, `T1/T2/T3` filter & masking guards, `SMOKE10` stamp atomicity, semantic fault-injection on `build_v037_ledgers.py`, and 4-cell mechanism attribution (`test_v21_mechanism_contribution_decomposition`).

## Privacy & Safety Note

This directory contains **code and synthetic unit-test fixtures only**. Real de-identified hospital stays, private `stats_v*.json` tables, and `fold 9` holdout data never enter this repository.
