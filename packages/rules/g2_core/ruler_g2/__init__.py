"""关2 跨文书时序一致尺子（ruler_g2）。核心规则由总经理编写，见 OWNERSHIP.md。依赖同目录下的 ruler_g1（v0.5，未改动）。"""
from .readings import HARD, K_GROUP, READINGS, RULE_VERSION, evaluate_admission, hit_span, k_any
from .timeline import Admission, Doc, build_real, build_synth, classify_progress, classify_progress_detail, core_set, surgery_day

__all__ = ["HARD", "K_GROUP", "READINGS", "RULE_VERSION", "evaluate_admission", "hit_span", "k_any",
           "Admission", "Doc", "build_real", "build_synth", "classify_progress", "classify_progress_detail", "core_set", "surgery_day"]
