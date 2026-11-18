"""关1 单篇事实一致尺子（R 层）。核心规则由总经理编写，见 OWNERSHIP.md。"""
from .adapters import render_summary7
from .compare import PAIRS, evaluate_document
from .extractors import RULE_VERSION, extract_facts
from .sections import is_first_progress, split_sections

__all__ = ["render_summary7", "PAIRS", "RULE_VERSION", "evaluate_document", "extract_facts",
           "is_first_progress", "split_sections"]
