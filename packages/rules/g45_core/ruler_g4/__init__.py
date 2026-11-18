"""关4 规范完整而不注水的尺子（g4-rules-0.2）。

三样读数同报（关卡谱关4）：
- 要素覆盖：coverage_care7（L1–L9 与 LLM-CARe 清单同一批事项，O1–O5 是规范有、它的清单没有的）；
- 具体度：info_units / analyze_care7 的 iu_per_kchar（每千字不重复的信息单元数）；
- 注水：empty_share（没有任何信息单元的句子占的字数比例）、template_rates（跨病历套话句）、dup_share（段间重复）。
整次住院用 analyze_admission（每篇一行，含关2 口径的篇间复制率）。
"""
from .ruler import (FIELDS, OFF_LIST, ON_LIST, analyze_admission, analyze_care7, analyze_text, containment,
                    coverage_care7, doc_role, info_units, nchars, norm_key, sentences, template_rates, time_anchors,
                    warning_spans)

RULES_VERSION = "g4-rules-0.2"
