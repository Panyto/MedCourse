# -*- coding: utf-8 -*-
"""规则层对比的自检：假规则、假模型，不连任何服务。规划用编的两行。"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_b import rule_compare as rc  # noqa: E402

RULES = {
    "anesthesia": lambda name: "局部麻醉" if "造影" in name else ("腰硬联合麻醉" if "痔" in name else "全身麻醉"),
    "no_incision": lambda anes: anes in ("局部麻醉", "静脉麻醉"),
    "sched": {"影像": 1, "培养": 3},
    "kind_of": {"CT": "影像", "血培养": "培养", "血常规": "检验"},
    "stop_reason": lambda c: {"抗菌药": "抗感染疗程已足，予停用"}.get(c, "症状改善，予停用"),
    "stop_classes": ("抗菌药",),
    "stop_options": ("抗感染疗程已足，予停用", "症状改善，予停用"),
}

JOURNEYS = [
    json.dumps({"procedures": [{"name": "冠脉造影", "kind": "操作"}], "meds": [{"class": "抗菌药", "chronic": False}, {"class": "降压药", "chronic": True}],
                "exams": [{"key": "CT"}]}, ensure_ascii=False),
    json.dumps({"procedures": [{"name": "混合痔切除术", "kind": "手术"}], "meds": [{"class": "止血药", "chronic": False}], "exams": [{"key": "血常规"}]},
               ensure_ascii=False),
]


class FakeClient:
    """按提示词里的名称给固定答案；sampling 必须是温度 0 的那组。"""
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def generate(self, *, sampling, seed, prompt, max_tokens, schema):
        assert sampling["temperature"] == 0.0 and seed == 20261110 and schema["type"] == "object"
        self.calls.append(prompt)
        for k, v in self.answers.items():
            if k in prompt:
                return {"text": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}
        return {"text": "不是 JSON"}


class TestRuleCompare(unittest.TestCase):
    def test_items(self):
        procs, classes, keys = rc.items_from_journeys(JOURNEYS)
        self.assertEqual(procs, {"冠脉造影", "混合痔切除术"})
        self.assertEqual(classes, {"抗菌药", "止血药"})          # 长期用药（降压药）不算
        self.assertEqual(keys, {"CT", "血常规"})

    def test_compare(self):
        procs, classes, keys = rc.items_from_journeys(JOURNEYS)
        client = FakeClient({
            "冠脉造影": {"anesthesia": "局部麻醉", "incision": False, "drain": False},
            "混合痔切除术": {"anesthesia": "其他椎管内麻醉", "incision": True, "drain": False},
            "开了CT": {"days": 1}, "开了血培养": {"days": 5}, "开了血常规": {"days": 0},
            "使用抗菌药": {"reason": "抗感染疗程已足，予停用"}, "使用止血药": {"reason": "抗感染疗程已足，予停用"},
        })
        rows, s = rc.compare(client, RULES, procs, classes, keys)
        a = s["anesthesia"]
        self.assertEqual((a["n"], a["agree"], a["agree_neuraxial_merged"], a["incision_agree"], a["drain_agree"], a["parse_fail"]), (2, 1, 2, 2, 1, 0))
        lag = s["lag"]["by_kind"]
        self.assertEqual((lag["影像"]["agree"], lag["影像"]["agree_in_a256"]), (1, 1))
        self.assertEqual((lag["培养"]["agree"], lag["培养"]["over"]), (0, 1))     # 5 天 > 规则上限 3
        self.assertEqual((lag["检验"]["agree"], lag["检验"]["rule_max"]), (1, 0))
        self.assertEqual(s["stop"], {"n": 2, "agree": 1, "parse_fail": 0})
        # 汇总里不带任何名称
        dumped = json.dumps(s, ensure_ascii=False)
        for name in ("冠脉造影", "混合痔", "止血药"):
            self.assertNotIn(name, dumped)
        self.assertEqual(len(rows), 2 + 3 + 2)

    def test_parse_fail(self):
        rows, s = rc.compare(FakeClient({}), RULES, {"某术"}, set(), set())
        self.assertEqual(s["anesthesia"]["parse_fail"], 1)
        self.assertEqual(s["lag"]["parse_fail"], 3)
        self.assertEqual(s["stop"]["parse_fail"], 1)
        self.assertTrue(all(r[-1] == "parse_fail" for r in rows))


if __name__ == "__main__":
    unittest.main()
