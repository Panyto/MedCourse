# -*- coding: utf-8 -*-
"""自检。示意文本是我自己编的，格式仿路径，不是任何真实路径的原文。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_b import provenance as pv  # noqa: E402
from route_b import schema, sections  # noqa: E402

TOY = """示意病临床路径（示意版）
一、示意病临床路径标准住院流程
（一）适用对象
第一诊断为示意病（ICD-10：X99.0）。
（二）诊断依据
1.症状：右上腹痛，可伴发热。
2.体征：局部压痛。
（三）治疗方案的选择
示意手术。
（四）标准住院日为≤７天。
（五）进入路径标准
（六）术前准备（术前评估）１－２天
1.必需的检查项目：血常规、肝功能、腹部超声。
（七）抗菌药物选择与使用时机
用至体温正常、症状消退后72-96小时。
（八）手术日为入院第３天
1.麻醉方式：全身麻醉。
（九）术后住院恢复３－４天。
（十）出院标准
（十一）变异及原因分析
二、示意病临床路径表单
时间 住院第1天 住院第2天（术前准备日） 住院第3天（手术日） 住院第4-6天（术后1-3日） 住院第7天（出院日）
主要诊疗工作 □询问病史 □手术
重点医嘱 长期医嘱 临时医嘱
"""


class TestSections(unittest.TestCase):
    def test_cn_numbers(self):
        self.assertEqual([sections.cn_to_int(s) for s in ("一", "九", "十", "十一", "十二", "二十")], [1, 9, 10, 11, 12, 20])

    def test_split(self):
        secs, form = sections.split_sections(TOY)
        self.assertEqual([s["no"] for s in secs], list(range(1, 12)))
        kinds = {s["no"]: s["kind"] for s in secs}
        self.assertEqual((kinds[1], kinds[2], kinds[4], kinds[7], kinds[8], kinds[10], kinds[11]),
                         ("target", "dx_basis", "los", "antibiotic", "surgery_day", "discharge", "variation"))
        self.assertIsNotNone(form)
        self.assertTrue(all(s["end"] <= form for s in secs))
        # 小节内容不串：第（八）节里有麻醉，第（九）节没有
        s8 = next(s for s in secs if s["no"] == 8)
        s9 = next(s for s in secs if s["no"] == 9)
        self.assertIn("全身麻醉", TOY[s8["start"]:s8["end"]])
        self.assertNotIn("麻醉", TOY[s9["start"]:s9["end"]])

    def test_form_days(self):
        _, form = sections.split_sections(TOY)
        self.assertEqual(sections.form_days(TOY, form), [(1, 1), (2, 2), (3, 3), (4, 6), (7, 7)])

    def test_inline_headings(self):
        # 网页抽出的文本没有换行，标题连在一起；正文里零散的"（三）"不连续，不算
        flat = TOY.replace("\n", " ")
        secs, form = sections.split_sections(flat)
        self.assertEqual([s["no"] for s in secs], list(range(1, 12)))
        noisy = "（一）适用对象 X （三）杂项 （二）诊断依据 Y 临床路径表单"
        self.assertEqual([s["no"] for s in sections.split_sections(noisy)[0]], [1, 2])

    def test_misnumbered_heading(self):
        # 第四节印成了（一）：跳过一个编号接着切，并标出来
        t = "（一）适用对象 A （二）诊断依据 B （三）治疗方案 C （一）标准住院日为7天 （五）进入路径标准 D （六）检查项目 E 临床路径表单"
        secs, _ = sections.split_sections(t)
        self.assertEqual([s["no"] for s in secs], [1, 2, 3, 5, 6])
        self.assertEqual([s["no"] for s in secs if s["gap_before"]], [5])

    def test_relative_days(self):
        t = "临床路径表单 时间 术前第1天 手术日 术后第1天 术后第2-3天 出院日 术后第1天"
        self.assertEqual(sections.form_days(t, 0), [])
        self.assertEqual(sections.form_days_relative(t, 0), [("pre", 1), ("op", 0), ("post", 1, 1), ("post", 2, 3), ("discharge",)])

    def test_no_form(self):
        secs, form = sections.split_sections("（一）适用对象\nX\n（二）诊断依据\nY")
        self.assertIsNone(form)
        self.assertEqual(len(secs), 2)
        self.assertEqual(sections.form_days("x", None), [])


class TestProvenance(unittest.TestCase):
    def test_quote_must_be_in_source(self):
        f = {"value": "全身麻醉", "source": "pathway", "quote": "麻醉方式：全身麻醉"}
        self.assertTrue(pv.check_fact(f, TOY))
        f2 = {"value": "全身麻醉", "source": "pathway", "quote": "麻醉方式：气管插管全身麻醉"}
        self.assertFalse(pv.check_fact(f2, TOY))   # 引文是编的，原文没有

    def test_value_must_be_in_quote(self):
        f = {"value": "局部麻醉", "source": "pathway", "quote": "麻醉方式：全身麻醉"}
        self.assertFalse(pv.check_fact(f, TOY))

    def test_fullwidth_and_numbers(self):
        self.assertTrue(pv.check_fact({"value": 7, "source": "pathway", "quote": "标准住院日为≤７天"}, TOY))
        self.assertTrue(pv.check_fact({"value": "3-4天", "source": "pathway", "quote": "术后住院恢复３－４天"}, TOY))
        self.assertTrue(pv.check_fact({"value": "72-96小时", "source": "pathway", "quote": "症状消退后72-96小时"}, TOY))
        self.assertFalse(pv.check_fact({"value": "5-6天", "source": "pathway", "quote": "术后住院恢复３－４天"}, TOY))

    def test_list_value(self):
        q = "必需的检查项目：血常规、肝功能、腹部超声"
        self.assertTrue(pv.check_fact({"value": ["血常规", "腹部超声"], "source": "pathway", "quote": q}, TOY))
        self.assertFalse(pv.check_fact({"value": ["血常规", "胸部CT"], "source": "pathway", "quote": q}, TOY))

    def test_model_prior_never_grounded(self):
        f = {"value": 2, "source": "model_prior", "quote": "标准住院日为≤７天"}
        self.assertFalse(pv.check_fact(f, TOY))

    def test_grounded_rate(self):
        facts = {"anesthesia": [{"grounded": True}], "exam_turnaround_days": [{"grounded": False}, {"grounded": False}]}
        self.assertEqual(pv.grounded_rate(facts), {"anesthesia": (1, 1), "exam_turnaround_days": (0, 2)})


class TestExtract(unittest.TestCase):
    def test_toy(self):
        from route_b import extract
        facts, info = extract.extract(TOY, exam_alias={"血常规": "血常规", "超声": "超声"})
        val = {k: [f["value"] for f in v] for k, v in facts.items()}
        self.assertEqual(val["los_max"], [7])
        self.assertEqual(val["chief_symptoms"], ["右上腹痛，可伴发热"])
        self.assertEqual(val["signs"], ["局部压痛"])
        self.assertEqual(val["exams_required"], [["血常规", "肝功能", "腹部超声"]])
        self.assertEqual(facts["exams_required"][0]["bench_key"], ["血常规", None, "超声"])
        self.assertEqual(val["surgery_day_max"], [3])
        self.assertEqual(val["anesthesia"], ["全身麻醉"])
        self.assertEqual(val["postop_days"], ["３－４天"])
        self.assertEqual(val["workup_within_days"], [2])
        self.assertIn("72-96小时", val["antibiotic_rule"][0])
        self.assertEqual(len(val["daily_form"]), 5)
        self.assertNotIn("exam_turnaround_days", facts)
        self.assertEqual(info["icd10"], "X99.0")
        # 规则抽出来的全部有出处
        self.assertTrue(all(f["grounded"] for fs in facts.values() for f in fs))

    def test_county_variants(self):
        from route_b import extract
        t = ("（一）适用对象\nX\n（二）诊断依据\n2.查体：腹胀。\n（三）治疗方案\nY\n（四）标准住院日。\n3-4天。\n（五）进入路径标准\nZ\n"
             "（六）住院期间检查项目\n1.建议必须检查项目：血常规、心电图。\n（七）药物\nW\n（八）手术日（入院当天急诊手术）。\n（九）术后恢复。恢复1～2天。\n")
        facts, _ = extract.extract(t)
        val = {k: [f["value"] for f in v] for k, v in facts.items()}
        self.assertEqual(val["los_max"], [4])
        self.assertEqual(val["signs"], ["腹胀"])
        self.assertEqual(val["exams_required"], [["血常规", "心电图"]])
        self.assertEqual(val["surgery_day_max"], ["入院当天急诊手术"])
        self.assertEqual(val["postop_days"], ["1～2天"])
        self.assertTrue(all(f["grounded"] for fs in facts.values() for f in fs))

    def test_turnaround_skips_intake_output(self):
        from route_b import extract
        facts, _ = extract.extract("（一）适用对象 记录24小时出入量 2－3天出现咳嗽 体温正常3天\n出院 血培养3天出报告")
        self.assertEqual([f["value"] for f in facts["exam_turnaround_days"]], ["3天出报告"])


class TestSchema(unittest.TestCase):
    META = {"pathway": "示意病临床路径", "version": "示意版", "icd10": "X99.0", "issuer": "示意", "url": "https://example.invalid",
            "retrieved": "2026-11-10", "text_sha256": "0" * 64}

    def test_ok(self):
        kb = {"meta": self.META, "facts": {"anesthesia": [{"value": "全身麻醉", "source": "pathway", "quote": "麻醉方式：全身麻醉"}],
                                           "exam_turnaround_days": [{"value": 1, "source": "model_prior"}]}}
        self.assertEqual(schema.validate(kb), [])

    def test_errors(self):
        kb = {"meta": {}, "facts": {"anesthesia": [{"value": "全身麻醉", "source": "pathway"}], "foo": [], "los_max": [{"source": "web"}]}}
        errs = schema.validate(kb)
        self.assertIn("meta_pathway", errs)
        self.assertIn("no_quote:anesthesia", errs)
        self.assertIn("unknown_field:foo", errs)
        self.assertIn("bad_source:los_max", errs)
        self.assertIn("no_value:los_max", errs)

    def test_fields_disjoint(self):
        self.assertFalse(set(schema.PATHWAY_FIELDS) & set(schema.GAP_FIELDS))


if __name__ == "__main__":
    unittest.main()
