"""g3-rules-0.3 的五类改动。全部是虚构句子，不含任何真实病历内容。"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ruler_g3 import evaluate, from_care7  # noqa: E402
from ruler_g3.numbers import n1_hits  # noqa: E402


def care(age="2个月", sex="男", dx="肺炎", hc="", di="门诊随诊。"):
    return {"Gender": sex, "Age": age, "Primary Diagnosis": dx, "Chief Complaint": "咳嗽3天",
            "History of Present Illness": "3天前受凉后咳嗽。", "Hospital Course": hc, "Discharge Instructions": di}


def age_hits(hc, age="2个月"):
    res = evaluate(from_care7(care(age=age, hc=hc), "T"))
    return [h for h in res["hits"] if h["rule_id"] == "D-AGE-LEVO"]


class TestSusceptibility(unittest.TestCase):
    def test_sentences(self):
        self.assertEqual(age_hits("痰培养示肺炎链球菌，药敏：头孢曲松：敏感，左氧氟沙星：敏感，青霉素：耐药。予头孢曲松抗感染。"), [])
        self.assertEqual(age_hits("血培养药敏结果：左氧氟沙星 S，头孢他啶 R。"), [])
        self.assertEqual(age_hits("药敏提示左氧氟沙星敏感，但患儿年龄小，未用。"), [])
        self.assertEqual(len(age_hits("予左氧氟沙星0.2g qd静滴。", age="8岁")), 1)
        self.assertEqual(len(age_hits("药敏提示对左氧氟沙星敏感，予左氧氟沙星0.2g qd静滴。", age="8岁")), 1)


class TestStopped(unittest.TestCase):
    def test_sentences(self):
        self.assertEqual(age_hits("会诊意见：停头孢他啶+左氧氟沙星，改用美罗培南。"), [])
        self.assertEqual(age_hits("停用头孢他啶、左氧氟沙星。"), [])
        self.assertEqual(len(age_hits("停头孢他啶，改用左氧氟沙星抗感染。")), 1)
        self.assertEqual(len(age_hits("停用头孢他啶后予左氧氟沙星静滴。")), 1)
        self.assertEqual(len(age_hits("患儿咳嗽不停，予左氧氟沙星静滴。")), 1)
        self.assertEqual(len(age_hits("停经后予左氧氟沙星静滴。")), 1)

    def test_start_then_stop(self):
        self.assertEqual(len(age_hits("予左氧氟沙星0.2g qd静滴，3天后停左氧氟沙星。", age="8岁")), 1)


class TestCompound(unittest.TestCase):
    def _dose(self, di, age="45岁"):
        res = evaluate(from_care7(care(age=age, di=di), "T"))
        return [h for h in res["hits"] if h["category"] == "DOSE"]

    def test_compound_not_single_dose(self):
        self.assertEqual(self._dose("阿利沙坦酯氨氯地平片 一次245mg qd。"), [])
        self.assertEqual(self._dose("缬沙坦氨氯地平片 一次80mg qd。"), [])

    def test_single_still_flagged(self):
        self.assertEqual(len(self._dose("氨氯地平片 50mg qd。")), 1)

    def test_regimen_with_plus_not_compound(self):
        res = evaluate(from_care7(care(age="8岁", hc="予头孢他啶+左氧氟沙星抗感染。"), "T"))
        self.assertEqual(len([h for h in res["hits"] if h["rule_id"] == "D-AGE-LEVO"]), 1)

    def test_compound_keeps_pregnancy_rule(self):
        rec = care(age="28岁", sex="女", dx="宫内妊娠20周；高血压", di="缬沙坦氨氯地平片 一次80mg qd。")
        res = evaluate(from_care7(rec, "T"))
        self.assertTrue(any(h["category"] == "PREG" for h in res["hits"]))


class TestHemoglobin(unittest.TestCase):
    def test_mch_is_not_hb(self):
        self.assertEqual(n1_hits("平均红细胞血红蛋白312g/L"), [])
        self.assertEqual(n1_hits("红细胞平均血红蛋白312g/L"), [])

    def test_real_hb_still_flagged(self):
        self.assertEqual(len(n1_hits("血红蛋白312g/L")), 1)


class TestNotCurrent(unittest.TestCase):
    def test_history_skipped(self):
        self.assertEqual(age_hits("既往曾用左氧氟沙星治疗。"), [])
        self.assertEqual(age_hits("曾服左氧氟沙星3天。"), [])

    def test_advice_without_order_skipped(self):
        self.assertEqual(age_hits("可考虑加用左氧氟沙星。"), [])
        self.assertEqual(age_hits("建议使用左氧氟沙星。"), [])

    def test_advice_with_order_flagged(self):
        self.assertEqual(len(age_hits("建议继续口服左氧氟沙星。", age="8岁")), 1)
        self.assertEqual(len(age_hits("建议左氧氟沙星0.2g qd。", age="8岁")), 1)

    def test_current_after_history_flagged(self):
        self.assertEqual(len(age_hits("既往肺炎病史，现予左氧氟沙星静滴。", age="8岁")), 1)

    def test_spec_excludes_not_current(self):
        res = evaluate(from_care7(care(age="45岁", hc="既往曾用左氧氟沙星。予头孢曲松2g qd。"), "T"))
        self.assertEqual(res["spec"]["mentions"], 1)


if __name__ == "__main__":
    unittest.main()
