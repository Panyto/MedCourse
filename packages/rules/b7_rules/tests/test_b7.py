"""B7 规则的虚构用例。运行：在 b7_rules 目录下 PYTHONPATH=.:<核心目录> python -m unittest -v tests/test_b7.py"""
import unittest

from b7_rules import check_record, parse_age_years


def rec(age, hpi="", sex="男", cc="咳嗽3天", hc="予抗感染治疗。", di="注意休息。", dx="肺炎"):
    return {"Gender": sex, "Age": age, "Primary Diagnosis": dx, "Chief Complaint": cc,
            "History of Present Illness": hpi, "Hospital Course": hc, "Discharge Instructions": di}


def rules(r):
    return sorted(h["rule"] for h in check_record(r))


class AgeParse(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_age_years("65岁"), 65)
        self.assertEqual(parse_age_years("1"), 1)
        self.assertAlmostEqual(parse_age_years("8个月"), 8 / 12)
        self.assertIsNone(parse_age_years("不详"))


class AgeHist(unittest.TestCase):
    def test_toddler_with_long_history(self):
        self.assertIn("AGE-HIST", rules(rec("1", "既往有高血压病史10年，长期服药控制良好。")))

    def test_adult_same_text_ok(self):
        self.assertEqual(rules(rec("65岁", "既往有高血压病史10年，长期服药控制良好。")), [])

    def test_yu_and_plus(self):
        self.assertIn("AGE-HIST", rules(rec("20岁", "反复咳嗽30余年。")))
        self.assertIn("AGE-HIST", rules(rec("5岁", "腰痛7+年。")))

    def test_chinese_numeral(self):
        self.assertIn("AGE-HIST", rules(rec("3岁", "患糖尿病十年。")))

    def test_calendar_year_not_duration(self):
        self.assertEqual(rules(rec("3岁", "2020年3月起出现咳嗽。")), [])
        self.assertEqual(rules(rec("3岁", "于2021年行手术。")), [])
        self.assertEqual(rules(rec("3岁", "20年3月起出现咳嗽。")), [])

    def test_family_sentence_skipped(self):
        self.assertEqual(rules(rec("3岁", "父亲有高血压病史10年。患儿咳嗽3天。")), [])
        self.assertEqual(rules(rec("3岁", "其外婆患糖尿病20年。")), [])

    def test_future_and_frequency_skipped(self):
        self.assertEqual(rules(rec("3岁", "咳嗽3天。", di="建议每5年复查一次。")), [])
        self.assertEqual(rules(rec("3岁", "咳嗽3天。", di="5年后复诊。")), [])
        self.assertEqual(rules(rec("3岁", "咳嗽3天。", di="随访10年。")), [])

    def test_not_more_than_age_plus_one(self):
        self.assertEqual(rules(rec("1", "2年前曾患肺炎。")), [])

    def test_grade(self):
        self.assertEqual(rules(rec("8岁", "现读小学3年级，学习成绩下降10余天。")), [])


class PedAdult(unittest.TestCase):
    def test_hits(self):
        self.assertIn("PED-ADULT", rules(rec("3岁", "既往有高血压病史，长期服用降压药。")))
        self.assertIn("PED-ADULT", rules(rec("4岁", "有吸烟史。")))

    def test_negated_list(self):
        self.assertEqual(rules(rec("3岁", "否认高血压、糖尿病、冠心病病史。")), [])
        self.assertEqual(rules(rec("3岁", "无吸烟、饮酒史。")), [])
        self.assertEqual(rules(rec("3岁", "咳嗽3天。", di="避免吸烟环境。")), [])

    def test_passive_smoke(self):
        self.assertEqual(rules(rec("3岁", "家中无人吸烟，有被动吸烟史。")), [])
        self.assertEqual(rules(rec("3岁", "有被动吸烟史。")), [])

    def test_family_and_pregnancy(self):
        self.assertEqual(rules(rec("3岁", "母亲孕期有高血压。")), [])
        self.assertEqual(rules(rec("2岁", "父亲吸烟。")), [])

    def test_adult_ok(self):
        self.assertEqual(rules(rec("40岁", "有吸烟史20年。")), [])

    def test_negation_does_not_leak_across_clause(self):
        self.assertIn("PED-ADULT", rules(rec("3岁", "无发热，有高血压病史。")))


class CoreRules(unittest.TestCase):
    def test_sex_content(self):
        self.assertIn("SEX-CONTENT", rules(rec("30岁", "前列腺增生病史。", sex="女")))

    def test_sex_body(self):
        self.assertIn("SEX-BODY", rules(rec("30岁", "患者男性，咳嗽3天。", sex="女")))

    def test_age_body(self):
        self.assertIn("AGE-BODY", rules(rec("3岁", "患者65岁，咳嗽3天。")))

    def test_unparsed_age(self):
        self.assertIn("AGE-UNPARSED", rules(rec("不详", "咳嗽3天。")))


class Slot068(unittest.TestCase):
    """G1-C 包里 SLOT068 改写后的版本（合成病历）。"""
    def test_hit(self):
        r = rec("1", "患者3天前无明显诱因出现发热，体温最高达39℃，伴有咳嗽。既往有高血压病史10年，长期服药控制良好。否认吸烟史，无药物过敏史。",
                cc="发热、咳嗽伴胸痛3天，加重1天",
                di="出院后继续口服阿莫西林克拉维酸钾0.5g，每日三次，疗程7天。建议1周后门诊复查胸部X线及血常规。",
                dx="支气管肺炎")
        self.assertEqual(rules(r), ["AGE-HIST", "PED-ADULT"])


if __name__ == "__main__":
    unittest.main()
