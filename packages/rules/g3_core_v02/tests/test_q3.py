"""关3 尺子和工具的测试。全部用虚构的句子和病历，不含任何真实病历内容。
运行：在 g3_core 目录下 python -B -m unittest discover -s tests -v
（可选：环境变量 G2_CORE 指向 g2_core 目录时，多测一条"评委渲染与 g2_core 逐字相同"。）
"""
import csv
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ruler_g3 import evaluate, summarize, from_care7, from_admission, RULES_VERSION  # noqa: E402
from ruler_g3.lexicon import lexicon, rules, weight_p50, DATA  # noqa: E402
from ruler_g3.parse import parse_age, parse_age_range, parse_sex, parse_dose, find_freq, is_pregnant  # noqa: E402
from ruler_g3.numbers import n1_hits, n2_numbers  # noqa: E402
from q3_kit import mines, floors, judge, stats  # noqa: E402


def care(age="45", sex="男", dx="社区获得性肺炎", cc="咳嗽3天", hpi="3天前受凉后出现咳嗽，咳少量白痰。",
         hc="入院后完善血常规、胸片，予抗感染治疗，症状好转。", di="注意休息，门诊随诊。"):
    return {"Gender": sex, "Age": age, "Primary Diagnosis": dx, "Chief Complaint": cc,
            "History of Present Illness": hpi, "Hospital Course": hc, "Discharge Instructions": di}


def run(rec):
    return evaluate(from_care7(rec, "T"))


def rule_ids(res):
    return sorted(h["rule_id"] for h in res["hits"])


def one(res, rule_id):
    hs = [h for h in res["hits"] if h["rule_id"] == rule_id]
    assert len(hs) == 1, (rule_id, res["hits"])
    return hs[0]


def ped_limit(drug, age_years):
    d = rules().dose[drug]
    lim = d["ped_max_mgkg"] * weight_p50(age_years)
    return min(lim, d["ped_cap"]) if d["ped_cap"] else lim


# ---------------------------------------------------------------- 数据表

class TestData(unittest.TestCase):
    def _rows(self, name):
        with open(os.path.join(DATA, name), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_version(self):
        self.assertEqual(RULES_VERSION, "g3-rules-0.2")
        with open(os.path.join(ROOT, "ruler_g3", "VERSION"), encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), RULES_VERSION)

    def test_alias_unique(self):
        seen = {}
        for r in self._rows("drug_lexicon.csv"):
            for a in r["aliases"].split("|"):
                a = a.strip()
                if a:
                    self.assertNotIn(a, seen, f"别名 {a} 同时属于 {seen.get(a)} 和 {r['drug_id']}")
                    seen[a] = r["drug_id"]

    def test_drug_ids_unique(self):
        ids = [r["drug_id"] for r in self._rows("drug_lexicon.csv")]
        self.assertEqual(len(ids), len(set(ids)))

    def test_sources_cover_rules_and_doses(self):
        src = {r["source_id"]: r for r in self._rows("sources.csv")}
        for r in src.values():
            self.assertTrue(r["url"].startswith("http"), r["source_id"])
            self.assertTrue(r["content"].strip(), r["source_id"])
        for r in self._rows("drug_rules.csv"):
            ids = r["source_ids"].split("|")
            if r["category"] in ("DOSE", "PEDADULT"):
                self.assertEqual(ids[0], "see dose_table", r["rule_id"])     # 剂量规则的出处逐药写在剂量表里
                ids = ids[1:]
            for s in ids:
                self.assertIn(s, src, r["rule_id"])
        for r in self._rows("dose_table.csv"):
            for s in r["source_ids"].split("|"):
                self.assertIn(s, src, r["drug_id"])

    def test_targets_exist(self):
        lex = lexicon()
        all_classes = set().union(*lex.classes.values())
        for r in self._rows("drug_rules.csv"):
            kind, _, val = r["target"].partition(":")
            if kind == "drug":
                self.assertIn(val, lex.classes, r["rule_id"])
            elif kind == "class":
                cls, _, excl = val.partition("-")
                self.assertIn(cls, all_classes, r["rule_id"])
                if excl:
                    self.assertIn(excl, lex.classes, r["rule_id"])
        for d in rules().dose:
            self.assertIn(d, lex.classes, d)

    def test_dose_rows_have_a_limit(self):
        for d, v in rules().dose.items():
            self.assertTrue(v["adult_max"] or v["ped_max_mgkg"] or v["ped_adult_threshold"], d)

    def test_weight_curve(self):
        self.assertAlmostEqual(weight_p50(2), 12.2)
        self.assertAlmostEqual(weight_p50(0), 3.3)
        self.assertAlmostEqual(weight_p50(10), 18.3)          # 60 月以后不再长
        self.assertAlmostEqual(weight_p50(1.5), 10.9)
        self.assertAlmostEqual(weight_p50(2.5), (12.2 + 14.3) / 2)


# ---------------------------------------------------------------- 解析

class TestParse(unittest.TestCase):
    def test_age(self):
        self.assertEqual(parse_age("65岁"), 65)
        self.assertAlmostEqual(parse_age("1岁3月"), 1.25)
        self.assertAlmostEqual(parse_age("8个月"), 8 / 12)
        self.assertAlmostEqual(parse_age("15天"), 15 / 365)
        self.assertAlmostEqual(parse_age("3周"), 21 / 365)
        self.assertEqual(parse_age("45"), 45)
        self.assertIsNone(parse_age("不详"))
        self.assertIsNone(parse_age("300"))

    def test_age_range(self):
        # 整岁、整月、整周按区间；写到天或带小数的上下限相同
        self.assertEqual(parse_age_range("2岁"), (2, 3))
        self.assertEqual(parse_age_range("2"), (2, 3))
        self.assertEqual(parse_age_range("0"), (0, 1))
        self.assertEqual(parse_age_range("2.5"), (2.5, 2.5))
        lo, hi = parse_age_range("1岁3月")
        self.assertAlmostEqual(lo, 1.25)
        self.assertAlmostEqual(hi, 1 + 4 / 12)
        lo, hi = parse_age_range("8个月")
        self.assertAlmostEqual(lo, 8 / 12)
        self.assertAlmostEqual(hi, 9 / 12)
        self.assertEqual(parse_age_range("15天"), (15 / 365, 15 / 365))
        self.assertEqual(parse_age_range("3周"), (21 / 365, 28 / 365))
        self.assertEqual(parse_age_range("不详"), (None, None))
        r = from_care7(care(age="0"))
        self.assertEqual((r.age, r.age_hi), (0, 1))
        hr = "性别：女\n年龄：1岁\n"
        r = from_admission({"docs": [{"doc_type": "hospital_record", "text": hr}]})
        self.assertEqual((r.age, r.age_hi, r.sex), (1, 2, "F"))

    def test_sex(self):
        self.assertEqual(parse_sex("女"), "F")
        self.assertEqual(parse_sex("Female"), "F")
        self.assertEqual(parse_sex("男性"), "M")
        self.assertIsNone(parse_sex("未知"))

    def test_mass_units_equal(self):
        self.assertEqual(parse_dose("片0.5g，每日三次")["daily_mg"], parse_dose("片500mg，每日三次")["daily_mg"])
        self.assertEqual(parse_dose("片0.5g，每日三次")["daily_mg"], 1500)

    def test_strength_times_count(self):
        d = parse_dose("0.25g×24粒 2粒 tid")
        self.assertEqual((d["single_mg"], d["daily_mg"]), (500, 1500))
        d = parse_dose("0.25g/粒，一次两粒，bid")
        self.assertEqual((d["single_mg"], d["daily_mg"]), (500, 1000))
        self.assertIsNone(parse_dose("0.25g×24粒，遵医嘱"))      # 只有规格没有用量：不抽

    def test_freq_tokens(self):
        self.assertEqual(parse_dose("0.5gtid")["daily_mg"], 1500)
        self.assertEqual(parse_dose("10mgqd")["daily_mg"], 10)
        self.assertEqual(parse_dose("0.5g，每6小时一次")["daily_mg"], 2000)
        self.assertEqual(parse_dose("250mg口服，1次/日")["daily_mg"], 250)
        self.assertEqual(parse_dose("80mg，每日两次")["daily_mg"], 160)
        self.assertEqual(parse_dose("0.3g st")["daily_mg"], 300)
        self.assertEqual(find_freq("心电图示ST段抬高"), (None, None))
        self.assertEqual(find_freq("ST-T改变"), (None, None))
        self.assertEqual(find_freq("每日一次，中午服"), (1.0, "每日一次"))

    def test_dose_of_next_drug_not_borrowed(self):
        # 复方丹参滴丸不在词表里，它的剂量和频次不能算到前一个药头上
        self.assertEqual(list(lexicon().find("复方丹参滴丸")), [])
        self.assertIsNone(parse_dose("、复方丹参滴丸270mg tid"))
        self.assertIsNone(parse_dose("，复方丹参滴丸270mg tid"))
        d = parse_dose("钙片20mg，复方丹参滴丸270mg tid")
        self.assertEqual((d["single_mg"], d["daily_mg"]), (20, None))
        self.assertEqual(rule_ids(run(care(di="氨氯地平、复方丹参滴丸270mg tid。"))), [])
        # 逗号后面是这个药自己的用法：照抽，频次在剂量前面也认
        d = parse_dose("片，每日3次，每次0.5g")
        self.assertEqual((d["single_mg"], d["daily_mg"]), (500, 1500))
        self.assertEqual(parse_dose("注射液，0.5g，qd")["daily_mg"], 500)

    def test_daily_markers(self):
        self.assertEqual(parse_dose("每日总量2g")["daily_mg"], 2000)
        self.assertEqual(parse_dose("1g/日，分两次")["daily_mg"], 1000)
        self.assertEqual(parse_dose("20mg/晚")["daily_mg"], 20)

    def test_rejected_amounts(self):
        self.assertIsNone(parse_dose("10mg/kg，每日一次"))
        self.assertIsNone(parse_dose("2mg/h泵入"))
        self.assertIsNone(parse_dose("5%葡萄糖500ml 10gtt/min"))
        self.assertIsNone(parse_dose("遵医嘱服用"))

    def test_pregnancy(self):
        self.assertTrue(is_pregnant("宫内妊娠20周", "", "F", 28))
        self.assertTrue(is_pregnant("先兆流产；宫内孕8周", "", "F", 28))
        self.assertFalse(is_pregnant("宫内妊娠39周", "行子宫下段剖宫产术", "F", 28))
        self.assertFalse(is_pregnant("宫内妊娠10周", "", "M", 28))
        self.assertFalse(is_pregnant("宫内妊娠10周", "", "F", 65))
        self.assertFalse(is_pregnant("不完全流产", "", "F", 28))
        self.assertFalse(is_pregnant("高血压病", "", "F", 28))


# ---------------------------------------------------------------- 规则

class TestRules(unittest.TestCase):
    def test_levo_age(self):
        r = run(care(age="17", di="出院带药：左氧氟沙星片0.5g，每日一次。"))
        h = one(r, "D-AGE-LEVO")
        self.assertEqual((h["severity"], h["hard"]), ("severe", True))
        self.assertEqual(rule_ids(run(care(age="18", di="出院带药：左氧氟沙星片0.5g，每日一次。"))), [])

    def test_levo_topical_and_negation(self):
        self.assertEqual(rule_ids(run(care(age="8", di="左氧氟沙星滴眼液滴双眼，每日四次。"))), [])
        self.assertEqual(rule_ids(run(care(age="8", hpi="对左氧氟沙星过敏。"))), [])
        self.assertEqual(rule_ids(run(care(age="8", hpi="否认使用左氧氟沙星。"))), [])
        self.assertEqual(rule_ids(run(care(age="8", hpi="其母曾服左氧氟沙星。"))), [])
        self.assertEqual(rule_ids(run(care(age="8", hc="外用左氧氟沙星凝胶。"))), [])

    def test_other_fq_and_tetra_are_warnings(self):
        h = one(run(care(age="10", di="莫西沙星片0.4g qd。")), "D-AGE-FQ")
        self.assertEqual((h["severity"], h["hard"]), ("warning", False))
        h = one(run(care(age="7", di="多西环素片0.1g bid。")), "D-AGE-TETRA")
        self.assertEqual((h["severity"], h["hard"]), ("warning", False))
        self.assertEqual(rule_ids(run(care(age="8", di="多西环素片0.1g bid。"))), [])
        s = summarize(run(care(age="10", di="莫西沙星片0.4g qd。")))
        self.assertFalse(s["any_hard"])
        self.assertTrue(s["any_warning"])

    def test_nimesulide_codeine_tramadol(self):
        one(run(care(age="11", di="尼美舒利颗粒50mg bid。")), "D-AGE-NIME")
        self.assertEqual(rule_ids(run(care(age="12", di="尼美舒利颗粒50mg bid。"))), [])
        one(run(care(age="9", di="可待因片15mg tid。")), "D-AGE-COD")
        one(run(care(age="9", di="曲马多缓释片50mg bid。")), "D-AGE-TRAM")

    def test_aspirin_not_ruled(self):
        # 川崎病大剂量阿司匹林是正规用法；阿司匹林不在任何规则里
        self.assertEqual(rule_ids(run(care(age="3", dx="川崎病", di="阿司匹林肠溶片100mg tid。"))), [])

    def test_pregnancy_rules(self):
        base = dict(age="28", sex="女", dx="宫内妊娠20周；妊娠期高血压")
        h = one(run(care(di="继续口服缬沙坦胶囊80mg，每日一次。", **base)), "D-PREG-RAS")
        self.assertTrue(h["hard"])
        one(run(care(di="出院带药：左氧氟沙星片0.5g qd。", **base)), "D-PREG-LEVO")
        one(run(care(hc="予异维A酸胶囊10mg qd。", **base)), "D-PREG-ISO")
        # 写明停用、换药：不算
        self.assertEqual(rule_ids(run(care(hc="停用缬沙坦，改用拉贝洛尔100mg bid。", **base))), [])
        # 已经分娩：不算妊娠
        self.assertEqual(rule_ids(run(care(hc="入院后行剖宫产术。", di="缬沙坦80mg qd。", **base))), [])
        # 男性、或诊断里没有妊娠：不算
        self.assertEqual(rule_ids(run(care(age="28", sex="男", dx="宫内妊娠20周", di="缬沙坦80mg qd。"))), [])
        self.assertEqual(rule_ids(run(care(age="28", sex="女", dx="高血压病", di="缬沙坦80mg qd。"))), [])

    def test_sex_rules(self):
        h = one(run(care(sex="男", hc="予缩宫素10U肌内注射。")), "D-SEX-OBST")
        self.assertEqual((h["severity"], h["hard"]), ("mild", False))
        self.assertEqual(rule_ids(run(care(sex="女", hc="予缩宫素10U肌内注射。"))), [])
        h = one(run(care(sex="女", di="度他雄胺软胶囊0.5mg qd。")), "D-SEX-5ARI")
        self.assertEqual(h["severity"], "warning")
        self.assertEqual(rule_ids(run(care(sex="女", di="非那雄胺片1mg qd。"))), [])   # 非那雄胺未核到明确禁忌，不判

    def test_adult_dose_boundaries(self):
        self.assertEqual(rule_ids(run(care(di="氨氯地平片15mg qd。"))), [])            # 1.5 倍：不报
        h = one(run(care(di="氨氯地平片16mg qd。")), "D-DOSE-ADULT")
        self.assertEqual(h["severity"], "mild")
        h = one(run(care(di="氨氯地平片20mg qd。")), "D-DOSE-ADULT")
        self.assertEqual((h["severity"], h["ratio"]), ("severe", 2.0))

    def test_single_dose_basis(self):
        # 没写频次：用单次量去比每日最大量
        h = one(run(care(di="阿托伐他汀钙片200mg。")), "D-DOSE-ADULT")
        self.assertEqual(h["basis"], "single")

    def test_ped_dose(self):
        # "2岁"按上限 3 岁取体重 14.3 kg
        lim = ped_limit("amoxicillin", 3)
        self.assertAlmostEqual(lim, 90 * 14.3)
        self.assertEqual(rule_ids(run(care(age="2", di="阿莫西林颗粒0.5g tid。"))), [])   # 1500/1287 = 1.17
        h = one(run(care(age="2", di="阿莫西林颗粒0.75g tid。")), "D-DOSE-PED")     # 2250/1287 = 1.75
        self.assertEqual(h["severity"], "mild")
        h = one(run(care(age="2", di="阿莫西林颗粒1.0g tid。")), "D-DOSE-PED")
        self.assertEqual(h["severity"], "severe")
        self.assertAlmostEqual(h["ratio"], round(3000 / lim, 2))
        self.assertEqual(h["weight_kg"], 14.3)
        self.assertEqual((h["age"], h["age_hi"]), (2, 3))

    def test_ped_dose_infant_upper(self):
        # Age 栏写"0"：按 0–1 岁的上限 9.6 kg，不按出生体重 3.3 kg
        h = one(run(care(age="0", di="阿莫西林颗粒1.0g tid。")), "D-DOSE-PED")
        self.assertEqual(h["weight_kg"], 9.6)
        self.assertEqual(rule_ids(run(care(age="0", di="阿莫西林颗粒0.4g tid。"))), [])   # 1200/864 = 1.39
        h = one(run(care(age="15天", di="阿莫西林颗粒0.2g tid。")), "D-DOSE-PED")          # 写到天：体重按 15 天
        self.assertAlmostEqual(h["weight_kg"], round(weight_p50(15 / 365), 1))

    def test_ped_cap(self):
        # 5 岁：头孢他啶 200 mg/kg/日 × 18.3 kg 超过上限 6000，取 6000
        self.assertEqual(ped_limit("ceftazidime", 5.9), 200 * weight_p50(5.9) if 200 * weight_p50(5.9) < 6000 else 6000)

    def test_pedadult(self):
        h = one(run(care(age="4", di="氯吡格雷片75mg qd。")), "D-PEDADULT")
        self.assertEqual((h["category"], h["hard"]), ("PEDADULT", False))
        self.assertEqual(rule_ids(run(care(age="4", di="氯吡格雷片25mg qd。"))), [])
        self.assertEqual(rule_ids(run(care(age="7", di="氯吡格雷片75mg qd。"))), [])
        self.assertTrue(summarize(run(care(age="4", di="氯吡格雷片75mg qd。")))["any_pedadult"])

    def test_not_drug_words(self):
        self.assertEqual(run(care(hc="留置肝素帽，胰岛素抵抗指数升高。"))["spec"]["mentions"], 0)

    def test_specificity(self):
        r = run(care(hc="予头孢曲松钠2g qd静滴，并予阿奇霉素抗感染。"))
        self.assertEqual(r["spec"], {"mentions": 2, "with_dose": 1})

    def test_age_unknown_uses_adult(self):
        rec = care(age="不详", di="氨氯地平片50mg qd。")
        self.assertEqual(one(run(rec), "D-DOSE-ADULT")["severity"], "severe")


# ---------------------------------------------------------------- 整次住院格式

HR = "姓名：某某\n性别：女\n年龄：28岁\n主诉：停经20周，头痛2天。\n初步诊断：\n1.宫内妊娠20周\n2.妊娠期高血压\n"
DS = "出院诊断：宫内妊娠20周\n出院医嘱：继续口服缬沙坦胶囊80mg，每日一次。"


class TestAdmission(unittest.TestCase):
    def adm(self, hr=HR, ds=DS, course="患者头痛好转，血压平稳。"):
        return {"admission_id": "A1", "docs": [
            {"doc_type": "hospital_record", "day_offset": 0, "text": hr},
            {"doc_type": "progress_note", "day_offset": 1, "text": course},
            {"doc_type": "discharge_summary", "day_offset": 3, "text": ds}]}

    def test_header_and_preg(self):
        rec = from_admission(self.adm())
        self.assertEqual((rec.age, rec.sex), (28, "F"))
        r = evaluate(rec)
        self.assertTrue(r["pregnant"])
        one(r, "D-PREG-RAS")

    def test_delivery_in_course(self):
        r = evaluate(from_admission(self.adm(course="今日行剖宫产术，娩出一活婴。")))
        self.assertFalse(r["pregnant"])
        self.assertEqual(rule_ids(r), [])

    def test_apply_admission_age(self):
        a = self.adm(hr=HR.replace("女", "男").replace("宫内妊娠20周\n2.妊娠期高血压", "社区获得性肺炎"),
                     ds="出院诊断：社区获得性肺炎\n出院医嘱：注意休息。")
        self.assertTrue(mines.admission_eligible(a, "M-QUIN"))
        m = mines.apply_admission(a, "M-QUIN", 0, "mined")
        rec = from_admission(m)
        self.assertEqual(rec.age, 8)
        self.assertTrue(mines.detected(evaluate(rec), {"variant": "mined", "expected_rule": "D-AGE-LEVO", "expected_severity": "severe"}))
        c = mines.apply_admission(a, "M-QUIN", 0, "control")
        self.assertEqual(rule_ids(evaluate(from_admission(c))), [])
        self.assertEqual(a["docs"][0]["text"].count("28岁"), 1)          # 原住院没被改

    def test_floor_and_degenerate_admission(self):
        child = self.adm(hr="性别：男\n年龄：4岁\n初步诊断：急性支气管炎\n", ds="出院医嘱：注意休息。", course="咳嗽好转。")
        adult = self.adm(hr="性别：男\n年龄：60岁\n初步诊断：高血压病\n", ds="出院医嘱：氨氯地平片5mg qd，阿托伐他汀钙片20mg qn。")
        out = floors.swap_floor_admissions("T", [("c", child), ("a", adult)])
        self.assertEqual([(h, d) for h, d, _ in out], [("c", "a")])
        r = evaluate(from_admission(out[0][2]))
        self.assertEqual(rule_ids(r), ["D-PEDADULT", "D-PEDADULT"])
        dg = evaluate(from_admission(floors.degenerate_admission(adult)))
        self.assertEqual((dg["hits"], dg["spec"]["mentions"]), ([], 0))


# ---------------------------------------------------------------- 数值

class TestNumbers(unittest.TestCase):
    def types(self, text):
        return sorted(h["type"] for h in n1_hits(text))

    def test_impossible(self):
        self.assertEqual(self.types("体温47.6℃"), ["T"])
        self.assertEqual(self.types("T 36.5℃，P 300次/分"), [])
        self.assertEqual(self.types("脉搏 350次/分"), ["P"])
        self.assertEqual(self.types("呼吸 2次/分"), ["R"])
        self.assertEqual(self.types("血压70/100mmHg"), ["SBP<=DBP"])
        self.assertEqual(self.types("BP 320/190mmHg"), ["SBP"])
        self.assertEqual(self.types("血压120/210mmHg"), ["DBP", "SBP<=DBP"])
        self.assertEqual(self.types("血钾45.3mmol/L，血钠140mmol/L"), ["K"])
        self.assertEqual(self.types("血钠 80mmol/L"), ["NA"])
        self.assertEqual(self.types("血红蛋白 300g/L"), ["HB_gL"])
        self.assertEqual(self.types("Hb 12 g/dL"), [])
        self.assertEqual(self.types("WBC 1200×10^9/L"), ["WBC"])
        self.assertEqual(self.types("血糖 200mmol/L"), ["GLU"])
        self.assertEqual(self.types("肌酐 6000μmol/L"), ["CR"])

    def test_plausible_and_zero(self):
        self.assertEqual(self.types("体温36.8℃，心率88次/分，呼吸20次/分，血压132/80mmHg，血钾4.5mmol/L"), [])
        self.assertEqual(self.types("心率 0 次/分，呼吸 0 次/分，血压 0/0mmHg"), [])

    def test_n2(self):
        total, unsourced, samples = n2_numbers("血钾3.2mmol/L，血红蛋白110g/L", '{"K": "3.2"}')
        self.assertEqual((total, unsourced, samples), (2, 1, ["110g/L"]))


# ---------------------------------------------------------------- 雷

def host_for(mine):
    if mine == "M-PREG":
        return care(age="30", sex="女", dx="高血压病", cc="头痛2天", hpi="2天前出现头痛。",
                    hc="入院后监测血压，症状好转。", di="门诊随诊。")
    if mine in ("M-QUIN", "M-PEDDOSE"):
        return care(age="5", sex="男", dx="急性支气管炎", hpi="患儿3天前受凉后出现咳嗽。")
    if mine == "M-SEX":
        return care(age="40", sex="男")
    return care(age="50", sex="男")


class TestMines(unittest.TestCase):
    def test_template_truth_by_formula(self):
        # M-PEDDOSE 写"2岁"，尺子按上限 3 岁取体重
        for mine, age in (("M-PEDDOSE", 3), ("M-DOSE10", 50)):
            for i, row in enumerate(mines.TEMPLATES[mine]["rows"]):
                drug, single, n = row[4]
                d = rules().dose[drug]
                lim = ped_limit(drug, age) if age < 6 else d["adult_max"]
                self.assertGreaterEqual(single * n / lim, 2.0, (mine, i))
                rec = mines.apply_care7(host_for(mine), mine, i, "mined")
                h = [x for x in run(rec)["hits"] if x["rule_id"] == row[2]]
                self.assertEqual(len(h), 1, (mine, i))
                self.assertEqual((h[0]["drug_id"], h[0]["single_mg"], h[0]["daily_mg"]), (drug, single, single * n), (mine, i))
                self.assertAlmostEqual(h[0]["limit_mg"], round(lim, 1), places=1)

    def test_every_template_detected_and_control_clean(self):
        for mine, t in mines.TEMPLATES.items():
            for i, row in enumerate(t["rows"]):
                host = host_for(mine)
                self.assertTrue(mines.care7_eligible(host, mine), mine)
                m = mines.apply_care7(host, mine, i, "mined")
                c = mines.apply_care7(host, mine, i, "control")
                case_m = {"variant": "mined", "expected_rule": row[2], "expected_severity": row[3]}
                self.assertTrue(mines.detected(run(m), case_m), (mine, i, run(m)["hits"], run(m)["n1"]))
                self.assertTrue(mines.detected(run(c), {"variant": "control"}), (mine, i, run(c)["hits"]))
                # 雷版和对照版只差那一句；注入自检全过
                self.assertEqual({k for k in m if m[k] != c[k]}, {"Discharge Instructions"})
                self.assertEqual(mines.check_care7(host, mine, i, m, c), [], (mine, i))
                span = mines._locate(from_care7(m), row[0])
                self.assertEqual(span[0], "DISCHARGE")
                self.assertTrue(mines.detected(run(m), dict(case_m, span=span)), (mine, i))

    def test_detect_needs_span(self):
        # 命中不在插进去的那一句里，不算检出
        host = care(age="40", hc="予左氧氟沙星片0.5g qd。")
        m = mines.apply_care7(host, "M-QUIN", 0, "mined")
        case = {"variant": "mined", "expected_rule": "D-AGE-LEVO", "expected_severity": "severe"}
        span = mines._locate(from_care7(m), mines.TEMPLATES["M-QUIN"]["rows"][0][0])
        self.assertTrue(mines.detected(run(m), dict(case, span=span)))
        self.assertFalse(mines.detected(run(m), dict(case, span=("CC", 0, 100))))

    def test_inject_check_catches(self):
        host = host_for("M-QUIN")
        m = mines.apply_care7(host, "M-QUIN", 0, "mined")
        c = mines.apply_care7(host, "M-QUIN", 0, "control")
        self.assertIn("age", mines.check_care7(host, "M-QUIN", 0, dict(m, Age="40"), c))
        self.assertIn("only_sentence", mines.check_care7(host, "M-QUIN", 0, m, dict(c, **{"Chief Complaint": "x"})))
        self.assertIn("rest_same", mines.check_care7(host, "M-QUIN", 0, dict(m, **{"Hospital Course": "x"}), c))
        # 没有入院记录的住院：年龄写在第一篇病程里，改不到，自检不过，宿主不进病例
        adm = {"docs": [{"doc_type": "progress_note", "text": "性别：男\n年龄：50岁\n"},
                        {"doc_type": "discharge_summary", "text": "出院医嘱：注意休息。"}]}
        rep = {}
        self.assertEqual(mines.build_admission_cases("P", [("A0", adm)], "M-QUIN", 1, report=rep), [])
        self.assertEqual(rep["inject_fail"], [("A0", ["age"])])

    def test_host_not_changed(self):
        host = host_for("M-PREG")
        before = dict(host)
        mines.apply_care7(host, "M-PREG", 0, "mined")
        self.assertEqual(host, before)

    def test_eligibility(self):
        self.assertFalse(mines.care7_eligible(care(age="5", hpi="患儿5岁，咳嗽3天。"), "M-QUIN"))
        self.assertFalse(mines.care7_eligible(care(age="5", dx="2型糖尿病"), "M-PEDDOSE"))
        self.assertFalse(mines.care7_eligible(care(age="60", sex="男", dx="前列腺增生"), "M-PREG"))
        self.assertFalse(mines.care7_eligible(care(age="60", hpi="患者58岁，咳嗽3天。"), "M-PREG"))
        self.assertFalse(mines.care7_eligible(care(age="4", hpi="患儿3天前受凉后出现咳嗽。"), "M-PREG"))
        self.assertFalse(mines.care7_eligible(care(age="60", hpi="吸烟史30年。"), "M-PREG"))
        self.assertTrue(mines.care7_eligible(care(age="60", hpi="咳嗽2年，2019年起加重。"), "M-PREG"))
        self.assertFalse(mines.care7_eligible(care(age="5", hpi="反复咳嗽2年。"), "M-PEDDOSE"))
        self.assertTrue(mines.care7_eligible(care(age="5", hpi="反复咳嗽2年。"), "M-QUIN"))
        self.assertFalse(mines.care7_eligible(care(age="30", sex="女", hc="既往剖宫产术后。"), "M-PREG"))
        # 男性宿主没有男性特有词也可以：两版都改成 28 岁女性
        host = care(age="60", sex="男")
        self.assertTrue(mines.care7_eligible(host, "M-PREG"))
        m = mines.apply_care7(host, "M-PREG", 0, "mined")
        self.assertEqual((m["Gender"], m["Age"]), ("女", "28岁"))
        self.assertEqual(mines.check_care7(host, "M-PREG", 0, m, mines.apply_care7(host, "M-PREG", 0, "control")), [])
        self.assertTrue(run(m)["pregnant"])
        self.assertFalse(mines.care7_eligible(care(age="16"), "M-DOSE10"))
        self.assertFalse(mines.care7_eligible(care(sex="女"), "M-SEX"))

    def test_build_cases(self):
        pool = [(f"H{i:02d}", host_for("M-QUIN")) for i in range(6)]
        pool.append(("HLEVO", care(age="40", hc="予左氧氟沙星0.5g qd抗感染。")))   # 改成 8 岁后对照版不干净，要被排除
        pool.append(("HADULT", care(age="40", hpi="高血压病史10年。")))           # 有成人特有词，不进年龄类雷
        a = mines.build_care7_cases("P", pool, "M-QUIN", 4)
        b = mines.build_care7_cases("P", pool, "M-QUIN", 4)
        self.assertEqual([c["case_id"] for c in a], [c["case_id"] for c in b])
        self.assertEqual(len(a), 8)
        hosts = {c["host_id"] for c in a}
        self.assertNotIn("HLEVO", hosts)
        self.assertNotIn("HADULT", hosts)
        for c in a:
            self.assertTrue(mines.detected(run(c["record"]), c), c["case_id"])
        self.assertEqual(sorted(c["variant"] for c in a), ["control"] * 4 + ["mined"] * 4)

    def test_build_admission_cases(self):
        hr = "性别：男\n年龄：50岁\n初步诊断：社区获得性肺炎\n"
        adms = [(f"A{i}", {"docs": [{"doc_type": "hospital_record", "text": hr},
                                    {"doc_type": "discharge_summary", "text": "出院医嘱：注意休息。"}]}) for i in range(5)]
        rep = {}
        cs = mines.build_admission_cases("P", adms, "M-DOSE10", 3, report=rep)
        self.assertEqual((len(cs), rep), (6, {}))
        for c in cs:
            self.assertIsNotNone(c["span"])
            self.assertTrue(mines.detected(evaluate(from_admission(c["adm"], c["host_id"])), c), c["case_id"])
        for mine in ("M-QUIN", "M-PEDDOSE", "M-PREG"):
            hrf = "性别：女\n年龄：30岁\n初步诊断：社区获得性肺炎\n"
            pool = [(f"B{i}", {"docs": [{"doc_type": "hospital_record", "text": hrf},
                                        {"doc_type": "discharge_summary", "text": "出院医嘱：注意休息。"}]}) for i in range(3)]
            rep = {}
            cs = mines.build_admission_cases("P", pool, mine, 2, report=rep)
            self.assertEqual((len(cs), rep), (4, {}), mine)
            for c in cs:
                self.assertTrue(mines.detected(evaluate(from_admission(c["adm"], c["host_id"])), c), c["case_id"])


# ---------------------------------------------------------------- 地板和退化解

class TestFloors(unittest.TestCase):
    def test_swap_floor(self):
        child = care(age="4", dx="急性支气管炎", hc="予雾化治疗，症状好转。", di="注意休息。")
        adult = care(age="60", dx="高血压病", hc="予降压治疗。", di="继续口服氨氯地平片5mg qd，阿托伐他汀钙片20mg qn。")
        out = floors.swap_floor_care7("T", [("c", child), ("a", adult), ("x", care(age="16"))])
        self.assertEqual([(h, d) for h, d, _ in out], [("c", "a")])
        self.assertEqual(rule_ids(run(child)), [])
        self.assertEqual(rule_ids(run(out[0][2])), ["D-PEDADULT", "D-PEDADULT"])
        self.assertEqual(out[0][2]["Age"], "4")

    def test_degenerate(self):
        rec = care(age="8", hc="予左氧氟沙星0.5g qd静滴，阿莫西林0.5g tid口服。", di="出院带药：氨氯地平片50mg qd。")
        self.assertTrue(run(rec)["hits"])
        dg = floors.degenerate_care7(rec)
        r = run(dg)
        self.assertEqual((r["hits"], r["spec"]["mentions"], r["spec"]["with_dose"]), ([], 0, 0))
        self.assertIn("药物治疗", dg["Hospital Course"])
        self.assertEqual(dg["Primary Diagnosis"], rec["Primary Diagnosis"])


# ---------------------------------------------------------------- 评委

class TestJudge(unittest.TestCase):
    def test_criteria(self):
        self.assertEqual([c for c, _ in judge.CRITERIA_Q3], ["K13", "K14", "K15", "K16", "K17", "X14"])
        self.assertEqual(judge.CARE_IDS, ("K13", "K14", "K15", "K16", "K17"))
        self.assertTrue(dict(judge.CRITERIA_Q3)["K17"].startswith("The discharge diagnosis should not conflict with medications"))

    def test_prompt(self):
        p = judge.prompt_for(care(age="8"), judge.CRITERIA_Q3[0][1])
        self.assertTrue(p.startswith("Below is an electronic medical record consisting of multiple sections:\n性别：男\n年龄：8岁\n"))
        self.assertTrue(p.endswith('Respond in the following JSON format:\n{"Meets Requirement": true/false}'))
        self.assertIn("sex or age.. Identify any conflicts", p)        # G1-D 原样：标准自带句号，模板又加一个

    def test_render_matches_g2_core(self):
        g2 = os.environ.get("G2_CORE")
        if not g2:
            self.skipTest("未设 G2_CORE")
        sys.path.insert(0, g2)
        from ruler_g1 import render_summary7 as ref
        for rec in (care(), care(age="8个月", sex="female"), {"性别": "女", "年龄": "30", "主诉": "a\nb"}):
            self.assertEqual(judge.render_summary7(rec), ref(rec))

    def test_seed_and_body(self):
        s = judge.seed_for("mine", "job1", 0)
        self.assertEqual(s, judge.seed_for("mine", "job1", 0))
        self.assertNotEqual(s, judge.seed_for("mine", "job1", 1))
        self.assertLess(s, 2 ** 32)
        b = judge.request_body("x", s)
        self.assertEqual((b["temperature"], b["top_p"], b["top_k"], b["repetition_penalty"], b["max_tokens"]), (0, 1.0, -1, 1.05, 512))
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})

    def test_jobs(self):
        cases = mines.build_care7_cases("P", [(f"H{i}", host_for("M-DOSE10")) for i in range(3)], "M-DOSE10", 2)
        jobs = judge.build_jobs(cases, "mine", synthetic=True)
        self.assertEqual(len(jobs), len(cases) * 6)
        self.assertEqual(len({j["job_id"] for j in jobs}), len(jobs))

    def test_parse(self):
        self.assertEqual(judge.parse_jcare('{"Meets Requirement": true}')[0], True)
        self.assertEqual(judge.parse_jcare('```json\n{"Meets Requirement": "false"}\n```')[0], False)
        self.assertIsNone(judge.parse_jcare('{"Meets Requirement": "true/false"}')[0])
        v = judge.parse_jcare('先 {"Meets Requirement": false}，再想想 {"Meets Requirement": true}')
        self.assertEqual((v[0], v[1], v[3], v[4]), (True, "last_valid_json", 1, 2))
        self.assertEqual(judge.parse_jcare("无法判断")[1], "failed")

    def test_parse_p2_alias(self):
        # 32B 常把键写成 "Meats Requirement"：G1-D P2 修订认它，前提是整段只有一个单键 JSON
        self.assertEqual(judge.parse_jcare('{"Meats Requirement": false}')[1], "failed")
        self.assertEqual(judge.parse_p2('{"Meats Requirement": false}'), (False, "P2_alias", 0, 0, 1))
        self.assertEqual(judge.parse_p2('{"Meats Requirement": "TRUE"}')[0], True)
        self.assertEqual(judge.parse_p2('{"Meets Requirement": true}'), judge.parse_jcare('{"Meets Requirement": true}'))
        self.assertIsNone(judge.parse_p2('{"Meats Requirement": false} {"x": 1}')[0])
        self.assertIsNone(judge.parse_p2('{"Meats Requirement": false, "reason": "x"}')[0])

    def test_body_bytes(self):
        import hashlib
        import json as _json
        b = judge.request_body("年龄：2岁", 7)
        raw = judge.request_bytes(b)
        self.assertIn("年龄：2岁".encode("utf-8"), raw)
        self.assertEqual(_json.loads(raw), b)
        self.assertEqual(judge.body_sha256(b), hashlib.sha256(raw).hexdigest())

    def test_scoring(self):
        ok = {k: True for k in judge.CARE_IDS}
        bad = dict(ok, K17=False)
        unk = dict(ok, K16=None)
        self.assertEqual((judge.all_pass(ok), judge.all_pass(bad), judge.all_pass(unk)), (True, False, None))
        self.assertEqual(judge.all_pass(dict(unk, K13=False)), False)
        self.assertEqual(judge.miss_bounds(ok), {"point": 1, "upper": 1, "lower": 1})
        self.assertEqual(judge.miss_bounds(unk), {"point": None, "upper": 1, "lower": 0})
        self.assertEqual(judge.paired_detect(ok, bad), {"control_fail": 0, "upper": 1, "lower": 1, "point": 1})
        self.assertEqual(judge.paired_detect(bad, bad)["point"], 0)
        self.assertEqual(judge.paired_detect(unk, bad), {"control_fail": 0, "upper": 1, "lower": 0, "point": None})

    def test_pair_miss(self):
        ok = {k: True for k in judge.CARE_IDS}
        bad = dict(ok, K17=False)
        unk = dict(ok, K16=None)
        P = judge.pair_miss
        self.assertEqual(P(ok, ok), {"fav": "miss", "point": "miss", "unfav": "miss"})
        self.assertEqual(P(ok, bad), {"fav": "caught", "point": "caught", "unfav": "caught"})
        self.assertEqual(P(bad, ok), {"fav": "excluded", "point": "excluded", "unfav": "excluded"})
        self.assertEqual(P(ok, unk), {"fav": "caught", "point": None, "unfav": "miss"})
        self.assertEqual(P(unk, ok), {"fav": "excluded", "point": None, "unfav": "miss"})
        self.assertEqual(P(unk, bad), {"fav": "caught", "point": None, "unfav": "excluded"})
        self.assertEqual(P(unk, unk), {"fav": "caught", "point": None, "unfav": "miss"})
        pairs = [P(ok, ok), P(ok, bad), P(unk, ok), P(ok, unk)]
        self.assertEqual(judge.miss_rate(pairs), (1, 3))
        self.assertEqual(judge.miss_rate(pairs, "point"), (1, 2))
        self.assertEqual(judge.miss_rate(pairs, "unfav"), (3, 4))


class TestStats(unittest.TestCase):
    def test_cp(self):
        lo, hi = stats.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.3085, places=3)
        lo, hi = stats.clopper_pearson(40, 40)
        self.assertAlmostEqual(lo, 0.9119, places=3)

    def test_paired(self):
        a = {i: (1 if i < 5 else 0, 1) for i in range(20)}
        b = {i: (0, 1) for i in range(20)}
        est, lo, hi, n = stats.paired_diff(a, b)
        self.assertEqual((est, n), (0.25, 20))
        self.assertGreater(lo, 0)


if __name__ == "__main__":
    unittest.main()
