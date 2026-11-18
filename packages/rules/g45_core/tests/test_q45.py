"""g45_core 自检。全部用编造的病历（张三、甲县一类），不碰任何真实病历。
在 g45_core 目录下：python -B -m unittest discover -s tests -v
需要 numpy、scikit-learn（判别器和年龄模型）；缺了这两个包，相关测试会报错而不是跳过（任务书要求 0 跳过）。
设了环境变量 G3_CORE（g3_core 0.2 解包目录）、G2_CORE（g2_core 0.4 解包目录）、RULER_G1（关1 尺子 ruler_g1_core 目录）时，
多做几条"与原件逐字相同"的核对。任务书要求三个都设，0 跳过。
"""
import hashlib
import json
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ruler_g4 import (analyze_admission, analyze_care7, analyze_text, containment, coverage_care7, doc_role,
                      info_units, norm_key, template_rates)
from q45_kit import c2st, demog, diffs, floors, judge, realtext, stats

REAL_LIKE = {
    "Gender": "男", "Age": "67岁", "Primary Diagnosis": "社区获得性肺炎",
    "Chief Complaint": "咳嗽、咳痰伴发热5天。",
    "History of Present Illness": "患者5天前受凉后出现咳嗽、咳黄痰，伴发热，最高体温38.9℃，无胸痛、咯血。"
                                  "曾于甲县人民医院门诊予头孢呋辛口服3天，症状无缓解，遂来我院。"
                                  "病程中精神差，食欲减退，睡眠一般，大小便正常，体重无明显变化。",
    "Hospital Course": "入院后查血常规示白细胞12.3×10^9/L，中性粒细胞85%，C反应蛋白56mg/L；胸部CT示右下肺斑片影。"
                       "予头孢曲松2g qd静滴抗感染，氨溴索化痰，治疗7天后体温正常，咳嗽明显减轻，复查胸部CT示病灶吸收。",
    "Discharge Instructions": "出院后口服头孢克肟0.1g bid，共5天；注意休息，戒烟；2周后门诊复查胸部CT。",
}


def _units(text, dx=""):
    return [(u["type"], u["key"]) for u in info_units(text, dx)]


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class _Foreign:
    """临时把另一个核心包放到 sys.path 最前面导入，用完撤掉。g2_core 自带一份新版 ruler_g1，和关1 原件同名，不隔开会互相顶掉。"""
    PREFIXES = ("ruler_g1", "ruler_g2", "q3_kit")

    def __init__(self, path):
        self.path = path

    def _drop(self):
        for name in [m for m in sys.modules if m.split(".")[0] in self.PREFIXES]:
            del sys.modules[name]

    def __enter__(self):
        self._drop()
        sys.path.insert(0, self.path)
        return self

    def __exit__(self, *exc):
        sys.path.remove(self.path)
        self._drop()
        return False


# g3_core v0.2 的 MANIFEST.sha256 里这些文件的 sha256（g45_core 原样带了一份 ruler_g3，和 stats.py、render7.py）
G3_MANIFEST = {
    "ruler_g3/__init__.py": "73743e053520000e99e41314d967963212cc220ef698e785ca541ccf08425c9a",
    "ruler_g3/VERSION": "e3b6c30ec576671dc43eb661e3ed6a7bd34159cc333f422efff679be1910d58f",
    "ruler_g3/adapters.py": "822b0211cb2425fec34590461517420393d3fb3a3e8dd1c5d20498e76b5c8bd1",
    "ruler_g3/engine.py": "372548b449067b7c2331d8825eba495fdbd74586936e20d7ff623e5c7de6eafd",
    "ruler_g3/lexicon.py": "f0eeda59acb2b4c5043e77975fc873fe9fb4a00b940e03cc02efeebd6a7daf99",
    "ruler_g3/numbers.py": "6e6a8d30566cda3192ab2232b36ccd456703312a5485246814816c9a93b7b9c7",
    "ruler_g3/parse.py": "3c3d7a2a014e8db9bca4c1ce52a973aee7c6283fa1029d0320df60d0f1315de4",
    "ruler_g3/data/dose_table.csv": "31e67c586519332817e611b1c7636b379010ab3aba488e457ceff9fc0b4f0a16",
    "ruler_g3/data/drug_lexicon.csv": "bff403406599427b9b458a300916e8cc4589735735245d766d2cd3336a599ae0",
    "ruler_g3/data/drug_rules.csv": "897f71c9389dc6ec4b0a23f687bda10ef0e828340b44fcf661e2c452ec90e469",
    "ruler_g3/data/sources.csv": "07ed61164c501464bb657696a3ba3e0c429e0d4b3efbced89ffefe1400c574b9",
    "q45_kit/stats.py": "593cad8090621c14f2e510534f713c1749ff4b9a46554832070d8a73772468a6",
    "q45_kit/render7.py": "4f189c1662f6ebc919449b40b98d21381b7232258cba7ba52415477e1bf3bcac",
}


class TestVendored(unittest.TestCase):
    def test_same_bytes_as_g3_core_02(self):
        for rel, want in G3_MANIFEST.items():
            self.assertEqual(_sha(os.path.join(ROOT, rel)), want, rel)

    @unittest.skipUnless(os.environ.get("G3_CORE"), "设 G3_CORE 才和 g3_core 原件逐字比")
    def test_judge_shared_parts_identical_to_q3(self):
        import importlib
        with _Foreign(os.environ["G3_CORE"]):
            q3 = importlib.import_module("q3_kit.judge")
            self.assertEqual(q3.JCARE_PROMPT, judge.JCARE_PROMPT)
            for t in ('{"Meets Requirement": true}', '{"Meats Requirement": false}', 'x {"meets_requirement": "true"} y',
                      '{"Meets Requirement": true} {"Meets Requirement": false}', "无法判断", ""):
                self.assertEqual(q3.parse_p2(t), judge.parse_p2(t), t)
            self.assertEqual(q3.request_body("p", 7), judge.request_body("p", 7))

    @unittest.skipUnless(os.environ.get("G2_CORE"), "设 G2_CORE 才和 g2_core 的篇间复制率比")
    def test_containment_matches_g2(self):
        a = "患者今日体温37.2℃，咳嗽较前减轻，继续目前治疗。"
        b = "患者今日体温36.8℃，咳嗽较前减轻，继续目前治疗，明日复查血常规。"
        with _Foreign(os.environ["G2_CORE"]):
            from ruler_g2.copyrate import containment as g2c
            for n in (5, 10):
                self.assertAlmostEqual(g2c(b, a, n), containment(b, a, n))


class TestInfoUnits(unittest.TestCase):
    def test_numbers_with_units(self):
        u = _units("咳嗽3天，最高体温38.5℃，持续5-10分钟，每日2次，每次50mg，十余年，心功能Ⅱ级")
        keys = [k for t, k in u if t == "NUM"]
        for k in ("3天", "38.5℃", "5分钟", "2次", "50mg", "十余年", "Ⅱ级"):
            self.assertIn(k, keys)

    def test_dates_and_age_are_not_units(self):
        self.assertEqual(_units("2019年3月5日入院，3月6日出院"), [])
        self.assertEqual(_units("患者68岁"), [])

    def test_exam_needs_a_result(self):
        self.assertNotIn(("EXAM", "血常规"), _units("入院后完善血常规、肝功能等检查。"))
        self.assertIn(("EXAM", "胸部CT"), _units("胸部CT示右下肺斑片影。"))
        # 结果属于后面那个检查，不算前面的
        u = _units("完善血常规、胸部CT，胸部CT示右下肺炎。")
        self.assertNotIn(("EXAM", "血常规"), u)
        self.assertIn(("EXAM", "胸部CT"), u)

    def test_negative_symptoms_count(self):
        u = _units("无发热、咳嗽。")
        self.assertIn(("SYM", "发热"), u)
        self.assertIn(("SYM", "咳嗽"), u)

    def test_procedures(self):
        self.assertIn(("PROC", "腹腔镜胆囊切除术"), _units("于3天前行腹腔镜胆囊切除术。"))
        self.assertFalse([k for t, k in _units("拟行手术治疗，术后恢复可。") if t == "PROC"])
        self.assertIn(("PROC", "血液透析"), _units("规律血液透析。"))

    def test_drugs(self):
        self.assertIn(("DRUG", "aspirin"), _units("阿司匹林100mg qd"))
        self.assertFalse([k for t, k in _units("对青霉素过敏。") if t == "DRUG"])

    def test_dx_string_is_not_information(self):
        dx = "2型糖尿病肾病V期"
        self.assertEqual(_units(f"因{dx}入院。", dx), [])
        self.assertIn(("NUM", "V期"), _units(f"因{dx}入院。"))

    def test_empty_sentences(self):
        a = analyze_text("予对症支持治疗，病情好转。予头孢曲松2g静滴抗感染。")
        self.assertEqual([s["empty"] for s in a["sentences"]], [1, 0])
        short = analyze_text("好转。")
        self.assertEqual(short["sentences"][0]["empty"], 0)   # 不到 6 个汉字的句子不参与


class TestCare7Readings(unittest.TestCase):
    def test_ladder_real_like_above_template_above_vague(self):
        r = analyze_care7(REAL_LIKE)
        t = analyze_care7(floors.ftpl(REAL_LIKE))
        v = analyze_care7(floors.vague("男", "67", "社区获得性肺炎"))
        self.assertGreater(r["iu_per_kchar"], t["iu_per_kchar"])
        self.assertGreater(t["iu_per_kchar"], v["iu_per_kchar"])
        self.assertLess(r["empty_share"], t["empty_share"])
        self.assertLess(t["empty_share"], v["empty_share"])

    def test_keyword_stuffing_adds_no_information(self):
        host = analyze_care7(REAL_LIKE)
        kw = analyze_care7(floors.fkw(REAL_LIKE))
        hk = {(u["type"], u["key"]) for f in host["fields"].values() for u in f["iu"]}
        kk = {(u["type"], u["key"]) for f in kw["fields"].values() for u in f["iu"]}
        self.assertEqual(hk, kk)
        self.assertLess(kw["iu_per_kchar"], host["iu_per_kchar"])
        self.assertGreater(kw["empty_share"], host["empty_share"])
        # 但按要素名算的覆盖会涨：这正是"覆盖率单独报会被刷"的例子
        self.assertGreaterEqual(kw["on_list"], host["on_list"])

    def test_template_rate(self):
        hosts = [dict(REAL_LIKE, **{"Primary Diagnosis": dx}) for dx in ("社区获得性肺炎", "脑梗死", "2型糖尿病", "胆囊结石")]
        pool = [(str(i), analyze_care7(floors.ftpl(h))["sent_keys"]) for i, h in enumerate(hosts)]
        tr = template_rates(pool)
        self.assertTrue(all(a == b and b > 0 for a, b in tr.values()))
        # 同一句只在 2 份里出现，不算套话
        two = [("a", [("句子甲甲甲甲甲甲", 8)]), ("b", [("句子甲甲甲甲甲甲", 8)]), ("c", [("句子乙乙乙乙乙乙", 8)])]
        self.assertEqual(template_rates(two)["a"], (0, 8))

    def test_norm_key_ignores_digits_and_dx(self):
        self.assertEqual(norm_key("咳嗽3天，加重1天。", "肺炎"), norm_key("咳嗽5天， 加重2天", "肺炎"))
        self.assertEqual(norm_key("因肺炎入院", "肺炎"), norm_key("因脑梗死入院", "脑梗死"))

    def test_dup_share(self):
        rec = dict(REAL_LIKE)
        rec["Hospital Course"] = rec["History of Present Illness"]
        self.assertGreater(analyze_care7(rec)["dup_share"], 0.3)
        self.assertLess(analyze_care7(REAL_LIKE)["dup_share"], 0.1)

    def test_coverage(self):
        c = coverage_care7(REAL_LIKE)
        for k in ("L1_主诉有症状或病名", "L2_主诉有时长", "L4_诱因", "L5_主要症状", "L6_一般情况五项", "L7_住院检查",
                  "L8_住院治疗", "L9_出院带药均有剂量", "O1_伴随或阴性症状", "O2_院外诊治经过", "O4_复查随访", "O5_生活饮食指导"):
            self.assertEqual(c[k], 1, k)
        t = coverage_care7(floors.ftpl(REAL_LIKE))
        for k in ("L5_主要症状", "L9_出院带药均有剂量", "O1_伴随或阴性症状", "O2_院外诊治经过"):
            self.assertEqual(t[k], 0, k)
        self.assertEqual(t["L6_一般情况五项"], 1)


class TestFloors(unittest.TestCase):
    def test_ftpl_is_identical_across_patients(self):
        a = floors.ftpl(dict(REAL_LIKE, **{"Primary Diagnosis": "X病"}))
        b = floors.ftpl({"性别": "女", "年龄": "3岁", "主要诊断": "X病", "主诉": "别的", "现病史": "别的"})
        for k in ("Chief Complaint", "History of Present Illness", "Hospital Course", "Discharge Instructions"):
            self.assertEqual(a[k], b[k])
        self.assertEqual((b["Gender"], b["Age"]), ("女", "3岁"))

    def test_fkw_keeps_host(self):
        k = floors.fkw(REAL_LIKE)
        for f in ("History of Present Illness", "Hospital Course", "Discharge Instructions"):
            self.assertTrue(k[f].startswith(REAL_LIKE[f].rstrip("。")))
            self.assertTrue(k[f].endswith(floors.KW[f]))
        self.assertEqual(k["Chief Complaint"], REAL_LIKE["Chief Complaint"])

    def test_vague_matches_g1b_p5(self):
        v = floors.vague("女", "63", "不稳定型心绞痛")
        self.assertEqual(v["History of Present Illness"], "患者因不稳定型心绞痛入院，入院后完善相关检查。")
        self.assertEqual(v["Discharge Instructions"], "出院后注意休息，门诊随诊。")


class TestJudge(unittest.TestCase):
    def test_criteria_are_the_nine_completeness_items(self):
        self.assertEqual(judge.COMP_IDS, tuple(f"C0{i}" for i in range(1, 10)))
        joined = "\n".join(r for _, r in judge.CRITERIA_Q4)
        # 与 LLM-CARe 复现代码 run_s2.py 的 COMPLETENESS、关1 G1-C T07 的 requirement 列逐字相同（总经理 10-27 核对）
        self.assertEqual(hashlib.sha256(joined.encode("utf-8")).hexdigest(),
                         "%s" % EXPECTED_CRITERIA_SHA)

    def test_jobs_and_bytes(self):
        jobs = judge.build_jobs([{"case_id": "F-TPL|S001", "record": floors.ftpl(REAL_LIKE)}], "q4", True)
        self.assertEqual(len(jobs), 9)
        self.assertIn("Explicitly lists all of the following general conditions", jobs[5]["prompt"])
        self.assertIn("性别：男\n年龄：67岁\n出院诊断：社区获得性肺炎\n主诉：社区获得性肺炎相关症状1周。", jobs[0]["prompt"])
        body = judge.request_body(jobs[0]["prompt"], judge.seed_for("q4", jobs[0]["job_id"], 0))
        self.assertEqual(json.loads(judge.request_bytes(body).decode("utf-8")), body)
        self.assertEqual(judge.body_sha256(body), hashlib.sha256(judge.request_bytes(body)).hexdigest())
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["repetition_penalty"], 1.05)

    def test_seed_prefix_differs_from_q3(self):
        raw = "Q3_J32|q4|x|0".encode("utf-8")
        self.assertNotEqual(judge.seed_for("q4", "x", 0), int(hashlib.sha256(raw).hexdigest()[:8], 16))
        self.assertEqual(judge.seed_for("q4", "x", 0), judge.seed_for("q4", "x", 0))

    def test_scoring_policies(self):
        v = {"C01": True, "C02": True, "C03": None, "C04": False, "C05": True, "C06": True, "C07": True, "C08": True, "C09": None}
        self.assertEqual(judge.score(v, None), (6, 7))
        self.assertEqual(judge.score(v, True), (8, 9))
        self.assertEqual(judge.score(v, False), (6, 9))
        self.assertIsNone(judge.all9(v, None))
        self.assertEqual(judge.all9(v, True), 0)      # 有一条明确的 False
        allt = {c: True for c in judge.COMP_IDS}
        allt["C09"] = None
        self.assertEqual(judge.all9(allt, True), 1)
        self.assertEqual(judge.all9(allt, False), 0)
        self.assertEqual(judge.POLICY["fav"], {"floor": False, "real": True})

    def test_parse(self):
        self.assertEqual(judge.parse_p2('{"Meets Requirement": true}')[0], True)
        self.assertEqual(judge.parse_p2('{"Meats Requirement": false}')[:2], (False, "P2_alias"))
        self.assertIsNone(judge.parse_p2("看不出来")[0])


class TestC2ST(unittest.TestCase):
    def _toy(self, n, a, b, seed):
        rng = random.Random(seed)
        pos = [(f"s{i}", "".join(rng.choice(a) for _ in range(80))) for i in range(n)]
        neg = [(f"s{i}", "".join(rng.choice(b) for _ in range(80))) for i in range(n)]
        return pos, neg

    def test_normalize(self):
        # NFKC 把 ℃ 拆成 °C、全角字母变半角，再统一小写：真、假两边同样处理，不会制造差别
        self.assertEqual(c2st.normalize("患者[患者]于 2019年3月5日 入院，T 38.5℃ ＡＢ"), "患者于0000年0月0日入院,t00.0°cab")
        self.assertEqual(c2st.normalize("血常规 | 白细胞 ｜ 正常"), "血常规白细胞正常")

    def test_separable_and_null(self):
        pos, neg = self._toy(60, "咳嗽发热胸痛予以", "咳嗽发热胸痛给予", 1)
        r = c2st.c2st(pos, neg, seeds=c2st.SEEDS[:2])
        self.assertGreater(r["auc_mean"], 0.9)
        self.assertGreater(r["ci_lo"], 0.8)
        # 两边同分布：单次抽样每组 60 对时 AUC 的波动约 ±0.09（总经理 10-27 跑过 20 次，均值 0.517），
        # 所以看 8 次独立抽样的平均，不看单次
        nulls = []
        for s in range(2, 10):
            pos, neg = self._toy(60, "咳嗽发热胸痛予以", "咳嗽发热胸痛予以", s)
            nulls.append(c2st.c2st(pos, neg, seeds=c2st.SEEDS[:1])["auc_mean"])
        self.assertLess(abs(sum(nulls) / len(nulls) - 0.5), 0.07)

    def test_null_split_is_centered(self):
        rng = random.Random(3)
        items, strata = [], {}
        for i in range(120):
            code = f"C{i % 6}"
            # 同一诊断的病历彼此相像：整体随机分两半会让留出 AUC 系统性偏低，配对设计不会
            items.append((f"s{i}", code * 5 + "".join(rng.choice("咳嗽发热胸痛予以给予") for _ in range(60))))
            strata[f"s{i}"] = code
        aucs = c2st.null_split(items, strata, reps=8)
        self.assertLess(abs(sum(aucs) / len(aucs) - 0.5), 0.08)

    def test_groups_stay_together(self):
        labels = [1, 0] * 30
        groups = [f"g{i // 2}" for i in range(60)]
        for tr, te in c2st.folds(labels, groups, 7):
            gte = {groups[i] for i in te}
            self.assertFalse(gte & {groups[i] for i in tr})

    def test_length_classifier(self):
        pos = [(f"s{i}", [100 + i, 300, 200, 80]) for i in range(40)]
        neg = [(f"s{i}", [30 + i, 300, 200, 80]) for i in range(40)]
        self.assertGreater(c2st.c2st(pos, neg, kind="length", seeds=c2st.SEEDS[:1])["auc_mean"], 0.95)

    def test_feature_mask(self):
        self.assertIsNone(c2st.feature_mask_reason("予以", 0, 50))
        self.assertIsNone(c2st.feature_mask_reason("咳嗽", 120, 90))
        self.assertEqual(c2st.feature_mask_reason("咳嗽", 7, 90), "真实病历里不到 20 份")
        self.assertEqual(c2st.feature_mask_reason("甲乙丙", 60, 0), "只在真实病历里常见")
        self.assertEqual(c2st.feature_mask_reason("人民医", 300, 40), "机构或地名字样")
        self.assertEqual(c2st.feature_mask_reason("甲县", 300, 40), "机构或地名字样")
        self.assertEqual(c2st.feature_mask_reason("0月0", 300, 300), "像日期")
        self.assertEqual(c2st.feature_mask_reason("[患", 300, 0), "占位符")

    def test_top_features_shape(self):
        pos, neg = self._toy(30, "予以患者", "给予病人", 4)
        tf = c2st.top_features([x for _, x in pos], [x for _, x in neg], k=3)
        self.assertEqual(len(tf), 6)
        self.assertTrue(all((f["ngram"] is None) == (f["masked"] is not None) for f in tf))

    def test_text_builders(self):
        t = c2st.care7_text(REAL_LIKE)
        self.assertEqual(t.count("｜"), 3)
        self.assertNotIn("社区获得性肺炎｜", t[:10])
        adm = {"docs": [{"doc_type": "hospital_record", "text": "入院记录 主诉：咳嗽3天"},
                        {"doc_type": "progress_note", "text": "首次病程记录 病例特点：……"},
                        {"doc_type": "discharge_summary", "text": "出院记录 出院医嘱：休息"}]}
        self.assertEqual(c2st.admission_lengths(adm)[-1], 3)
        self.assertEqual(c2st.admission_text(adm, ("hr",)), c2st.normalize("入院记录 主诉：咳嗽3天"))


class TestDemog(unittest.TestCase):
    def test_bands(self):
        for age, b in ((0, "0-5"), (5.99, "0-5"), (6, "6-13"), (13.5, "6-13"), (14, "14-17"), (17.9, "14-17"),
                       (18, "18-44"), (44, "18-44"), (45, "45-64"), (64.9, "45-64"), (65, "65+"), (101, "65+")):
            self.assertEqual(demog.band(age), b)
        self.assertIsNone(demog.band(None))

    def test_stated_age(self):
        self.assertEqual(demog.stated_band("8个月"), "0-5")     # 规划器把它当 8 岁，这里不跟
        self.assertEqual(demog.stated_band("65"), "65+")
        self.assertEqual(demog.stated_band("47岁"), "45-64")
        self.assertIsNone(demog.stated_band("不详"))
        self.assertTrue(demog.changed_far("68", "52"))
        self.assertFalse(demog.changed_far("63", "66"))          # 换了年龄段，但只差 3 岁
        self.assertTrue(demog.changed_far("50", "70岁"))
        self.assertFalse(demog.changed_far("47", "61"))          # 同一段
        self.assertFalse(demog.changed_far("不详", "3"))
        self.assertEqual(demog.explicit_age(REAL_LIKE), 0)
        self.assertEqual(demog.explicit_age(dict(REAL_LIKE, **{"History of Present Illness": "患者，男，67岁，咳嗽5天。"})), 1)

    def test_self_auc(self):
        qbar = {"A": {"65+": 0.5, "18-44": 0.5}}
        rows = [{"65+": 0.9, "18-44": 0.1}] * 12 + [{"65+": 0.2, "18-44": 0.8}] * 12
        own = ["65+"] * 12 + ["18-44"] * 12
        r = demog.self_auc(rows, own, ["A"] * 24, qbar)
        self.assertAlmostEqual(r["macro"], 1.0)
        flat = demog.self_auc([{"65+": 0.5, "18-44": 0.5}] * 24, own, ["A"] * 24, qbar)
        self.assertAlmostEqual(flat["macro"], 0.5)
        self.assertIsNone(demog.self_auc(rows[:5], own[:5], ["A"] * 5, qbar)["macro"])

    def test_tvd_w1(self):
        self.assertAlmostEqual(demog.tvd({"男": 3, "女": 1}, {"男": 1, "女": 1}), 0.25)
        self.assertAlmostEqual(demog.w1_bins({10: 1}, {12: 1}), 10.0)       # 两档相差 2 档 × 5 岁
        self.assertAlmostEqual(demog.w1_bins({10: 1, 12: 1}, {10: 1, 12: 1}), 0.0)

    def test_typicality_rewards_marginal_matching(self):
        from collections import Counter
        ref = {"E11": {"sex": Counter({"男": 50, "女": 50}), "age": Counter({2: 10, 12: 90})}}
        before = [("E11", "男", 62)] * 10                       # 全是 60 多岁的男性
        after = [("E11", "男", 62)] * 4 + [("E11", "女", 62)] * 5 + [("E11", "女", 12)]   # 按配额改了性别、年龄
        tb, ta = demog.typicality(before, ref), demog.typicality(after, ref)
        self.assertGreater(ta["t_sex"], tb["t_sex"])
        self.assertGreater(ta["t_age"], tb["t_age"])
        self.assertEqual(demog.typicality([("X99", "男", 3)], ref)["skipped_codes"], ["X99"])

    def test_age_model_sees_through_field_rewrite(self):
        rng = random.Random(5)
        old = "老年患者既往高血压冠心病多年退休在家"
        kid = "患儿幼儿园就读家长代诉哭闹喂养"
        texts, labels = [], []
        for i in range(60):
            texts.append("".join(rng.choice(old) for _ in range(40)))
            labels.append("65+")
            texts.append("".join(rng.choice(kid) for _ in range(40)))
            labels.append("0-5")
        m = demog.fit(texts, labels)
        t = "".join(rng.choice(old) for _ in range(40))
        p_old, p_new = demog.p_stated(m, [t, t], ["65+", "0-5"])
        self.assertGreater(p_old - p_new, 0.5)          # 年龄栏改成幼儿，正文还是老人：给自己年龄栏的概率掉下去
        self.assertEqual(demog.p_stated(m, [t], ["14-17"]), [None])
        p, prior = demog.crossfit(texts, labels)
        self.assertGreater(sum(p) / len(p) - sum(prior) / len(prior), 0.3)


    def test_prefers_old_cancels_disease_level_information(self):
        qbar = {"65+": 0.6, "45-64": 0.3, "18-44": 0.1}
        # 正文只透露病种：概率和同病种平均一样，二选一打平
        self.assertEqual(demog.prefers_old(dict(qbar), "65+", "18-44", qbar), 0.5)
        # 正文有"老人"线索：比值偏向 65+
        self.assertEqual(demog.prefers_old({"65+": 0.8, "45-64": 0.15, "18-44": 0.05}, "65+", "18-44", qbar), 1.0)
        # 改写后那一段更像：0
        self.assertEqual(demog.prefers_old({"65+": 0.3, "45-64": 0.2, "18-44": 0.5}, "65+", "18-44", qbar), 0.0)
        self.assertIsNone(demog.prefers_old(dict(qbar), "65+", "65+", qbar))
        self.assertIsNone(demog.prefers_old(dict(qbar), "65+", "0-5", qbar))
        self.assertIsNone(demog.prefers_old(dict(qbar), "65+", "18-44", {}))

    def test_two_choice_on_simulated_rewrite(self):
        """模拟：两个病种，A 多见于老人、B 多见于年轻人。一半正文只写病种，一半还写了这个人的年龄线索。
        把年龄改成同病种里的另一段以后：有线索的正文大多认得出原来那一段，只写病种的正文五五开。"""
        rng = random.Random(11)
        cue = {"65+": "退休老伴老年", "18-44": "上班工作学校"}
        dis = {"A": "胸闷气短心悸", "B": "腹痛恶心呕吐"}

        def make(code, band, with_cue):
            pool = dis[code] + (cue[band] if with_cue else "")
            return "".join(rng.choice(pool) for _ in range(60))
        texts, labels, codes, cues = [], [], [], []
        for i in range(400):
            code = "A" if i % 2 else "B"
            old_share = 0.75 if code == "A" else 0.25
            band = "65+" if rng.random() < old_share else "18-44"
            wc = i % 4 < 2
            texts.append(make(code, band, wc))
            labels.append(band)
            codes.append(code)
            cues.append(wc)
        rows, _ = demog.crossfit_rows(texts, labels)
        qbar = demog.qbar_by_group(rows, codes)
        flip = {"65+": "18-44", "18-44": "65+"}
        with_cue = [demog.prefers_old(r, l, flip[l], qbar[c]) for r, l, c, w in zip(rows, labels, codes, cues) if w]
        without = [demog.prefers_old(r, l, flip[l], qbar[c]) for r, l, c, w in zip(rows, labels, codes, cues) if not w]
        self.assertGreater(sum(with_cue) / len(with_cue), 0.8)
        self.assertLess(abs(sum(without) / len(without) - 0.5), 0.12)
        # 不扣同病种平均（直接比概率）时，只写病种的正文也会"偏向原来那一段"：这正是要扣掉的病种层面信息
        raw = [1.0 if r[l] > r[flip[l]] else 0.0 for r, l, w in zip(rows, labels, cues) if not w]
        self.assertGreater(sum(raw) / len(raw), sum(without) / len(without))


class _Sec:
    def __init__(self, name, text):
        self.name, self.text = name, text


class _Fact:
    def __init__(self, field, value_norm):
        self.field, self.value_norm = field, value_norm


def _fake_split(text):
    out = []
    for line in text.split("\n"):
        if "：" in line:
            k, v = line.split("：", 1)
            out.append(_Sec(k, v))
    return out


def _fake_facts(age):
    def f(text, aid, doc_no, kind):
        facts = [_Fact("header_sex", "男")]
        if age is not None:
            facts.append(_Fact("header_age", f"{age:g}|ge=0"))
        return facts, []
    return f


class TestRealText(unittest.TestCase):
    ADM = {"admission_id": "X1", "docs": [
        {"doc_type": "progress_note", "text": "主诉：不是这一篇"},
        {"doc_type": "hospital_record", "text": "主诉：咳嗽3天\n现病史：受凉后咳嗽。"},
        {"doc_type": "hospital_record", "text": "主诉：第二篇入院记录不用"},
        {"doc_type": "discharge_summary", "text": "出院诊断：肺炎\n诊疗经过：抗感染。\n出院医嘱：休息"}]}

    def test_cut_like_r7(self):
        rec, why = realtext.care7_from_admission(self.ADM, _fake_split, _fake_facts(67))
        self.assertEqual(why, "ok")
        self.assertEqual(rec, {"Gender": "男", "Age": "67", "Primary Diagnosis": "肺炎", "Chief Complaint": "咳嗽3天",
                               "History of Present Illness": "受凉后咳嗽。", "Hospital Course": "抗感染。",
                               "Discharge Instructions": "休息"})

    def test_reasons(self):
        rec, why = realtext.care7_from_admission(self.ADM, _fake_split, _fake_facts(0.5))
        self.assertEqual(why, "age_lt1")
        self.assertIsNone(rec["Age"])
        self.assertEqual(rec["History of Present Illness"], "受凉后咳嗽。")    # 读不出年龄也照样截，截法核对要用
        self.assertEqual(realtext.care7_from_admission(self.ADM, _fake_split, _fake_facts(None))[1], "no_age")
        no_ds = {"docs": self.ADM["docs"][:2]}
        self.assertEqual(realtext.care7_from_admission(no_ds, _fake_split, _fake_facts(67)), (None, "missing_doc"))

    @unittest.skipUnless(os.environ.get("RULER_G1"), "设 RULER_G1 才用关1 尺子原件核对")
    def test_with_ruler_g1(self):
        with _Foreign(os.environ["RULER_G1"]):
            from ruler_g1 import extract_facts, split_sections
            self._check(extract_facts, split_sections)

    def _check(self, extract_facts, split_sections):
        adm = {"admission_id": "X2", "docs": [
            {"doc_type": "hospital_record", "text": "入院记录\n姓名：[姓名]  性别：女  年龄：8岁\n主诉：发热2天。\n现病史：患儿2天前受凉后发热。\n既往史：体健。"},
            {"doc_type": "discharge_summary", "text": "出院记录\n出院诊断：急性扁桃体炎\n诊疗经过：予抗感染治疗。\n出院医嘱：注意休息。"}]}
        rec, why = realtext.care7_from_admission(adm, split_sections, extract_facts)
        self.assertEqual(why, "ok")
        self.assertEqual((rec["Gender"], rec["Age"]), ("女", "8"))
        self.assertEqual(rec["History of Present Illness"], "患儿2天前受凉后发热。")
        self.assertEqual(rec["Primary Diagnosis"], "急性扁桃体炎")
        self.assertEqual(rec["Discharge Instructions"], "注意休息。")


class TestDiffs(unittest.TestCase):
    def test_loop_patch(self):
        before = dict(REAL_LIKE)
        after = dict(REAL_LIKE)
        after["History of Present Illness"] = before["History of Present Illness"].replace("食欲减退", "食欲减退，饮食量减少")
        d = diffs.record_diff(before, after)
        self.assertEqual(d["_total"]["changed_fields"], 1)
        self.assertIn("C06_一般情况", d["hpi"]["items"])
        self.assertEqual(d["hc"]["changed"], 0)
        self.assertGreater(d["hpi"]["added_chars"], 0)


class TestAdmission(unittest.TestCase):
    def test_roles_copy_headers(self):
        p = "患者今日体温37.2℃，咳嗽较前减轻，双肺呼吸音粗，继续目前抗感染治疗。"
        adm = {"docs": [
            {"doc_type": "hospital_record", "text": "主诉：咳嗽3天。\n现病史：……\n既往史：体健。\n个人史：无。\n婚育史：已婚。\n家族史：无。\n体格检查：T 36.8℃\n专科检查：无。\n辅助检查：无。\n初步诊断：肺炎"},
            {"doc_type": "progress_note", "text": "首次病程记录\n病例特点：……\n诊断依据：……\n鉴别诊断：……\n诊疗计划：……"},
            {"doc_type": "progress_note", "text": p},
            {"doc_type": "progress_note", "text": p},
            {"doc_type": "discharge_summary", "text": "入院情况：……\n入院诊断：肺炎\n诊疗经过：……\n出院诊断：肺炎\n出院情况：好转\n出院医嘱：休息"}]}
        rows = analyze_admission(adm)
        self.assertEqual([r["role"] for r in rows], ["hr", "fp", "course", "course", "discharge"])
        self.assertEqual(rows[0]["header_cov"], 1.0)
        self.assertEqual(rows[1]["header_cov"], 1.0)
        self.assertEqual(rows[4]["header_cov"], 1.0)
        self.assertIsNone(rows[0]["copy5_prev"])
        self.assertEqual(rows[3]["copy5_prev"], 1.0)
        self.assertEqual(doc_role({"role": "FP", "text": "x"}), "fp")
        self.assertEqual(doc_role({"role": "DS", "text": "x"}), "discharge")


class TestStats(unittest.TestCase):
    def test_paired_mean_diff(self):
        a = {f"s{i}": 1.0 for i in range(30)}
        b = {f"s{i}": 0.5 for i in range(30)}
        est, lo, hi, n, ref = stats.paired_mean_diff(a, b)
        self.assertAlmostEqual(est, 0.5)
        self.assertEqual(n, 30)
        self.assertEqual(stats.band_verdict(-0.01, 0.01, 0.5), "带内")


EXPECTED_CRITERIA_SHA = "3d1bff826da606126780753e0c95756240ab59dc739754688290b8a9f48ab8cf"

if __name__ == "__main__":
    unittest.main()
