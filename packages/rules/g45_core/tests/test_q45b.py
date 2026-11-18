"""g45_core 0.2 新加部分的自检（尺子 0.2、q45b、agepair）。全部用编造的病历，不碰真实病历。
在 g45_core 目录下：python -B -m unittest discover -s tests -v
"""
import json
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ruler_g4 import RULES_VERSION, analyze_care7, analyze_text, info_units, time_anchors, warning_spans
from q45_kit import agepair, c2st, floors, judge, q45b

from test_q45 import REAL_LIKE


def _units(text, dx=""):
    return [(u["type"], u["key"]) for u in info_units(text, dx)]


def _rec(cc, hpi, hc, di, dx="社区获得性肺炎", sex="男", age="60"):
    return {"Gender": sex, "Age": age, "Primary Diagnosis": dx, "Chief Complaint": cc,
            "History of Present Illness": hpi, "Hospital Course": hc, "Discharge Instructions": di}


class TestRuler02(unittest.TestCase):
    def test_version(self):
        self.assertEqual(RULES_VERSION, "g4-rules-0.2")
        with open(os.path.join(ROOT, "ruler_g4", "VERSION"), encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), "g4-rules-0.2")

    def test_treatment_classes_count(self):
        # Q45 审查第 3 节：合成空句判错的第二类
        u = _units("予抗感染、止咳化痰治疗，病情稳定。")
        self.assertIn(("PROC", "抗感染"), u)
        self.assertIn(("PROC", "止咳化痰"), u)
        u = _units("住院期间予限水、降钾、降压、利尿及纠正电解质紊乱等治疗。")
        for k in ("限水", "降钾", "降压", "利尿", "纠正电解质"):
            self.assertIn(("PROC", k), u)
        # 太泛的仍然不算
        a = analyze_text("予药物治疗及对症支持治疗，病情好转。")
        self.assertEqual(a["sentences"][0]["empty"], 1)

    def test_developmental_symptoms(self):
        u = _units("近期家长发现患儿对复杂指令理解困难，语言发育较同龄儿童明显滞后。")
        self.assertIn(("SYM", "理解困难"), u)
        self.assertIn(("SYM", "语言发育"), u)
        self.assertIn(("SYM", "明显滞后"), u)

    def test_extra_drugs_and_grams(self):
        self.assertIn(("DRUG", "维生素B12"), _units("遵医嘱服用营养神经药物，如维生素B12，每日一次。"))
        self.assertIn(("NUM", "2克"), _units("每日钠摄入量不超过2克。"))
        # 词表里有的药不因补充表重复计
        u = [x for x in _units("口服甲钴胺0.5mg tid。") if x[0] == "DRUG"]
        self.assertEqual(len(u), 1)

    def test_warning_sentences_in_di(self):
        di = "注意休息，避免劳累，如出现发热、咳嗽或呼吸困难应及时复诊。口服头孢克肟0.1g bid。"
        self.assertEqual(len(warning_spans(di)), 1)
        a = analyze_text(di, field="di")
        self.assertFalse([u for u in a["iu"] if u["type"] == "SYM"])
        self.assertTrue([u for u in a["iu"] if u["type"] == "DRUG"])
        # 现病史里同样的字不受影响
        h = analyze_text("如出现发热，及时就诊。", field="hpi")
        self.assertTrue([u for u in h["iu"] if u["type"] == "SYM"])

    def test_time_anchors(self):
        t = "2019年3月5日入院，3月6日行胃镜，术后第2天进食，次日出院。[日期]复查。"
        a = time_anchors(t)
        self.assertEqual(a["date_n"], 3)
        self.assertEqual(a["relday_n"], 2)
        r = analyze_care7(_rec("咳嗽3天", "患者3天前咳嗽。", t, "定期复查。"))
        self.assertEqual(r["anchor_n"], 5)
        self.assertEqual(time_anchors("咳嗽1周，1年前确诊")["date_n"], 0)

    def test_floors_unchanged_by_02(self):
        # 填表病历在 0.2 下仍然只有 3 个信息单元（1周、1片、1次一类的数），关键词堆砌仍然不加信息
        t = analyze_care7(floors.ftpl(REAL_LIKE))
        self.assertEqual(t["iu_by_type"]["SYM"] + t["iu_by_type"]["PROC"] + t["iu_by_type"]["DRUG"], 0)
        host = analyze_care7(REAL_LIKE)
        kw = analyze_care7(floors.fkw(REAL_LIKE))
        hk = {(u["type"], u["key"]) for f in host["fields"].values() for u in f["iu"]}
        kk = {(u["type"], u["key"]) for f in kw["fields"].values() for u in f["iu"]}
        self.assertEqual(hk, kk)


class TestNewSlots(unittest.TestCase):
    def test_first_dx(self):
        self.assertEqual(q45b.first_dx("1.2型糖尿病\n2.高血压病3级"), "2型糖尿病")
        self.assertEqual(q45b.first_dx("出院诊断：1、社区获得性肺炎（重症）；2、慢性阻塞性肺疾病"), "社区获得性肺炎")
        self.assertEqual(q45b.first_dx("[疾病名称]急性胆囊炎"), "急性胆囊炎")
        self.assertEqual(q45b.first_dx("1. "), "")
        self.assertEqual(q45b.first_dx("冠状动脉粥样硬化性心脏病 不稳定型心绞痛 心功能Ⅱ级"),
                         "冠状动脉粥样硬化性心脏病 不稳定型心绞痛 心功能Ⅱ级"[:30])

    def test_eligible_and_pick(self):
        rec = dict(REAL_LIKE)
        self.assertEqual(q45b.eligible("a1", rec, "ok", set()), (True, "ok"))
        self.assertEqual(q45b.eligible("a1", rec, "ok", {"a1"}), (False, "used_slot"))
        self.assertEqual(q45b.eligible("a1", rec, "no_age", set()), (False, "no_age"))
        short = dict(rec, **{"Chief Complaint": "咳"})
        self.assertEqual(q45b.eligible("a1", short, "ok", set()), (False, "short_field"))
        ids = [f"x{i:04d}" for i in range(1000)]
        p1 = q45b.pick_new(ids)
        p2 = q45b.pick_new(list(reversed(ids)))
        self.assertEqual(p1, p2)
        self.assertEqual(len(p1), 256)
        self.assertEqual(len(set(p1)), 256)
        self.assertEqual(len(q45b.pick_new(ids[:10])), 10)

    def test_template_for(self):
        rec = dict(REAL_LIKE, **{"Primary Diagnosis": "1.社区获得性肺炎\n2.高血压病"})
        t = q45b.template_for(rec)
        self.assertEqual(t["Primary Diagnosis"], "社区获得性肺炎")
        self.assertEqual(t["Chief Complaint"], "社区获得性肺炎相关症状1周。")
        self.assertEqual(t["Gender"], rec["Gender"])
        self.assertEqual(t["Age"], rec["Age"])
        self.assertEqual(t["Hospital Course"], floors.TPL["Hospital Course"])

    def test_unpaired_mean_diff(self):
        rng = random.Random(1)
        a = [rng.gauss(24, 5) for _ in range(256)]
        b = [rng.gauss(48, 10) for _ in range(256)]
        est, lo, hi, na, nb, mb = q45b.unpaired_mean_diff(a, b)
        self.assertLess(hi, -20)
        self.assertLess(lo, est)
        self.assertLess(est, hi)
        same = q45b.unpaired_mean_diff(a, [x + 0.0 for x in a])
        self.assertLess(same[1], 0)
        self.assertGreater(same[2], 0)


class TestEqualLengthC2ST(unittest.TestCase):
    def test_equal_pair_lengths(self):
        short = _rec("咳嗽3天", "患者3天前咳嗽。", "予抗感染治疗。", "注意休息。")
        xa, xb = q45b.equal_pair(REAL_LIKE, short)
        self.assertEqual(q45b.equal_lengths(xa), q45b.equal_lengths(xb))
        self.assertEqual(xb, c2st.care7_text(short))   # 较短那份不被截
        self.assertTrue(xa.startswith(c2st.normalize(REAL_LIKE["Chief Complaint"])[:4]))

    def test_strip_dates(self):
        r = _rec("咳嗽3天", "2019年3月5日起咳嗽，术后第2天加重。", "次日好转。", "复查。")
        xa, _ = q45b.equal_pair(r, r, no_dates=True)
        self.assertNotIn("年0月", xa)
        self.assertNotIn("次日", xa)

    def test_length_is_uninformative_after_equalizing(self):
        rng = random.Random(3)
        pos, neg = [], []
        for i in range(80):
            long_r = _rec("咳嗽" * rng.randint(3, 8), "患者咳嗽" * rng.randint(20, 40), "予治疗" * rng.randint(20, 40),
                          "注意休息" * rng.randint(5, 10))
            short_r = _rec("咳嗽" * rng.randint(2, 4), "患者咳嗽" * rng.randint(5, 10), "予治疗" * rng.randint(5, 10),
                           "注意休息" * rng.randint(1, 4))
            xs, xl = q45b.equal_pair(short_r, long_r)
            pos.append((f"S{i}", q45b.equal_lengths(xs)))
            neg.append((f"S{i}", q45b.equal_lengths(xl)))
        res = c2st.c2st(pos, neg, kind="length", seeds=c2st.SEEDS[:1])
        self.assertLess(res["auc_mean"], 0.6)

    def test_null_split_equal_centered(self):
        rng = random.Random(5)
        words = ["咳嗽", "发热", "胸痛", "腹痛", "头晕", "乏力", "气促", "呕吐"]
        items, strata = [], {}
        for i in range(120):
            r = _rec("".join(rng.choice(words) for _ in range(3)), "".join(rng.choice(words) for _ in range(30)),
                     "".join(rng.choice(words) for _ in range(30)), "".join(rng.choice(words) for _ in range(10)))
            items.append((f"S{i}", r))
            strata[f"S{i}"] = f"C{i % 4}"
        aucs = q45b.null_split_equal(items, strata, reps=6)
        m = sum(aucs) / len(aucs)
        self.assertGreater(m, 0.38)
        self.assertLess(m, 0.62)


class TestAgePair(unittest.TestCase):
    def test_prompt_has_no_demographic_fields(self):
        rec = dict(REAL_LIKE, **{"Age": "67岁", "Primary Diagnosis": "社区获得性肺炎甲"})
        p = agepair.prompt_for(rec, "67岁", "22", "ON")
        self.assertIn("A. 67岁", p)
        self.assertIn("B. 22岁", p)
        self.assertNotIn("社区获得性肺炎甲", p)
        self.assertNotIn("性别：", p)
        self.assertIn("【现病史】" + REAL_LIKE["History of Present Illness"], p)
        q = agepair.prompt_for(rec, "67岁", "22", "NO")
        self.assertIn("A. 22岁", q)
        self.assertIn("B. 67岁", q)

    def test_age_label(self):
        self.assertEqual(agepair.age_label("8个月"), "8个月")
        self.assertEqual(agepair.age_label("37"), "37岁")
        self.assertEqual(agepair.age_label("62岁"), "62岁")
        self.assertIsNone(agepair.age_label("不详"))

    def test_request_body(self):
        b = agepair.request_body("问题", judge.seed_for("agepair", "x", 0))
        self.assertEqual(b["max_tokens"], 8)
        for k in ("temperature", "top_p", "top_k", "repetition_penalty", "chat_template_kwargs", "model"):
            self.assertEqual(b[k], judge.request_body("问题", 1)[k])
        self.assertEqual(json.loads(judge.request_bytes(b)), b)

    def test_parse_ab(self):
        for s, want in (("A", "A"), (" B。", "B"), ("答案：A", "A"), ("**B**", "B"), ("选B", "B"),
                        ("A或B都有可能", None), ("我无法判断", None), ("ABC", None), ("", None)):
            self.assertEqual(agepair.parse_ab(s), want, s)

    def test_chose_old_and_orders(self):
        self.assertEqual(agepair.chose_old("A", "ON"), 1)
        self.assertEqual(agepair.chose_old("B", "ON"), 0)
        self.assertEqual(agepair.chose_old("A", "NO"), 0)
        self.assertEqual(agepair.chose_old("B", "NO"), 1)
        self.assertIsNone(agepair.chose_old(None, "ON"))

    def test_swap_partners(self):
        code_of = {f"S{i}": ("C1" if i < 5 else "C2") for i in range(8)}
        p = agepair.swap_partners("S1", code_of)
        self.assertEqual(len(p), 2)
        self.assertTrue(all(code_of[x] == "C1" and x != "S1" for x in p))
        self.assertEqual(p, agepair.swap_partners("S1", dict(reversed(list(code_of.items())))))
        self.assertEqual(len(agepair.swap_partners("S5", {"S5": "C9"})), 0)

    def test_build_jobs(self):
        code_of = {"S1": "C1", "S2": "C1", "S3": "C1"}
        texts = {s: REAL_LIKE for s in code_of}
        jobs = agepair.build_jobs([{"unit_id": "u1", "slot": "S1", "old": "67", "new": "20"}], texts, code_of, "R", 0)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(len({j["job_id"] for j in jobs}), 6)
        self.assertEqual({j["role"] for j in jobs}, {"own", "swap0", "swap1"})
        self.assertTrue(all(j["text_slot"] != "S1" for j in jobs if j["role"] != "own"))

    def _simulate(self, p_own, p_swap, n_slots=100, seed=7):
        """模拟读者：本人正文以 p_own 的概率选原年龄，换来的以 p_swap 的概率选。"""
        rng = random.Random(seed)
        results = []
        for i in range(n_slots):
            for role, p in (("own", p_own), ("swap0", p_swap), ("swap1", p_swap)):
                for order in agepair.ORDERS:
                    old = rng.random() < p
                    ans = ("A" if old else "B") if order == "ON" else ("B" if old else "A")
                    results.append({"unit_id": f"u{i}", "slot": f"S{i}", "role": role, "order": order, "answer": ans})
        return agepair.own_vs_swap(agepair.unit_scores(results))

    def test_own_vs_swap_detects_text_specific_age(self):
        r = self._simulate(0.85, 0.55)
        self.assertGreater(r["D"], 0.2)
        self.assertGreater(r["ci_lo"], 0)
        self.assertEqual(r["units"], 100)

    def test_own_vs_swap_cancels_reader_bias(self):
        # 读者天生偏好原年龄（比如原年龄都是中年、读者爱选中年），但正文里没有这个人的线索：本人和换来的一样，D≈0
        r = self._simulate(0.75, 0.75)
        self.assertLess(r["ci_lo"], 0)
        self.assertGreater(r["ci_hi"], 0)
        self.assertGreater(r["own_mean"], 0.65)

    def test_unit_scores_handles_failures(self):
        res = [{"unit_id": "u", "slot": "S", "role": "own", "order": "ON", "answer": None},
               {"unit_id": "u", "slot": "S", "role": "own", "order": "NO", "answer": "B"},
               {"unit_id": "u", "slot": "S", "role": "swap0", "order": "ON", "answer": None},
               {"unit_id": "u", "slot": "S", "role": "swap0", "order": "NO", "answer": None}]
        s = agepair.unit_scores(res)
        self.assertEqual(s["u"]["own"], 1.0)
        self.assertEqual(s["u"]["swap"], [None])
        r = agepair.own_vs_swap(s)
        self.assertEqual(r["units"], 0)
        self.assertEqual(r["dropped"], 1)


if __name__ == "__main__":
    unittest.main()
