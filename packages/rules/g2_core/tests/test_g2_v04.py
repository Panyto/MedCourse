"""g2-core 0.4 新增部分的自检：泄漏检查 v2、C2 三分量、相邻篇 tf-idf、G-E′ 生成。只用虚构病历。

运行：在 g2_core 目录下  python -B -m unittest tests.test_g2_v04 -v
伙计第 0 步必须和 tests.test_g2 一起全部通过；任何一条不过，停下来写进 GAPS，不要自己改代码。
"""
import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
sys.dont_write_bytecode = True

from test_g2 import gb_slot, good_draft, real  # noqa: E402
from g2_kit import gen as G  # noqa: E402
from g2_kit import gen2 as G2  # noqa: E402
from g2_kit import oracle2 as O2  # noqa: E402
from ruler_g2 import core_set, evaluate_admission  # noqa: E402
from ruler_g2.c2parts import c2_components  # noqa: E402
from ruler_g2.similarity import RefTfidf, adjacent_tfidf, cosine, progress_between  # noqa: E402


def _doc(slot, role, text, nth=0):
    spec = [x for x in slot["skeleton"] if x["role"] == role][nth]
    return {"slot_id": slot["slot_id"], "k": spec["k"], "role": role, "doc_type_cn": spec["doc_type_cn"],
            "day": spec["day"], "date": G.doc_date(slot, spec["day"]), "text": text}


# ---------- 泄漏检查 v2 ----------

class TestOracle2(unittest.TestCase):
    def setUp(self):
        self.slot, self.dr = gb_slot(), good_draft()   # FP 第0天，SUP 第1天，PRE 第2天，手术第3天

    def counts(self, role, text, nth=0):
        per, _ = O2.oracle_leaks_v2(self.dr, self.slot, [_doc(self.slot, role, text, nth)])
        h = per[0]
        return h["O"], h["A"], h["B"], h["B_loose"]

    def test_same_as_v1_on_v1_cases(self):
        clean = [_doc(self.slot, "FP", "初步诊断：急性胆囊炎？胆总管结石待排。完善MRCP。"),
                 _doc(self.slot, "SUP", "腹部CT示胆囊壁增厚。急性胆囊炎诊断明确。拟行腹腔镜胆囊切除术。MRCP结果待回报。"),
                 _doc(self.slot, "PRE", "拟施手术名称和方式：腹腔镜胆囊切除术。术后可能出现出血等并发症。")]
        _, summ = O2.oracle_leaks_v2(self.dr, self.slot, clean)
        self.assertEqual((summ["O"], summ["A"], summ["B"]), (0, 0, 0))
        leak = [_doc(self.slot, "FP", "胆总管结石已排除。"),
                _doc(self.slot, "SUP", "MRCP示胆总管未见结石。"),
                _doc(self.slot, "PRE", "患者已行腹腔镜胆囊切除术，手术顺利。")]
        per, summ = O2.oracle_leaks_v2(self.dr, self.slot, leak)
        self.assertEqual([(h["O"], h["A"], h["B"]) for h in per], [(0, 0, 1), (1, 0, 0), (0, 1, 0)])
        self.assertEqual(summ["any"], 1)

    # A：pkg1b 里读到的假报警
    def test_a_plan_and_past_not_counted(self):
        for t in ("需行腹腔镜胆囊切除术以解除症状。", "必要时行腹腔镜胆囊切除术。", "考虑行腹腔镜胆囊切除术。",
                  "术前做好各项准备，确保手术顺利进行。", "术后需定期随访，评估恢复情况。",
                  "安排第3天行腹腔镜胆囊切除术。", "适时进行腹腔镜胆囊切除术。", "评估是否需要进行腹腔镜胆囊切除术。"):
            self.assertEqual(self.counts("PRE", t)[1], 0, t)
        hr_like = "既往史：1年前因胆囊息肉行腹腔镜胆囊切除术，术后恢复良好。"
        self.assertEqual(self.counts("FP", hr_like)[1], 0)
        self.assertEqual(self.counts("FP", "既往史：曾行阑尾切除术，术后恢复良好。")[1], 0)

    def test_a_true_leaks(self):
        self.assertEqual(self.counts("PRE", "患者术后第1天，切口干燥。")[1], 1)
        self.assertEqual(self.counts("PRE", "今日在全麻下行腹腔镜胆囊切除术。")[1], 1)
        self.assertEqual(self.counts("PRE", "患者已行胆囊切除术，恢复可。")[1], 1)        # 去掉"腹腔镜"后的名字
        self.assertEqual(self.counts("SUP", "已于今日上午行LC胆囊切除术。")[1], 1)       # "已行……术"与手术名共有 3 字

    # B：严格档才进判定
    def test_b_hedged_not_strict(self):
        for t in ("初步诊断为急性胆囊炎，仍需进一步明确。", "目前考虑的主要诊断为急性胆囊炎。", "需进一步排除急性胰腺炎。",
                  "不能完全排除急性胰腺炎。", "尚不能完全排除急性胰腺炎的可能。", "胆总管结石待排除。",
                  "需进一步评估以排除胆总管结石的可能性。", "若急性胆囊炎明确，考虑手术。", "安排血淀粉酶检查，排除急性胰腺炎可能。",
                  "目前诊断为急性胆囊炎可能性较大。", "需进一步明确诊断，排除急性胰腺炎及胆总管结石的可能性。"):
            self.assertEqual(self.counts("FP", t)[2], 0, t)

    def test_b_loose(self):
        self.assertEqual(self.counts("FP", "腹部超声示胆囊增大，符合急性胆囊炎。")[2:], (0, 1))
        self.assertEqual(self.counts("FP", "暂不考虑急性胰腺炎。")[2:], (0, 1))
        self.assertEqual(self.counts("FP", "可以暂时排除急性胰腺炎。")[2:], (0, 1))
        self.assertEqual(self.counts("FP", "目前诊断为急性胆囊炎。")[2:], (0, 1))
        self.assertEqual(self.counts("FP", "症状不支持急性胰腺炎。")[2:], (0, 1))

    def test_b_strict(self):
        for t in ("急性胆囊炎诊断明确。", "经专科检查确诊为急性胆囊炎。", "已排除急性胰腺炎。", "排除了急性胰腺炎的可能性。",
                  "腹部CT检查证实急性胆囊炎。", "诊断为急性胆囊炎明确。"):
            self.assertEqual(self.counts("FP", t)[2], 1, t)
        # 同一天确诊的不算
        self.assertEqual(self.counts("SUP", "急性胆囊炎诊断明确。")[2], 0)

    # O
    def test_o_pending_and_generic_not_counted(self):
        for t in ("待腹部CT及血淀粉酶结果回报后调整方案。", "腹部CT：已安排，尚未出结果。", "完善相关检查，包括腹部CT结果回报，进一步明确诊断。",
                  "需通过腹部CT进一步确认。", "急性胰腺炎：通常伴有血淀粉酶升高，腹部CT可见胰腺水肿。", "MRCP：今日已完成，结果将在明日出具。"):
            self.assertEqual(self.counts("FP", t)[0], 0, t)

    def test_o_dx_name_overlap_not_counted(self):
        dr = copy.deepcopy(self.dr)
        dr["observations"].append({"id": "o6", "name": "病理报告", "kind": "病理", "order_day": 3, "result_day": 6,
                                   "result": "提示急性胆囊炎"})
        per, _ = O2.oracle_leaks_v2(dr, self.slot, [_doc(self.slot, "FP", "初步诊断：急性胆囊炎（疑似）。")])
        self.assertEqual(per[0]["O"], 0)

    def test_o_visible_result_wording_not_counted(self):
        dr = copy.deepcopy(self.dr)
        dr["observations"].append({"id": "o0", "name": "腹部超声", "kind": "影像", "order_day": 0, "result_day": 0,
                                   "result": "胆囊增大，胆囊壁增厚，胆囊内多发结石"})
        per, _ = O2.oracle_leaks_v2(dr, self.slot, [_doc(self.slot, "FP", "腹部超声示胆囊增大，胆囊壁增厚，胆囊内多发结石。")])
        self.assertEqual(per[0]["O"], 0)       # 第1天的 CT 结果与第0天已回报的超声同文，不算 CT 泄漏

    def test_o_true_leaks(self):
        self.assertEqual(self.counts("FP", "腹部CT示胆囊增大，胆囊壁增厚，胆囊内多发结石。")[0], 1)
        self.assertEqual(self.counts("FP", "血淀粉酶：正常。")[0], 1)
        self.assertEqual(self.counts("PRE", "术后病理：慢性胆囊炎急性发作。")[0], 1)
        dr = copy.deepcopy(self.dr)
        dr["observations"].append({"id": "o7", "name": "24小时尿蛋白定量", "kind": "检验", "order_day": 0, "result_day": 2,
                                   "result": "24小时尿蛋白定量 1.8 g"})
        per, _ = O2.oracle_leaks_v2(dr, self.slot, [_doc(self.slot, "FP", "24小时尿蛋白定量1.8g。")])
        self.assertEqual(per[0]["O"], 1)
        per, _ = O2.oracle_leaks_v2(dr, self.slot, [_doc(self.slot, "FP", "完善24小时尿蛋白定量。")])
        self.assertEqual(per[0]["O"], 0)

    def test_ds_skipped_and_summary_keys(self):
        per, summ = O2.oracle_leaks_v2(self.dr, self.slot, [_doc(self.slot, "DS", "术后病理：慢性胆囊炎急性发作。")])
        self.assertEqual(per, [])
        self.assertEqual(set(summ), {"O", "A", "B", "B_loose", "any", "n_docs"})


# ---------- C2 分量、相邻篇 tf-idf ----------

class TestC2Sim(unittest.TestCase):
    def test_c2_components_sum(self):
        for name in ("GB", "KNEE", "PNA"):
            adm = real(name)
            parts = c2_components(adm)
            c2 = evaluate_admission(adm)["readings"]["C2"]["value"]
            self.assertEqual(parts["C2"], c2, name)
            self.assertEqual(parts["exam_kinds"] + parts["surgery"] + parts["dx_changes"], c2, name)

    def test_tfidf(self):
        ref = ["患者今日精神可，无发热。", "复查血常规正常，继续抗感染。", "切口愈合良好，予出院。"]
        m = RefTfidf(ref)
        a = m.vector("患者今日精神可，无发热。")
        self.assertAlmostEqual(cosine(a, a), 1.0, places=9)
        self.assertEqual(m.vector("ＸＹＺ"), {})                       # 不在 R 词表里的 gram 丢掉
        self.assertAlmostEqual(adjacent_tfidf(m, ["患者今日精神可，无发热。"] * 3), 1.0, places=9)
        self.assertIsNone(adjacent_tfidf(m, ["只有一篇"]))
        texts = progress_between(real("GB"))
        self.assertTrue(all(isinstance(t, str) for t in texts))


# ---------- G-E′ 生成 ----------

class TestGen2(unittest.TestCase):
    def setUp(self):
        self.slot, self.dr = gb_slot(), good_draft()
        self.times = G2.doc_times(self.slot)

    def test_slot_filter(self):
        ok = {"dx_name": "急性胆囊炎", "dx_code3": "K81", "age": 58, "sex": "男"}
        self.assertEqual(G2.slot_filter(ok), [])
        self.assertTrue(G2.slot_filter({"dx_name": "早产儿", "dx_code3": "P07", "age": 45, "sex": "男"}))
        self.assertTrue(G2.slot_filter({"dx_name": "新生儿颤抖", "dx_code3": "P96", "age": 24, "sex": "女"}))
        self.assertEqual(G2.slot_filter({"dx_name": "胎儿宫内窘迫", "dx_code3": "O36", "age": 29, "sex": "女"}), [])
        self.assertTrue(G2.slot_filter({"dx_name": "子宫平滑肌瘤", "dx_code3": "D25", "age": 40, "sex": "男"}))
        self.assertTrue(G2.slot_filter({"dx_name": "前列腺增生", "dx_code3": "N40", "age": 70, "sex": "女"}))
        self.assertEqual(G2.slot_filter({"dx_name": "新生儿黄疸", "dx_code3": "P59", "age": 0, "sex": "女"}), [])

    def test_times(self):
        self.assertEqual(self.times, G2.doc_times(self.slot))            # 固定种子
        sk = self.slot["skeleton"]
        by_day = {}
        for x in sk:
            by_day.setdefault(x["day"], []).append(self.times[x["k"]])
        for day, ts in by_day.items():
            self.assertEqual(ts, sorted(ts), day)
            self.assertEqual(len(set(ts)), len(ts), day)
        hr = next(x for x in sk if x["role"] == "HR")
        fp = next(x for x in sk if x["role"] == "FP")
        gap = (int(self.times[fp["k"]][:2]) * 60 + int(self.times[fp["k"]][3:])) - \
              (int(self.times[hr["k"]][:2]) * 60 + int(self.times[hr["k"]][3:]))
        self.assertTrue(60 <= gap <= 300, gap)
        other = dict(self.slot, slot_id="T-OTHER")
        self.assertNotEqual(G2.doc_times(other), self.times)

    def test_template_header_roundtrip(self):
        for spec in self.slot["skeleton"]:
            text = G2.template2(self.slot, spec, self.times)
            self.assertEqual(G2.header_time(spec["doc_type_cn"], text), self.times[spec["k"]], spec)

    def _prompts(self, arm):
        prior, out = [], []
        for spec in self.slot["skeleton"]:
            k = spec["k"]
            msgs, mt, info = G2.doc_messages2(arm, self.slot, k, prior, self.dr, self.times)
            out.append((spec, msgs[1]["content"], info))
            prior.append({"k": k, "doc_type_cn": spec["doc_type_cn"], "day": spec["day"],
                          "text": f"{spec['doc_type_cn']}正文第{k}篇独有句子ABC{k}。\n诊疗计划：1. 继续抗感染。"})
        return out

    def test_visibility(self):
        """非出院记录的提示词里，没有当天以后才回报的结果、当天以后才做的手术、当天以后才有的诊断变化。"""
        for spec, p, _ in self._prompts("G-E2"):
            if spec["role"] == "DS":
                self.assertIn("慢性胆囊炎急性发作", p)
                continue
            day = spec["day"]
            for o in self.dr["observations"]:
                if o["result_day"] > day:
                    self.assertNotIn(o["result"], p, (spec, o["id"]))
            for j in self.dr["judgments"]:
                if j["day"] > day and j["state"] != "疑似":
                    self.assertNotIn(f"{j['diagnosis']}已于第{j['day']}天", p, (spec, j))
            for a in self.dr["actions"]:
                if a["do_day"] > day:
                    self.assertNotIn(f"已行{a['name']}", p, spec)
                    self.assertNotIn(f"今天实施的处置：{a['name']}", p, spec)

    def test_no_prior_full_text_in_ge2(self):
        rs = self._prompts("G-E2")
        for spec, p, info in rs:
            self.assertNotIn("【已写好的前几篇文书】", p)
            for j in range(spec["k"]):
                self.assertNotIn(f"独有句子ABC{j}", p)        # 前几篇正文不进提示词
            if spec["k"] > 0:
                self.assertIn("继续抗感染", p)               # 只进上一篇的处理意见
        rp = self._prompts("G-E2P")
        self.assertIn("独有句子ABC0", rp[2][1])
        self.assertIn("【已写好的前几篇文书】", rp[2][1])

    def test_today_and_state_blocks(self):
        rs = {spec["k"]: (spec, p) for spec, p, _ in self._prompts("G-E2")}
        sup = rs[2][1]                                              # 第1天查房
        self.assertIn("今天回报的检查：", sup)
        self.assertIn("- 腹部CT：胆囊增大，胆囊壁增厚，胆囊内多发结石", sup)
        self.assertIn("急性胆囊炎已于第1天明确（根据血常规、腹部CT结果）。", sup)
        self.assertIn("此前已回报的检查", sup)
        self.assertNotIn("=疑似", sup)
        self.assertNotIn("（依据：", sup)
        post = rs[4][1]                                             # 第3天术后首次病程
        self.assertIn("今天实施的处置：腹腔镜胆囊切除术", post)
        self.assertIn(self.times[4], post)
        hr = rs[0][1]
        self.assertNotIn("正文围绕今天的新情况写", hr)
        self.assertIn("入院初步考虑急性胆囊炎，需要鉴别急性胰腺炎、胆总管结石。", hr)

    def test_before_surgery_same_day(self):
        slot = copy.deepcopy(self.slot)
        # 在手术日（第3天）术后首次病程之前插一篇查房
        sk = slot["skeleton"]
        i = next(i for i, x in enumerate(sk) if x["role"] == "POST")
        sk.insert(i, {"k": None, "role": "SUP", "doc_type_cn": "主治医师查房记录", "day": 3})
        for j, x in enumerate(sk):
            x["k"] = j
        times = G2.doc_times(slot)
        msgs, _, _ = G2.doc_messages2("G-E2", slot, i, [], self.dr, times)
        p = msgs[1]["content"]
        self.assertIn("今天拟行的手术（本篇写在手术之前）：腹腔镜胆囊切除术", p)
        self.assertNotIn("今天实施的处置：腹腔镜胆囊切除术", p)

    def test_coverage(self):
        sup_k = 2
        self.assertEqual(G2.coverage_missing(self.dr, self.slot, sup_k, "腹部CT示胆囊增大。血淀粉酶正常。"), [])
        self.assertEqual(G2.coverage_missing(self.dr, self.slot, sup_k, "腹部CT示胆囊增大。"), ["检查:血淀粉酶"])
        self.assertEqual(G2.coverage_missing(self.dr, self.slot, 4, "今日在全麻下行LC术。"), ["处置:腹腔镜胆囊切除术"])
        self.assertEqual(G2.coverage_missing(self.dr, self.slot, 4, "今日在全麻下行腹腔镜胆囊切除术。"), [])
        ds_k = self.slot["skeleton"][-1]["k"]
        self.assertEqual(G2.coverage_missing(self.dr, self.slot, ds_k, "行胆囊切除术。出院诊断：急性胆囊炎。"), [])
        self.assertEqual(G2.pick_attempt([("a", ["x"]), ("b", []), ("c", [])]), 1)
        self.assertEqual(G2.pick_attempt([("a", ["x", "y"]), ("b", ["x"]), ("c", ["z"])]), 1)

    def test_format_checks2(self):
        sk = self.slot["skeleton"]
        docs = []
        for spec in sk:
            text = G2.template2(self.slot, spec, self.times).replace("…", "内容")
            docs.append({"k": spec["k"], "day": spec["day"], "doc_type_cn": spec["doc_type_cn"], "text": text})
        out = G2.format_checks2(self.slot, docs, self.times)
        self.assertTrue(all(x["time_copied"] and not x["same_day_inversion"] and not x["state_leak"] for x in out))
        docs[1]["text"] = docs[1]["text"].replace(self.times[1], "09:00") + "\n今天的新情况：无"
        out = G2.format_checks2(self.slot, docs, self.times)
        self.assertFalse(out[1]["time_copied"])
        self.assertTrue(out[1]["same_day_inversion"])               # 首次病程早于入院记录
        self.assertEqual(out[1]["state_leak"], ["今天的新情况"])

    def test_plan_excerpt(self):
        t = "2025-05-25 09:05　主治医师查房记录\n患者腹痛减轻。\n\n**分析与处理意见：**\n1. 继续抗感染。\n2. 完善MRCP。"
        self.assertEqual(G2.plan_excerpt(t), "1. 继续抗感染。\n2. 完善MRCP。")
        long = "日常病程记录\n诊疗计划：" + "继续观察病情变化。" * 60
        self.assertLessEqual(len(G2.plan_excerpt(long)), G2.PLAN_MAX)
        self.assertEqual(G2.plan_excerpt("没有标题的一段话。"), "没有标题的一段话。")

    def test_seed(self):
        self.assertEqual(G2.seed_for("G2-001", "G-E2", 3, 0), G2.seed_for("G2-001", "G-E2", 3, 0))
        self.assertNotEqual(G2.seed_for("G2-001", "G-E2", 3, 0), G2.seed_for("G2-001", "G-E2", 3, 1))

    def test_prompt_lengths(self):
        for arm in G2.ARMS2:
            for spec, p, info in self._prompts(arm):
                self.assertLess(G.estimate_tokens(p), G.MAX_MODEL_LEN - G.MAX_TOKENS[spec["doc_type_cn"]] - 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
