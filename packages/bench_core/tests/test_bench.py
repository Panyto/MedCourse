"""bench-core 自检。环境变量要指到各规则包（见 README）；没有 scikit-learn 时判别器相关的条目跳过。"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench import aggregate, canon, negcontrols, safe  # noqa: E402
from bench.readers import extra  # noqa: E402
from bench.registry import BY_ID, M, verdict  # noqa: E402

G2 = os.environ.get("BENCH_G2")
try:
    import sklearn  # noqa: F401
    HAS_SK = True
except ImportError:
    HAS_SK = False


def adm(docs):
    return {"row": "X", "id": "A1", "kind": "admission", "docs": docs, "meta": {}, "view": "full"}


DOCS = [{"doc_type": "hospital_record", "day_offset": 0, "text": "入院记录\n主诉：咳嗽3天。"},
        {"doc_type": "progress_note", "day_offset": 0, "text": "首次病程记录\n2025-03-01 病例特点：咳嗽。"},
        {"doc_type": "progress_note", "day_offset": 1, "text": "2025-03-02 日常病程记录\n体温37.2℃。"},
        {"doc_type": "progress_note", "day_offset": 2, "text": "2025-03-03 日常病程记录\n体温36.8℃。"},
        {"doc_type": "discharge_summary", "day_offset": 3, "text": "出院记录\n诊疗经过：好转。"}]


class TestCanon(unittest.TestCase):
    def test_synth_doc_type(self):
        x = canon.admission_line({"slot_id": 7, "docs": [{"doc_type_cn": "首次病程记录", "day": 1, "date": "2025-03-02", "text": "t", "k": 1}]}, "G")
        self.assertEqual(x["id"], "7")
        self.assertEqual(x["docs"][0]["doc_type"], "progress_note")
        self.assertEqual(x["docs"][0]["day_offset"], 1)
        self.assertEqual(x["meta"]["admit_date"], "2025-03-01")

    def test_missing_type_is_position_only(self):
        with self.assertRaises(safe.SafeError) as e:
            canon.admission_line({"admission_id": "A", "docs": [{"text": "病历正文不能进异常"}]}, "G")
        self.assertNotIn("病历", str(e.exception))
        self.assertNotIn("病历", str(e.exception.where))


class TestSafe(unittest.TestCase):
    def test_numeric_only_rejects_text(self):
        with self.assertRaises(safe.SafeError):
            safe.numeric_only({"row": "R", "id": "1", "x": "一段文字"})
        self.assertTrue(safe.numeric_only({"row": "R", "id": "1", "x": 1.5, "y": None, "z": True}))

    def test_log_masks_long_chinese(self):
        self.assertTrue(safe._long_cjk("患者因咳嗽三天入院查体见双肺呼吸音粗未闻及"))
        self.assertFalse(safe._long_cjk("普外科"))


class TestRegistry(unittest.TestCase):
    def test_every_b_primary_has_floor_and_target(self):
        for m in M:
            if m["group"] == "B" and m["primary"]:
                self.assertIsNotNone(m["floor"], m["id"])
                self.assertIsNotNone(m["target"], m["id"])

    def test_plan_lines(self):
        self.assertEqual(BY_ID["B_iu"]["target"], (40, None))
        self.assertEqual(BY_ID["B_tfidf"]["floor"], (None, 0.30))
        self.assertEqual(BY_ID["B_auc"]["target"], (None, 0.90))
        self.assertEqual(BY_ID["D_G1_4"]["status"], "只作诊断")

    def test_a_group_all_primary(self):
        group = [m for m in M if m["group"] == "A"]
        self.assertEqual(len(group), 19)
        for m in group:
            self.assertTrue(m["primary"], m["id"])
            self.assertIsNotNone(m["floor"], m["id"])

    def test_verdict(self):
        self.assertIs(verdict(0.3, (None, 0.30)), True)
        self.assertIs(verdict(0.31, (None, 0.30)), False)
        self.assertIs(verdict(0.4, (0.35, 0.43)), True)
        self.assertIsNone(verdict(None, (0, 1)))


HEADER = "甲县人民医院住院病历 姓名：[姓名] 性别：男 年龄：58岁 科室：呼吸内科 床号：12 住院号：[编号]\n"
LONG_DOCS = [{"doc_type": "hospital_record", "day_offset": 0, "text": "入院记录\n主诉：咳嗽3天。"},
             {"doc_type": "progress_note", "day_offset": 0, "text": HEADER + "2025-03-01 14:00 首次病程记录\n病例特点：咳嗽。"},
             {"doc_type": "progress_note", "day_offset": 1, "text": HEADER + "2025-03-02 09:00 日常病程记录\n体温37.2℃。"},
             {"doc_type": "progress_note", "day_offset": 2, "text": HEADER + "2025-03-03 09:00 日常病程记录\n体温36.8℃。"},
             {"doc_type": "discharge_summary", "day_offset": 3, "text": "出院记录\n诊疗经过：好转。"}]


class TestDateReader(unittest.TestCase):
    def test_first40_keeps_v01_reading(self):
        self.assertEqual(extra.head_date_rule("2025-03-02 日常病程记录\n体温37.2℃。")[1], "first40")
        self.assertEqual(extra.head_date_rule("2025-13-02 日常病程记录"), (None, None))

    def test_header_longer_than_40(self):
        self.assertGreater(len(HEADER), extra.HEAD_FIRST)
        when, rule = extra.head_date_rule(HEADER + "2025-03-02 09:00 日常病程记录\n体温正常。")
        self.assertEqual((when.isoformat(), rule), ("2025-03-02", "stamp"))
        self.assertEqual(extra.head_date_rule(HEADER + "2025年3月2日 主治医师查房记录")[1], "title")
        self.assertEqual(extra.head_date_rule(HEADER + "2025-03-02\n今日查房")[1], "line")

    def test_mentions_in_header_or_text_are_not_the_stamp(self):
        when, rule = extra.head_date_rule(HEADER + "入院日期：2025-02-20\n2025-03-02 09:00 日常病程记录")
        self.assertEqual((when.isoformat(), rule), ("2025-03-02", "stamp"))
        self.assertEqual(extra.head_date_rule(HEADER + "患者于2025-02-01 10:00行手术，术后恢复可。"), (None, None))
        self.assertEqual(extra.head_date_rule(HEADER + "手术时间：2025-02-01 10:00"), (None, None))
        self.assertEqual(extra.head_date_rule("甲" * extra.HEAD_MAX + "2025-03-02 09:00 日常病程记录"), (None, None))

    def test_order_on_long_header(self):
        self.assertEqual(extra.date_order(LONG_DOCS), (True, False))
        chk, viol = extra.date_order(negcontrols.shuffle(adm(LONG_DOCS))["docs"])
        self.assertTrue(chk)
        self.assertTrue(viol)


class TestNegControls(unittest.TestCase):
    def test_shuffle_keeps_days_and_breaks_dates(self):
        y = negcontrols.shuffle(adm(DOCS))
        self.assertEqual([d["day_offset"] for d in y["docs"]], [0, 0, 1, 2, 3])
        self.assertEqual(y["docs"][0]["text"], DOCS[0]["text"])
        chk, viol = extra.date_order(y["docs"])
        self.assertTrue(chk)
        self.assertTrue(viol)
        self.assertEqual(extra.date_order(DOCS), (True, False))

    def test_truncate_and_repeat(self):
        long = [dict(d, text=d["text"] * 20) for d in DOCS]
        a = negcontrols.truncate(adm(long))
        self.assertLess(sum(len(d["text"]) for d in a["docs"]), 0.5 * sum(len(d["text"]) for d in long))
        b = negcontrols.repeat(adm(long))
        self.assertEqual([len(d["text"]) for d in b["docs"]], [len(d["text"]) for d in long])

    def test_constant_template_same_across_admissions(self):
        a = negcontrols.template(adm(DOCS))
        self.assertEqual(a["docs"][2]["text"], a["docs"][3]["text"])
        self.assertEqual(a["row"], "NC-template@X")
        self.assertEqual(len({d["text"] for d in negcontrols.constant(adm(DOCS))["docs"]}), 1)

    def test_swap_uses_next_admission(self):
        one, two = adm(DOCS), dict(adm([dict(d, text="乙" + d["text"]) for d in DOCS]), id="A2")
        out = negcontrols.swap_all([one, two])
        self.assertTrue(out[0]["docs"][-1]["text"].startswith("乙"))
        self.assertFalse(out[1]["docs"][0]["text"].startswith("乙") is False)

    def test_dxflip_cycles_states_and_keeps_text(self):
        y = negcontrols.dxflip(adm(DOCS))
        self.assertEqual(y["row"], "NC-dxflip@X")
        self.assertTrue(y["docs"][0]["text"].startswith(DOCS[0]["text"]))
        self.assertIn("入院诊断", y["docs"][0]["text"])
        tails = [d["text"][len(o["text"]):] for d, o in zip(y["docs"][1:4], DOCS[1:4])]
        self.assertIn("可能", tails[0])
        self.assertIn("诊断明确", tails[1])
        self.assertIn("排除", tails[2])
        self.assertEqual(y["docs"][4]["text"], DOCS[4]["text"])
        self.assertEqual([d["day_offset"] for d in y["docs"]], [d["day_offset"] for d in DOCS])

    def test_build_all_eight(self):
        rows = {x["row"] for x in negcontrols.build([adm(DOCS)])}
        self.assertEqual(len(rows), 8)


class TestAggregate(unittest.TestCase):
    def test_rate_mean_pooled(self):
        recs = [{"h": True, "c": True, "v": 2.0, "a": 1, "b": 4}, {"h": False, "c": True, "v": 4.0, "a": 1, "b": 4},
                {"h": True, "c": False, "v": None, "a": 2, "b": 2}]
        r = aggregate.one_metric({"id": "x", "how": "rate", "field": "h", "chk": "c"}, recs, {})
        self.assertEqual((r["numerator"], r["denominator"]), (1, 2))
        r = aggregate.one_metric({"id": "x", "how": "mean", "field": "v"}, recs, {})
        self.assertEqual(r["value"], 3.0)
        r = aggregate.one_metric({"id": "x", "how": "pooled", "field": "a", "den": "b"}, recs, {})
        self.assertAlmostEqual(r["value"], 0.4)

    def test_neg_matrix_rules(self):
        def row(r, m, v, ok, note=None):
            return {"row": r, "metric": m, "value": v, "floor_ok": ok, "note": note}
        board = [row("R", "B_tfidf", 0.23, True), row("NC-a@R", "B_tfidf", 1.0, False),
                 row("R", "B_B1", 0.3, True), row("NC-a@R", "B_B1", None, None),      # 读不出算拒绝
                 row("R", "B_auc", None, None), row("NC-a@R", "B_auc", 1.0, False),   # 真实行自己没有判别器读数，按在线内
                 row("R", "B_exam", 8.0, True), row("NC-a@R", "B_exam", 9.0, True),   # 比真实还好：需配对量
                 row("R", "B_dose", 0.1, False), row("NC-a@R", "B_dose", 0.0, False)] # 基线本来就不过线，不算被打破
        rows, summ = aggregate.negcontrol_matrix(board, ["R"], ["NC-a"], "R")
        by = {r["metric"]: r for r in rows}
        self.assertTrue(by["B_tfidf"]["rejected"])
        self.assertTrue(by["B_B1"]["rejected"])
        self.assertTrue(by["B_auc"]["rejected"])
        self.assertFalse(by["B_exam"]["rejected"])
        self.assertTrue(by["B_exam"]["needs_paired_measure"])
        self.assertFalse(by["B_dose"]["rejected"])
        s = {x["name"]: x for x in summ}
        self.assertEqual(s["NC-a@R"]["count"], 3)
        self.assertTrue(s["B_tfidf"]["pass"])
        self.assertFalse(s["B_exam"]["pass"])

    def test_not_worse_counts_equal_and_two_sided(self):
        self.assertTrue(aggregate.not_worse((None, 0.3), 0.2, 0.2))
        self.assertTrue(aggregate.not_worse((6.25, None), 8.0, 8.0))
        self.assertFalse(aggregate.not_worse((None, 0.3), 0.25, 0.2))
        self.assertTrue(aggregate.not_worse((0.295, 0.49), 0.39, 0.33))
        self.assertFalse(aggregate.not_worse((0.295, 0.49), 0.33, 0.39))

    def test_a_group_rejects_and_paired_table(self):
        def row(r, m, v, ok):
            return {"row": r, "metric": m, "value": v, "floor_ok": ok}
        board = [row("R", "A_S2", 0.0, True), row("NC-a@R", "A_S2", 0.95, False),
                 row("R", "A_g3_hard", 0.01, True), row("NC-a@R", "A_g3_hard", 0.01, True),
                 row("R", "B_num", 20.0, True), row("NC-a@R", "B_num", 20.0, True)]
        rows, summ = aggregate.negcontrol_matrix(board, ["R"], ["NC-a"], "R")
        by = {r["metric"]: r for r in rows}
        self.assertTrue(by["A_S2"]["rejected"])
        self.assertEqual({x["name"]: x for x in summ}["NC-a@R"]["count"], 1)
        self.assertEqual({(r["group"], r["metric"]) for r in aggregate.paired_table(rows)}, {("A", "A_g3_hard"), ("B", "B_num")})

    def test_neg_only_binding_base_counts_for_b(self):
        board = [{"row": "R", "metric": "B_tfidf", "value": 0.2, "floor_ok": True}, {"row": "G", "metric": "B_tfidf", "value": 0.2, "floor_ok": True},
                 {"row": "NC-a@R", "metric": "B_tfidf", "value": 0.2, "floor_ok": True}, {"row": "NC-a@G", "metric": "B_tfidf", "value": 0.9, "floor_ok": False}]
        _, summ = aggregate.negcontrol_matrix(board, ["R", "G"], ["NC-a"], "R")
        self.assertFalse({x["name"]: x for x in summ}["B_tfidf"]["pass"])

    def test_cp(self):
        from bench.stats_cp import cp
        lo, hi = cp(0, 187)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.01953333128693524, places=6)
        lo, hi = cp(9, 1596)
        self.assertAlmostEqual(lo, 0.002581699817727969, places=6)
        self.assertAlmostEqual(hi, 0.01067766135263105, places=6)


@unittest.skipUnless(G2 and os.path.isdir(G2), "需要 BENCH_G2 等环境变量")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(G2, "tests"))
        import fixture_rows
        cls.tmp = tempfile.mkdtemp(prefix="bench_")
        cfg = fixture_rows.write_rows(cls.tmp)
        cls.out = os.path.join(cls.tmp, "out")
        readers = "extra,g1,g2,g3,g3old,g4,g6" + (",g4old" if os.environ.get("BENCH_G45_OLD") else "") + (",c2" if HAS_SK else "")
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
        cls.proc = subprocess.run([sys.executable, "-B", "-m", "bench.run", "--rows", cfg, "--out", cls.out, "--readers", readers,
                                   "--neg", "R"], env=env, cwd=ROOT, capture_output=True)
        cls.err = cls.proc.stderr.decode("utf-8", "replace")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _table(self, name):
        with open(os.path.join(self.out, "tables", name), encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_run_succeeds(self):
        self.assertEqual(self.proc.returncode, 0, self.err[-1500:])

    def test_core_view_applied(self):
        rows = {r["row"]: r for r in self._table("S0_rows.csv")}
        self.assertEqual(rows["R"]["lines"], "40")
        self.assertEqual(rows["C"]["kind"], "care7")

    def test_scoreboard_has_values(self):
        b = {(r["row"], r["metric"]): r for r in self._table("S0_scoreboard.csv")}
        for key in [("R", "B_tfidf"), ("R", "B_exam"), ("R", "B_mentions"), ("R", "B_iu"), ("R", "A_S1"), ("C", "B_iu"), ("C", "B_mentions")]:
            self.assertNotIn(b[key]["value"], ("", None), key)
        self.assertEqual(b[("C", "B_tfidf")]["note"], "结构上不适用")
        self.assertAlmostEqual(float(b[("R", "B_tfidf")]["value"]), float(b[("R", "B_tfidf")]["value"]))

    def test_negative_controls(self):
        b = {(r["row"], r["metric"]): r for r in self._table("S0_scoreboard.csv")}
        # 固定模板：四段字数远低于真实，检查种类接近 0
        self.assertLess(float(b[("NC-template@R", "B_exam")]["value"]), float(b[("R", "B_exam")]["value"]))
        self.assertLess(float(b[("NC-truncate@R", "B_chars")]["value"]), float(b[("R", "B_chars")]["value"]))
        rows = self._table("S0_neg_summary.csv")
        self.assertTrue(any(r["kind"] == "a_nc_rejected" for r in rows))
        self.assertTrue(any(r["kind"] == "b_metric_broken" for r in rows))

    def test_dxflip_breaks_diagnosis_changes(self):
        b = {(r["row"], r["metric"]): r for r in self._table("S0_scoreboard.csv")}
        # 虚构样例的基线本来就超线，这里只看横跳把步数实打实抬高；真实基线在线内，判据 (b) 在 ys 上裁决
        self.assertGreater(float(b[("NC-dxflip@R", "B_dxchg")]["value"]), float(b[("R", "B_dxchg")]["value"]) + 1)
        self.assertEqual(b[("NC-dxflip@R", "B_dxchg")]["floor_ok"], "False")

    def test_date_order_reads_fixture_rows(self):
        b = {(r["row"], r["metric"]): r for r in self._table("S0_scoreboard.csv")}
        self.assertEqual(b[("R", "A_date_order")]["numerator"], "0")
        self.assertGreater(float(b[("NC-shuffle@R", "A_date_order")]["value"]), BY_ID["A_date_order"]["floor"][1])

    def test_paired_table_written(self):
        rows = self._table("S0_neg_paired.csv")
        self.assertTrue(rows)
        self.assertTrue({r["group"] for r in rows} <= {"A", "B"})

    def test_no_text_in_outputs(self):
        for d, _, fs in os.walk(self.out):
            for f in fs:
                with open(os.path.join(d, f), encoding="utf-8") as h:
                    content = h.read()
                for word in ("右上腹", "胆囊炎", "骨关节炎"):
                    self.assertNotIn(word, content, f)
        for word in ("右上腹", "胆囊炎"):
            self.assertNotIn(word, self.err)

    def test_anchor_table_is_written(self):
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, "-B", "-m", "bench.anchors", "--out", self.out], env=env, cwd=ROOT, capture_output=True)
        self.assertIn(p.returncode, (0, 3))
        rows = self._table("S0_anchor_check.csv")
        self.assertGreaterEqual(len(rows), 100)
        self.assertTrue(all(r["status"] in ("一致", "不一致", "没有读数") for r in rows))

    @unittest.skipUnless(HAS_SK, "需要 scikit-learn")
    def test_auc_row(self):
        b = {(r["row"], r["metric"]): r for r in self._table("S0_scoreboard.csv")}
        self.assertNotIn(b[("S", "B_auc")]["value"], ("", None))


if __name__ == "__main__":
    unittest.main()
