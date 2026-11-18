"""三个小实验、S4b 重数、MAUVE、锚点、算力探针的自检。全部虚构数据。"""
import csv
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench import anchors, copyops, date_diag, mauve, perf_probe, rvr  # noqa: E402

G2 = os.environ.get("BENCH_G2")
try:
    import sklearn  # noqa: F401
    HAS_SK = True
except ImportError:
    HAS_SK = False


def run_mod(mod, *args):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-B", "-m", mod, *args], env=env, cwd=ROOT, capture_output=True)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestCopyOps(unittest.TestCase):
    def test_classify(self):
        prev = "患者一般情况可。体温37.2℃，继续抗感染治疗。双肺呼吸音清晰对称，未闻及明显干湿啰音及胸膜摩擦音。"
        cur = "患者一般情况可。体温36.8℃，继续抗感染治疗。双肺呼吸音清晰对称，未闻及明显干湿啰音及胸膜摩擦音，律齐。今日复查胸片示炎症吸收。"
        chars, exact = copyops.classify_pair(prev, cur)
        self.assertEqual(exact, ["患者一般情况可"])
        self.assertGreater(chars["exact"], 0)
        self.assertGreater(chars["numeric_changed"], 0)
        self.assertGreater(chars["near"], 0)
        self.assertGreater(chars["new"], 0)

    def test_gap(self):
        self.assertEqual(copyops.gap_label(1, 2), "1")
        self.assertEqual(copyops.gap_label(1, 9), "3+")
        self.assertEqual(copyops.gap_label(None, 2), "空")


class TestMauve(unittest.TestCase):
    def test_frontier_ends(self):
        self.assertAlmostEqual(mauve.frontier_area([0.5, 0.5], [0.5, 0.5]), 1.0, places=6)
        self.assertAlmostEqual(mauve.frontier_area([1.0, 0.0], [0.0, 1.0]), 0.0, places=2)
        mid = mauve.frontier_area([0.7, 0.3], [0.3, 0.7])
        self.assertTrue(0.0 < mid < 1.0)

    def test_kt_smoothing_keeps_positive(self):
        h = mauve.kt_hist([0, 0, 1], 3)
        self.assertAlmostEqual(sum(h), 1.0)
        self.assertTrue(all(x > 0 for x in h))

    @unittest.skipUnless(HAS_SK, "需要 scikit-learn")
    def test_compare_orders_similarity(self):
        a = [f"患者咳嗽{i}天伴发热，予抗感染治疗好转。" for i in range(40)]
        b = [f"患者腹痛{i}小时伴呕吐，行手术治疗顺利。" for i in range(40)]
        same, _ = mauve.compare(a[:20], a[20:], per=10)
        far, _ = mauve.compare(a, b, per=10)
        self.assertGreater(same, far)


class TestAnchors(unittest.TestCase):
    def test_file_loaded_and_tolerances(self):
        rows = anchors.load_anchors()
        self.assertGreaterEqual(len(rows), 100)
        self.assertTrue(all(float(r["abs_tol"]) > 0 for r in rows))
        by = {(r["source"], r["row"], r["field"]): r for r in rows}
        self.assertAlmostEqual(float(by[("G3/T44", "G-E2", "tfidf_adj")]["value"]), 0.4631228540128505)
        self.assertAlmostEqual(float(by[("Q45b/Q4b_01", "CARe-S1", "iu_distinct")]["value"]), 25.49609375)
        self.assertEqual(by[("Q3/Q03", "R-full", "any_hard")]["numerator"], "9")

    def test_no_text_columns(self):
        for r in anchors.load_anchors():
            float(r["value"])

    def test_within(self):
        self.assertTrue(anchors.within(0.5000001, 0.5, 1e-6, 1e-6))
        self.assertFalse(anchors.within(0.51, 0.5, 1e-6, 1e-6))

    def test_q4_09_uses_historical_reader_values_unchanged(self):
        rows = anchors.load_anchors()
        self.assertEqual(len(rows), 132)
        old = [r for r in rows if r["reader"] == "g4old"]
        self.assertEqual({r["source"] for r in old}, {"Q45/Q4_09"})
        self.assertEqual(len(old), 33)
        by = {(r["row"], r["field"]): r for r in old}
        self.assertEqual(by[("R", "course_empty")]["value"], "0.3213576317193159")
        self.assertEqual(by[("G-E2", "course_empty")]["value"], "0.4393053239474205")


class TestDept(unittest.TestCase):
    def test_strip_and_major(self):
        self.assertEqual(rvr.dept_stripped("东院区心内一科二病区"), "心内科")
        self.assertEqual(rvr.dept_stripped("普外科（甲状腺）"), "普外科")
        self.assertEqual(rvr.dept_major("心血管外科一病区"), "心血管外科")
        self.assertEqual(rvr.dept_major("心血管内科"), "心内科")
        self.assertEqual(rvr.dept_major("心胸外科"), "胸外科")
        self.assertEqual(rvr.dept_major("肝胆外科"), "普外科")
        self.assertEqual(rvr.dept_major("消化内二科"), "消化内科")
        self.assertEqual(rvr.dept_major("神经外科"), "神经外科")
        self.assertEqual(rvr.dept_major("某罕见专科"), "某罕见专科")
        self.assertIsNone(rvr.dept_major(None))

    def test_groups_do_not_share_admissions(self):
        depts = {f"{d}{i}": d for d in "ABC" for i in range(60)}
        codes = {i: "J18" for i in depts}
        out = rvr.dept_pairs(depts, codes)
        seen = [i for _, _, pairs in out for pair in pairs for i in pair]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0][2]), 60)

    def test_min_pairs_not_lowered(self):
        self.assertEqual(rvr.MIN_PAIRS, 30)
        depts = {f"{d}{i}": d for d in "AB" for i in range(29)}
        self.assertEqual(rvr.dept_pairs(depts, {i: "J18" for i in depts}), [])

    def test_diag_has_counts_only(self):
        depts = {f"a{i}": "心内一科一病区" for i in range(40)}
        depts.update({f"b{i}": "心内二科" for i in range(40)})
        depts.update({f"c{i}": "呼吸内科" for i in range(40)})
        codes = {i: "J18" for i in depts}
        rows = {r["level"]: r for r in rvr.dept_diag(depts, codes)}
        self.assertEqual((rows["raw"]["distinct"], rows["major"]["distinct"]), (3, 2))
        self.assertEqual(rows["major"]["top1_pairs"], 40)
        self.assertEqual(rows["major"]["qualifying_groups"], 1)
        for r in rows.values():
            for v in r.values():
                self.assertNotIn("内科", str(v))


class TestDateDiag(unittest.TestCase):
    HEAD = "甲县人民医院住院病历 姓名：[姓名] 性别：男 年龄：58岁 科室：呼吸内科 床号：12 住院号：[编号]\n"

    def _write(self, tmp, name, adms):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            for a in adms:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        return path

    def _adm(self, i, dates):
        docs = [{"doc_type": "hospital_record", "day_offset": 0, "text": "入院记录\n主诉：咳嗽。"}]
        docs += [{"doc_type": "progress_note", "day_offset": n, "text": self.HEAD + f"{d} 09:00 日常病程记录\n体温正常。"} for n, d in enumerate(dates)]
        return {"admission_id": f"A{i}", "docs": docs}

    def test_pass_and_constant_date_stops(self):
        tmp = tempfile.mkdtemp(prefix="date_")
        try:
            good = self._write(tmp, "good.jsonl", [self._adm(i, ["2025-03-01", "2025-03-02", "2025-03-03"]) for i in range(3)])
            bad = self._write(tmp, "bad.jsonl", [self._adm(i, ["2025-03-01", "2025-03-01", "2025-03-01"]) for i in range(3)])
            p = run_mod("bench.date_diag", "--row", f"R={good}", "--real", "R", "--out", os.path.join(tmp, "o1"))
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            r = read_csv(os.path.join(tmp, "o1", "S0_date_diag.csv"))[0]
            self.assertEqual((r["notes"], r["read"], r["adm_ge2"], r["adm_viol"], r["pairs"], r["pairs_agree"]), ("9", "9", "3", "0", "6", "6"))
            self.assertEqual(r["tautology_flag"], "True")
            self.assertEqual(r["rule_stamp"], "9")
            self.assertEqual(r["pos_40_59"], "9")
            p = run_mod("bench.date_diag", "--row", f"R={bad}", "--real", "R", "--out", os.path.join(tmp, "o2"))
            self.assertEqual(p.returncode, 3)
            r = read_csv(os.path.join(tmp, "o2", "S0_date_diag.csv"))[0]
            self.assertEqual((r["d1"], r["d2"]), ("True", "False"))
            p = run_mod("bench.date_diag", "--row", f"S={bad}", "--out", os.path.join(tmp, "o3"))
            self.assertEqual(p.returncode, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_output_has_no_dates(self):
        c = date_diag.judge(date_diag.tally([{"docs": [{"doc_type": "progress_note", "day_offset": 0, "text": "2025-03-01 09:00 日常病程记录"}]}]), "R", set())
        self.assertNotIn("2025", json.dumps(c, default=str))


class TestExtractSchema(unittest.TestCase):
    def _load(self, name):
        with open(os.path.join(ROOT, "data", name), encoding="utf-8") as f:
            return json.load(f)

    def test_no_max_length_and_bound_only_on_events(self):
        free, bounded = self._load("extract_schema.json"), self._load("extract_schema_bounded.json")
        self.assertNotIn("maxLength", json.dumps(free))
        self.assertNotIn("maxLength", json.dumps(bounded))
        self.assertNotIn("maxItems", json.dumps(free))
        self.assertEqual(bounded["properties"]["events"].pop("maxItems"), 20)
        self.assertEqual(free, bounded)


class Fake(http.server.BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        Fake.seen.append((self.headers.get("Authorization"), body))
        content = '{"events": [1, 2]}' if "response_format" in body else "正文"
        out = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 7}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class TestPerfProbe(unittest.TestCase):
    def test_prompts(self):
        self.assertIn("改写", perf_probe.prompt(100, 1))
        text = perf_probe.prompt(600, 3, "extract")
        self.assertIn("抽取", text)
        self.assertNotIn("改写", text)
        self.assertEqual(text, perf_probe.prompt(600, 3, "extract"))
        self.assertGreater(len(set(text.split("。"))), 5)

    def test_against_fake_server(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        tmp = tempfile.mkdtemp(prefix="probe_")
        try:
            schema = os.path.join(tmp, "s.json")
            with open(schema, "w", encoding="utf-8") as f:
                json.dump({"type": "object"}, f)
            out = os.path.join(tmp, "p.csv")
            url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
            env = dict(os.environ, VLLM_API_KEY="abc-not-printed")
            p = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "bench", "perf_probe.py"), "--base-url", url, "--model", "M", "--concurrency", "1,4",
                                "--requests", "8", "--input-chars", "100", "--max-tokens", "16", "--out", out, "--label", "x"], env=env, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            self.assertNotIn(b"abc-not-printed", p.stdout + p.stderr)
            p = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "bench", "perf_probe.py"), "--base-url", url, "--model", "M", "--concurrency", "2",
                                "--requests", "4", "--input-chars", "100", "--json-schema", schema, "--out", out], env=env, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            p = subprocess.run([sys.executable, "-B", os.path.join(ROOT, "bench", "perf_probe.py"), "--base-url", url, "--model", "M", "--concurrency", "1",
                                "--requests", "2", "--input-chars", "100", "--max-tokens", "16", "--raw", "--out", os.path.join(tmp, "raw.csv")], env=env, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            self.assertTrue(any("prompt" in b and "messages" not in b for _, b in Fake.seen))
            rows = read_csv(out)
            self.assertEqual([r["concurrency"] for r in rows], ["1", "4", "2"])
            self.assertEqual(rows[0]["ok"], "8")
            self.assertEqual(rows[2]["json_ok"], "4")
            self.assertEqual((rows[0]["task"], rows[0]["max_tokens"]), ("rewrite", "16"))
            self.assertEqual((rows[2]["task"], rows[2]["max_tokens"]), ("extract", "2048"))
            self.assertEqual((rows[2]["finish_stop"], rows[2]["finish_length"], rows[2]["mean_events"]), ("4", "0", "2.0"))
            self.assertGreater(float(rows[2]["json_ok_per_s"]), 0)
            self.assertTrue(any(b.get("max_tokens") == 2048 and "抽取" in b["messages"][0]["content"] for _, b in Fake.seen if "messages" in b))
            self.assertTrue(all(a == "Bearer abc-not-printed" for a, _ in Fake.seen))
            self.assertTrue(any("response_format" in b for _, b in Fake.seen))
        finally:
            srv.shutdown()
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(G2 and os.path.isdir(G2), "需要 BENCH_G2 等环境变量")
class TestScripts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(G2, "tests"))
        import fixture_rows
        cls.tmp = tempfile.mkdtemp(prefix="tools_")
        fixture_rows.write_rows(cls.tmp)
        cls.r = os.path.join(cls.tmp, "r.jsonl")
        cls.s = os.path.join(cls.tmp, "s.jsonl")
        cls.codes = os.path.join(cls.tmp, "codes.csv")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _ok(self, p):
        self.assertEqual(p.returncode, 0, p.stderr.decode("utf-8", "replace")[-800:])

    def test_copyops(self):
        out = os.path.join(self.tmp, "c")
        self._ok(run_mod("bench.copyops", "--row", f"R={self.r}=core", "--row", f"S={self.s}", "--out", out))
        rows = read_csv(os.path.join(out, "S0_copyops.csv"))
        self.assertEqual({r["row"] for r in rows}, {"R", "S"})
        for r in rows:
            total = sum(float(r[f"{o}_share"]) for o in copyops.OPS)
            self.assertAlmostEqual(total, 1.0, places=6)
        self.assertTrue({"全部", "文书细类", "天数间隔", "手术"} <= {r["group"] for r in rows})

    def test_profile(self):
        out = os.path.join(self.tmp, "p")
        self._ok(run_mod("bench.profile_rows", "--path", self.r, "--out", out))
        rows = read_csv(os.path.join(out, "S0_profile.csv"))
        by = {(r["table"], r["key"]): r for r in rows}
        self.assertEqual(by[("counts", "admissions")]["n"], "40")
        self.assertIn(("n_docs", "每住院文书数"), by)

    def test_s4b_recount(self):
        out = os.path.join(self.tmp, "s4b")
        self._ok(run_mod("bench.s4b_recount", "--path", self.r, "--out", out))
        rows = read_csv(os.path.join(out, "S0_s4b_recount.csv"))
        self.assertEqual({r["class"] for r in rows}, {"手术日", "别的操作", "认不出"})

    def test_run_with_relative_out_dir(self):
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, "-B", "-m", "bench.run", "--rows", "rows.json", "--out", "rel_out", "--readers", "extra"],
                           env=env, cwd=self.tmp, capture_output=True)
        self._ok(p)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "rel_out", "raw", "all.extra.jsonl")))

    def test_tfidf_reference_fits_only(self):
        with open(os.path.join(self.tmp, "rows.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        base = {"reference": "R", "rows": [cfg["rows"][0]]}
        ids = os.path.join(self.tmp, "ref_ids.txt")
        with open(ids, "w", encoding="utf-8") as f:
            f.write("".join(f"F{i:03d}\n" for i in range(40)))
        same = dict(base, tfidf_reference={"path": self.r, "kind": "admission", "view": "core", "filter_ids": ids, "n_expected": 40})
        with open(ids.replace(".txt", "3.txt"), "w", encoding="utf-8") as f:
            f.write("F000\n")
        other = dict(base, tfidf_reference={"path": self.r, "kind": "admission", "view": "core", "filter_ids": ids.replace(".txt", "3.txt"), "n_expected": 1})

        def value(name, c):
            path, out = os.path.join(self.tmp, name + ".json"), os.path.join(self.tmp, "t_" + name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(c, f, ensure_ascii=False)
            self._ok(run_mod("bench.run", "--rows", path, "--out", out, "--readers", "g2"))
            rows = {r["row"]: r for r in read_csv(os.path.join(out, "tables", "S0_scoreboard.csv")) if r["metric"] == "B_tfidf"}
            self.assertNotIn("__tfidf_ref__", rows)
            seen = {json.loads(x)["row"] for x in open(os.path.join(out, "raw", "all.g2.jsonl"), encoding="utf-8")}
            self.assertEqual(seen, {"R"})
            return float(rows["R"]["value"])

        plain = value("plain", base)
        self.assertAlmostEqual(value("same", same), plain, places=12)
        self.assertNotAlmostEqual(value("other", other), plain, places=6)

    def test_date_diag_on_fixture(self):
        out = os.path.join(self.tmp, "dd")
        self._ok(run_mod("bench.date_diag", "--row", f"R={self.r}", "--real", "R", "--out", out))
        r = read_csv(os.path.join(out, "S0_date_diag.csv"))[0]
        self.assertEqual(r["admissions"], "40")
        self.assertEqual(r["adm_viol"], "0")

    def test_sealcheck(self):
        path = os.path.join(self.tmp, "seal.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(6):
                f.write(json.dumps({"admission_id": f"A{i}", "split_fold": i % 2, "docs": [{"doc_type": "hospital_record", "text": "x"}]}) + "\n")
        out = os.path.join(self.tmp, "seal")
        self._ok(run_mod("bench.sealcheck", "--path", path, "--out", out))
        rows = read_csv(os.path.join(out, "S0_folds.csv"))
        self.assertEqual([r["admissions"] for r in rows], ["3", "3"])
        self.assertEqual(rows[0]["role"], "开发")
        self.assertEqual(len(rows[0]["ids_sha256"]), 64)

    @unittest.skipUnless(HAS_SK, "需要 scikit-learn")
    def test_mauve_cli(self):
        out = os.path.join(self.tmp, "m")
        self._ok(run_mod("bench.mauve", "--rows", os.path.join(self.tmp, "rows.json"), "--out", out))
        rows = [json.loads(x) for x in open(os.path.join(out, "raw", "all.mauve.row.jsonl"), encoding="utf-8")]
        self.assertEqual({r["row"] for r in rows}, {"S", "C"})
        self.assertTrue(all(0.0 <= r["value"] <= 1.0 for r in rows))

    @unittest.skipUnless(HAS_SK, "需要 scikit-learn")
    def test_rvr(self):
        out = os.path.join(self.tmp, "rvr")
        self._ok(run_mod("bench.rvr", "--path", self.r, "--dx-map", self.codes, "--out", out, "--reps", "2"))
        rows = read_csv(os.path.join(out, "S0_rvr.csv"))
        names = {r["experiment"] for r in rows}
        self.assertTrue({"null_7field", "null_admission", "null_hr", "null_fp", "null_course", "null_ds"} <= names)
        self.assertEqual({r["level"] for r in read_csv(os.path.join(out, "S0_rvr_dept_diag.csv"))}, {"raw", "stripped", "major"})

    @unittest.skipUnless(HAS_SK, "需要 scikit-learn")
    def test_rvr_dept_only_leaves_null_table_alone(self):
        out = os.path.join(self.tmp, "rvr_dept")
        self._ok(run_mod("bench.rvr", "--path", self.r, "--dx-map", self.codes, "--out", out, "--dept-only", "--dept-map", os.path.join(self.tmp, "map.csv")))
        self.assertFalse(os.path.exists(os.path.join(out, "S0_rvr.csv")))
        self.assertTrue(os.path.exists(os.path.join(out, "S0_rvr_dept.csv")))
        self.assertTrue(all(r["experiment"].startswith("dept_") for r in read_csv(os.path.join(out, "S0_rvr_dept.csv"))))


if __name__ == "__main__":
    unittest.main()
