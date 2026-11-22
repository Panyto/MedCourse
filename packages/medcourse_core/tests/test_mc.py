"""medcourse-core 自检。环境变量要指到规则包（同 bench_core）。全部虚构数据。"""
import collections
import copy
import csv
import hashlib
import http.server
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fixtures_mc  # noqa: E402
from medcourse import compiler, contract, gate, icd, kb, llm, out_diag, pipeline, planner, stats, synth_export, writer  # noqa: E402
from medcourse.extract import run as xrun  # noqa: E402
from medcourse.extract import schema, segments, verify  # noqa: E402


def env():
    return dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")


def run_mod(mod, *args):
    return subprocess.run([sys.executable, "-B", "-m", mod, *args], env=env(), cwd=ROOT, capture_output=True)


def make(jid="J1", seed=1, slot=None, st=None):
    st = st or fixtures_mc.stats()
    rng = random.Random(seed)
    j = planner.plan_journey(st, rng, jid, slot)
    return j, planner.schedule(j, st, rng), st, rng


class TestContract(unittest.TestCase):
    def test_journey_errors_are_codes(self):
        j, specs, _, _ = make()
        self.assertEqual(contract.validate_journey(j), [])
        self.assertEqual(contract.validate_docspecs(specs, j["los"]), [])
        bad = copy.deepcopy(j)
        bad["exams"][0]["order_day"], bad["exams"][0]["result_day"] = 3, 1
        self.assertIn("exam_days", contract.validate_journey(bad))
        bad = copy.deepcopy(j)
        bad["patient"]["sex"] = "未知"
        self.assertIn("patient_sex", contract.validate_journey(bad))

    def test_surgery_flag_must_match(self):
        j, _, _, _ = make()
        bad = copy.deepcopy(j)
        bad["surgical"] = not bad["surgical"]
        self.assertIn("surgical_flag", contract.validate_journey(bad))

    def test_event_contract(self):
        ok = {"event_type": "lab", "name": "白细胞", "ev_start": 3, "ev_end": 9, "assertion": "done"}
        self.assertEqual(contract.validate_event(ok), [])
        self.assertIn("event_type", contract.validate_event(dict(ok, event_type="x")))
        self.assertIn("event_span", contract.validate_event(dict(ok, ev_end=3)))


class TestSegments(unittest.TestCase):
    def test_sections_and_header_skipped(self):
        doc = {"doc_type": "hospital_record", "text": fixtures_mc.HR}
        segs = segments.segments(doc)
        names = [s["section"] for s in segs]
        self.assertIn("现病史", names)
        self.assertNotIn("抬头", names)
        for s in segs:
            self.assertEqual(doc["text"][s["start"]:s["end"]], s["text"])

    def test_note_without_titles_drops_header_only(self):
        doc = {"doc_type": "progress_note", "text": fixtures_mc.NOTE}
        segs = segments.segments(doc)
        self.assertEqual(segs[0]["section"], "正文")
        self.assertIn("今日主治医师查房", segs[0]["text"])
        self.assertNotIn("住院号", segs[0]["text"])

    def test_long_section_is_cut_on_sentences(self):
        body = "现病史：" + "患者咳嗽三天伴有发热症状。" * 120
        segs = segments.segments({"doc_type": "hospital_record", "text": body})
        self.assertGreater(len(segs), 1)
        self.assertTrue(all(len(s["text"]) <= segments.MAX_SEG for s in segs))
        self.assertEqual("".join(s["text"] for s in segs), body[len("现病史："):])

    def test_sentence_longer_than_limit_is_hard_cut(self):
        body = "现病史：" + "甲" * 1500
        segs = segments.segments({"doc_type": "hospital_record", "text": body})
        self.assertTrue(all(len(s["text"]) <= segments.MAX_SEG for s in segs))
        self.assertEqual("".join(s["text"] for s in segs), "甲" * 1500)

    def test_halve_keeps_positions(self):
        seg = {"section": "现病史", "start": 100, "end": 100 + 390, "text": "患者咳嗽三天伴有发热症状。" * 30}
        a, b = segments.halve(seg)
        self.assertEqual(a["text"] + b["text"], seg["text"])
        self.assertEqual((a["start"], a["end"], b["start"], b["end"]), (100, 100 + len(a["text"]), 100 + len(a["text"]), seg["end"]))
        self.assertIsNone(segments.halve({"section": "x", "start": 0, "end": 5, "text": "短句。"}))


class TestVerify(unittest.TestCase):
    def test_substring_required_ignoring_whitespace(self):
        seg = {"section": "x", "start": 10, "text": "血常规示 白细胞 12.1×10^9/L，血钾4.1mmol/L。"}
        tally = {}
        ok = verify.ground({"event_type": "lab", "name": "白细胞", "value": "12.1", "unit": "×10^9/L", "evidence": "白细胞12.1×10^9/L"}, seg, tally)
        self.assertEqual(seg["text"][ok["ev_start"] - 10:ok["ev_end"] - 10].replace(" ", ""), "白细胞12.1×10^9/L")
        # v0.3：引文对不上、名称在原文里的，按名称找回（ground=name），数值不在找回的引文里就清掉
        fb = verify.ground({"event_type": "lab", "name": "白细胞", "value": "13", "unit": None, "evidence": "白细胞13"}, seg, tally)
        self.assertEqual((fb["ground"], fb["value"]), ("name", None))
        self.assertEqual(tally["grounded_by_name"], 1)
        self.assertIsNone(verify.ground({"event_type": "lab", "name": "红细胞", "value": None, "unit": None, "evidence": "红细胞4.5"}, seg, tally))
        self.assertEqual((tally["dropped_not_substring"], tally["dropped_type:lab"]), (1, 1))

    def test_list_item_name_fallback(self):
        # S2G3：模型给列表后面几项拼出"否认糖尿病"当引文；按名称找回，引文从分句开头到名称为止，带上前面的否定词
        seg = {"section": "既往史", "start": 0, "text": "否认高血压、糖尿病、冠心病病史，无手术史。"}
        tally = {}
        e = verify.ground({"event_type": "diagnosis", "name": "糖尿病", "value": None, "unit": None, "evidence": "否认糖尿病"}, seg, tally)
        self.assertEqual((e["ground"], e["evidence"]), ("name", "否认高血压、糖尿病"))
        self.assertEqual(seg["text"][e["ev_start"]:e["ev_end"]], "否认高血压、糖尿病")
        e2 = verify.ground({"event_type": "diagnosis", "name": "高血压", "value": None, "unit": None, "evidence": "否认高血压"}, seg, tally)
        self.assertNotIn("ground", e2)                                                      # 本来就是子串的照旧
        from medcourse.extract import normalize as N
        r = N.normalize_event(dict(e, assertion=None), seg["text"], e["ev_start"], e["ev_end"], "hospital_record", "既往史")
        self.assertEqual((r["assertion"], r["norm"][:1]), ("negated", ["fill_missing"]))

    def test_value_not_in_evidence_is_cleared(self):
        seg = {"section": "x", "start": 0, "text": "血钾4.1mmol/L。"}
        tally = {}
        e = verify.ground({"event_type": "lab", "name": "血钾", "value": "5.0", "unit": "mmol/L", "evidence": "血钾4.1mmol/L"}, seg, tally)
        self.assertIsNone(e["value"])
        self.assertEqual(e["unit"], "mmol/L")
        self.assertEqual(tally["value_cleared"], 1)


class TestSchema(unittest.TestCase):
    def test_no_max_length(self):
        for s in (schema.PASS1_SCHEMA, schema.PASS2_SCHEMA):
            self.assertNotIn("maxLength", json.dumps(s))
        self.assertEqual(schema.PASS1_SCHEMA["properties"]["events"]["maxItems"], schema.CAP)

    def test_reason_comes_before_assertion(self):
        keys = list(schema.PASS2_SCHEMA["properties"]["items"]["items"]["properties"])
        self.assertLess(keys.index("reason"), keys.index("assertion"))


class Fake(http.server.BaseHTTPRequestHandler):
    mode = "ok"
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        Fake.seen.append(body)
        props = (body.get("response_format") or {}).get("json_schema", {}).get("schema", {}).get("properties", {})
        user = body["messages"][-1]["content"] if "messages" in body else body.get("prompt", "")
        finish = "stop"
        if "events" in props:
            text = user.split("【文字】\n")[1]
            if Fake.mode == "cap" and len(text) > 150:
                content, finish = '{"events": [', "length"
            else:
                ev = text[:8]
                content = json.dumps({"events": [{"event_type": "symptom", "name": "咳嗽", "value": None, "unit": None, "evidence": ev},
                                                 {"event_type": "lab", "name": "假", "value": "9", "unit": None, "evidence": "原文里没有这句"}]}, ensure_ascii=False)
        elif "items" in props:
            content = json.dumps({"items": [{"id": 0, "reason": "写明已有", "assertion": "done", "time_text": None}]}, ensure_ascii=False)
        else:
            content = "患者一般情况可，继续当前治疗，密切观察病情变化。"
        msg = {"message": {"content": content}, "text": content, "finish_reason": finish}
        out = json.dumps({"choices": [msg], "usage": {"prompt_tokens": 10, "completion_tokens": 7}}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class TestLLM(unittest.TestCase):
    def test_only_loopback(self):
        with self.assertRaises(llm.LLMError):
            llm.Client("http://10.0.0.1:8102/v1", "M")
        with self.assertRaises(llm.LLMError):
            llm.Client("https://example.com/v1", "M")
        llm.Client("http://127.0.0.1:8102/v1", "M")

    def test_sampling_always_explicit(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
            Fake.mode, Fake.seen = "ok", []
            llm.Client(url, "M").generate(prompt="你好", max_tokens=16, sampling=llm.WRITE, seed=7)
            llm.Client(url, "M", raw=True).generate(prompt="你好", max_tokens=16, sampling=llm.EXTRACT, seed=8)
            chat, raw = Fake.seen
            for body, want in ((chat, llm.WRITE), (raw, llm.EXTRACT)):
                for k, v in {**want, **llm.FIXED, "max_tokens": 16}.items():
                    self.assertEqual(body[k], v, k)
            self.assertEqual((chat["seed"], raw["seed"]), (7, 8))
            self.assertEqual(chat["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("chat_template_kwargs", raw)
            with self.assertRaises(TypeError):
                llm.Client(url, "M").generate(prompt="你好", seed=1)
            with self.assertRaises(llm.LLMError):
                llm.Client(url, "M").generate(prompt="你好", sampling=llm.WRITE, seed=None)
            with self.assertRaises(llm.LLMError):
                llm.Client(url, "M").generate(prompt="你好", sampling={"temperature": 0.7}, seed=1)
            self.assertEqual(len(Fake.seen), 2)
        finally:
            srv.shutdown()


class TestExtractRun(unittest.TestCase):
    def setUp(self):
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        self.tmp = tempfile.mkdtemp(prefix="x_")
        self.path = os.path.join(self.tmp, "a.jsonl")
        with open(self.path, "w", encoding="utf-8") as f:
            for i in range(2):
                f.write(json.dumps(fixtures_mc.admission(i), ensure_ascii=False) + "\n")
        Fake.mode, Fake.seen = "ok", []

    def tearDown(self):
        self.srv.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, out, *extra):
        return run_mod("medcourse.extract.run", "--path", self.path, "--base-url", self.url, "--model", "M", "--out", out, "--concurrency", "2", *extra)

    def test_real_output_must_stay_in_derived_root(self):
        p = self._run(os.path.join(self.tmp, "out"))
        self.assertEqual(p.returncode, 2)
        self.assertIn(b"out_not_in_derived_root", p.stderr)

    def test_two_passes_grounded_and_resumable(self):
        out = os.path.join(self.tmp, "out")
        p = self._run(out, "--synthetic")
        self.assertEqual(p.returncode, 0, p.stderr[-400:])
        counts = json.load(open(os.path.join(out, "counts.json"), encoding="utf-8"))
        self.assertGreater(counts["segments"], 4)
        self.assertEqual(counts["dropped_not_substring"], counts["segments"])
        rows = [json.loads(x) for x in open(os.path.join(out, "events.jsonl"), encoding="utf-8")]
        self.assertEqual(len(rows), counts["events_kept"])
        self.assertTrue(all(r["assertion"] == "done" for r in rows))
        doc = fixtures_mc.admission()["docs"]
        for r in rows[:5]:
            self.assertEqual("".join(doc[r["doc_idx"]]["text"][r["ev_start"]:r["ev_end"]].split()), "".join(r["evidence"].split()))
        n = len(Fake.seen)
        self.assertTrue(all(b["temperature"] == 0.0 and b["top_p"] == 1.0 and b["top_k"] == -1 and "seed" in b for b in Fake.seen))
        self.assertEqual(self._run(out, "--synthetic").returncode, 0)
        self.assertEqual(len(Fake.seen), n)
        self.assertEqual(json.load(open(os.path.join(out, "counts.json"), encoding="utf-8"))["skipped_done"], counts["segments"])

    def test_truncated_segment_is_halved(self):
        adm = {"admission_id": "L1", "docs": [{"doc_type": "hospital_record", "day_offset": 0, "text": "现病史：" + "患者咳嗽三天伴有发热症状。" * 40}]}
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps(adm, ensure_ascii=False) + "\n")
        Fake.mode = "cap"
        out = os.path.join(self.tmp, "cap")
        self.assertEqual(self._run(out, "--synthetic", "--passes", "1").returncode, 0)
        c = json.load(open(os.path.join(out, "counts.json"), encoding="utf-8"))
        self.assertGreaterEqual(c["seg_split"], 1)
        self.assertGreaterEqual(c["finish_length"], 1)
        self.assertGreater(c["events_kept"], 0)

    def test_no_text_in_counts_or_stderr(self):
        out = os.path.join(self.tmp, "t")
        p = self._run(out, "--synthetic")
        self.assertNotIn("咳嗽".encode("utf-8"), p.stderr)
        self.assertNotIn("咳嗽", open(os.path.join(out, "counts.json"), encoding="utf-8").read())


class TestSegDiag(unittest.TestCase):
    def test_counts_only(self):
        tmp = tempfile.mkdtemp(prefix="d_")
        try:
            path = os.path.join(tmp, "a.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(fixtures_mc.admission(), ensure_ascii=False) + "\n")
            p = run_mod("medcourse.extract.seg_diag", "--path", path, "--out", tmp)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            rows = list(csv.DictReader(open(os.path.join(tmp, "S2_seg_diag.csv"), encoding="utf-8")))
            self.assertEqual({r["doc_type"] for r in rows}, {"hospital_record", "progress_note", "discharge_summary"})
            self.assertNotIn("咳嗽", open(os.path.join(tmp, "S2_seg_diag.csv"), encoding="utf-8").read())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestStats(unittest.TestCase):
    def test_k_threshold_suppresses(self):
        lines = [fixtures_mc.admission(i) for i in range(12)]
        s = stats.build_stats(lines, {f"F{i:03d}": "J18" for i in range(12)}, k=11)
        self.assertEqual(s["n_admissions"], 12)
        self.assertIn("男|40-65|J", s["joint"]["items"])
        s2 = stats.build_stats(lines[:5], {f"F{i:03d}": "J18" for i in range(5)}, k=11)
        self.assertEqual(s2["joint"]["items"], {})
        self.assertGreater(s2["suppressed"], 0)

    def test_labs_and_meds_read(self):
        lines = [fixtures_mc.admission(i) for i in range(12)]
        s = stats.build_stats(lines, {f"F{i:03d}": "J18" for i in range(12)}, k=11)
        self.assertEqual(s["labs"]["WBC"]["q"][2], 12.1)
        self.assertEqual(s["labs"]["K"]["q"][2], 4.1)
        self.assertIn("ceftriaxone", s["meds"]["J"]["items"])
        self.assertIn("高血压", s["past"]["40-65"]["items"])
        self.assertNotIn("糖尿病", s["past"]["40-65"]["items"])

    def test_v02_tables(self):
        lines = [fixtures_mc.admission(i) for i in range(12)]
        s = stats.build_stats(lines, {f"F{i:03d}": "J18" for i in range(12)}, k=11)
        self.assertEqual(s["version"], stats.STATS_VERSION)
        self.assertEqual(s["med_alias"]["ceftriaxone"]["items"], {"头孢曲松": 12})
        self.assertEqual(s["med_dose"]["ceftriaxone"]["items"], {"2g": 12})
        self.assertEqual(s["med_freq"]["ceftriaxone"]["items"], {"qd": 12})
        self.assertEqual(s["med_route"]["ceftriaxone"]["items"], {"静滴": 12})
        self.assertIn("社区获得性肺炎", s["dx_names"]["J18"]["items"])
        self.assertIn("社区获得性肺炎", s["dx_names_ch"]["J"]["items"])
        for t in ("exams", "meds", "symptoms", "dept"):
            self.assertIn("n:社区获得性肺炎", s[t], t)
            self.assertIn("J18", s[t], t)
        rows = {r["table"]: r for r in stats.summary_rows(s)}
        self.assertEqual(rows["joint"]["cells"], len(s["joint"]["items"]))
        self.assertGreater(rows["labs"]["cells"], 0)

    def test_hgb_reading_skips_mch_mchc(self):
        self.assertEqual(stats.LAB_RE["HGB"].search("血红蛋白112g/L").group(1), "112")
        self.assertEqual(stats.LAB_RE["HGB"].search("Hb 98 g/L").group(1), "98")
        for t in ("平均红细胞血红蛋白浓度330g/L", "红细胞平均血红蛋白浓度 340g/L", "平均血红蛋白浓度335g/L"):
            self.assertIsNone(stats.LAB_RE["HGB"].search(t), t)

    def test_implausible_lab_readings_dropped(self):
        lines = []
        for i in range(12):
            a = fixtures_mc.admission(i)
            a["docs"][0]["text"] += "复查：血红蛋白300g/L，血红蛋白120g/L。\n"
            lines.append(a)
        s = stats.build_stats(lines, {f"F{i:03d}": "J18" for i in range(12)}, k=11)
        self.assertEqual(s["labs"]["HGB"]["q"], [120.0] * 5)
        self.assertEqual(s["labs_dropped"], {"HGB": 12})
        self.assertEqual(s["surgery_meta"], {"surgical_admissions": 0, "proc_key_found": 0, "proc_family_found": 0})
        rows = {r["table"]: r["cells"] for r in stats.summary_rows(s)}
        self.assertEqual(rows["labs_dropped:HGB"], 12)

    def test_proc_key(self):
        self.assertEqual(stats.proc_key("在全麻下行腹腔镜下右侧腹股沟疝修补术（TAPP）"), "腹腔镜{S侧}腹股沟疝修补术")
        self.assertEqual(stats.proc_key("经腹腔镜胆囊切除术"), stats.proc_key("腹腔镜下胆囊切除术"))
        self.assertEqual(stats.proc_key("腹腔镜胆囊切除术"), "腹腔镜胆囊切除术")
        self.assertEqual(stats.proc_key("左肾切除术"), "{S}肾切除术")
        self.assertIsNone(stats.proc_key("行 术"))

    def test_lab_regex(self):
        self.assertEqual(stats.LAB_RE["WBC"].search("白细胞12.1×10^9/L").group(1), "12.1")
        self.assertEqual(stats.LAB_RE["WBC"].search("WBC 9.8 10^9/L").group(1), "9.8")
        self.assertIsNone(stats.LAB_RE["K"].search("血钾正常"))

    def test_copy_table(self):
        tmp = tempfile.mkdtemp(prefix="c_")
        try:
            p = os.path.join(tmp, "c.csv")
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["row", "group", "value", "pairs", "exact_share", "numeric_changed_share", "near_share"])
                w.writeheader()
                w.writerow({"row": "R", "group": "文书细类", "value": "术后首次病程", "pairs": 500, "exact_share": 0.07, "numeric_changed_share": 0.0, "near_share": 0.05})
                w.writerow({"row": "G-E2", "group": "全部", "value": "全部", "pairs": 500, "exact_share": 0.9, "numeric_changed_share": 0.0, "near_share": 0.0})
            t = stats.read_copy_table(p)
            self.assertAlmostEqual(t["文书细类"]["术后首次病程"]["reuse"], 0.12)
            self.assertNotIn("全部", t)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPlanner(unittest.TestCase):
    def test_deterministic_by_seed(self):
        a, sa, _, _ = make("J1", 7)
        b, sb, _, _ = make("J1", 7)
        self.assertEqual(a, b)
        self.assertEqual(sa, sb)

    def test_many_journeys_are_valid(self):
        for i in range(60):
            j, specs, _, _ = make(f"J{i}", i)
            self.assertEqual(planner.check_plan(j, specs), [], i)

    def test_no_impossible_patient(self):
        st = fixtures_mc.stats()
        st["joint"]["items"] = {"男|40-65|O": 50}
        st["code3"]["O"] = {"n": 50, "items": {"O80": 50}}
        st["dx_names"]["O80"] = {"n": 50, "items": {"单胎顺产": 50}}
        self.assertIsNone(planner.plan_journey(st, random.Random(1), "J", None))

    def test_meds_pass_rules(self):
        self.assertFalse(planner.med_allowed({"age": 8, "sex": "男", "pregnant": False, "primary_dx": {"name": "肺炎"}}, "左氧氟沙星", "0.5g", "qd", "静滴"))
        self.assertFalse(planner.med_allowed({"age": 30, "sex": "男", "pregnant": False, "primary_dx": {"name": "肺炎"}}, "缩宫素", "10U", "qd", "静滴"))
        self.assertTrue(planner.med_allowed({"age": 30, "sex": "女", "pregnant": False, "primary_dx": {"name": "肺炎"}}, "缩宫素", "10U", "qd", "静滴"))
        self.assertFalse(planner.med_allowed({"age": 30, "sex": "女", "pregnant": True, "primary_dx": {"name": "妊娠合并高血压"}}, "卡托普利", "25mg", "tid", "口服"))
        self.assertTrue(planner.med_allowed({"age": 40, "sex": "男", "pregnant": False, "primary_dx": {"name": "肺炎"}}, "左氧氟沙星", "0.5g", "qd", "静滴"))
        for i in range(60):
            j, _, _, _ = make(f"J{i}", i)
            if j["patient"]["age"] < 18:
                self.assertNotIn("levofloxacin", [m["drug_id"] for m in j["meds"]])

    def test_paired_mode_uses_slot_not_skeleton(self):
        slot = {"slot_id": "S9", "sex": "女", "age": 66, "dept": "普外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80", "los": 6, "surgical": True, "surgery_day": 2,
                "skeleton": [{"k": 0, "role": "HR", "doc_type_cn": "入院记录", "day": 0}]}
        j, specs, _, _ = make("S9", 3, slot)
        self.assertEqual((j["patient"]["sex"], j["patient"]["age"], j["los"], j["surgery_day"]), ("女", 66, 6, 2))
        self.assertEqual([p["kind"] for p in j["procedures"]], ["手术"])
        self.assertIn("POST", [s["role"] for s in specs])
        post = next(s for s in specs if s["role"] == "POST")
        self.assertEqual(post["day"], 2)
        self.assertGreater(len(specs), 1)

    def test_times_rise_within_day(self):
        j, specs, _, _ = make("J5", 5)
        last = {}
        for s in specs:
            self.assertGreaterEqual(s["time"], last.get(s["day"], "00:00"))
            last[s["day"]] = s["time"]


class TestPlannerV012(unittest.TestCase):
    """监督员第二轮审查的几条，一条一测。"""

    def test_table_skips_empty_stratum(self):
        st = fixtures_mc.stats()
        self.assertEqual(planner._table(st, "meds", ["n:无此病", "K40", "K", "*"])["items"], {"omeprazole": 150})
        self.assertIn("ceftriaxone", planner._table(st, "meds", planner.strata_of("社区获得性肺炎", "J18"))["items"])
        self.assertEqual(planner._table(st, "meds", planner.strata_of("社区获得性肺炎", "J18"))["n"], 80)

    def test_recheck_when_first_cbc_on_last_day(self):
        st = fixtures_mc.stats()
        st["los"] = {"*": {"n": 300, "items": {"3": 300}}}
        st["exams"] = {"*": {"n": 300, "items": {"血常规|": 300}}}
        st["exam_days"] = {"血常规|": {"5": 100}}
        for i in range(80):
            j = planner.plan_journey(st, random.Random(i), f"R{i}", None)
            self.assertFalse(any(e.get("recheck") for e in j["exams"]), i)

    def test_past_history_never_denies_listed_dx(self):
        seen = 0
        for i in range(120):
            j, _, _, _ = make(f"P{i}", i)
            p = j["patient"]
            dxs = [p["primary_dx"]["name"]] + [c["name"] for c in p["comorbidities"]]
            txt = compiler.past_text(p)
            denied = txt.split("否认", 1)[1] if "否认" in txt else ""
            for t in compiler.NEG_PAST:
                if any(t in d for d in dxs):
                    seen += 1
                    self.assertIn(t, p["past_history"], i)
                    self.assertNotIn(t, denied, i)
        self.assertGreater(seen, 10)

    def test_sex_and_age_filters(self):
        for i in range(150):
            j, _, _, _ = make(f"S{i}", i)
            p = j["patient"]
            names = [c["name"] for c in p["comorbidities"]] + j["differentials"]
            if p["sex"] == "男":
                self.assertNotIn("子宫肌瘤", names, i)
            else:
                self.assertNotIn("前列腺增生", names, i)
            if p["age"] < 18:
                self.assertFalse(any(planner.ADULT_DX.search(n) for n in names + [p["primary_dx"]["name"]]), i)
            self.assertTrue(planner.dept_ok(p["dept"], p["sex"], p["age"], p["pregnant"]), i)
        self.assertFalse(planner.dx_ok("前列腺增生", "女", 50))
        self.assertFalse(planner.dx_ok("子宫肌瘤", "男", 50))
        self.assertTrue(planner.dx_ok("高血压病", "男", 50))
        self.assertFalse(planner.dept_ok("妇科", "男", 50, False))
        self.assertFalse(planner.dept_ok("儿科", "女", 40, False))
        self.assertFalse(planner.dept_ok("产科", "女", 40, False))
        self.assertTrue(planner.dept_ok("妇产科", "女", 30, True))

    def test_procedure_follows_diagnosis_and_side(self):
        rng = random.Random(1)
        self.assertEqual(planner.realize_proc("{S侧}腹股沟疝修补术", "左侧腹股沟斜疝", rng), ("左侧腹股沟疝修补术", "左"))
        self.assertEqual(planner.realize_proc("{S侧}腹股沟疝修补术", "双侧腹股沟疝", rng), ("双侧腹股沟疝修补术", "双"))
        self.assertEqual(planner.realize_proc("腹腔镜下{S}肾切除术", "右肾癌", rng), ("腹腔镜下右肾切除术", "右"))
        self.assertIn(planner.realize_proc("{S侧}腹股沟疝修补术", "腹股沟疝", rng)[1], ("左", "右"))
        self.assertEqual(planner.realize_proc("腹腔镜胆囊切除术", "胆囊结石", rng), ("腹腔镜胆囊切除术", "无"))
        hernia = gall = 0
        for i in range(400):
            j, _, _, _ = make(f"H{i}", i)
            name = j["patient"]["primary_dx"]["name"]
            for a in j["procedures"]:
                # 0.1.4：K40 没有凑够 k 次的诊断名，取主干"{S侧}腹股沟斜疝"，侧别随机，术式侧别跟着诊断
                if name.endswith("侧腹股沟斜疝"):
                    hernia += 1
                    self.assertEqual((a["name"], a["side"]), (f"{name[0]}侧腹股沟疝无张力修补术", name[0]))
                else:
                    gall += 1
                    self.assertEqual(name, "胆囊结石伴急性胆囊炎")
                    self.assertEqual(a["name"], "腹腔镜胆囊切除术")
            if not j["patient"]["primary_dx"]["code3"].startswith("K"):
                self.assertFalse(j["surgical"])
        self.assertGreater(hernia, 0)
        self.assertGreater(gall, 0)

    def test_chapter_only_dx_does_not_borrow_other_code3(self):
        for i in range(200):
            j, _, _, _ = make(f"C{i}", i)
            p = j["patient"]
            if p["primary_dx"]["code3"] == "K" and p["primary_dx"]["name"] == "胆囊结石伴急性胆囊炎":
                self.assertFalse(j["surgical"], i)

    def test_med_allowed_sees_comorbidities_and_past(self):
        got = {}
        real = planner.evaluate

        def spy(rec):
            got["dx"] = rec.dx_text
            return real(rec)
        planner.evaluate = spy
        try:
            planner.med_allowed({"age": 60, "sex": "男", "pregnant": False, "primary_dx": {"name": "肺炎"}, "comorbidities": [{"name": "慢性肾衰竭"}], "past_history": ["痛风"]},
                                "头孢曲松", "2g", "qd", "静滴")
        finally:
            planner.evaluate = real
        self.assertIn("慢性肾衰竭", got["dx"])
        self.assertIn("痛风", got["dx"])

    def test_pregnancy_age_and_menses(self):
        rng = random.Random(3)
        for _ in range(50):
            self.assertTrue(40 <= planner._band_age(rng, "40-65", True) <= 49)
        self.assertIsNone(planner._band_age(rng, "65+", True))
        p = {"pregnant": True, "gest_weeks": 30, "age": 55}
        self.assertIn("停经30周", compiler.menses_text(p))
        self.assertNotIn("绝经", compiler.menses_text(p))
        self.assertEqual(compiler.menses_text({"pregnant": False, "age": 60}), "已绝经。")

    def test_lab_values_stay_plausible(self):
        st = fixtures_mc.stats()
        st["labs"]["HGB"] = {"q": [200, 250, 300, 330, 340], "n": 900, "unit": "g/L"}
        rng = random.Random(5)
        for _ in range(300):
            v = float(planner._lab_value(st, rng, "HGB", rng.random() < 0.5)["value"])
            self.assertLessEqual(v, stats.PLAUSIBLE["HGB"][1])

    def test_old_stats_rejected(self):
        st = fixtures_mc.stats()
        st["version"] = "mc-stats-0.1"
        with self.assertRaises(planner.StatsVersionError):
            planner.require_stats(st)

    def test_plan_diag_counts_only(self):
        tmp = tempfile.mkdtemp(prefix="d_")
        try:
            sp, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "d.csv")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(fixtures_mc.stats(), f, ensure_ascii=False)
            p = run_mod("medcourse.plan_diag", "--stats", sp, "--n", "60", "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            rows = dict(csv.reader(open(out, encoding="utf-8")))
            self.assertEqual(rows["planned"], "60")
            for k in ("comorb_wrong_sex_or_age", "diff_wrong_sex_or_age", "dept_wrong_sex_or_age", "past_denies_listed_dx", "proc_side_conflict"):
                self.assertEqual(rows.get(k, "0"), "0", k)
            self.assertNotIn("exception_ValueError", rows)
            self.assertIn("proc_available", rows)
            self.assertIn("surgery_expected_per_mille_sum", rows)
            text = open(out, encoding="utf-8").read()
            for w in ("肺炎", "胆囊", "呼吸内科", "头孢"):
                self.assertNotIn(w, text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCompiler(unittest.TestCase):
    def _surgical(self):
        slot = {"slot_id": "S1", "sex": "男", "age": 50, "dept": "普外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80", "los": 6, "surgical": True, "surgery_day": 3}
        return make("S1", 11, slot)

    def test_future_results_invisible(self):
        j, specs, st, rng = self._surgical()
        for spec in specs:
            if spec["role"] == "DS":
                continue
            for pl in compiler.compile_doc(j, spec, specs, st, rng):
                joined = " ".join(f["text"] for f in pl["facts"]) + (pl.get("program_text") or "")
                for e in j["exams"]:
                    if e["result_day"] > spec["day"] and e["items"]:
                        self.assertNotIn(f"{e['items'][0]['item']}{e['items'][0]['value']}", joined)

    def test_surgery_not_done_before_its_day(self):
        j, specs, st, rng = self._surgical()
        for spec in specs:
            if spec["day"] < j["surgery_day"] and spec["role"] != "PRE":
                for pl in compiler.compile_doc(j, spec, specs, st, rng):
                    self.assertFalse(any(f["kind"] == "procedure" and f["must"] for f in pl["facts"]))

    def test_plan_has_section_contract(self):
        j, specs, st, rng = make("J2", 2)
        for spec in specs:
            for pl in compiler.compile_doc(j, spec, specs, st, rng):
                self.assertEqual(contract.validate_section_plan(pl), [])

    def test_budget_varies_by_type_and_gap(self):
        st = fixtures_mc.stats()
        a = compiler.copy_budget(st, {"doc_type_cn": "日常病程记录", "day": 5}, 4)
        b = compiler.copy_budget(st, {"doc_type_cn": "主任医师查房记录", "day": 5}, 4)
        c = compiler.copy_budget(st, {"doc_type_cn": "日常病程记录", "day": 8}, 4)
        self.assertGreater(a, b)
        self.assertGreater(c, a)
        self.assertEqual(compiler.copy_budget({}, {"doc_type_cn": "日常病程记录", "day": 5}, 4), 0.0)

    def test_apply_copy_only_stable_sentences(self):
        prev = "患者一般情况可，精神睡眠好。白细胞12.1×10^9/L。拟明日复查血常规。今日查房未见异常情况。继续当前治疗方案不变。"
        out, added = compiler.apply_copy("今天症状减轻。复查结果已回。", prev, 0.5, random.Random(1))
        self.assertGreater(added, 0)
        self.assertNotIn("12.1", out)
        self.assertNotIn("拟明日", out)
        self.assertTrue(out.endswith("复查结果已回。"))

    def test_budget_limits_copy(self):
        prev = "患者一般情况可精神睡眠好。继续当前治疗方案不变。心肺听诊未见明显异常。"
        out, added = compiler.apply_copy("今天症状减轻，复查结果已回。" * 3, prev, 0.2, random.Random(1))
        self.assertLessEqual(added / len(out), 0.2)
        self.assertEqual(compiler.apply_copy("甲。", prev, 0.0, random.Random(1)), ("甲。", 0))


class TestGate(unittest.TestCase):
    def _plan(self, **kw):
        base = {"section": "正文", "target_chars": 10, "allowed": ["37.2", "80"], "facts": []}
        base.update(kw)
        return base

    def test_section_checks(self):
        self.assertEqual(gate.check_section(self._plan(), "体温37.2℃，脉搏80次/分，一般情况可。"), [])
        self.assertIn("number_unsourced", gate.check_section(self._plan(), "体温39.9℃，一般情况可以的。"))
        self.assertIn("prompt_leak", gate.check_section(self._plan(), "【患者】一般情况可以的。"))
        self.assertIn("too_short", gate.check_section(self._plan(target_chars=100), "一般可。"))
        must = self._plan(facts=[{"text": "t", "must": True, "key": "胸部CT"}])
        self.assertIn("fact_missing", gate.check_section(must, "患者一般情况可以的。"))
        self.assertEqual(gate.check_section(must, "胸部CT未见新发病灶。"), [])

    def test_routine_numbers_not_unsourced(self):
        ok = ("双侧瞳孔等大等圆，直径约3mm，对光反射灵敏。", "予0.9%氯化钠100ml加头孢曲松静滴。", "血氧饱和度98%，一般情况可以。", "予5%葡萄糖注射液250ml静滴。")
        for t in ok:
            self.assertNotIn("number_unsourced", gate.check_section(self._plan(), t), t)
        bad = ("手术切口约5cm，手术顺利完成。", "复查白细胞12.5×10^9/L，较前升高。", "中性粒细胞比例85%偏高，继续抗感染。", "瞳孔直径8mm，需警惕病情变化。")
        for t in bad:
            self.assertIn("number_unsourced", gate.check_section(self._plan(), t), t)

    def test_rewrite_keeps_previous_note(self):
        specs = [{"role": r, "day": d} for r, d in (("HR", 0), ("FP", 0), ("SUP", 1), ("PRE", 1), ("DAILY", 2), ("DAILY", 3))]
        bodies = {2: "甲", 4: "乙"}
        self.assertEqual(pipeline.prev_note_for(5, specs, bodies), {"day": 2, "text": "乙"})
        self.assertEqual(pipeline.prev_note_for(4, specs, bodies), {"day": 1, "text": "甲"})
        self.assertEqual(pipeline.prev_note_for(3, specs, bodies), {"day": 1, "text": "甲"})
        self.assertIsNone(pipeline.prev_note_for(2, specs, bodies))

    def _docs(self, j, specs, st, rng):
        w = writer.TemplateWriter()
        tally = {}
        docs = []
        for spec in specs:
            text, _ = pipeline.write_doc(j, spec, specs, st, w, rng, tally, None)
            docs.append({"doc_type": spec["doc_type"], "day_offset": spec["day"], "text": text, "doc_type_cn": spec["doc_type_cn"], "k": spec["k"], "role": spec["role"]})
        return docs

    def test_clean_admission_passes_and_mines_are_caught(self):
        j, specs, st, rng = make("J3", 3)
        docs = self._docs(j, specs, st, rng)
        self.assertEqual(gate.check_admission(j, docs), [])
        bad = copy.deepcopy(docs)
        bad[1]["text"] += "\n予美托洛尔5000mg qd口服。"
        self.assertTrue(any(c.startswith("g3_") for _, c in gate.check_admission(j, bad)))
        bad = copy.deepcopy(docs)
        bad[1]["text"] += "\n体温85℃。"
        self.assertTrue(any(c.startswith("n1_") for _, c in gate.check_admission(j, bad)))

    def test_surgical_admissions_pass_gate(self):
        """S1-B 在真实统计表上可能一例手术都没有，手术流程（术前小结、术后首程、侧别 S3、手术日 S4b）在这里补测。"""
        slots = [{"slot_id": "G1", "sex": "女", "age": 62, "dept": "普外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80", "los": 6, "surgical": True, "surgery_day": 2},
                 {"slot_id": "G2", "sex": "男", "age": 50, "dept": "普外科", "dx_name": "左侧腹股沟斜疝", "dx_code3": "K40", "los": 5, "surgical": True, "surgery_day": 1},
                 {"slot_id": "G3", "sex": "男", "age": 45, "dept": "普外科", "dx_name": "左侧腹股沟斜疝", "dx_code3": "K40", "los": 4, "surgical": True, "surgery_day": 0}]
        for slot in slots:
            for seed in range(4):
                j, specs, st, rng = make(slot["slot_id"], seed, slot)
                self.assertIn("POST", [s["role"] for s in specs])
                if slot["slot_id"] != "G1":
                    self.assertEqual((j["procedures"][0]["name"], j["procedures"][0]["side"]), ("左侧腹股沟疝无张力修补术", "左"))
                docs = self._docs(j, specs, st, rng)
                self.assertEqual(gate.check_admission(j, docs), [], (slot["slot_id"], seed))
                bad = copy.deepcopy(docs)
                post = next(i for i, s in enumerate(specs) if s["role"] == "POST")
                bad[post]["text"] = bad[post]["text"].replace("左侧", "右侧")
                if slot["slot_id"] != "G1":
                    self.assertIn("S3", [c for _, c in gate.check_admission(j, bad)], (slot["slot_id"], seed))

    def test_locked_diagnosis_has_no_state_change(self):
        from ruler_g2 import build_real
        from ruler_g2.c2parts import c2_components
        n = 0
        for i in range(40):
            j, specs, st, rng = make(f"J{i}", i)
            if any(x["state"] != "确诊" for x in j["dx_events"]):
                continue
            n += 1
            self.assertEqual(c2_components(build_real({"admission_id": j["journey_id"], "docs": self._docs(j, specs, st, rng)}))["dx_changes"], 0, i)
        self.assertGreater(n, 5)

    def test_same_value_already_known_is_not_future(self):
        j, specs, st, rng = make("J3", 3)
        e = next(x for x in j["exams"] if x["items"])
        j["exams"].append(dict(e, id="e99", order_day=j["los"], result_day=j["los"]))
        docs = self._docs(j, specs, st, rng)
        self.assertNotIn("future_result", [c for _, c in gate.check_admission(j, docs)])

    def test_future_result_caught(self):
        for i in range(30):
            j, specs, st, rng = make(f"J{i}", i)
            later = [e for e in j["exams"] if e["items"] and e["result_day"] > 0]
            if later:
                break
        docs = self._docs(j, specs, st, rng)
        e = later[0]
        hr = copy.deepcopy(docs)
        hr[0]["text"] += f"\n{e['items'][0]['item']}{e['items'][0]['value']}{e['items'][0]['unit']}。"
        self.assertIn((0, "future_result"), gate.check_admission(j, hr))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="p_")
        cls.st = os.path.join(cls.tmp, "stats.json")
        with open(cls.st, "w", encoding="utf-8") as f:
            json.dump(fixtures_mc.stats(), f, ensure_ascii=False)
        cls.out = os.path.join(cls.tmp, "out")
        cls.proc = run_mod("medcourse.pipeline", "--stats", cls.st, "--mode", "free", "--n", "30", "--writer", "template", "--out", cls.out, "--concurrency", "2")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _rows(self, name="admissions"):
        return [json.loads(x) for x in open(os.path.join(self.out, f"{name}.jsonl"), encoding="utf-8")]

    def test_runs_and_keeps(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-800:])
        c = json.load(open(os.path.join(self.out, "counts.json"), encoding="utf-8"))
        self.assertEqual(c["kept"] + c["discarded"], 30)
        self.assertEqual(c["kept"], 30)
        self.assertEqual(len(self._rows()), c["kept"])

    def test_admission_contract(self):
        for a in self._rows():
            types = [d["doc_type"] for d in a["docs"]]
            self.assertEqual(types[0], "hospital_record")
            self.assertEqual(types[-1], "discharge_summary")
            self.assertGreaterEqual(len(a["docs"]), 3)
            self.assertEqual([d["day_offset"] for d in a["docs"]], sorted(d["day_offset"] for d in a["docs"]))

    def test_same_seed_same_output(self):
        out2 = os.path.join(self.tmp, "again")
        p = run_mod("medcourse.pipeline", "--stats", self.st, "--mode", "free", "--n", "6", "--writer", "template", "--out", out2, "--concurrency", "1")
        out3 = os.path.join(self.tmp, "again3")
        run_mod("medcourse.pipeline", "--stats", self.st, "--mode", "free", "--n", "6", "--writer", "template", "--out", out3, "--concurrency", "1")
        self.assertEqual(p.returncode, 0)
        self.assertEqual(open(os.path.join(out2, "admissions.jsonl"), encoding="utf-8").read(), open(os.path.join(out3, "admissions.jsonl"), encoding="utf-8").read())

    def test_drafts_for_oracle(self):
        rows = self._rows("drafts")
        self.assertTrue(rows)
        d = rows[0]["draft"]
        self.assertEqual(set(d), {"final_diagnoses", "presentation", "observations", "judgments", "actions"})

    def test_llm_writer_against_fake(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
            out = os.path.join(self.tmp, "llm")
            Fake.mode, Fake.seen = "ok", []
            p = run_mod("medcourse.pipeline", "--stats", self.st, "--mode", "free", "--n", "3", "--writer", "llm", "--base-url", url, "--model", "M", "--out", out, "--concurrency", "1")
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            c = json.load(open(os.path.join(out, "counts.json"), encoding="utf-8"))
            self.assertGreater(c["calls"], 3)
            self.assertIn("sec_fact_missing", c)
            self.assertEqual(len(Fake.seen), c["calls"])
            self.assertTrue(all(b["temperature"] == 0.7 and b["top_p"] == 0.8 and b["top_k"] == 20 and "seed" in b for b in Fake.seen))
        finally:
            srv.shutdown()


class TestV014(unittest.TestCase):
    """0.1.4：外科、产科章被规划器跳过（诊断名兜底、公开词典补 code3）；检查漏写和偏少；病程相邻太像；出院记录"入院情况"写进住院后的事。"""

    def test_icd_lookup(self):
        from medcourse import icd
        self.assertEqual(icd.code3_exact("2型糖尿病"), "E11")
        self.assertEqual(icd.code3_exact("社区获得性肺炎"), "J18")         # 词典没有，查常用写法对照
        self.assertEqual(icd.code3_exact("高血压病3级（极高危）"), "I10")    # 病名打头加分级
        self.assertIsNone(icd.code3_exact("胆囊结石伴急性胆囊炎"))
        self.assertEqual(icd.code3_stem("胆囊结石伴急性胆囊炎"), "K80")
        self.assertEqual(icd.code3_stem("左侧股骨颈骨折"), "S72")
        self.assertEqual(icd.stem_key("左侧腹股沟斜疝伴嵌顿"), "{S侧}腹股沟斜疝")
        self.assertEqual(icd.category("I63"), "脑梗死")
        self.assertEqual(icd.category("K80"), "胆囊结石")
        d = icd.self_check()
        self.assertGreaterEqual(d["agree"], stats.DX_MAP_MIN * d["found"])
        self.assertGreaterEqual(d["chapter_agree"], stats.DX_MAP_MIN_CH * d["found"])

    def test_dx_map_check_gates_on_agreement(self):
        names = {f"A{i}": "胆囊结石伴急性胆囊炎" for i in range(40)}
        m = stats.dx_map_check(names, {f"A{i}": "K80" for i in range(40)})
        self.assertEqual((m["use_stem"], m["stem_basis_dictionary"], m["stem_agree"]), (1, 0, 40))
        m = stats.dx_map_check(names, {f"A{i}": "K81" for i in range(40)})        # 真实住院上对不上：不启用
        self.assertEqual(m["use_stem"], 0)
        m = stats.dx_map_check({"A1": "胆囊结石伴急性胆囊炎"}, {})                    # 样本不够：按公开词典自身判
        self.assertEqual((m["use_stem"], m["stem_basis_dictionary"], m["use_exact"]), (1, 1, 1))

    def test_stats_fills_code3_and_counts(self):
        lines = []
        for i in range(12):
            a = fixtures_mc.admission(i)
            a["docs"][0]["text"] += "血常规：平均红细胞血红蛋白浓度330g/L，血红蛋白120g/L。\n"
            lines.append(a)
        s = stats.build_stats(lines, {}, k=11)          # dx_map 全空：社区获得性肺炎按常用写法补成 J18
        self.assertEqual(s["dx_map_meta"]["added_exact"], 12)
        self.assertIn("J18", s["dx_names"])
        self.assertEqual(s["exam_kinds"]["J18"]["n"], 12)
        lm = s["labs_meta"]
        self.assertEqual((lm["HGB_old_only"], lm["HGB_old_only_out_of_range"]), (12, 12))
        self.assertEqual(s["labs"]["HGB"]["q"], [120.0] * 5)
        rows = {r["table"]: r["cells"] for r in stats.summary_rows(s)}
        self.assertEqual(rows["labs_meta:HGB_old_only"], 12)
        self.assertEqual(rows["dx_map_meta:added_exact"], 12)

    def test_proc_family(self):
        self.assertEqual(stats.proc_family("腹腔镜胆囊切除术"), "胆囊切除")
        self.assertEqual(stats.proc_family("腹腔镜子宫肌瘤剔除术"), "子宫肌瘤剔除")      # 不能写成子宫切除
        self.assertEqual(stats.proc_family("胆总管切开取石术"), "胆道取石")             # 不能写成泌尿系碎石
        self.assertEqual(stats.proc_family("{S}乳腺癌改良根治术"), "乳腺癌根治")
        self.assertIsNone(stats.proc_family("清创缝合术"))
        st = fixtures_mc.stats()
        st["procedures"] = {"K": {"n": 130, "items": {}}}
        st["proc_family"] = {"K80": {"n": 100, "items": {"胆囊切除": 60}}}
        got = set()
        for i in range(60):
            j, _, _, _ = make(f"F{i}", i, None, st)
            for a in j["procedures"]:
                got.add((j["patient"]["primary_dx"]["code3"], a["name"], a["source"]))
        self.assertEqual(got, {("K80", "腹腔镜胆囊切除术", "family")})

    def test_dx_name_fallback_order(self):
        st = fixtures_mc.stats()
        rng = random.Random(0)
        self.assertEqual(planner.dx_name_for(st, rng, "K80", "K"), ("胆囊结石伴急性胆囊炎", "name"))
        name, src = planner.dx_name_for(st, rng, "K40", "K")
        self.assertEqual((name[1:], src), ("侧腹股沟斜疝", "stem"))
        st["dx_stems"] = {}
        self.assertEqual(planner.dx_name_for(st, rng, "K40", "K"), ("腹股沟斜疝", "icd"))
        self.assertEqual(planner.dx_name_for(st, rng, None, "K")[1], "chapter")
        # 0.1.3 的问题：code3 下没有凑够 k 次的诊断名就整章跳过。现在 K 章照样抽得到，而且有手术
        chapters, surgical = set(), 0
        for i in range(200):
            j, _, _, _ = make(f"C{i}", i, None, st)
            chapters.add(j["patient"]["primary_dx"]["code3"][:1])
            surgical += j["surgical"]
        self.assertIn("K", chapters)
        self.assertGreater(surgical, 0)

    def test_paired_surgery_without_procedure_not_invented(self):
        slot = {"slot_id": "P1", "sex": "男", "age": 50, "dept": "普外科", "dx_name": "股疝", "dx_code3": "K41", "los": 5, "surgical": True, "surgery_day": 2}
        j, specs, _, _ = make("P1", 1, slot)
        self.assertFalse(j["surgical"])
        self.assertEqual(j["procedures"], [])
        self.assertEqual(contract.validate_journey(j), [])

    def test_exam_fill_to_real_kind_count(self):
        st = fixtures_mc.stats()
        st["exam_kinds"] = {"J": {"q": [7, 7, 7, 7, 7], "n": 130}, "K": {"q": [6, 6, 6, 6, 6], "n": 130}}
        for i in range(40):
            j, specs, _, _ = make(f"E{i}", i, None, st)
            kinds = {(e["key"], e["region"] or "") for e in j["exams"]}
            want = 7 if j["patient"]["primary_dx"]["code3"].startswith("J") else 6
            self.assertGreaterEqual(len(kinds), want, i)
            self.assertEqual(contract.validate_journey(j), [])

    def test_symptoms_only_resolve(self):
        for i in range(40):
            j, _, _, _ = make(f"Y{i}", i)
            days = [d["symptoms"] for d in j["daily"]]
            for a, b in zip(days, days[1:]):
                self.assertEqual(b[0], days[0][0])
                self.assertTrue(set(b) <= set(a))

    def test_every_later_result_is_must_somewhere(self):
        for i in range(60):
            j, specs, st, rng = make(f"R{i}", i)
            plans = [pl for s in specs for pl in compiler.compile_doc(j, s, specs, st, rng)]
            must = {f["key"] for pl in plans for f in pl["facts"] if f["must"] and f.get("key")}
            for e in j["exams"]:
                if e["result_day"] > 0:
                    self.assertIn(e["name"], must, (i, e["name"], e["result_day"]))

    def test_course_notes_follow_changes(self):
        for i in range(40):
            j, specs, st, rng = make(f"N{i}", i)
            for s in specs:
                if s["role"] not in compiler.COURSE:
                    continue
                texts = [f["text"] for f in compiler.compile_doc(j, s, specs, st, rng)[0]["facts"]]
                self.assertFalse(any(t.startswith("目前用药") or t.startswith("仍在等结果") for t in texts))
                self.assertLessEqual(sum(t.startswith("继续") for t in texts), 2)     # 0.1.5 轮换提两种

    def test_admission_section_only_day0(self):
        j, specs, st, rng = make("A1", 3)
        ds = next(s for s in specs if s["role"] == "DS")
        plans = compiler.compile_doc(j, ds, specs, st, rng)
        pl = next(x for x in plans if x["section"] == "入院情况")
        self.assertTrue(pl["admission_only"])
        later = {x["value"] for e in j["exams"] if e["result_day"] > 0 for x in e["items"]} - {x["value"] for e in j["exams"] if e["result_day"] == 0 for x in e["items"]}
        self.assertFalse(later & set(pl["allowed"]))
        self.assertIn("只写患者入院当时", writer.section_prompt(j, ds, pl))
        self.assertIn("只写入院当时", writer.raw_prompt(j, ds, pl))
        other = next(x for x in plans if x["section"] == "诊疗经过")
        self.assertNotIn("只写患者入院当时", writer.section_prompt(j, ds, other))
        self.assertNotIn("只写入院当时", writer.raw_prompt(j, ds, other))
        base = {"section": "入院情况", "facts": [], "target_chars": 10, "allowed": [], "admission_only": True}
        self.assertIn("admission_leak", gate.check_section(base, "患者因咳嗽3天入院，入院后完善相关检查，予抗感染治疗。"))
        self.assertNotIn("admission_leak", gate.check_section(base, "患者因咳嗽3天入院，3年前曾行阑尾切除术，查体双肺可闻及湿啰音。"))
        self.assertNotIn("admission_leak", gate.check_section(dict(base, admission_only=False), "入院后完善相关检查。"))

    def test_plan_diag_v014_structure_zero(self):
        tmp = tempfile.mkdtemp(prefix="d_")
        try:
            sp, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "d.csv")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(fixtures_mc.stats(), f, ensure_ascii=False)
            p = run_mod("medcourse.plan_diag", "--stats", sp, "--n", "80", "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            rows = dict(csv.reader(open(out, encoding="utf-8")))
            for k in ("results_never_must", "ds_admission_only_missing", "proc_name_concat", "surgery_without_procedure"):
                self.assertEqual(rows.get(k, "0"), "0", k)
            self.assertGreater(int(rows.get("dx_source_stem", 0)), 0)
            self.assertGreater(int(rows.get("chapter_K", 0)), 0)
            self.assertGreater(int(rows.get("proc_source_key", 0)), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestV015(unittest.TestCase):
    """0.1.5：术式大类漏分母；O 章这类没有 code3 凑够 k 次的章；药名次数改过头；入院情况的事实自带 K4 触发词；S4b 没有日历日期；
    检查补齐只在本层；无数值的检查结果写法被改成计划口吻；病程每篇都由程序写全生命体征。"""

    def _surgical_lines(self, n=12):
        post = "2025-03-03 10:00　术后首次病程记录\n手术时间：2025-03-03 10时前完成。\n手术方式：腹腔镜下胆囊切除术。\n麻醉方式：全身麻醉。\n手术顺利。\n"
        lines = []
        for i in range(n):
            a = fixtures_mc.admission(i)
            a["docs"].insert(2, {"doc_type": "progress_note", "day_offset": 2, "text": post})
            lines.append(a)
        return lines

    def test_proc_family_table_built(self):
        s = stats.build_stats(self._surgical_lines(), {}, k=11)
        self.assertEqual(s["surgery_meta"]["proc_family_found"], 12)
        self.assertEqual(s["proc_family"]["*"]["items"], {"胆囊切除": 12})      # 0.4 漏登记分母，这张表是空的
        self.assertEqual(s["procedures"]["*"]["items"], {"腹腔镜胆囊切除术": 12})
        self.assertIn("J", s["dx_stems"])                                       # 章层主干
        self.assertEqual(s["note_vitals"]["n"], 12)

    def test_chapter_stem_fallback(self):
        st = fixtures_mc.stats()
        st["dx_names_ch"]["O"] = {"n": 40, "items": {}}
        st["dx_stems"]["O"] = {"n": 40, "items": {"瘢痕子宫": 20}}
        self.assertEqual(planner.dx_name_for(st, random.Random(0), None, "O"), ("瘢痕子宫", "chapter_stem"))
        st["joint"]["items"]["女|18-40|O"] = 200
        st["code3"]["O"] = {"n": 40, "items": {}}
        got = [make(f"O{i}", i, None, st)[0]["patient"] for i in range(80)]
        preg = [p for p in got if p["primary_dx"]["code3"] == "O"]
        self.assertTrue(preg)
        self.assertTrue(all(p["pregnant"] and p["primary_dx"]["source"] == "chapter_stem" for p in preg))

    def test_duration_not_longer_than_age(self):
        self.assertFalse(planner.duration_ok("10年", 8))
        self.assertTrue(planner.duration_ok("3天", 0.5))
        self.assertFalse(planner.duration_ok("2月", 0.1))
        self.assertTrue(planner.duration_ok("不详", 5))

    def test_day0_pathology_is_prior_and_k4_clean(self):
        e = planner._exam(fixtures_mc.stats(), random.Random(0), 1, "病理", "", 5, day=0)
        self.assertEqual((e["name"], e["prior"]), ("外院病理", True))
        st = fixtures_mc.stats()
        # 0.1.9：有创检查（病理）只从节层以内抽，所以放在 code3 层（0.1.8 这条放在章层）
        st["exams"]["J18"] = {"n": 100, "items": dict(st["exams"]["J"]["items"], **{"病理|": 95})}
        st["exam_days"]["病理|"] = {"0": 100}
        n = 0
        for i in range(40):
            j, specs, st2, rng = make(f"P{i}", i, None, st)
            ds = next(s for s in specs if s["role"] == "DS")
            pl = next(x for x in compiler.compile_doc(j, ds, specs, st2, rng) if x["section"] == "入院情况")
            n += any(e["key"] == "病理" for e in j["exams"] if e["result_day"] == 0)
            self.assertEqual(gate.admission_leaks("。".join(f["text"] for f in pl["facts"])), [])
        self.assertGreater(n, 0)

    def test_gate_counts_each_missing_fact_and_result_tone(self):
        pl = {"section": "正文", "target_chars": 5, "allowed": [], "facts": [
            {"kind": "result", "text": "胸部CT示：未见明显异常", "must": True, "key": "胸部CT", "values": []},
            {"kind": "result", "text": "心电图示：未见明显异常", "must": True, "key": "心电图", "values": []},
            {"kind": "med", "text": "继续头孢曲松治疗", "must": True, "key": "头孢曲松", "values": []}]}
        self.assertEqual(gate.check_section(pl, "病情平稳，继续观察。").count("fact_missing"), 3)
        self.assertIn("result_tone_missing", gate.check_section(pl, "完善胸部CT检查，心电图示未见异常，继续头孢曲松治疗。"))
        self.assertEqual(gate.check_section(pl, "胸部CT示未见明显异常，心电图示未见明显异常，继续头孢曲松治疗。"), [])

    def test_strip_admission_leak(self):
        t = gate.strip_admission_leak("患者因咳嗽3天入院，入院后完善相关检查，查体双肺可闻及湿啰音。")
        self.assertEqual(t, "患者因咳嗽3天入院，查体双肺可闻及湿啰音。")
        self.assertEqual(gate.admission_leaks(t), [])
        self.assertEqual(gate.strip_admission_leak("患者3年前曾行阑尾切除术，此次因腹痛入院。"), "患者3年前曾行阑尾切除术，此次因腹痛入院。")

        class LeakWriter(writer.TemplateWriter):
            def write(self, j, spec, plan, attempt):
                return "患者因咳嗽入院，入院后完善相关检查。" if plan["section"] == "入院情况" else super().write(j, spec, plan, attempt)

        j, specs, st, rng = make("L1", 3)
        tally = {}
        ds = next(s for s in specs if s["role"] == "DS")
        _, texts = pipeline.write_doc(j, ds, specs, st, LeakWriter(), rng, tally, None)
        self.assertEqual(gate.admission_leaks(texts["入院情况"]), [])
        self.assertEqual(tally["sec_admission_leak_stripped"], 1)

    def test_s4b_checkable_with_calendar_date(self):
        from ruler_g2 import build_real, evaluate_admission
        slot = {"slot_id": "G1", "sex": "女", "age": 62, "dept": "普外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80", "los": 6, "surgical": True, "surgery_day": 3}
        for seed in range(3):
            j, specs, st, rng = make("G1", seed, slot)
            w, docs, tally = writer.TemplateWriter(), [], {}
            for spec in specs:
                text, _ = pipeline.write_doc(j, spec, specs, st, w, rng, tally, None)
                docs.append({"doc_type": spec["doc_type"], "day_offset": spec["day"], "text": text, "doc_type_cn": spec["doc_type_cn"], "k": spec["k"], "role": spec["role"]})
            r = evaluate_admission(build_real({"admission_id": "G1", "docs": docs}))["readings"]["S4b"]
            self.assertEqual((r["n_checkable"], r["adm_conflict"]), (1, False), seed)

    def test_course_vitals_and_med_rotation(self):
        st = fixtures_mc.stats()
        st["note_vitals"] = {"rate": 0.0, "n": 300}
        seen_two = False
        for i in range(40):
            j, specs, st2, rng = make(f"V{i}", i, None, st)
            for s in specs:
                if s["role"] not in compiler.COURSE:
                    continue
                pl = compiler.compile_doc(j, s, specs, st2, rng)[0]
                d = j["daily"][s["day"]]
                abnormal = d["T"] >= 37.3 or d["SBP"] >= 160 or d["SBP"] < 90
                self.assertEqual(bool(pl["program_prefix"]), abnormal)
                cont = [f for f in pl["facts"] if f["text"].startswith("继续")]
                if len(cont) == 2:
                    seen_two = True
                    self.assertEqual([f["must"] for f in cont], [True, False])
        self.assertTrue(seen_two)

    def test_exam_fill_reaches_coarser_layer(self):
        st = fixtures_mc.stats()
        st["exam_kinds"] = {"J": {"q": [12, 12, 12, 12, 12], "n": 130}}
        st["exams"]["*"] = {"n": 300, "items": {"血常规|": 290, "尿常规|": 200, "便常规|": 150, "X线|胸部": 120, "超声|心脏": 100}}
        for i in range(20):
            j, _, _, _ = make(f"X{i}", i, None, st)
            if j["patient"]["primary_dx"]["code3"].startswith("J"):
                kinds = {(e["key"], e["region"] or "") for e in j["exams"]}
                self.assertGreaterEqual(len(kinds), 12, i)

    def test_out_diag_counts_only(self):
        tmp = tempfile.mkdtemp(prefix="o_")
        try:
            sp, out, run = os.path.join(tmp, "s.json"), os.path.join(tmp, "o.csv"), os.path.join(tmp, "run")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(fixtures_mc.stats(), f, ensure_ascii=False)
            p = run_mod("medcourse.pipeline", "--stats", sp, "--mode", "free", "--n", "12", "--writer", "template", "--out", run)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            p = run_mod("medcourse.out_diag", "--run", run, "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            with open(out, encoding="utf-8") as f:
                rows = dict(csv.reader(f))
            self.assertEqual(rows["admissions"], "12")
            self.assertEqual(rows["exam_planned"], rows["exam_planned_read"])       # 模板照抄"X示："，规划的检查全读成结果口吻
            for k in ("tfidf_adj_full_x1000", "tfidf_adj_no_header_x1000", "tfidf_adj_no_header_vitals_x1000", "k4_checkable", "med_mentions", "course_body_chars_p50"):
                self.assertIn(k, rows)
            text = open(out, encoding="utf-8").read()
            for w in ("肺炎", "胆囊结石", "头孢", "呼吸内科", "患者"):
                self.assertNotIn(w, text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_plan_diag_v015_structure_zero(self):
        tmp = tempfile.mkdtemp(prefix="d_")
        try:
            sp, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "d.csv")
            st = fixtures_mc.stats()
            st["exams"]["J"]["items"]["病理|"] = 120
            st["exam_days"]["病理|"] = {"0": 100}
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False)
            p = run_mod("medcourse.plan_diag", "--stats", sp, "--n", "80", "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            with open(out, encoding="utf-8") as f:
                rows = dict(csv.reader(f))
            self.assertEqual(rows.get("ds_admission_facts_leak", "0"), "0")
            self.assertIn("exam_target_total", rows)
            self.assertIn("course_vitals_prefix", rows)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestV016(unittest.TestCase):
    """0.1.6：术式按适应证关键词放开到章层、分侧术式配分侧诊断；产科孕周孕产次占位符；病程按当天的事写重点；W-8B 上一篇对照臂；
    out_diag 套话计数；合成病历抽样带回（synth_export）。"""

    def _stats(self):
        st = fixtures_mc.stats()
        st["dx_names"]["K40"] = {"n": 30, "items": {"腹股沟斜疝": 25}}
        st["procedures"]["K40"] = {"n": 30, "items": {}}
        del st["procedures"]["n:左侧腹股沟斜疝"]
        st["proc_family"] = {"K": {"n": 130, "items": {"腹股沟疝修补": 40, "胆囊切除": 60}}}
        return st

    def _obstetric_stats(self):
        st = fixtures_mc.stats()
        st["joint"]["items"]["女|18-40|O"] = 300
        st["code3"]["O"] = {"n": 40, "items": {}}
        st["dx_names_ch"]["O"] = {"n": 40, "items": {}}
        st["dx_stems"]["O"] = {"n": 40, "items": {"孕{W}周G{G}P{P}待产": 20, "先兆早产孕{W}周": 12}}
        st["surgery"]["O"] = {"n": 40, "items": {"yes": 30, "no": 10}}
        st["proc_family"] = {"O": {"n": 40, "items": {"剖宫产": 25}}}
        st["dept"]["O"] = {"n": 40, "items": {"产科": 40}}
        return st

    def test_obstetric_stem_key_and_realize(self):
        self.assertEqual(icd.stem_key("孕39+2周G1P0LOA待产"), "孕{W}周G{G}P{P}待产")
        self.assertEqual(icd.stem_key("孕1产0宫内孕38+5周头位待产"), icd.stem_key("孕2产1宫内孕40周头位待产"))
        self.assertEqual(icd.stem_key("左侧腹股沟斜疝伴嵌顿"), "{S侧}腹股沟斜疝")                  # 非产科不变
        self.assertEqual(icd.code3_stem("妊娠期糖尿病（孕32周）"), "O24")                         # 去掉孕周再查词典
        rng = random.Random(3)
        for _ in range(30):
            x = icd.realize("孕{W}周G{G}P{P}待产", rng)
            self.assertNotIn("{", x)
            self.assertTrue(37 <= icd.gest_weeks(x) <= 41, x)
            g, p = map(int, re.search(r"G(\d)P(\d)", x).groups())
            self.assertLess(p, g)
        self.assertTrue(28 <= icd.gest_weeks(icd.realize("先兆早产孕{W}周", rng)) <= 36)
        self.assertIsNone(icd.gest_weeks("瘢痕子宫"))

    def test_obstetric_plan_syncs_gest_weeks(self):
        st = self._obstetric_stats()
        n_o = n_cs = 0
        for i in range(120):
            j, specs, _, _ = make(f"O{i}", i, None, st)
            p = j["patient"]
            if p["primary_dx"]["code3"] != "O":
                continue
            n_o += 1
            name = p["primary_dx"]["name"]
            self.assertNotIn("{", name)
            self.assertTrue(p["pregnant"])
            self.assertEqual(icd.gest_weeks(name), p["gest_weeks"])
            for a in j["procedures"]:
                n_cs += a["name"] == "子宫下段剖宫产术"
                self.assertGreaterEqual(p["gest_weeks"], 28)
        self.assertGreater(n_o, 10)
        self.assertGreater(n_cs, 0)

    def test_family_keyword_gate(self):
        self.assertTrue(planner.family_ok("腹股沟疝修补", "腹股沟斜疝"))
        self.assertFalse(planner.family_ok("腹股沟疝修补", "胆囊结石伴急性胆囊炎"))
        self.assertFalse(planner.family_ok("剖宫产", "孕20周瘢痕子宫"))                         # 28 周以前不剖
        self.assertFalse(planner.family_ok("清宫", "先兆流产"))                                 # 先兆流产保胎，不清宫
        self.assertTrue(planner.family_ok("清宫", "稽留流产"))
        self.assertFalse(planner.family_ok("气管切开", "呼吸衰竭"))                             # 没有适应证写法的大类不进章层
        items, how = planner.proc_for(self._stats(), planner.strata_of("腹股沟斜疝", "K40"), "腹股沟斜疝")
        self.assertEqual((items, how), ({"腹股沟疝修补": 40}, "family_kw"))
        items, how = planner.proc_for(self._stats(), planner.strata_of("急性胰腺炎", "K"), "急性胰腺炎")
        self.assertEqual((items, how), ({}, None))

    def test_lateralize_rules(self):
        self.assertEqual(planner.lateralize_dx("腹股沟斜疝", "左"), "左侧腹股沟斜疝")
        self.assertEqual(planner.lateralize_dx("肺癌", "右"), "右肺癌")
        self.assertEqual(planner.lateralize_dx("老年性白内障", "右", "白内障"), "右眼老年性白内障")
        self.assertEqual(planner.lateralize_dx("慢性肾脏病5期", "左"), "慢性肾脏病5期")             # 肾脏病不是一侧的病变
        self.assertFalse(planner.same_organ("左前臂动静脉内瘘成形术", "慢性肾脏病5期"))
        p = {"primary_dx": {"name": "股骨颈骨折"}, "age": 70}
        pname, side, name = planner.realize_surgery("骨折内固定", "family_kw", p, random.Random(1), True)
        self.assertEqual(pname, name + "切开复位内固定术")
        self.assertIn(side, ("左", "右"))
        self.assertEqual(planner.side_of(name), side)
        p = {"primary_dx": {"name": "腰椎压缩性骨折"}, "age": 70}
        self.assertEqual(planner.realize_surgery("骨折内固定", "family_kw", p, random.Random(1), True), ("腰椎压缩性骨折切开复位内固定术", "无", "腰椎压缩性骨折"))
        p = {"primary_dx": {"name": "腹股沟斜疝"}, "age": 5}
        self.assertIn("疝囊高位结扎", planner.realize_surgery("腹股沟疝修补", "family_kw", p, random.Random(1), True)[0])

    def test_lateral_surgery_makes_s3_checkable(self):
        from ruler_g2 import build_real, evaluate_admission
        st = self._stats()
        checked = 0
        for i in range(60):
            j, specs, st2, rng = make(f"H{i}", i, None, st)
            if not any(a["source"] == "family_kw" for a in j["procedures"]):
                continue
            a = j["procedures"][0]
            dx = j["patient"]["primary_dx"]
            self.assertTrue(dx.get("side_added"))
            self.assertEqual(planner.side_of(dx["name"]), a["side"])
            w, docs, tally = writer.TemplateWriter(), [], {}
            for spec in specs:
                text, _ = pipeline.write_doc(j, spec, specs, st2, w, rng, tally, None)
                docs.append({"doc_type": spec["doc_type"], "day_offset": spec["day"], "text": text, "doc_type_cn": spec["doc_type_cn"], "k": spec["k"], "role": spec["role"]})
            r = evaluate_admission(build_real({"admission_id": j["journey_id"], "docs": docs}))["readings"]["S3"]
            self.assertFalse(r["adm_conflict"])
            checked += r["n_checkable"] > 0
        self.assertGreater(checked, 0)

    def test_paired_dx_not_lateralized(self):
        slot = {"slot_id": "S1", "sex": "男", "age": 50, "dept": "普外科", "dx_name": "腹股沟斜疝", "dx_code3": "K40", "los": 5, "surgical": True, "surgery_day": 2}
        for seed in range(10):
            j, _, _, _ = make("S1", seed, slot, self._stats())
            self.assertEqual(j["patient"]["primary_dx"]["name"], "腹股沟斜疝")

    def test_course_focus(self):
        st = self._stats()
        kinds = collections.Counter()
        boiler = [re.compile(rx) for _, rx in out_diag.BOILER]
        for i in range(40):
            j, specs, st2, rng = make(f"C{i}", i, None, st)
            for s in specs:
                if s["role"] not in compiler.COURSE:
                    continue
                pl = compiler.compile_doc(j, s, specs, st2, rng)[0]
                self.assertTrue(1 <= len(pl["focus"]) <= 2)
                kinds.update(pl["focus"])
                if s["role"] == "SUP":
                    self.assertEqual(pl["focus"][0], "round")
                for f in pl["facts"]:
                    self.assertFalse(any(w in f["text"] for w in gate.LEAK), f["text"])
                    if f["kind"] == "focus":
                        self.assertFalse(any(rx.search(f["text"]) for rx in boiler), f["text"])
                if any(a["do_day"] < s["day"] for a in j["procedures"]):
                    self.assertIn("postop", pl["focus"])
        for k in ("round", "result", "quiet", "postop"):
            self.assertGreater(kinds[k], 0, k)

    def test_prev_note_prompt_only_when_asked(self):
        j, specs, st, rng = make("N1", 2)
        s = next(x for x in specs if x["role"] == "DAILY")
        pl = compiler.compile_doc(j, s, specs, st, rng)[0]
        plain = writer.section_prompt(j, s, pl)
        self.assertNotIn("上一篇", plain)
        pl2 = dict(pl, prev_text="患者咳嗽较前减轻。" * 100)
        got = writer.section_prompt(j, s, pl2)
        self.assertIn("【上一篇病程】", got)
        self.assertIn(writer.PREV_RULE, got)
        self.assertLessEqual(got.count("患者咳嗽较前减轻"), writer.PREV_CHARS // 9 + 1)
        self.assertEqual(got.replace(writer.PREV_NOTE.format(prev=pl2["prev_text"][-writer.PREV_CHARS:]), "").replace(writer.PREV_RULE, ""), plain)

    def test_prev_note_arm_against_fake(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        tmp = tempfile.mkdtemp(prefix="n_")
        try:
            sp = os.path.join(tmp, "s.json")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(fixtures_mc.stats(), f, ensure_ascii=False)
            url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
            for arm, flag in (("a", []), ("b", ["--prev-note"])):
                Fake.mode, Fake.seen = "ok", []
                p = run_mod("medcourse.pipeline", "--stats", sp, "--mode", "free", "--n", "2", "--writer", "llm", "--base-url", url, "--model", "M",
                            "--out", os.path.join(tmp, arm), "--concurrency", "1", *flag)
                self.assertEqual(p.returncode, 0, p.stderr[-400:])
                n_prev = sum("【上一篇病程】" in b["messages"][-1]["content"] for b in Fake.seen)
                c = json.load(open(os.path.join(tmp, arm, "counts.json"), encoding="utf-8"))
                self.assertEqual(c["prev_note"], bool(flag))
                self.assertEqual(c["model"], "M")
                if flag:
                    self.assertGreater(n_prev, 0)
                    self.assertTrue(all("病程" in b["messages"][-1]["content"].split("”一节")[0] or "查房" in b["messages"][-1]["content"].split("”一节")[0]
                                        for b in Fake.seen if "【上一篇病程】" in b["messages"][-1]["content"]))
                else:
                    self.assertEqual(n_prev, 0)
        finally:
            srv.shutdown()
            shutil.rmtree(tmp, ignore_errors=True)

    def _run(self, tmp, n=24):
        sp, run = os.path.join(tmp, "s.json"), os.path.join(tmp, "run")
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(self._stats(), f, ensure_ascii=False)
        p = run_mod("medcourse.pipeline", "--stats", sp, "--mode", "free", "--n", str(n), "--writer", "template", "--out", run)
        self.assertEqual(p.returncode, 0, p.stderr[-400:])
        return run

    def test_out_diag_boilerplate(self):
        tmp = tempfile.mkdtemp(prefix="b_")
        try:
            run = self._run(tmp)
            path = os.path.join(run, "admissions.jsonl")
            rows = [json.loads(x) for x in open(path, encoding="utf-8")]
            notes = 0
            for r in rows:
                for d in r["docs"]:
                    if d["role"] in ("SUP", "DAILY"):
                        d["text"] += "患者精神、睡眠、饮食可，二便正常。\n"
                        notes += 1
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            got = dict(out_diag.diagnose(run))
            self.assertEqual(got["boiler_notes:精神睡眠饮食"], notes)
            self.assertEqual(got["boiler_notes:二便"], notes)
            self.assertEqual(got["boiler_notes_any"], notes)
            self.assertGreater(got["course_sent_common_x1000"], 0)
            self.assertIn("course_sent_prev_repeat_x1000", got)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_synth_export(self):
        """0.1.7 口径：同一行里连续 30 字、数字换 0。"""
        tmp = tempfile.mkdtemp(prefix="x_")
        try:
            run = self._run(tmp)
            with open(os.path.join(run, "admissions.jsonl"), encoding="utf-8") as f:
                rows = sorted((json.loads(x) for x in f), key=lambda r: r["admission_id"])   # 流程多线程写，行序不固定
            n = synth_export.norm

            def grams30(text):
                t = n(text)
                return {t[i:i + 30] for i in range(len(t) - 29)}

            fx = grams30("".join(d["text"] for d in fixtures_mc.admission(0)["docs"]))
            rare_src = None
            for ti, target in enumerate(rows):                                   # 找一段带数字、只属于一份住院的（同一行）片段
                others = fx.union(*(grams30(line) for r in rows if r is not target for _, line in synth_export.lines_of(r)))
                rare_src = next((line[i:i + 30] for _, line in synth_export.lines_of(target) for i in range(len(line) - 29)
                                 if re.search(r"\d", line[i:i + 30]) and n(line[i:i + 30]) not in others), None)
                if rare_src:
                    break
            self.assertIsNotNone(rare_src)
            rows = [target] + [r for r in rows if r is not target]
            shifted = re.sub(r"\d", lambda m: str((int(m.group(0)) + 1) % 10), rare_src)   # 数字不同也算同一段
            common_src = next(x for _, x in synth_export.lines_of(rows[1]) if len(x) >= 30)[:30]
            real = [fixtures_mc.admission(i) for i in range(12)]
            real[0]["docs"][0]["text"] += "\n" + shifted
            for r in real[1:12]:
                r["docs"][1]["text"] += "\n" + common_src
            src = os.path.join(tmp, "real.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in real)
            got, st = synth_export.rare_span_counts(rows[:2], real)
            self.assertGreater(got[30][rows[0]["admission_id"]], 0)             # 罕见片段命中（数字换过也认得）
            self.assertEqual(got[30][rows[1]["admission_id"]], 0)               # 11 份里都有的是常见写法
            self.assertIn("g20_rare_in_real", st)
            # 跨行拼出来的片段不算：两行各取 15 字拼起来写进真实语料，合成那份的命中数不变
            two = [x for _, x in synth_export.lines_of(rows[2]) if len(x) >= 15]
            cross = two[0][-15:] + two[1][:15]
            before, _ = synth_export.rare_span_counts([rows[2]], real)
            real2 = copy.deepcopy(real)
            real2[0]["docs"][2]["text"] += "\n" + cross
            after, _ = synth_export.rare_span_counts([rows[2]], real2)
            self.assertEqual(before[30], after[30])
            out = os.path.join(tmp, "export", "SYNTH_t")
            p = run_mod("medcourse.synth_export", "--run", run, "--source", src, "--out", out, "--label", "t", "--n", "5")
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            self.assertNotIn(rare_src[:8].encode(), p.stderr)
            with open(os.path.join(out, "check.csv"), encoding="utf-8") as f:
                chk = dict(csv.reader(f))
            self.assertEqual((chk["exported"], chk["digits_to_zero"], chk["cross_line"], chk["gram_chars"]), ("5", "1", "0", "30"))
            texts = ""
            for x in os.listdir(os.path.join(out, "docs")):
                with open(os.path.join(out, "docs", x), encoding="utf-8") as f:
                    texts += f.read()
            self.assertNotIn(n(rare_src), n(texts))
            self.assertTrue(os.path.exists(out + ".tar.gz"))
            self.assertFalse(any("journey" in x or "draft" in x for x in os.listdir(out)))
            with open(os.path.join(out, "MANIFEST.sha256"), encoding="utf-8") as f:
                for line in f:
                    h, rel = line.split("  ./")
                    with open(os.path.join(out, rel.strip()), "rb") as fb:
                        self.assertEqual(hashlib.sha256(fb.read()).hexdigest(), h)
            ck = os.path.join(tmp, "export", "CHECK_t")
            p = run_mod("medcourse.synth_export", "--run", run, "--source", src, "--out", ck, "--label", "t", "--check-only")
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            self.assertEqual(os.listdir(ck), ["check.csv"])
            self.assertFalse(os.path.exists(ck + ".tar.gz"))
            p = run_mod("medcourse.synth_export", "--run", run, "--source", src, "--out", os.path.join(tmp, "e2"), "--label", "t", "--n", "21")
            self.assertIn(b"n_out_of_range", p.stderr)
            p = run_mod("medcourse.synth_export", "--run", run, "--source", src, "--out", os.path.join(tmp, "derived", "x"), "--label", "t")
            self.assertIn(b"out_under_derived_or_data", p.stderr)
            with open(os.path.join(run, "counts.json"), encoding="utf-8") as f:
                c = json.load(f)
            with open(os.path.join(run, "counts.json"), "w", encoding="utf-8") as f:
                json.dump(dict(c, mode="paired"), f)
            p = run_mod("medcourse.synth_export", "--run", run, "--source", src, "--out", os.path.join(tmp, "e3"), "--label", "t")
            self.assertIn(b"only_free_mode", p.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_plan_diag_v016_structure_zero(self):
        tmp = tempfile.mkdtemp(prefix="d_")
        try:
            st = self._obstetric_stats()
            st["dx_names"]["K40"] = {"n": 30, "items": {"腹股沟斜疝": 25}}
            st["procedures"]["K40"] = {"n": 30, "items": {}}
            del st["procedures"]["n:左侧腹股沟斜疝"]
            st["proc_family"]["K"] = {"n": 130, "items": {"腹股沟疝修补": 40, "胆囊切除": 60}}
            sp, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "d.csv")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False)
            p = run_mod("medcourse.plan_diag", "--stats", sp, "--n", "200", "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            rows = dict(csv.reader(open(out, encoding="utf-8")))
            for k in ("lateral_proc_dx_unsided_same_organ", "proc_kw_mismatch", "gest_weeks_mismatch", "cesarean_before_28w", "curettage_after_13w",
                      "placeholder_left", "proc_side_conflict", "ds_admission_facts_leak", "results_never_must"):
                self.assertEqual(rows.get(k, "0"), "0", k)
            self.assertGreater(int(rows["proc_source_family_kw"]), 0)
            self.assertGreater(int(rows["dx_side_added"]), 0)
            self.assertGreater(int(rows["chapter_O"]), 0)
            self.assertGreater(int(rows["course_focus_quiet"]), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestV017(unittest.TestCase):
    """0.1.7：合成包查重口径（见 TestV016.test_synth_export）；真实病程同口径基线；S3 探针；被 k 挡住的章；套话约束臂；拷贝不带停用的药和变化口吻。"""

    def test_probe_slots_make_s3_checkable(self):
        from medcourse import probe
        from ruler_g2 import build_real, evaluate_admission
        rows = probe.slots()
        self.assertEqual(len(rows), 2 * len(probe.BASE))
        self.assertEqual(len({r["slot_id"] for r in rows}), len(rows))
        checked = 0
        for slot in rows:
            self.assertNotIn("{", slot["dx_name"] + slot["procedure"])
            self.assertEqual(planner.side_of(slot["dx_name"]), planner.side_of(slot["procedure"]))
            j, specs, st, rng = make(slot["slot_id"], 1, slot)
            a = j["procedures"][0]
            self.assertEqual((a["name"], a["source"], a["side"]), (slot["procedure"], "slot", planner.side_of(slot["procedure"])))
            self.assertEqual(j["patient"]["primary_dx"]["name"], slot["dx_name"])
            w, docs, tally = writer.TemplateWriter(), [], {}
            for spec in specs:
                text, _ = pipeline.write_doc(j, spec, specs, st, w, rng, tally, None)
                docs.append({"doc_type": spec["doc_type"], "day_offset": spec["day"], "text": text, "doc_type_cn": spec["doc_type_cn"], "k": spec["k"], "role": spec["role"]})
            r = evaluate_admission(build_real({"admission_id": j["journey_id"], "docs": docs}))["readings"]["S3"]
            self.assertFalse(r["adm_conflict"], slot["slot_id"])
            checked += r["n_checkable"] > 0
        self.assertGreaterEqual(checked, len(rows) - 2)

    def test_probe_cli_and_first_fail_counts(self):
        tmp = tempfile.mkdtemp(prefix="pr_")
        try:
            sp, sl, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "slots.jsonl"), os.path.join(tmp, "run")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(fixtures_mc.stats(), f, ensure_ascii=False)
            p = run_mod("medcourse.probe", "--out", sl)
            self.assertEqual(p.returncode, 0, p.stderr[-300:])
            p = run_mod("medcourse.pipeline", "--stats", sp, "--mode", "paired", "--slots", sl, "--writer", "template", "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-300:])
            with open(os.path.join(out, "counts.json"), encoding="utf-8") as f:
                c = json.load(f)
            self.assertEqual((c["planned"], c["kept"] + c["discarded"]), (30, 30))
            self.assertGreater(c["kept"], 25)
            self.assertNotIn("first_fail_S3", c)                                # 模板照抄事实，侧别不会写反
            self.assertTrue(all(isinstance(v, int) for k, v in c.items() if k.startswith("first_fail_")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_blocked_chapters(self):
        code3, names_ch, stems_t = stats.Tally(), stats.Tally(), stats.Tally()
        for _ in range(40):
            code3.adm(("*", "O"))
        for c3, n in (("O80", 7), ("O82", 5)):
            code3.add(("O",), c3, n)
        names_ch.add(("O",), "孕39周待产", 3)
        stems_t.add(("O",), "孕{W}周待产", 6)
        for _ in range(30):
            code3.adm(("*", "K"))
        code3.add(("K",), "K80", 20)
        got = stats.blocked_chapters(code3, names_ch, stems_t, 11)
        self.assertEqual(set(got), {"O"})
        self.assertEqual(got["O"], {"admissions": 40, "code3_kinds": 2, "code3_max": 7, "name_kinds": 1, "name_max": 3, "stem_kinds": 1, "stem_max": 6})

    def test_no_filler_prompt_only_when_asked(self):
        j, specs, st, rng = make("F1", 2)
        s = next(x for x in specs if x["role"] == "DAILY")
        pl = compiler.compile_doc(j, s, specs, st, rng)[0]
        plain = writer.section_prompt(j, s, pl)
        self.assertNotIn("笼统", plain)
        got = writer.section_prompt(j, s, dict(pl, no_filler=True))
        self.assertEqual(got.replace(writer.NO_FILLER, ""), plain)

        class Seen(writer.TemplateWriter):
            no_filler = True
            seen = []

            def write(self, j, spec, plan, attempt):
                Seen.seen.append((spec["role"], bool(plan.get("no_filler"))))
                return super().write(j, spec, plan, attempt)

        tally = {}
        for spec in specs:
            pipeline.write_doc(j, spec, specs, st, Seen(), rng, tally, None)
        self.assertTrue(all(flag == (role in compiler.COURSE) for role, flag in Seen.seen))
        self.assertTrue(any(flag for _, flag in Seen.seen))

    def test_copy_skips_stopped_meds_and_day_bound(self):
        prev = "继续头孢曲松治疗。查体：咳嗽相关体征较前好转。结合血常规结果分析病情。向患者及家属交代病情，患者配合治疗。"
        # 0.1.9：病情趋势句（"较前好转"）不再拷
        self.assertEqual(compiler.stable_sentences(prev, ["头孢曲松"]), ["结合血常规结果分析病情。", "向患者及家属交代病情，患者配合治疗。"])
        self.assertIn("继续头孢曲松治疗。", compiler.stable_sentences(prev))
        self.assertTrue(compiler.trend_clash("查体：咳嗽相关体征较前好转。", "咳嗽较前减轻。查体：咳嗽相关体征与前相仿。"))
        self.assertFalse(compiler.trend_clash("查体：咳嗽相关体征较前好转。", "咳嗽较前减轻。"))
        got, _ = compiler.apply_copy("咳嗽较前减轻。查体：咳嗽相关体征与前相仿。", prev, 0.9, random.Random(0), ["头孢曲松"])
        self.assertNotIn("较前好转", got)
        self.assertIn("患者配合治疗", got)
        j = {"meds": [{"name": "头孢曲松", "start_day": 0, "stop_day": 2}, {"name": "氨溴索", "start_day": 1, "stop_day": 5}]}
        self.assertEqual(compiler.inactive_meds(j, 3), ["头孢曲松"])
        self.assertEqual(compiler.inactive_meds(j, 0), ["氨溴索"])
        st = fixtures_mc.stats()
        for i in range(30):
            j, specs, st2, rng = make(f"CP{i}", i, None, st)
            w, tally, prev_note = writer.TemplateWriter(), {}, None
            for spec in specs:
                _, texts = pipeline.write_doc(j, spec, specs, st2, w, rng, tally, prev_note)
                if spec["role"] in compiler.COURSE and "正文" in texts:
                    body = texts["正文"]
                    for name in compiler.inactive_meds(j, spec["day"]):
                        self.assertNotIn("继续" + name, body)
                    prev_note = {"day": spec["day"], "text": body}

    def test_out_diag_real_mode_same_keys(self):
        tmp = tempfile.mkdtemp(prefix="rl_")
        try:
            src, out = os.path.join(tmp, "real.jsonl"), os.path.join(tmp, "o.csv")
            with open(src, "w", encoding="utf-8") as f:
                for i in range(12):
                    a = fixtures_mc.admission(i)
                    a["docs"].insert(2, dict(a["docs"][1], text=a["docs"][1]["text"].replace("继续当前治疗。", "病情平稳，密切观察。")))
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
            p = run_mod("medcourse.out_diag", "--real", src, "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-300:])
            with open(out, encoding="utf-8") as f:
                real = dict(csv.reader(f))
            self.assertEqual(real["pb_notes"], "24")
            self.assertEqual(real["pb_boiler_notes:密切观察"], "12")
            self.assertEqual(real["pb_admissions_with_2plus_notes"], "12")
            self.assertNotEqual(real["pb_tfidf_adj_full_x1000"], "NA")
            with open(out, encoding="utf-8") as f:
                text = f.read()
            for w in ("咳嗽", "头孢", "呼吸内科"):
                self.assertNotIn(w, text)
            run = TestV016._run(TestV016(), tmp, 12)
            syn = dict(out_diag.diagnose(run))
            self.assertTrue({k for k in real if k.startswith("pb_")} - {"pb_admissions_with_2plus_notes"} <= set(syn))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestV018(unittest.TestCase):
    """0.1.8：第七次合成病历审阅的 11 条和监督员第七轮的 4 条，能在程序里测的一条一测。"""

    def test_stats_helpers(self):
        self.assertEqual(stats.clean_dept("心血管内科南院病区年龄"), "心血管内科病区")
        self.assertEqual(stats.clean_dept("呼吸与危重症医学科一区年龄"), "呼吸与危重症医学科一区")
        self.assertEqual(stats.clean_dept("南院口腔颌面外科"), "口腔颌面外科")
        self.assertEqual(stats.clean_dept("胃肠外科一区(急腹症与疝专科)"), "胃肠外科一区(急腹症与疝专科)")
        self.assertIsNone(stats.clean_dept("年龄"))
        rx = re.compile("发热|恶心|呕吐|腹痛|寒战|胸闷")
        self.assertEqual(stats.positive_symptoms("3天前无明显诱因出现腹痛，伴恶心、呕吐，无发热、寒战。", rx), ["腹痛", "恶心", "呕吐"])
        self.assertEqual(stats.positive_symptoms("否认发热，不伴胸闷，未诉腹痛", rx), [])
        self.assertEqual(stats.route_tail("头孢曲松2g静滴，布地奈德雾化", 4), "2g静滴")
        self.assertEqual(stats.route_tail("头孢曲松、布地奈德雾化吸入", 4), "")

    def test_stats_tables_v08(self):
        lines = []
        for i in range(12):
            a = fixtures_mc.admission(i)
            hr = a["docs"][0]
            hr["text"] = hr["text"].replace("科室：呼吸内科", "科室：呼吸内科南院病区年龄：58岁").replace("无胸痛。", "无胸痛、发热、恶心。")
            lines.append(a)
        s = stats.build_stats(lines, {f"F{i:03d}": "J18" for i in range(12)}, k=11)
        self.assertEqual(s["version"], stats.STATS_VERSION)
        self.assertEqual(set(s["dept"]["J"]["items"]), {"呼吸内科病区"})
        self.assertIn("blk:J09-J18", s["dept"])
        self.assertIn("咳嗽", s["chief_symptoms"]["J"]["items"])
        self.assertNotIn("发热", s["symptoms"]["J"]["items"])
        self.assertNotIn("恶心", s["symptoms"]["J"]["items"])
        self.assertEqual(s["pb_subtypes"]["admissions"], 12)
        self.assertEqual(s["pb_subtypes"]["by_subtype"]["主治医师查房"]["notes"], 12)

    def test_block_layer_and_dept(self):
        self.assertEqual(icd.block_of("I20"), "blk:I20-I25")
        self.assertEqual(icd.block_of("K85"), "blk:K80-K87")
        self.assertEqual(icd.block_of("K05"), "blk:K00-K14")
        self.assertIsNone(icd.block_of("K"))
        self.assertEqual(planner.strata_of("胆囊结石", "K80"), ["n:胆囊结石", "K80", "blk:K80-K87", "K", "*"])
        slot = {"slot_id": "B1", "sex": "女", "age": 60, "dept": "普外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80", "los": 5, "surgical": False}
        st = fixtures_mc.stats()
        seen = set()
        for i in range(40):
            j, _, _, _ = make(f"D{i}", i, None, st)
            if j and j["patient"]["primary_dx"]["code3"] == "K80":
                seen.add(j["patient"]["dept"])
        self.assertEqual(seen, {"肝胆外科"})
        self.assertEqual(make("D1", 1, slot, st)[0]["patient"]["dept"], "普外科")

    def test_chief_from_chief_table_not_code_point(self):
        st = fixtures_mc.stats()
        # 0.1.9：主诉表只到节层，所以放在 code3 层（0.1.8 这条放在章层）
        for c3 in ("J18", "J44"):
            st["symptoms"][c3] = {"n": 100, "items": {"发热": 120, "咳嗽": 110, "咳痰": 60, "乏力": 70, "胸痛": 100}}
            st["chief_symptoms"][c3] = {"n": 100, "items": {"胸痛": 100}}
        st["dx_names_ch"]["J"]["items"] = {"社区获得性肺炎": 80}
        firsts = collections.Counter()
        for i in range(60):
            j, _, _, _ = make(f"C{i}", i, None, st)
            if j and j["patient"]["primary_dx"]["code3"].startswith("J"):
                firsts[j["chief"]["symptoms"][0]] += 1
                self.assertEqual(j["chief"]["symptoms"], ["胸痛"])
                self.assertEqual(j["daily"][0]["symptoms"][0], "胸痛")
                self.assertLessEqual(len(j["daily"][0]["symptoms"]), 5)
        self.assertGreater(firsts["胸痛"], 10)

    def test_meds_stop_at_chapter(self):
        st = fixtures_mc.stats()
        for key in ("n:社区获得性肺炎", "J", "K80"):
            st["meds"].pop(key, None)
            st["med_class"].pop(key, None)       # 0.1.9：先抽大类，大类表也一样只退到章层
        for i in range(30):
            j, _, _, _ = make(f"M{i}", i, None, st)
            if j:
                self.assertNotIn("omeprazole", [m["drug_id"] for m in j["meds"]])

    def test_anesthesia_by_procedure(self):
        cases = {"单根导管的冠状动脉造影术": "局部麻醉", "腹腔镜胆囊切除术": "全身麻醉", "左侧腹股沟疝无张力修补术": "腰硬联合麻醉", "子宫下段剖宫产术": "腰硬联合麻醉",
                 "清宫术": "静脉麻醉", "左侧大隐静脉高位结扎剥脱术": "腰硬联合麻醉", "白内障超声乳化摘除伴人工晶状体植入术": "局部麻醉", "阑尾切除术": "全身麻醉"}
        for name, want in cases.items():
            self.assertEqual(planner.anesthesia_for(name), want, name)
        for i in range(40):
            j, _, _, _ = make(f"A{i}", i)
            for a in (j or {}).get("procedures", []):
                self.assertEqual(a["anesthesia"], planner.anesthesia_for(a["name"]))

    def test_exam_findings_no_placeholder(self):
        st = fixtures_mc.stats()
        st["exams"]["*"]["items"].update({"CT|胸部": 110, "心电图|": 200})
        # 0.1.9：诊断相关只在节层以内比，所以给 J18 一张表
        st["exams"]["J18"] = {"n": 100, "items": dict(st["exams"]["J"]["items"])}
        diag = normal = 0
        for i in range(40):
            j, _, _, _ = make(f"E{i}", i, None, st)
            if not j:
                continue
            for e in j["exams"]:
                text = planner.exam_result_text(e)
                self.assertNotIn("与主要诊断相符的异常改变", text)
                if e.get("diagnostic"):
                    diag += 1
                    self.assertTrue(e["abnormal"])
                    # 0.1.9：按检查类型写"符合××表现"，不以"提示"开头（会被写成"示提示"）
                    self.assertTrue(text.startswith(("符合", "培养出")), text)
                    self.assertNotIn("？", text)
                elif "abnormal" in e and not e["items"] and e["key"] not in ("病理", "活检"):
                    normal += 1
                    self.assertEqual(text, "未培养出致病菌" if e["kind"] == "培养" else "未见明显异常")
        self.assertGreater(diag, 0)
        self.assertGreater(normal, 0)
        self.assertIsNone(planner.finding_name("恶性肿瘤维持性化学治疗"))

    def test_compiler_caps_hints_and_flags(self):
        st = fixtures_mc.stats()
        st["section_chars"]["入院记录|体格检查"] = {"q": [500, 600, 700, 800, 900], "n": 200}
        st["section_chars"]["入院记录|专科情况"] = {"q": [300, 400, 500, 600, 700], "n": 200}
        j, specs, st, rng = make("H1", 3, None, st)
        hr = compiler.compile_doc(j, specs[0], specs, st, rng)
        by = {pl["section"]: pl for pl in hr}
        self.assertLessEqual(by["体格检查"]["target_chars"], 220)
        self.assertLessEqual(by["专科情况"]["target_chars"], 100)
        self.assertIn("生命体征", by["体格检查"]["hint"])
        self.assertIn("既往史", by["现病史"]["hint"])
        prompt = writer.section_prompt(j, specs[0], by["体格检查"])
        self.assertIn("不要写“未测”", prompt)
        out = writer.TemplateWriter().write(j, specs[0], by["体格检查"], 0)
        self.assertNotIn("未测", out)
        fp = compiler.compile_doc(j, specs[1], specs, st, rng)
        self.assertTrue(any(f["kind"] == "past" and f["text"].startswith("既往") for pl in fp if pl["section"] == "病例特点" for f in pl["facts"]))
        nonsurg = 0
        for i in range(30):
            j, specs, st2, rng = make(f"N{i}", i)
            if not j:
                continue
            for s in specs:
                for pl in compiler.compile_doc(j, s, specs, st2, rng):
                    flag = pl.get("nonsurgical", False)
                    done_ops, pending_ops = compiler.op_state(j, s, specs)
                    want = pl["mode"] == "llm" and not (done_ops + pending_ops) and s["role"] in ("DS", "SUP", "DAILY")
                    self.assertEqual(flag, want)
                    nonsurg += flag
        self.assertGreater(nonsurg, 0)
        # 0.1.9：趋势句也不拷，这里换一句不带趋势词的
        self.assertEqual(compiler.stable_sentences("主任医师强调需警惕病情变化并继续监测。患者配合治疗，按时服药。"), ["患者配合治疗，按时服药。"])

    def test_gate_truncated_repeat_postop_and_repair(self):
        loop = "双肺呼吸音粗，" + "双肺底可闻及细湿啰音，双肺底可闻及Velcro啰音，" * 25 + "双肺"
        fails = gate.check_section({"target_chars": 100, "facts": [], "allowed": []}, loop)
        self.assertEqual(set(fails), {"truncated", "repeat"})
        fixed = gate.repair_section(loop, fails)
        self.assertEqual(fixed, "双肺呼吸音粗，双肺底可闻及细湿啰音，双肺底可闻及Velcro啰音。")
        self.assertEqual(gate.check_section({"target_chars": 10, "facts": [], "allowed": []}, fixed), [])
        ok = "胸片未见明显异常，心电图未见明显异常，腹部超声未见明显异常。"
        self.assertEqual(gate.check_section({"target_chars": 10, "facts": [], "allowed": []}, ok), [])
        self.assertEqual(gate.check_section({"target_chars": 10, "facts": [], "allowed": [], "nonsurgical": True}, "既往胆囊切除术后5年。"), [])
        bad = "症状好转。患者术后恢复顺利，予出院。"
        self.assertEqual(gate.check_section({"target_chars": 10, "facts": [], "allowed": [], "nonsurgical": True}, bad), ["nonsurg_postop"])
        self.assertEqual(gate.repair_section(bad, ["nonsurg_postop"]), "症状好转。")
        self.assertEqual(gate.strip_admission_leak("因胸痛入院，入院时查体无异常。入院后行冠脉造影检查，示未见明显异常。心电图示窦性心律。"),
                         "因胸痛入院，入院时查体无异常。心电图示窦性心律。")
        self.assertFalse(gate.truncated("查体合作（神志清楚。）"))

    def test_pipeline_repairs_and_no_filler2(self):
        class Loop(writer.TemplateWriter):
            def write(self, j, spec, plan, attempt):
                if plan["section"] == "体格检查":
                    return "神志清楚，" + "双肺底可闻及细湿啰音，" * 6 + "双肺"
                return super().write(j, spec, plan, attempt)
        j, specs, st, rng = make("P1", 2)
        tally = {}
        text, texts = pipeline.write_doc(j, specs[0], specs, st, Loop(), rng, tally, None)
        self.assertEqual(texts["体格检查"].count("双肺底可闻及细湿啰音"), 1)
        self.assertTrue(texts["体格检查"].endswith("。"))
        self.assertEqual(tally["sec_repeat_repaired"], 1)
        self.assertEqual(tally["sec_truncated_repaired"], 1)
        plan = {"section": "正文", "facts": [{"text": "咳嗽较前减轻"}], "target_chars": 120, "no_filler2": True}
        p = writer.section_prompt(j, specs[2], plan)
        self.assertIn("结尾不写泛泛的总结句", p)
        self.assertNotIn("无特殊不适", p)
        tmp = tempfile.mkdtemp(prefix="nf2_")
        try:
            sp, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "run")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(fixtures_mc.stats(), f, ensure_ascii=False)
            p = run_mod("medcourse.pipeline", "--stats", sp, "--mode", "free", "--n", "6", "--writer", "template", "--no-filler2", "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-300:])
            with open(os.path.join(out, "counts.json"), encoding="utf-8") as f:
                counts = json.load(f)
            self.assertEqual((counts["no_filler"], counts["no_filler2"], counts["version"]), (False, True, pipeline.VERSION))
            p = run_mod("medcourse.pipeline", "--stats", sp, "--mode", "free", "--n", "2", "--writer", "template", "--no-filler", "--no-filler2", "--out", out + "2")
            self.assertEqual(p.returncode, 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_out_diag_qa_flags(self):
        hr = ("入院记录\n姓名：[患者]　性别：女　年龄：85岁　科室：心内科年龄\n主诉：发热、呕吐1年。\n现病史：患者无明显诱因发热，既往无特殊病史。\n"
              "既往史：有高血压病史。\n体格检查：体温36.5℃，脉搏91次/分。体温未测，血压未测，神志清楚。\n")
        ds = "出院记录\n入院情况：查体所见与主要症状相符。示可见与主要诊断相符的异常改变。\n出院医嘱：术后恢复顺利，" + "注意休息，" * 3 + "定期复查\n"
        daily = "2025-02-04 13:30　日常病程记录\n患者咳嗽减轻。主任医师强调需警惕病情变化。\n"
        docs = [{"doc_type_cn": "入院记录", "text": hr}, {"doc_type_cn": "日常病程记录", "text": daily}, {"doc_type_cn": "出院记录", "text": ds}]
        flags = out_diag.qa_flags({"surgical": False}, docs)
        # 0.1.9：新口径的占位句 path_placeholder 也算上
        self.assertEqual(flags, {"truncated", "headless", "pe_untested", "placeholder", "dept_glued", "chief_fever_first", "nonsurg_postop", "past_conflict", "senior_in_daily",
                                 "path_placeholder"})
        loop = [{"doc_type_cn": "入院记录", "text": "体格检查：" + "双肺底可闻及细湿啰音，" * 4 + "心律齐。\n"}]
        self.assertEqual(out_diag.qa_flags({"surgical": True}, loop), {"repeat"})

    def test_plan_diag_v018_structure_zero_and_template_run(self):
        tmp = tempfile.mkdtemp(prefix="pd18_")
        try:
            st = TestV016()._stats()
            st["dept"]["*"]["items"]["心内科南院病区"] = 30
            sp, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "d.csv")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False)
            p = run_mod("medcourse.plan_diag", "--stats", sp, "--n", "120", "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            with open(out, encoding="utf-8") as f:
                rows = dict(csv.reader(f))
            for k in ("chief_not_in_table", "anesthesia_mismatch", "exam_placeholder", "level_meds_global", "dept_label_glued"):
                self.assertEqual(rows.get(k, "0"), "0", k)
            self.assertIn("exam_diagnostic", rows)
            self.assertIn("level_dept_block", rows)
            run = TestV016._run(TestV016(), tmp, 24)
            rows = dict(out_diag.diagnose(run))
            self.assertEqual(rows["qa_admissions"], 24)
            for f in ("repeat", "truncated", "headless", "pe_untested", "placeholder", "dept_glued", "nonsurg_postop", "senior_in_daily"):
                self.assertEqual(rows[f"qa_adm_{f}"], 0, f)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestV019(unittest.TestCase):
    """0.1.9：第八次合成病历审阅的十条、审查员第八轮补充的五条，能在程序里测的一条一测。"""

    def _st(self):
        """夹具加三种住院：脑梗死（I63，检查表在 code3 层）、化疗（Z51，检查表只在章层）、儿童肺炎沿用夹具。全部虚构。"""
        st = fixtures_mc.stats()
        st["joint"]["items"].update({"男|65+|I": 80, "女|40-65|Z": 60})
        st["code3"]["I"] = {"n": 80, "items": {"I63": 80}}
        st["code3"]["Z"] = {"n": 60, "items": {"Z51": 60}}
        st["dx_names"]["I63"] = {"n": 80, "items": {"脑梗死": 80}}
        st["dx_names"]["Z51"] = {"n": 60, "items": {"恶性肿瘤维持性化学治疗": 60}}
        st["comorbid"]["Z51"] = {"n": 60, "items": {"肺恶性肿瘤": 40}}
        st["symptoms"]["I63"] = {"n": 80, "items": {"肢体无力": 60, "头晕": 40}}
        st["chief_symptoms"]["I63"] = {"n": 80, "items": {"肢体无力": 60}}
        st["exams"]["I63"] = {"n": 80, "items": {"血常规|": 78, "心脏超声|": 60, "超声|泌尿": 50, "磁共振|头": 70, "CT|头": 60, "尿常规|": 60, "心电图|": 75}}
        st["exams"]["*"]["items"].update({"心脏超声|": 40, "超声|泌尿": 30, "磁共振|头": 20, "CT|头": 30, "尿常规|": 150, "心电图|": 200, "骨髓穿刺|": 15, "活检|": 15})
        st["exams"]["Z"] = {"n": 60, "items": {"血常规|": 58, "骨髓穿刺|": 30, "活检|": 30, "CT|腹": 30}}
        st["med_class"]["Z51"] = {"n": 60, "items": {"胃肠动力和止吐药": 50, "糖皮质激素": 45, "抗肿瘤药": 35}}
        st["med_class"]["I63"] = {"n": 80, "items": {"抗血小板药": 70, "调脂药": 65}}
        st["med_class"]["*"]["items"].update({"胃肠动力和止吐药": 90, "糖皮质激素": 80, "抗肿瘤药": 36, "抗血小板药": 100, "调脂药": 90})
        st["med_class_drug"].update({"胃肠动力和止吐药": {"n": 90, "items": {"ondansetron": 80}}, "糖皮质激素": {"n": 80, "items": {"dexamethasone": 70}},
                                     "抗肿瘤药": {"n": 36, "items": {"oxaliplatin": 20, "docetaxel": 14}}, "抗血小板药": {"n": 100, "items": {"aspirin": 90}},
                                     "调脂药": {"n": 90, "items": {"atorvastatin": 80}}})
        for did, alias, dose, freq, route in (("ondansetron", "昂丹司琼", "8mg", "bid", "静推"), ("dexamethasone", "地塞米松", "5mg", "qd", "静推"),
                                              ("oxaliplatin", "奥沙利铂", "200mg", "", "静滴"), ("docetaxel", "多西他赛", "120mg", "", "静滴"),
                                              ("aspirin", "阿司匹林", "100mg", "qd", "口服"), ("atorvastatin", "阿托伐他汀", "20mg", "qn", "口服")):
            for part, v in (("alias", alias), ("dose", dose), ("freq", freq), ("route", route)):
                st[f"med_{part}"][did] = {"n": 50, "items": {v: 50}}
        return st

    def _journeys(self, st, n=150, prefix="V"):
        out = []
        for i in range(n):
            j, specs, st2, rng = make(f"{prefix}{i}", i, None, st)
            if j:
                out.append((j, specs, st2, rng))
        return out

    def test_drug_class_covers_lexicon(self):
        from medcourse import drug_class
        with open(os.path.join(os.environ["BENCH_G3"], "ruler_g3", "data", "drug_lexicon.csv"), encoding="utf-8") as f:
            ids = [r["drug_id"] for r in csv.DictReader(f)]
        self.assertEqual(len(ids), 280)
        self.assertEqual([d for d in ids if drug_class.drug_class(d) == "其他"], [])
        for d in ("aspirin", "clopidogrel", "metoprolol", "atorvastatin", "amlodipine", "insulin", "warfarin"):
            self.assertTrue(drug_class.is_chronic(d), d)
        for d in ("ceftriaxone", "omeprazole", "dexamethasone", "lmwh", "haloperidol"):
            self.assertFalse(drug_class.is_chronic(d), d)
        self.assertNotIn("疗程已足", drug_class.stop_reason("抗血小板药"))

    def test_stats_v09_tables(self):
        lines = []
        for i in range(12):
            a = fixtures_mc.admission(i)
            a["docs"][0]["text"] = a["docs"][0]["text"].replace("主诉：咳嗽3天。", "主诉：肛门肿物脱出3天，伴便血。")
            a["docs"][2]["text"] = a["docs"][2]["text"].replace("出院情况：无咳嗽。", "出院情况：无咳嗽，血压88/60mmHg，脉搏80次/分。")
            lines.append(a)
        s = stats.build_stats(lines, {f"F{i:03d}": "J18" for i in range(12)}, k=11)
        self.assertEqual(s["version"], stats.STATS_VERSION)
        self.assertIn("肛门肿物", s["chief_symptoms"]["J"]["items"])
        self.assertIn("便血", s["chief_symptoms"]["J"]["items"])
        self.assertEqual(s["lab_state"]["J"]["items"].get("WBC|abn"), 12)                    # 白细胞 12.1 在范围外
        self.assertEqual(s["lab_state"]["J"]["items"].get("K|norm"), 12)
        self.assertEqual(set(s["med_class"]["J"]["items"]), {"抗菌药", "祛痰镇咳药"})
        self.assertEqual(s["med_class_drug"]["抗菌药"]["items"], {"ceftriaxone": 12})
        self.assertEqual(s["note_general"]["n"], 12)
        self.assertEqual(s["real_rates"]["cbc_admissions"], 12)
        self.assertEqual(s["real_rates"]["cbc_any_extreme"], 12)
        self.assertEqual(s["real_rates"]["zero_meds"], 0)
        self.assertEqual((s["real_rates"]["ds_vitals_adult_read"], s["real_rates"]["ds_vitals_adult_off"]), (12, 12))
        self.assertEqual(s["pb_subtypes"]["by_los"]["4-7"]["sub"]["主治医师查房"], {"admissions": 12, "notes": 12})
        rows = {r["table"]: r for r in stats.summary_rows(s)}
        self.assertIn("note_general:二便_per_mille", rows)

    def test_chief_stops_at_block_and_treatment(self):
        st = self._st()
        sources = collections.Counter()
        for j, specs, st2, rng in self._journeys(st, 160, "CH"):
            c = j["chief"]
            sources[c["source"]] += 1
            self.assertNotIn(c["source"], ("chapter", "global"))
            code = j["patient"]["primary_dx"]["code3"]
            if code == "J18":
                self.assertEqual(c["source"], "code3")
            if code == "K80":
                self.assertEqual(c["source"], "block")
            if code == "Z51":
                self.assertEqual((c["source"], c["kind"], c["key"]), ("treatment", "化疗", "化疗"))
                self.assertTrue(planner.chief_text(j).startswith("确诊"))
                self.assertIn("为行第", planner.chief_text(j))
                hr = compiler.compile_doc(j, specs[0], specs, st2, rng)
                by = {pl["section"]: pl for pl in hr}
                self.assertEqual(by["主诉"]["program_text"], planner.chief_text(j) + "。")
                self.assertEqual(by["现病史"]["facts"][0]["key"], "化疗")
                w, tally = writer.TemplateWriter(), {}
                for s in specs:
                    pipeline.write_doc(j, s, specs, st2, w, rng, tally, None)
        self.assertGreater(sources["treatment"], 3)
        self.assertGreater(sources["code3"], 3)

    def test_exam_site_routine_guarantee(self):
        self.assertFalse(planner.diag_eligible("尿常规"))
        self.assertFalse(planner.diag_eligible("心电图"))
        self.assertFalse(planner.diag_eligible("血气"))
        self.assertTrue(planner.diag_eligible("心脏超声"))
        self.assertEqual(planner.dx_organs("脑梗死", "I63"), {"脑"})
        self.assertEqual(planner.dx_organs("慢性肾脏病5期", "N18"), {"泌尿"})
        self.assertEqual(planner.dx_organs("支气管哮喘", "J"), {"肺"})
        self.assertFalse(planner.site_ok("心脏超声", None, {"脑"}))
        self.assertFalse(planner.site_ok("超声", "泌尿", {"脑"}))
        self.assertTrue(planner.site_ok("磁共振", "头", {"脑"}))
        self.assertTrue(planner.site_ok("CT", "胸部", {"肺"}))                                  # 部位写法按评测台词表归一
        self.assertFalse(planner.site_ok("CT", "", {"肺"}))
        st = self._st()
        strata = planner.strata_of("脑梗死", "I63")
        cbc = [{"key": "血常规", "region": None, "items": [1]}]
        self.assertEqual(planner.ensure_diagnostic(st, strata, cbc, "脑梗死", {"脑"}), ("磁共振", "头"))
        n_stroke = 0
        for j, *_ in self._journeys(st, 160, "EX"):
            if j["patient"]["primary_dx"]["code3"] != "I63":
                continue
            n_stroke += 1
            diag = {(e["key"], e["region"]) for e in j["exams"] if e.get("diagnostic")}
            self.assertTrue(diag and diag <= {("磁共振", "头"), ("CT", "头")}, diag)
            for e in j["exams"]:
                text = planner.exam_result_text(e)
                self.assertFalse(text.startswith("提示"), text)
                if e["key"] in ("心脏超声", "尿常规") or e["region"] == "泌尿":
                    self.assertEqual(text, "未见明显异常")
        self.assertGreater(n_stroke, 5)

    def test_bare_dedupe_and_invasive(self):
        mk = lambda k, r: {"key": k, "region": r, "name": (planner.REGION_NAME.get(r, r) if r else "") + k}  # noqa: E731
        got = planner.dedupe_bare([mk("CT", "胸"), mk("CT", None)], {"肺"})
        self.assertEqual([e["name"] for e in got], ["胸部CT"])
        got = planner.dedupe_bare([mk("CT", None)], {"肺"})
        self.assertEqual((got[0]["region"], got[0]["name"]), ("胸", "胸部CT"))
        self.assertEqual(len(planner.dedupe_bare([mk("CT", "腹"), mk("CT", None)], set())), 1)
        self.assertEqual(len(planner.dedupe_bare([mk("CT", None)], set())), 1)
        st = self._st()
        n_z = 0
        for j, *_ in self._journeys(st, 160, "IV"):
            kinds = [e["kind"] for e in j["exams"] if e["kind"] in planner.INVASIVE and e["name"] != "术后病理"]
            if j["patient"]["primary_dx"]["code3"] == "Z51":
                n_z += 1
                self.assertEqual(kinds, [])                                                    # 章层的骨髓穿刺、活检不拿来凑数
            self.assertEqual(len(kinds), len(set("组织" if k in ("取材", "病理") else k for k in kinds)))
            regional = {e["key"] for e in j["exams"] if e["region"]}
            self.assertFalse(any(e["key"] in planner.REGIONAL_KEYS and not e["region"] and e["key"] in regional for e in j["exams"]))
        self.assertGreater(n_z, 3)

    def test_labs_rates_correlation_recheck(self):
        st = fixtures_mc.stats()
        jst = planner.strata_of("社区获得性肺炎", "J18")
        self.assertAlmostEqual(planner.lab_rate(st, jst, "WBC"), 50 / 120)
        self.assertAlmostEqual(planner.lab_rate(st, jst, "HGB"), 10 / 120)                   # 章层只有"正常"一格：上界 (k-1)/(n+k-1)
        self.assertEqual(planner.lab_rate({}, jst, "WBC"), planner.P_LAB_ABN_DEFAULT)
        rng = random.Random(5)
        for _ in range(200):
            x = planner._lab_value(st, rng, "WBC", False)
            self.assertEqual(x["flag"], "正常")
            y = planner._lab_value(st, rng, "HGB", True, rng.random(), 0.1)
            self.assertNotEqual(y["flag"], "正常")
        both = w = n_ = 0
        for i in range(3000):
            ctx = planner.lab_ctx(random.Random(i), st, jst)
            r = random.Random(10 ** 6 + i)
            a, b = planner.draw_lab(r, "WBC", ctx)["flag"] != "正常", planner.draw_lab(r, "NEUT", ctx)["flag"] != "正常"
            w += a
            n_ += b
            both += a and b
        self.assertAlmostEqual(w / 3000, 50 / 120, delta=0.04)                                 # 单项比例不变
        self.assertGreater(both / 3000, (w / 3000) * (n_ / 3000) + 0.03)                       # 各项一起偏
        hi = {"key": "WBC", "value": "15.0"}
        for d in range(1, 6):
            v = float(planner.recheck_value(random.Random(d), hi, d, 5)["value"])
            self.assertTrue(9.5 <= v < 15.0, v)
        ok = {"key": "WBC", "value": "6.0"}
        self.assertEqual(planner.recheck_value(random.Random(1), ok, 3, 5)["flag"], "正常")
        for j, *_ in self._journeys(st, 120, "LB"):
            hgb = [float(x["value"]) for e in j["exams"] if e["key"] == "血常规" for x in e["items"] if x["key"] == "HGB"]
            if len(hgb) >= 2:
                self.assertLess(max(hgb) - min(hgb), 30)

    def test_meds_core_class_and_chronic(self):
        st = self._st()
        n_z = n_i = 0
        for j, specs, st2, rng in self._journeys(st, 160, "MD"):
            code = j["patient"]["primary_dx"]["code3"]
            classes = {m["class"] for m in j["meds"]}
            if code == "Z51":
                n_z += 1
                self.assertIn("抗肿瘤药", classes)                                             # 按 Lift 取主治大类，不是频率最高的止吐药
            if code == "J18" and j["patient"]["primary_dx"]["name"] == "社区获得性肺炎" and j["patient"]["age"] >= 18:
                self.assertIn("抗菌药", classes)
            for m in j["meds"]:
                if m["chronic"]:
                    self.assertEqual((m["start_day"], m["stop_day"]), (0, j["los"]))
                    self.assertTrue(m["discharge"])
                if m["core"] and not m["chronic"]:
                    self.assertGreaterEqual(m["stop_day"], min(j["los"], round(planner.CORE_MIN_SHARE * j["los"])))
            if code == "I63":
                n_i += 1
                self.assertTrue(any(m["chronic"] for m in j["meds"]))
                for s in specs:
                    for pl in compiler.compile_doc(j, s, specs, st2, rng):
                        self.assertFalse(any("疗程已足" in f["text"] for f in pl["facts"]))
        self.assertGreater(n_z, 3)
        self.assertGreater(n_i, 3)

    def test_vitals_discharge_and_children(self):
        st = fixtures_mc.stats()
        n_child = 0
        for j, *_ in self._journeys(st, 160, "VT"):
            p, last = j["patient"], j["daily"][-1]
            if p["age"] >= 18 and j["los"] >= 1:
                self.assertFalse(stats.ds_vitals_off(last["SBP"], last["P"], last["R"]))
                self.assertTrue(36.0 <= last["T"] <= 37.2 and 60 <= last["P"] <= 100)
            cr = planner.child_range(p["age"])
            if cr:
                n_child += 1
                d0 = j["daily"][0]
                self.assertTrue(cr["SBP"][0] <= d0["SBP"] <= cr["SBP"][1], (p["age"], d0["SBP"]))
                self.assertTrue(cr["P"][0] <= d0["P"] <= cr["P"][1])
        self.assertGreater(n_child, 3)
        st["vitals"]["child:1-6"] = {"SBP": {"q": [88, 90, 92, 94, 96], "n": 50}}
        for j, *_ in self._journeys(st, 160, "VT"):
            if 1 <= j["patient"]["age"] < 6:
                self.assertTrue(88 <= j["daily"][0]["SBP"] <= 96)                                # 统计表有本年龄段的先用

    def test_compiler_timeline_anesthesia_and_facts(self):
        st = fixtures_mc.stats()
        before = after = general = notes = 0
        for j, specs, st2, rng in self._journeys(st, 200, "TL"):
            post = next((s["k"] for s in specs if s["role"] == "POST"), None)
            for s in specs:
                pls = compiler.compile_doc(j, s, specs, st2, rng)
                if s["role"] not in compiler.COURSE:
                    continue
                facts = [f for pl in pls for f in pl["facts"]]
                notes += 1
                general += any(f["kind"] == "general" for f in facts)
                self.assertFalse(any("相关体征" in f["text"] for f in facts))
                for a in j["procedures"]:
                    if s["day"] == a["do_day"] and s["k"] < post:
                        before += 1
                        self.assertFalse(any(f["kind"] == "procedure" for f in facts))
                        self.assertTrue(any(f["text"].startswith("今日拟行") for f in facts))
                    if s["day"] == a["do_day"] and s["k"] > post:
                        after += 1
                        self.assertIn(f"今日在{a['anesthesia']}下行{a['name']}", [f["text"] for f in facts])
        self.assertGreater(before, 0)
        self.assertGreater(after, 0)
        self.assertTrue(0.15 < general / notes < 0.5, general / notes)
        self.assertIn("穿刺点", compiler.postop_watch({"anesthesia": "局部麻醉"}, 1))
        self.assertNotIn("切口", compiler.postop_watch({"anesthesia": "局部麻醉"}, 4))
        self.assertIn("切口", compiler.postop_watch({"anesthesia": "全身麻醉"}, 1))
        slot = {"slot_id": "PCI1", "sex": "男", "age": 62, "dept": "心内科", "dx_name": "冠状动脉粥样硬化性心脏病", "dx_code3": "I25", "los": 5,
                "surgical": True, "surgery_day": 2, "procedure": "经皮冠状动脉支架植入术"}
        j, specs, st2, rng = make("PCI1", 3, slot, st)
        self.assertEqual(j["procedures"][0]["anesthesia"], "局部麻醉")
        for s in specs:
            for pl in compiler.compile_doc(j, s, specs, st2, rng):
                self.assertFalse(any(re.search("切口|引流", f["text"]) for f in pl["facts"]), s["role"])
        self.assertEqual(compiler.stable_sentences("咳嗽与前日相仿。患者配合治疗，按时服药。"), ["患者配合治疗，按时服药。"])
        j, specs, st2, rng = make("CB1", 1, None, st)
        self.assertTrue(set(j["chief"]["symptoms"]) <= set(compiler.copy_banned(j, 1)))
        self.assertNotIn("另", compiler.HINT_HPI)
        self.assertIn("病史", compiler.HINT_PE)
        self.assertIn("心电图", compiler.HINT_PE)

    def test_compiler_shared_facts(self):
        st = fixtures_mc.stats()
        n_diff = n_round = 0
        for j, specs, st2, rng in self._journeys(st, 120, "SF"):
            fp = {pl["section"]: pl for pl in compiler.compile_doc(j, specs[1], specs, st2, rng)}
            dx = fp["拟诊讨论"]["facts"][0]["text"]
            self.assertIn(planner.chief_text(j), dx)
            for e in compiler.visible(j, 0)["results"]:
                self.assertIn(e["name"], dx)
            self.assertEqual(fp["拟诊讨论"]["hint"], compiler.HINT_DISCUSS)
            if j["differentials"]:
                n_diff += 1
                pl = fp["鉴别诊断"]
                self.assertEqual(pl["hint"], compiler.HINT_DIFF)
                self.assertIn(planner.chief_text(j), pl["facts"][0]["text"])
                for n in j["differentials"]:
                    self.assertTrue(any(f"没有{n}的典型症状" in f["text"] for f in pl["facts"]))
            ds = next(s for s in specs if s["role"] == "DS")
            adm = next(pl for pl in compiler.compile_doc(j, ds, specs, st2, rng, {"hr_pe": "双肺呼吸音粗，可闻及湿啰音。"}) if pl["section"] == "入院情况")
            texts = [f["text"] for f in adm["facts"]]
            self.assertIn("入院时体格检查：双肺呼吸音粗，可闻及湿啰音。", texts)
            self.assertFalse(any("专科查体与主要症状相符" in t for t in texts))
            hr = {pl["section"]: pl for pl in compiler.compile_doc(j, specs[0], specs, st2, rng)}
            self.assertTrue(hr["专科情况"]["pe_ref"])
            for s in specs:
                if s["role"] != "SUP":
                    continue
                prev = compiler.prev_course_day(s, specs)
                new = {m["name"] for m in j["meds"] if prev < m["start_day"] <= s["day"]}
                rnd = compiler.course_focus(j, s, specs, random.Random(0))[0][1]
                n_round += 1
                self.assertFalse(any(f"继续{n}" in rnd for n in new), rnd)
        self.assertGreater(n_diff, 5)
        self.assertGreater(n_round, 5)

    def test_gate_pe_history_hint_leak(self):
        pe = {"section": "体格检查", "target_chars": 10, "facts": [], "allowed": [], "hint": "x"}
        text = "神志清楚，否认高血压病史，双肺呼吸音清。"
        self.assertEqual(gate.check_section(pe, text), ["pe_history"])
        self.assertEqual(gate.repair_section(text, ["pe_history"]), "神志清楚，双肺呼吸音清。")
        self.assertEqual(gate.repair_section("双肺呼吸音清。患者自述胸闷。", ["pe_history"]), "双肺呼吸音清。")    # 不补出"。。"
        # 0.1.9 体格检查写检查结果只计数、不进闸门；0.1.12 老板批准进节级闸门（pe_result，见 TestV0112）
        self.assertEqual(gate.check_section(pe, "双肺呼吸音清。心电图示窦性心律。"), ["pe_result"])
        hpi = {"section": "现病史", "target_chars": 10, "facts": [], "allowed": [], "hint": "x"}
        text = "患者咳嗽3天。既往史及过敏史详见另节。"
        self.assertEqual(gate.check_section(hpi, text), ["hint_leak"])
        self.assertEqual(gate.repair_section(text, ["hint_leak"]), "患者咳嗽3天。")
        self.assertEqual(gate.check_section(dict(hpi, hint=None), text), [])
        self.assertEqual(writer.clean("患者既往史：有高血压病史。", "既往史"), "有高血压病史。")

    def test_pipeline_pe_ref_ctx_and_ds_rewrite(self):
        from unittest import mock

        class Seen(writer.TemplateWriter):
            def __init__(self):
                super().__init__()
                self.seen = []

            def write(self, j, spec, plan, attempt):
                self.seen.append((spec["role"], plan["section"], [f["text"] for f in plan["facts"]]))
                self.hints = getattr(self, "hints", {})
                self.hints[plan["section"]] = plan.get("hint") or ""
                if plan["section"] == "体格检查":
                    return "神志清楚，双肺可闻及湿啰音。"
                return super().write(j, spec, plan, attempt)

        w = Seen()
        j, docs, fails = pipeline.generate_one("PR1", fixtures_mc.stats(), w, 7, None, {})
        self.assertEqual(fails, [])
        spec_facts = next(f for r, s, f in w.seen if s == "专科情况")
        # 0.1.11：体格检查不再当事实给，整段放进提示只作对照
        self.assertFalse(any("体格检查所见" in x for x in spec_facts), spec_facts)
        self.assertIn("同一篇已写好的体格检查（只作对照，不要照抄）：神志清楚，双肺可闻及湿啰音。", w.hints["专科情况"])
        adm_facts = next(f for r, s, f in w.seen if s == "入院情况")
        self.assertTrue(any(x.startswith("入院时体格检查：神志清楚，双肺可闻及湿啰音。") for x in adm_facts), adm_facts)
        w2 = Seen()
        real = gate.check_admission
        calls = []

        def fake(j, docs):
            calls.append(1)
            return [(0, "x")] if len(calls) == 1 else real(j, docs)

        with mock.patch.object(pipeline.gate, "check_admission", side_effect=fake):
            j, docs, fails = pipeline.generate_one("PR1", fixtures_mc.stats(), w2, 7, None, {})
        ds_secs = {"入院情况", "诊疗经过", "出院情况", "出院医嘱"}
        self.assertEqual({s for r, s, _ in w.seen if r == "DS"}, ds_secs)
        self.assertEqual(sum(1 for r, s, _ in w.seen if r == "DS" and s == "入院情况"), 1)
        self.assertEqual(sum(1 for r, s, _ in w2.seen if r == "DS" and s == "入院情况"), 2)      # 入院记录重写，出院记录跟着重写
        self.assertEqual({s for r, s, _ in w2.seen[len(w.seen):] if r == "DS"}, ds_secs)

    def test_out_diag_v019_flags(self):
        j = {"surgical": True, "patient": {"primary_dx": {"name": "脑梗死", "organs": ["脑"]}}, "meds": [{"name": "阿托伐他汀", "drug_id": "atorvastatin"}],
             "procedures": [{"name": "冠状动脉造影术", "anesthesia": "局部麻醉", "do_day": 2}]}
        docs = [{"doc_type_cn": "入院记录", "role": "HR", "day_offset": 0,
                 "text": "入院记录\n现病史：患者头晕3天。既往史及过敏史详见另节。\n体格检查：体温36.5℃。患者自述头晕，心电图示窦性心律。\n辅助检查：尿常规提示脑梗死；心脏超声提示脑梗死。\n"},
                {"doc_type_cn": "主治医师查房记录", "role": "SUP", "day_offset": 2,
                 "text": "2025-01-03 09:00　主治医师查房记录\n今日主治医师查房。手术过程顺利。继续阿托伐他汀治疗，加用阿托伐他汀20mg。\n"},
                {"doc_type_cn": "术后首次病程记录", "role": "POST", "day_offset": 2, "text": "2025-01-03 19:00　术后首次病程记录\n切口敷料干燥。全身麻醉下行冠脉造影。\n"},
                {"doc_type_cn": "日常病程记录", "role": "DAILY", "day_offset": 3, "text": "2025-01-04 09:00　日常病程记录\n阿托伐他汀疗程已足，予停用。血红蛋白为120g/L。\n"},
                {"doc_type_cn": "出院记录", "role": "DS", "day_offset": 4, "text": "出院记录\n诊疗经过：血红蛋白85g/L。病理结果与术前诊断相符。\n"}]
        self.assertEqual(out_diag.qa_flags(j, docs), {"path_placeholder", "pe_history", "hint_leak", "routine_says_dx", "before_op_done", "chronic_course_done",
                                                      "lab_swing", "anes_conflict", "interv_incision", "pe_result", "med_conflict", "exam_site_mismatch"})
        tmp = tempfile.mkdtemp(prefix="od19_")
        try:
            run = TestV016._run(TestV016(), tmp, 24)
            rows = dict(out_diag.diagnose(run))
            for f in ("path_placeholder", "pe_history", "hint_leak", "routine_says_dx", "before_op_done", "chronic_course_done", "anes_conflict",
                      "interv_incision", "pe_result", "med_conflict", "exam_site_mismatch", "placeholder"):
                self.assertEqual(rows[f"qa_adm_{f}"], 0, f)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_plan_diag_v019_structure_zero(self):
        tmp = tempfile.mkdtemp(prefix="pd19_")
        try:
            sp, out = os.path.join(tmp, "s.json"), os.path.join(tmp, "d.csv")
            st = self._st()
            st["real_rates"] = {"cbc_admissions": 300, "cbc_any_extreme": 90}
            st["pb_subtypes"] = {"admissions": 300, "notes_q": [1, 2, 4, 6, 12], "by_subtype": {},
                                 "by_los": {"4-7": {"admissions": 120, "notes": 500, "sub": {"主治医师查房": {"admissions": 100, "notes": 160}}}}}
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False)
            p = run_mod("medcourse.plan_diag", "--stats", sp, "--n", "160", "--out", out)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            with open(out, encoding="utf-8") as f:
                rows = dict(csv.reader(f))
            for k in ("chief_from_chapter", "chief_not_in_table", "exam_diag_routine", "exam_result_tishi", "path_placeholder", "chronic_stop_before_discharge",
                      "ds_vitals_adult_off", "before_op_done_fact", "interv_incision_fact", "exam_diag_site_mismatch", "exam_bare_dup", "exam_invasive_dup",
                      "child_bp_out_of_range", "level_meds_global", "exam_placeholder", "anesthesia_mismatch"):
                self.assertEqual(rows.get(k, "0"), "0", k)
            self.assertEqual(rows["fp_diff_has_chief"], rows["fp_diff"])
            for k in ("level_chief_treatment", "level_chief_code3", "exam_diagnostic", "plan_cbc_any_extreme", "stats_real_cbc_any_extreme",
                      "stats_pb_los_4-7_主治医师查房_notes", "plan_pb_los_4-7_admissions", "meds_antineoplastic", "course_general_facts"):
                self.assertIn(k, rows)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestExtractV02(unittest.TestCase):
    """抽取 v0.2（金标 S2G v1 之后）：定义一处、两个出口和新状态；确定性纠偏。全部虚构句子。"""

    def test_definitions_single_source(self):
        self.assertEqual(set(schema.TYPE_DEFS), set(contract.EVENT_TYPES))
        self.assertEqual(set(schema.ASSERT_DEFS), set(contract.ASSERTIONS))
        self.assertEqual(contract.EVENT_TYPES[:7], contract.CORE_TYPES)
        msg = schema.pass1_messages({"section": "现病史", "text": "咳嗽3天。"}, "入院记录")[1]["content"]
        for cn, d in schema.TYPE_DEFS.values():
            self.assertIn(d, msg)
        msg2 = schema.pass2_messages({"section": "现病史", "text": "咳嗽3天。"}, "入院记录", [{"event_type": "symptom", "name": "咳嗽", "evidence": "咳嗽"}])[1]["content"]
        for cn, d in schema.ASSERT_DEFS.values():
            self.assertIn(d, msg2)
        self.assertIn("not_patient", schema.PASS2_SCHEMA["properties"]["items"]["items"]["properties"]["assertion"]["enum"])
        self.assertNotIn("rule_out", schema.PASS2_SCHEMA["properties"]["items"]["items"]["properties"]["assertion"]["enum"])
        self.assertTrue(contract.is_core({"event_type": "lab", "assertion": "done"}))
        self.assertFalse(contract.is_core({"event_type": "sign", "assertion": "done"}))
        self.assertFalse(contract.is_core({"event_type": "symptom", "assertion": "not_patient"}))
        self.assertEqual(contract.validate_event({"event_type": "symptom", "name": "咳嗽", "assertion": "rule_out", "ev_start": 0, "ev_end": 2}), [])

    def test_normalize_rules(self):
        from medcourse.extract import normalize as N

        def run(typ, name, text, ev, sec, st="done", doc="hospital_record", val=None, unit=None):
            s = text.index(ev)
            r = N.normalize_event({"event_type": typ, "name": name, "assertion": st, "value": val, "unit": unit}, text, s, s + len(ev), doc, sec)
            return r["event_type"], r["name"], r["assertion"], r.get("value"), r.get("unit")

        cases = [
            (("symptom", "腹软", "腹软，无压痛。", "腹软", "体格检查"), ("sign", "腹软", "done", None, None)),
            (("symptom", "湿啰音", "今日查房，查体：双肺可闻及湿啰音。", "双肺可闻及湿啰音", "正文", "done", "progress_note"), ("sign", "湿啰音", "done", None, None)),
            (("symptom", "咳嗽", "3天前出现咳嗽。", "咳嗽", "现病史", "history"), ("symptom", "咳嗽", "done", None, None)),
            (("symptom", "偏瘫", "脑梗死常表现为偏瘫、意识障碍。", "偏瘫", "鉴别诊断"), ("symptom", "偏瘫", "not_patient", None, None)),
            (("exam", "血常规", "血常规示白细胞12.1。", "血常规", "辅助检查"), ("lab", "血常规", "history", None, None)),
            # 0.2.1：认不出名字的检查不再归出去（彩超、胸片不在评测台词表里，0.2 在开发检查里误删 8 题），只有笼统说法归出去
            (("exam", "右肺下叶结节", "胸部CT示右肺下叶结节。", "右肺下叶结节", "辅助检查"), ("exam", "右肺下叶结节", "history", None, None)),
            (("exam", "彩超", "彩超示胆囊壁毛糙。", "彩超", "正文", "done", "progress_note"), ("exam", "彩超", "done", None, None)),
            (("exam", "相关检查", "完善相关检查。", "相关检查", "诊疗计划", "planned", "progress_note"), ("other", "相关检查", "planned", None, None)),
            (("symptom", "头晕", "既往有头晕病史。", "头晕", "既往史", "history"), ("symptom", "头晕", "history", None, None)),
            (("diagnosis", "肺结核", "不考虑肺结核。", "肺结核", "鉴别诊断", "rule_out", "progress_note"), ("diagnosis", "肺结核", "excluded", None, None)),
            (("diagnosis", "肺结核", "肺结核待排。", "肺结核", "初步诊断", "rule_out"), ("diagnosis", "肺结核", "uncertain", None, None)),
            (("exam", "胸部CT", "门诊胸部CT示右下肺斑片影。", "胸部CT", "现病史"), ("exam", "胸部CT", "history", None, None)),
            (("procedure", "气压治疗", "予气压治疗。", "气压治疗", "正文", "done", "progress_note"), ("other", "气压治疗", "done", None, None)),
            (("medication", "头孢克肟", "出院后口服头孢克肟。", "头孢克肟", "出院医嘱", "done", "discharge_summary"), ("medication", "头孢克肟", "planned", None, None)),
            # 0.3（D1）：现在仍有的慢性病算已发生，写了结束的才算既往
            (("diagnosis", "高血压病史", "有高血压病史5年。", "高血压病史", "既往史"), ("diagnosis", "高血压", "done", None, None)),
            (("diagnosis", "糖尿病", "有糖尿病10年，规律服药。", "糖尿病", "既往史", "history"), ("diagnosis", "糖尿病", "done", None, None)),
            (("diagnosis", "阑尾炎", "20年前患阑尾炎，已治愈。", "阑尾炎", "既往史", "done"), ("diagnosis", "阑尾炎", "history", None, None)),
            (("medication", "二甲双胍", "长期口服二甲双胍。", "二甲双胍", "既往史", "history"), ("medication", "二甲双胍", "done", None, None)),
            (("procedure", "阑尾切除术", "行阑尾切除术。", "阑尾切除术", "既往史"), ("procedure", "阑尾切除术", "history", None, None)),
            # 0.3：第二趟没作答的补状态
            (("diagnosis", "社区获得性肺炎", "1.社区获得性肺炎", "社区获得性肺炎", "入院诊断", None), ("diagnosis", "社区获得性肺炎", "done", None, None)),
            (("diagnosis", "肺结核", "2.肺结核？", "肺结核", "初步诊断", None), ("diagnosis", "肺结核", "uncertain", None, None)),
            (("exam", "胃镜", "拟行胃镜检查。", "胃镜", "正文", None, "progress_note"), ("exam", "胃镜", "planned", None, None)),
            # 0.3（D2）：检查做了、结果正常是已发生；没做的仍是否认
            (("exam", "胸部CT", "胸部CT未见异常。", "胸部CT", "正文", "negated", "progress_note"), ("exam", "胸部CT", "done", None, None)),
            (("exam", "胃镜", "患者拒绝行胃镜检查。", "胃镜", "正文", "negated", "progress_note"), ("exam", "胃镜", "negated", None, None)),
            # 0.3（D4）：生命体征只有五项
            (("vital", "体重", "体重65kg。", "体重65kg", "体格检查"), ("other", "体重", "done", None, None)),
            (("vital", "眼压", "眼压15mmHg。", "眼压15mmHg", "专科情况"), ("exam", "眼压", "done", None, None)),
            (("vital", "脉搏", "脉搏80次/分。", "脉搏80次/分", "体格检查"), ("vital", "脉搏", "done", None, None)),
            # 0.3：辅助检查一节改既往只管检查、化验，不碰诊断
            (("diagnosis", "肺炎", "胸部CT示肺炎。", "肺炎", "辅助检查"), ("diagnosis", "肺炎", "done", None, None)),
            # 0.3：手术操作的边界
            (("procedure", "化疗", "行第2次化疗。", "化疗", "正文", "done", "progress_note"), ("medication", "化疗", "done", None, None)),
            (("procedure", "奥沙利铂", "予奥沙利铂静滴。", "奥沙利铂", "正文", "done", "progress_note"), ("medication", "奥沙利铂", "done", None, None)),
            (("procedure", "化疗泵置入术", "行化疗泵置入术。", "化疗泵置入术", "正文", "done", "progress_note"), ("procedure", "化疗泵置入术", "done", None, None)),
            (("procedure", "免疫组化", "免疫组化示CK阳性。", "免疫组化", "正文", "done", "progress_note"), ("exam", "免疫组化", "done", None, None)),
            (("procedure", "高压氧治疗", "予高压氧治疗。", "高压氧治疗", "正文", "done", "progress_note"), ("other", "高压氧治疗", "done", None, None)),
            (("symptom", "其他不适", "无其他不适。", "其他不适", "现病史", "negated"), ("other", "其他不适", "negated", None, None)),
            (("diagnosis", "高血压病史", "否认高血压病史。", "否认高血压病史", "既往史", "negated"), ("diagnosis", "高血压", "negated", None, None)),
            (("symptom", "药物过敏史", "否认药物过敏史。", "否认药物过敏史", "既往史", "negated"), ("other", "药物过敏史", "negated", None, None)),
            (("diagnosis", "糖尿病", "父亲有糖尿病。", "糖尿病", "家族史"), ("diagnosis", "糖尿病", "not_patient", None, None)),
            (("vital", "血压", "血压130/80mmHg。", "血压130", "体格检查", "done", "hospital_record", "130"), ("vital", "血压", "done", "130/80", "mmHg")),
            (("medication", "头孢曲松", "0.9%氯化钠100ml+头孢曲松2g qd静滴。", "0.9%氯化钠100ml+头孢曲松2g", "正文", "done", "progress_note", "100", "ml"),
             ("medication", "头孢曲松", "done", "2", "g")),
            (("medication", "头孢曲松", "头孢曲松2g qd静滴。", "头孢曲松2g qd", "正文", "done", "progress_note", "2 qd", "g"), ("medication", "头孢曲松", "done", "2", "g")),
            (("diagnosis", "肺炎", "考虑肺炎可能。", "肺炎", "初步诊断", "rule_out"), ("diagnosis", "肺炎", "uncertain", None, None)),
            (("diagnosis", "肺结核", "已排除肺结核。", "肺结核", "诊疗经过", "rule_out", "discharge_summary"), ("diagnosis", "肺结核", "excluded", None, None)),
            (("symptom", "无发热", "无发热、寒战。", "无发热", "现病史"), ("symptom", "发热", "negated", None, None)),
            (("exam", "胸部CT", "拟行胸部CT检查。", "胸部CT", "正文", "done", "progress_note"), ("exam", "胸部CT", "planned", None, None)),
            (("lab", "胸部CT", "胸部CT示双肺纹理增多。", "胸部CT", "正文", "done", "progress_note"), ("exam", "胸部CT", "done", None, None)),
            (("vital", "血压", "监测血压。", "监测血压", "诊疗计划", "done", "progress_note"), ("vital", "血压", "planned", None, None)),
        ]
        for args, want in cases:
            self.assertEqual(run(*args), want, args[:2])

    def test_normalize_cli_counts_only(self):
        tmp = tempfile.mkdtemp(prefix="nm_")
        try:
            a = fixtures_mc.admission(0)
            text = a["docs"][0]["text"]
            s = text.index("双肺可闻及湿啰音")
            ev = {"contract": "mc-contract-0.1", "admission_id": a["admission_id"], "doc_idx": 0, "doc_type": "hospital_record", "event_type": "symptom",
                  "name": "湿啰音", "value": None, "unit": None, "evidence": "双肺可闻及湿啰音", "ev_start": s, "ev_end": s + 8, "section": "体格检查",
                  "seg_start": 0, "seg_end": len(text), "assertion": "done", "time_text": None}
            src, evp, out = os.path.join(tmp, "s.jsonl"), os.path.join(tmp, "e.jsonl"), os.path.join(tmp, "n.jsonl")
            for path, rows in ((src, [a]), (evp, [ev])):
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            p = run_mod("medcourse.extract.normalize", "--events", evp, "--source", src, "--out", out)
            self.assertEqual(p.returncode, 2)                                  # 真实产物只能写在派生目录下
            p = run_mod("medcourse.extract.normalize", "--events", evp, "--source", src, "--out", out, "--synthetic")
            self.assertEqual(p.returncode, 0, p.stderr[-300:])
            got = [json.loads(x) for x in open(out, encoding="utf-8")]
            self.assertEqual((got[0]["event_type"], got[0]["norm"], got[0]["norm_version"]), ("sign", ["pe_section"], "mc-norm-0.3"))
            counts = open(os.path.join(tmp, "normalize_counts.csv"), encoding="utf-8").read()
            self.assertIn("rule:pe_section,1", counts)
            self.assertNotIn("啰音", counts + p.stderr.decode("utf-8", "replace"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestV0191(unittest.TestCase):
    """0.1.9.1：第九次 S1-B 按 PREREG v10 第 3 节停止（模板 qa_adm_med_conflict 12/499）；S1-A 主诉写"不适" 49/500。"""

    def test_copy_skips_med_action_sentences(self):
        self.assertEqual(compiler.stable_sentences("加用氨溴索口服。停用头孢曲松。继续氨溴索治疗。"), ["继续氨溴索治疗。"])
        st = fixtures_mc.stats()
        # 真实统计表里很多药没有剂量：剂量为空时"加用某药口服"不带数字，0.1.9 会被当成稳定句拷进下一篇
        for did in st["med_dose"]:
            st["med_dose"][did] = {"n": 50, "items": {"": 50}}
        tmp = tempfile.mkdtemp(prefix="mc191_")
        try:
            sp, run = os.path.join(tmp, "s.json"), os.path.join(tmp, "run")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False)
            p = run_mod("medcourse.pipeline", "--stats", sp, "--mode", "free", "--n", "80", "--writer", "template", "--out", run)
            self.assertEqual(p.returncode, 0, p.stderr[-300:])
            rows = dict(out_diag.diagnose(run))
            self.assertGreater(rows["qa_admissions"], 60)
            self.assertEqual(rows["qa_adm_med_conflict"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_chief_chapter_fallback_only_when_organ_matches(self):
        st = TestV019()._st()
        # 痔（I84，节 I80-I89 → 血管）在循环系统章（→ 心），章层主诉表不能用；慢阻肺（J44 → 肺）在呼吸章，可以用
        st["joint"]["items"]["女|40-65|I"] = 120
        st["code3"]["I"]["items"]["I84"] = 120
        st["dx_names"]["I84"] = {"n": 120, "items": {"混合痔": 120}}
        st["chief_symptoms"]["I"] = {"n": 200, "items": {"胸闷": 150, "头晕": 60}}
        seen = collections.Counter()
        for i in range(200):
            j, *_ = make(f"CF{i}", i, None, st)
            if not j:
                continue
            code, src = j["patient"]["primary_dx"]["code3"], j["chief"]["source"]
            seen[(code, src)] += 1
            if code == "I84":
                self.assertEqual(src, "none")
                self.assertNotIn("胸闷", j["chief"]["symptoms"])
            if code == "J44":
                self.assertEqual(src, "chapter_organ")
                self.assertIn(j["chief"]["symptoms"][0], st["chief_symptoms"]["J"]["items"])
        self.assertGreater(seen[("I84", "none")], 3)
        self.assertGreater(seen[("J44", "chapter_organ")], 3)


class TestV0111(unittest.TestCase):
    """0.1.11（老板批准的 14 条，pkg10 验收第 5、7 节）。全部虚构数据。"""

    def _journeys(self, st, n, prefix, seed=0):
        out = []
        for i in range(n):
            rng = random.Random(seed + i)
            j = planner.plan_journey(st, rng, f"{prefix}{i:03d}")
            if j is not None:
                out.append((j, planner.schedule(j, st, rng), rng))
        return out

    def test_diff_fact_no_initial_dx(self):
        # 第 1 条：鉴别诊断只给主诉，不写"初步诊断××"（评测台的疑似触发词）
        st = fixtures_mc.stats()
        seen = 0
        for j, specs, rng in self._journeys(st, 80, "DF"):
            if not j["differentials"]:
                continue
            fp = next(s for s in specs if s["role"] == "FP")
            for pl in compiler.compile_doc(j, fp, specs, st, rng):
                if pl["section"] == "鉴别诊断":
                    seen += 1
                    self.assertFalse(any("初步诊断" in f["text"] for f in pl["facts"]), pl["facts"])
                    self.assertTrue(pl["facts"][0]["text"].startswith("患者因"))
        self.assertGreater(seen, 3)

    def test_finding_label(self):
        # 第 8 条：检查结论用类目名，去掉限定尾巴
        self.assertEqual(planner.finding_label("大脑动脉的闭塞或狭窄引起的脑梗死", "I63"), "脑梗死")
        self.assertEqual(planner.finding_label("痔不伴有并发症", "I84"), "痔")
        self.assertEqual(planner.finding_label("胆囊结石伴急性胆囊炎", "K80"), "胆囊结石")
        self.assertEqual(planner.finding_label("细菌性肺炎", "J15"), "细菌性肺炎")
        self.assertEqual(planner.finding_label("社区获得性肺炎", "J"), "社区获得性肺炎")      # 章层诊断没有类目
        self.assertIsNone(planner.finding_label("恶性肿瘤维持性化学治疗", "Z51"))

    def test_proc_exam_merge_and_pathology(self):
        # 第 2 条：检查名在术式里的，结果在手术之后；没取组织的操作不开术后病理
        self.assertTrue(planner.proc_has_exam("冠状动脉造影术", "冠脉造影"))
        self.assertTrue(planner.proc_has_exam("ERCP下胆总管取石术", "ERCP"))
        self.assertFalse(planner.proc_has_exam("腹腔镜胆囊切除术", "超声"))
        self.assertFalse(planner.takes_tissue({"name": "冠状动脉造影术"}))
        self.assertTrue(planner.takes_tissue({"name": "肠镜下结肠息肉切除术"}))
        self.assertTrue(planner.takes_tissue({"name": "腹腔镜胆囊切除术"}))
        st = fixtures_mc.stats()
        st["exams"]["K80"] = {"n": 100, "items": {"血常规|": 95, "ERCP|": 90, "超声|腹部": 60}}
        st["procedures"]["K80"] = {"n": 100, "items": {"ERCP下胆总管取石术": 80}}
        merged = 0
        for j, specs, rng in self._journeys(st, 120, "PX"):
            for a in j["procedures"]:
                for e in j["exams"]:
                    if planner.proc_has_exam(a["name"], e["key"]):
                        merged += 1
                        self.assertTrue(e["result_day"] > a["do_day"] or e["result_day"] == j["los"], (e, a))
                        self.assertEqual(e["order_day"], a["decide_day"])
                        hr = next(s for s in specs if s["role"] == "HR")
                        self.assertNotIn(e, compiler.visible(j, 0)["results"] if a["do_day"] >= 0 and e["result_day"] > 0 else [])
                self.assertFalse(any(e["name"] == "术后病理" for e in j["exams"]) and not planner.takes_tissue(a))
            self.assertEqual(planner.check_plan(j, specs), [])
        self.assertGreater(merged, 3)

    def test_differentials_same_organ(self):
        # 第 3 条：鉴别诊断和主诊断同一器官系统
        st = fixtures_mc.stats()
        st["dx_names_ch"]["K"]["items"]["牙周炎"] = 30
        st["dx_names"]["K05"] = {"n": 30, "items": {"慢性牙周炎": 30}}
        got = collections.Counter()
        for j, specs, rng in self._journeys(st, 200, "DO"):
            organs = set(j["patient"]["primary_dx"]["organs"])
            for n in j["differentials"]:
                got[n] += 1
                self.assertNotIn(n, ("牙周炎", "慢性牙周炎"))
                o = planner.dx_organs(n, "")
                self.assertTrue(not o or o & organs, (n, o, organs))
        self.assertGreater(sum(got.values()), 5)

    def test_chronic_indication_and_subclass(self):
        from medcourse import drug_class as dc
        # 第 4 条：长期药要有适应证
        self.assertFalse(dc.indicated("metformin", "急性心力衰竭；高血压"))
        self.assertTrue(dc.indicated("metformin", "急性心力衰竭；2型糖尿病"))
        self.assertTrue(dc.indicated("ceftriaxone", "急性心力衰竭"))                    # 不是长期药
        self.assertTrue(dc.indicated("olanzapine", "言语和语言发育障碍", "F80"))         # 精神和行为障碍章
        self.assertFalse(dc.indicated("olanzapine", "社区获得性肺炎", "J18"))
        self.assertTrue(dc.indicated("amlodipine", "胆囊结石；高血压"))
        # 第 9 条：亚类按 INN 词干，规则包药名词表的类别标记都对得上
        self.assertEqual(dc.subclass("ceftriaxone"), dc.subclass("ceftazidime"))
        self.assertEqual(dc.subclass("levofloxacin"), "fq")
        self.assertEqual(dc.subclass("penicillin_g"), "penicillin")
        self.assertIsNone(dc.subclass("ambroxol"))
        path = os.path.join(os.environ["BENCH_G3"], "ruler_g3", "data", "drug_lexicon.csv")
        want = {"ceph": {"ceph"}, "fq": {"fq"}, "tetra": {"tetra"}, "ras": {"pril", "sartan"}}
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for t in r["classes"].split("|"):
                    if t in want:
                        self.assertIn(dc.subclass(r["drug_id"]), want[t], r["drug_id"])
        # 规划：同类第二种药不和第一种同亚类；非主治的长期药都有适应证
        st = fixtures_mc.stats()
        st["med_class_multi"]["抗菌药"]["items"]["2+"] = 240
        st["med_class_drug"]["抗菌药"]["items"]["ceftazidime"] = 50
        st["med_alias"]["ceftazidime"] = {"n": 50, "items": {"头孢他啶": 50}}
        st["med_dose"]["ceftazidime"] = {"n": 50, "items": {"2g": 50}}
        st["med_freq"]["ceftazidime"] = {"n": 50, "items": {"bid": 50}}
        st["med_route"]["ceftazidime"] = {"n": 50, "items": {"静滴": 50}}
        pairs = 0
        for j, specs, rng in self._journeys(st, 120, "MS"):
            abx = [m for m in j["meds"] if m["class"] == "抗菌药"]
            pairs += len(abx) >= 2
            subs = [dc.subclass(m["drug_id"]) for m in abx if dc.subclass(m["drug_id"])]
            self.assertEqual(len(subs), len(set(subs)), [m["drug_id"] for m in abx])
        self.assertGreater(pairs, 3)

    def test_labs_stratified_child_and_diabetes(self):
        # 第 5、12 条：分位数节 → 章，14 岁以下取儿童层；糖化血红蛋白、血糖没有糖尿病一律正常
        st = fixtures_mc.stats()
        st["labs_by"] = {"blk:J09-J18": {"WBC": {"q": [5.0, 6.0, 7.0, 8.0, 30.0], "n": 40}},
                         "kid:J": {"WBC": {"q": [5.5, 7.0, 9.0, 11.0, 13.0], "n": 40}}}
        st["lab_state"]["J"]["items"].update({"HBA1C|abn": 100, "HBA1C|norm": 20})
        st["lab_state"]["kid:J"] = {"n": 40, "items": {"WBC|abn": 20, "WBC|norm": 20}}
        jst = planner.strata_of("社区获得性肺炎", "J18")
        adult = {"age": 50, "primary_dx": {"name": "社区获得性肺炎", "code3": "J18"}, "comorbidities": [], "past_history": []}
        kid = dict(adult, age=4)
        dm = dict(adult, past_history=["糖尿病"])
        ca = planner.lab_ctx(random.Random(1), st, jst, adult)
        self.assertEqual(ca["q_layers"], ["blk:J09-J18", "J"])
        ck = planner.lab_ctx(random.Random(1), st, jst, kid)
        self.assertEqual(ck["q_layers"], ["kid:J", "kid:*"])
        self.assertEqual(planner._lab_q(st, "WBC", ca["q_layers"])[-1], 30.0)
        self.assertEqual(planner._lab_q(st, "WBC", ck["q_layers"])[-1], 13.0)
        self.assertEqual(planner._lab_q(st, "K", ca["q_layers"]), st["labs"]["K"]["q"])   # 分层没有的退到全院
        hi_kid, abn_dm, abn_no = [], 0, 0
        for i in range(400):
            r = random.Random(i)
            x = planner.draw_lab(r, "WBC", planner.lab_ctx(random.Random(10 ** 5 + i), st, jst, kid))
            if x["flag"] == "偏高":
                hi_kid.append(float(x["value"]))
            abn_dm += planner.draw_lab(r, "HBA1C", planner.lab_ctx(random.Random(2 * 10 ** 5 + i), st, jst, dm))["flag"] != "正常"
            abn_no += planner.draw_lab(r, "HBA1C", planner.lab_ctx(random.Random(3 * 10 ** 5 + i), st, jst, adult))["flag"] != "正常"
        self.assertTrue(hi_kid and max(hi_kid) <= 13.0, max(hi_kid or [0]))                 # 儿童层的尾部，不是全院的
        self.assertGreater(abn_dm, 200)
        self.assertEqual(abn_no, 0)

    def test_child_meds_from_child_layer(self):
        # 第 12 条：14 岁以下只开儿童层用够 k 次的药
        st = fixtures_mc.stats()
        st["meds"]["kid:J"] = {"n": 30, "items": {"ambroxol": 20, "ceftriaxone": 15}}
        st["meds"]["kid:*"] = {"n": 40, "items": {"ambroxol": 25, "ceftriaxone": 20}}
        p = {"age": 4, "sex": "男", "pregnant": False, "weight_kg": 16.0, "primary_dx": {"name": "社区获得性肺炎", "code3": "J18"}, "comorbidities": [], "past_history": []}
        jst = planner.strata_of("社区获得性肺炎", "J18")
        got = set()
        for i in range(150):
            for m in planner.plan_meds(st, random.Random(i), p, jst, 6):
                got.add(m["drug_id"])
        self.assertTrue(got)
        self.assertLessEqual(got, {"ambroxol", "ceftriaxone"})
        self.assertNotIn("levofloxacin", got)

    def test_chief_term_completion(self):
        # 第 13 条：部位＋疼痛、部位＋不适、词＋修饰词连着记
        rx = re.compile("|".join(re.escape(s) for s in sorted(["疼痛", "语言发育", "语言发育迟缓", "腹痛", "发热", "便血"], key=len, reverse=True)))
        sites = stats.site_regex({"腹部": "腹", "上腹部": "腹", "腹": "腹", "胸": "胸", "头": "头"})
        self.assertEqual(stats.positive_symptoms("右上腹疼痛3天", rx, sites), ["右上腹疼痛"])
        self.assertEqual(stats.positive_symptoms("肛周疼痛3天", rx, sites), ["肛周疼痛"])
        self.assertEqual(stats.positive_symptoms("语言发育落后5月", rx, sites), ["语言发育落后"])
        self.assertEqual(stats.positive_symptoms("语言发育迟缓5月", rx, sites), ["语言发育迟缓"])
        self.assertEqual(stats.positive_symptoms("肛门不适1年，便血2天", rx, sites), ["肛门不适", "便血"])
        self.assertEqual(stats.positive_symptoms("否认肛门不适，无腹痛", rx, sites), [])
        self.assertEqual(stats.positive_symptoms("疼痛2天", rx, sites), ["疼痛"])
        self.assertEqual(stats.positive_symptoms("右上腹疼痛3天", rx), ["疼痛"])                 # 不给 sites 照旧
        lines = [fixtures_mc.admission(i) for i in range(12)]
        s = stats.build_stats(lines, {f"F{i:03d}": "J18" for i in range(12)}, k=11)
        self.assertEqual(s["chief_cov"]["J18"]["items"]["hr"], 12)
        self.assertEqual(s["chief_cov"]["J18"]["items"]["cc"], 12)
        self.assertIn("labs_by", s)

    def test_culture_bowel_and_none_reason(self):
        # 第 14 条：阳性培养至少在出院前一天回报；第 10 条：病程不给二便；第 11 条：写"不适"记原因
        st = fixtures_mc.stats()
        st["exams"]["*"]["items"]["痰培养|"] = 20
        st["exams"]["J18"] = {"n": 100, "items": {"血常规|": 95, "痰培养|": 60, "CT|胸部": 80}}
        st["chief_symptoms"].pop("K")
        pos, reasons = 0, collections.Counter()
        for j, specs, rng in self._journeys(st, 200, "CB"):
            for e in j["exams"]:
                if e["kind"] == "培养" and e.get("abnormal"):
                    pos += 1
                    self.assertTrue(j["los"] == 0 or e["result_day"] < j["los"], (e, j["los"]))
            for s in specs:
                if s["role"] in compiler.COURSE:
                    for pl in compiler.compile_doc(j, s, specs, st, rng):
                        self.assertFalse(any(f["kind"] == "general" and f["text"] in compiler.GENERAL_BOWEL for f in pl["facts"]))
            if j["chief"]["source"] == "none":
                reasons[(j["patient"]["primary_dx"]["code3"], j["chief"].get("none_reason"))] += 1
        self.assertGreater(pos, 3)
        self.assertTrue(reasons)
        self.assertTrue(all(r in ("no_chapter", "chapter_no_organ", "organ_mismatch", "chapter_table_empty") for _, r in reasons), reasons)
        self.assertTrue(any(c == "K40" and r == "chapter_table_empty" for c, r in reasons), reasons)

    def test_plan_diag_v0111_structural_zero(self):
        from medcourse import plan_diag
        rows = dict(plan_diag.diagnose(fixtures_mc.stats(), 120, 20261102))
        for k in ("diff_fact_initial_dx", "exam_before_its_proc", "path_without_tissue", "diff_organ_mismatch", "chronic_without_indication",
                  "dm_marker_without_dm", "same_subclass_pair", "child_med_not_in_child_layer", "culture_pos_on_discharge", "course_bowel_fact"):
            self.assertEqual(rows.get(k, 0), 0, k)


class TestV0112(unittest.TestCase):
    """0.1.12（老板 11-10 批准的 8 条，pkg11 验收第 9 节）。全部虚构数据。"""

    def _journeys(self, st, n, prefix, seed=0):
        return TestV0111._journeys(self, st, n, prefix, seed)

    def test_pe_result_gate_and_repair(self):
        # 第 1 条：体格检查、专科情况写检查结果进节级闸门，三次不过删分句
        for sec in ("体格检查", "专科情况"):
            pl = {"section": sec, "target_chars": 5, "facts": [], "allowed": [], "hint": None}
            text = "双肺呼吸音粗。尿常规未见异常，心电图示窦性心律。腹软。"
            self.assertIn("pe_result", gate.check_section(pl, text))
            self.assertEqual(gate.repair_section(text, ["pe_result"]), "双肺呼吸音粗。腹软。")
            self.assertNotIn("pe_result", gate.check_section(pl, "神经系统检查示双侧肌力Ⅳ级，双肺呼吸音粗。"))
        self.assertIn("pe_result", gate.FIXABLE)
        hpi = {"section": "现病史", "target_chars": 5, "facts": [], "allowed": [], "hint": None}
        self.assertNotIn("pe_result", gate.check_section(hpi, "患者咳嗽3天，入院当天血常规示白细胞升高。"))
        # 紧跟着的从属分句（"示……"）一起删
        self.assertEqual(gate.drop_pe_results("双肺呼吸音粗，心电图示窦性心律，示未见ST段改变，腹软。"), "双肺呼吸音粗，腹软。")
        self.assertEqual(gate.drop_pe_results("腹软，无压痛，血常规示白细胞升高。"), "腹软，无压痛。")

    def test_hpi_workup_facts_and_hint(self):
        # 第 1 条：现病史不再说"诊治情况"；入院当天已回报的结果当事实给，没有就说入院前未做过检查
        self.assertNotIn("诊治", compiler.HINT_HPI)
        st = fixtures_mc.stats()
        seen = collections.Counter()
        for j, specs, rng in self._journeys(st, 60, "HP"):
            hr = next(s for s in specs if s["role"] == "HR")
            pl = next(x for x in compiler.compile_doc(j, hr, specs, st, rng) if x["section"] == "现病史")
            res = [e for e in j["exams"] if e["result_day"] <= 0]
            texts = [f["text"] for f in pl["facts"]]
            if res:
                f = next(f for f in pl["facts"] if f["text"].startswith("入院当天已回报："))
                self.assertFalse(f["must"])
                self.assertTrue(set(f["values"]) <= set(pl["allowed"]))
                for e in res:
                    self.assertIn(e["name"], f["text"])
                seen["res"] += 1
            else:
                self.assertIn("入院前未做过检查", texts)
                seen["none"] += 1
        self.assertGreater(seen["res"], 5)

    def test_oral_ok(self):
        # 第 2 条：途径为空时只有有口服剂型的药出院带走
        from medcourse import drug_class
        self.assertTrue(drug_class.NO_ORAL <= set(drug_class.DRUG_CLASS), drug_class.NO_ORAL - set(drug_class.DRUG_CLASS))
        self.assertFalse(drug_class.oral_ok("cisplatin", ""))
        self.assertFalse(drug_class.oral_ok("ceftriaxone", ""))
        self.assertTrue(drug_class.oral_ok("capecitabine", ""))
        self.assertTrue(drug_class.oral_ok("omeprazole", "口服"))
        self.assertFalse(drug_class.oral_ok("omeprazole", "静滴"))

    def test_home_meds_need_oral_form(self):
        st = fixtures_mc.stats()
        for d in ("ceftriaxone", "ambroxol"):
            st["med_route"].pop(d)      # 途径为空
        home = collections.Counter()
        for j, specs, rng in self._journeys(st, 150, "HM"):
            for m in j["meds"]:
                if m["discharge"]:
                    home[m["drug_id"]] += 1
                    self.assertTrue(m.get("chronic") or drug_class_oral(m), m)
        self.assertEqual(home["ceftriaxone"], 0, home)
        self.assertGreater(home["ambroxol"], 0, home)

    def test_treatment_primary_not_suspected(self):
        # 第 6 条：化疗、放疗、复查住院的主诊断不走疑似（随机数照抽）
        from unittest import mock
        st = fixtures_mc.stats()
        with mock.patch.object(planner, "P_DX_CHANGE", 1.0):
            ctrl = [j for j, _, _ in self._journeys(st, 40, "TX") if j["los"] >= 2]
            self.assertTrue(any(d["state"] == "疑似" and d["name"] == j["patient"]["primary_dx"]["name"] for j in ctrl for d in j["dx_events"]))
            with mock.patch.object(planner, "treatment_kind", lambda name, code3: "化疗"):
                tx = [j for j, _, _ in self._journeys(st, 40, "TX") if j["los"] >= 2]
        self.assertTrue(tx)
        for j in tx:
            name = j["patient"]["primary_dx"]["name"]
            self.assertFalse(any(d["name"] == name and d["state"] == "疑似" for d in j["dx_events"]), j["dx_events"])

    def test_enforce_parts(self):
        # 第 7 条：同组化验部分 ≤ 总量
        rng = random.Random(1)
        ck, ckmb = planner._fmt_lab("CK", 29), planner._fmt_lab("CKMB", 33)
        out = planner.enforce_parts(rng, [ck, ckmb, planner._fmt_lab("LDH", 200)])
        self.assertLessEqual(float(out[1]["value"]), 29)
        self.assertEqual(out[0], ck)
        out = planner.enforce_parts(rng, [planner._fmt_lab("TC", 3.1), planner._fmt_lab("TG", 1.2), planner._fmt_lab("LDL", 3.4)])
        self.assertLessEqual(float(out[2]["value"]), 3.1)
        ok = [planner._fmt_lab("CK", 120), planner._fmt_lab("CKMB", 15)]
        state = rng.getstate()
        self.assertEqual(planner.enforce_parts(rng, ok), ok)
        self.assertEqual(rng.getstate(), state)     # 不违反时不用随机数

    def test_diag_none_reason_and_cbc_sources(self):
        # 第 4、5 条：只记不改
        e_img = {"items": [], "key": "CT", "abnormal": False}
        e_lab = {"items": [{"key": "WBC"}], "key": "血常规"}
        self.assertEqual(planner.diag_none_reason([e_img], "化疗", "x", {"肺"}), "treatment")
        self.assertEqual(planner.diag_none_reason([e_img], None, None, {"肺"}), "no_label")
        self.assertEqual(planner.diag_none_reason([e_lab, {"items": [], "key": "病理", "abnormal": True}], None, "肺炎", {"肺"}), "labs_only")
        self.assertEqual(planner.diag_none_reason([e_img], None, "肺炎", set()), "no_organ")
        self.assertEqual(planner.diag_none_reason([e_img], None, "肺炎", {"肺"}), "no_candidate")
        self.assertIsNone(planner.diag_none_reason([dict(e_img, diagnostic=True)], None, "肺炎", {"肺"}))
        st = fixtures_mc.stats()
        for j, specs, rng in self._journeys(st, 30, "CS"):
            ed = j["exam_diag"]
            self.assertEqual(ed["none_reason"] is None, any(e.get("diagnostic") for e in j["exams"]))
            self.assertEqual(set(ed["cbc"]), set(planner.PANELS["血常规"]))
            for v in ed["cbc"].values():
                self.assertIn(v["flag"], ("偏高", "偏低", "正常"))
                self.assertIn(v["q"], ("block", "chapter", "kid_all", "kid_chapter", "global", "ref"))

    def test_chief_organ_layer(self):
        # 第 3 条：章层表空、器官对不上时退到器官合并层
        st = fixtures_mc.stats()
        st["exams"]["*"]["items"]["痰培养|"] = 20
        st["chief_symptoms"].pop("K")
        organs = planner.dx_organs("", "K40")
        self.assertTrue(organs)
        o = sorted(organs)[0]
        base = collections.Counter(j["chief"]["source"] for j, _, _ in self._journeys(st, 200, "CO") if j["patient"]["primary_dx"]["code3"] == "K40")
        self.assertGreater(base["none"], 0, base)
        st["chief_symptoms"][f"org:{o}"] = {"n": 200, "items": {"腹股沟肿物": 40}}
        got = collections.Counter()
        for j, _, _ in self._journeys(st, 200, "CO"):
            if j["patient"]["primary_dx"]["code3"] == "K40":
                got[j["chief"]["source"]] += 1
                if j["chief"]["source"] == "organ":
                    self.assertEqual(j["chief"]["symptoms"][0], "腹股沟肿物")
                    self.assertEqual(j["chief"]["organ"], o)
        self.assertEqual(got["none"], 0, got)
        self.assertGreater(got["organ"], 0, got)

    def test_stats_011_tables(self):
        lines = [fixtures_mc.admission(i) for i in range(12)]
        s = stats.build_stats(lines, {f"F{i:03d}": "J18" for i in range(12)}, k=11)
        self.assertEqual(s["version"], "mc-stats-0.21")
        orgs = [x for x in s["chief_symptoms"] if x.startswith("org:")]
        self.assertTrue(orgs, list(s["chief_symptoms"]))
        self.assertEqual(s["chief_symptoms"][orgs[0]]["items"].get("咳嗽"), 12)
        self.assertEqual(s["chief_cov"][orgs[0]]["items"].get("term"), 12)
        for t in ("chief_probe", "chief_lat", "chief_base"):
            self.assertIn(t, s)
        self.assertEqual(s["chief_lat"]["J18"]["items"].get("咳嗽"), 12)

    def test_qa_meta_and_cross_repeat(self):
        # 第 8 条：先计数
        rep = "结合甲状腺功能检查结果未见明显异常提示甲状腺功能在正常范围内"
        docs = [{"role": "DAILY", "text": f"2025-01-0{i} 日常病程记录\n患者病情平稳。{rep}。"} for i in range(1, 4)]
        self.assertEqual(out_diag.qa_flags_v0112(docs), {"cross_repeat"})
        self.assertEqual(out_diag.qa_flags_v0112(docs[:2]), set())
        docs2 = [{"role": "SUP", "text": "2025-01-01 主治医师查房记录\n本次查房记录完整，内容真实，符合病历书写规范。"}]
        self.assertEqual(out_diag.qa_flags_v0112(docs2), {"meta_text"})
        self.assertIn("meta_text", out_diag.QA_FLAGS)

    def test_plan_diag_v0112(self):
        from medcourse import plan_diag
        st = fixtures_mc.stats()
        st["med_route"].pop("ceftriaxone")
        rows = dict(plan_diag.diagnose(st, 120, 20261102))
        for k in ("home_no_oral", "tx_primary_suspected", "lab_part_gt_total"):
            self.assertEqual(rows.get(k, 0), 0, k)
        self.assertTrue(any(k.startswith("cbc_WBC_q_") for k in rows))
        self.assertTrue(any(k.startswith("chief_cov_I84_") or k.startswith("chief_cov_N13_") for k in rows))


def drug_class_oral(m):
    from medcourse.drug_class import oral_ok
    return oral_ok(m["drug_id"], m["route"])


class TestV01121(unittest.TestCase):
    """0.1.12.1：result_unsourced（老板 11-10 批准）。全部虚构数据。"""

    def test_check_and_repair(self):
        pl = {"section": "病程", "target_chars": 5, "facts": [], "allowed": [], "hint": None, "result_keys": ["血常规"]}
        text = "患者咳嗽减轻。血常规示白细胞较前下降，心电图示窦性心律。继续抗感染。"
        self.assertIn("result_unsourced", gate.check_section(pl, text))
        self.assertEqual(gate.repair_section(text, ["result_unsourced"], pl), "患者咳嗽减轻。血常规示白细胞较前下降。继续抗感染。")
        self.assertEqual(gate.repair_section(text, ["result_unsourced"]), text)            # 不给 plan 不删
        self.assertNotIn("result_unsourced", gate.check_section(pl, "患者咳嗽减轻，拟行心电图检查，待结果回报。"))   # 计划口吻不算
        self.assertNotIn("result_unsourced", gate.check_section(dict(pl, result_keys=None), text))          # 出院记录不查
        self.assertIn("result_unsourced", gate.FIXABLE)
        # 程序给的待回报事实不算（评测台口吻判断见"回报后"会判成结果口吻）
        for t in ("待胸部CT、痰培养回报后进一步明确。", "已开具心电图、痰培养，待回报。", "完善相关检查：心电图、胸部CT。", "痰培养回报后调整抗生素。"):
            self.assertNotIn("result_unsourced", gate.check_section(pl, t), t)
        self.assertIn("result_unsourced", gate.check_section(pl, "复查心电图示窦性心律。"))
        self.assertIn("result_unsourced", gate.check_section(pl, "心电图示窦性心律，余检查待回报。"))

    def test_result_keys_alias(self):
        keys = gate.result_keys([{"key": "心肌肌钙蛋白", "name": "心肌肌钙蛋白"}, {"key": "CT", "name": "胸部CT"}])
        self.assertIn("肌钙蛋白", keys)
        self.assertIn("CT", keys)
        pl = {"section": "病程", "target_chars": 5, "facts": [], "allowed": [], "hint": None, "result_keys": keys}
        self.assertNotIn("result_unsourced", gate.check_section(pl, "肌钙蛋白示阴性，胸部CT示右下肺斑片影。"))

    def test_pipeline_sets_keys_and_strips_unsourced(self):
        st = fixtures_mc.stats()
        seen = []

        class Rec(writer.TemplateWriter):
            def write(self, j, spec, pl, seed):
                seen.append((spec["role"], spec["day"], pl.get("result_keys"), j))
                text = super().write(j, spec, pl, seed)
                if spec["role"] in ("SUP", "DAILY"):
                    text = text.rstrip("。") + "。脑电图示未见明显异常。"       # 规划里没有的检查，写成已回报
                return text

        tally, n = {}, 0
        for i in range(30):
            j, docs, fails = pipeline.generate_one(f"RU{i:03d}", st, Rec(), 20261102, None, tally)
            if docs is None:
                continue
            n += 1
            for d in docs:
                self.assertNotIn("脑电图示", d["text"], d["doc_type_cn"])
        self.assertGreater(n, 10)
        self.assertGreater(tally.get("sec_result_unsourced_repaired", 0), 0)
        for role, day, keys, j in seen:
            if role == "DS":
                self.assertIsNone(keys)
            else:
                vis = {e["key"] for e in j["exams"] if e["result_day"] <= day}
                self.assertTrue(vis <= set(keys), (role, day))

    def test_template_has_no_unsourced(self):
        st = fixtures_mc.stats()
        tally = {}
        for i in range(60):
            pipeline.generate_one(f"RT{i:03d}", st, writer.TemplateWriter(), 20261102, None, tally)
        self.assertEqual(tally.get("sec_result_unsourced", 0), 0, {k: v for k, v in tally.items() if k.startswith("sec_")})
        self.assertEqual(pipeline.VERSION, "s1-pipeline-0.3.4")


class TestV0113(unittest.TestCase):
    """0.1.13：S2 准入四类建表 vs 规则读数的对比（s2_table_compare）。全部虚构数据。"""

    @classmethod
    def setUpClass(cls):
        from medcourse import s2_table_compare as T
        from medcourse.rules import setup
        setup(("BENCH_G2",))
        from ruler_g2 import build_real
        cls.T, cls.build_real = T, staticmethod(build_real)
        cls.vocab = T.Vocab()

    def _ev(self, obj, adm, i, typ, needle, name, assertion="done", value=None):
        text = obj["docs"][i]["text"]
        a = text.find(needle)
        self.assertGreaterEqual(a, 0, needle)
        d = next(x for x in adm.docs if x.idx == i)
        return {"admission_id": obj["admission_id"], "doc_idx": i, "doc_type": obj["docs"][i]["doc_type"], "section": d.section_name_at(a),
                "event_type": typ, "name": name, "assertion": assertion, "evidence": needle, "ev_start": a, "ev_end": a + len(needle), "value": value}

    def _fixture(self, n=12):
        lines = [fixtures_mc.admission(i) for i in range(n)]
        for obj in lines:
            # 0.2：辅助检查里加一项非检验类的（心电图），检查的对比只比非检验类
            obj["docs"][0]["text"] = obj["docs"][0]["text"].replace("血钾4.1mmol/L。", "血钾4.1mmol/L。心电图示窦性心律。")
        events = {}
        for obj in lines:
            adm = self.build_real({"admission_id": obj["admission_id"], "docs": obj["docs"]})
            events[obj["admission_id"]] = [
                self._ev(obj, adm, 0, "symptom", "咳嗽3天", "咳嗽"),
                self._ev(obj, adm, 0, "exam", "胸部CT示右下肺斑片影", "胸部CT"),
                self._ev(obj, adm, 0, "lab", "血常规示白细胞12.1×10^9/L", "血常规"),
                self._ev(obj, adm, 0, "exam", "心电图示窦性心律", "心电图"),
                self._ev(obj, adm, 0, "vital", "T：37.8℃", "体温", value="37.8"),
                self._ev(obj, adm, 0, "vital", "BP：130/80mmHg", "血压", value="130/80"),
                self._ev(obj, adm, 0, "diagnosis", "有高血压病史5年", "高血压", assertion="history"),
                self._ev(obj, adm, 2, "diagnosis", "1.社区获得性肺炎", "社区获得性肺炎"),
                self._ev(obj, adm, 1, "lab", "头孢曲松2g", "头孢曲松")]      # 非准入类型，不进对比
        return lines, events

    def test_normalize_exam_symptom_vital(self):
        T, v = self.T, self.vocab
        text = "辅助检查：胸部CT示右下肺斑片影。患者右上腹疼痛2天。"
        a = text.find("CT示")
        e = {"event_type": "exam", "name": "CT", "evidence": "CT示右下肺斑片影", "ev_start": a, "ev_end": a + 9}
        self.assertEqual(T.normalize(v, e, text), {"jia": "CT|胸", "yi": "CT|"})      # 甲在原文位置上补出部位
        a = text.find("右上腹疼痛")
        e = {"event_type": "symptom", "name": "腹痛", "evidence": "右上腹疼痛", "ev_start": a, "ev_end": a + 5}
        got = T.normalize(v, e, text)
        self.assertEqual(got["jia"], "右上腹疼痛")
        self.assertEqual(got["yi"], "腹痛")
        self.assertEqual(T.vital_values("BP", "130/80", ""), [("SBP", 130.0), ("DBP", 80.0)])
        self.assertEqual(T.vital_values("T", None, "体温36.5℃"), [])
        self.assertEqual(T.vital_item("脉搏88次/分"), "P")
        self.assertEqual(T.vital_item("BP：120/80mmHg"), "BP")

    def test_dx_vocab(self):
        v = self.T.Vocab()
        self.assertIsNone(v.dx_from_evidence("2.高血压病3级", "高血压"))      # 没有词表时配不上
        v.set_dx_vocab({"社区获得性肺炎", "高血压病", "2型糖尿病"})
        self.assertEqual(v.dx_from_evidence("2.高血压病3级", "高血压"), "高血压病")
        self.assertEqual(v.dx_from_evidence("1.社区获得性肺炎 2.2型糖尿病", "2型糖尿病"), "2型糖尿病")
        self.assertIsNone(v.dx_from_evidence("否认冠心病", "冠心病"))

    def test_neg_bucket_and_status(self):
        T = self.T
        t = "否认高血压、糖尿病、高脂血症、慢性支气管炎、冠心病等病史。有乙肝病史。否认外伤，糖尿病多年"
        self.assertEqual(T.neg_distance_bucket(t, t.find("冠心病")), "gt10")      # 长清单，规则 10 字窗口够不着
        self.assertEqual(T.neg_distance_bucket(t, t.find("乙肝")), "none")
        self.assertEqual(T.neg_distance_bucket(t, t.rfind("糖尿病")), "le10")      # 隔着逗号
        self.assertEqual(T._status_cat(None), "absent")
        self.assertEqual(T._status_cat({"negated", "done"}), "negated")
        self.assertEqual(T._status_cat({"pre_done"}, pre=True), "pre_only")
        self.assertEqual(T._status_cat({"pre_done", "planned"}, pre=True), "planned")
        self.assertEqual(T._status_cat({"none"}), "other")

    def test_strata_and_spearman(self):
        T = self.T
        rows = [(("J",), {"a", "b"}, {"a", "c"})] * 12
        ov, sp, sk = T.strata_compare(rows, 2, 11, 3)
        self.assertEqual(ov, [0.5])
        self.assertEqual(sk, 0)
        ov, _, sk = T.strata_compare(rows[:5], 2, 11)
        self.assertEqual(ov, [])        # 层的住院数 < k 不比
        self.assertAlmostEqual(T.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertIsNone(T.spearman([1, 2], [2, 1]))

    def test_compare_fixture(self):
        T = self.T
        lines, events = self._fixture()
        rows = dict(T.compare(lines, {f"F{i:03d}": "J18" for i in range(12)}, events, k=11))
        self.assertEqual(rows["admissions"], 12)
        self.assertEqual(rows["version"] if "version" in rows else T.VERSION, "s2t-compare-0.2")
        self.assertEqual(rows["exam_chosen"], "jia")
        self.assertEqual(rows["symptom_jia_jaccard_p50"], 1.0)
        # 0.2：检查只比非检验类、去掉入院前段落。现病史里的门诊胸部 CT 两边都不算（规则记 H，S2 记在入院前段落），血常规是检验类
        self.assertEqual(rows["exam_jia_kinds_rule_mean"], 1.0)
        self.assertEqual(rows["exam_jia_kinds_ratio"], 1.0)
        self.assertEqual(rows["verdict_exam_kinds_ratio"], 1)
        self.assertEqual(rows["exam_jia_rule_all_kinds_mean"], 2.0)          # 规则全部（含检验类）：心电图、血常规
        self.assertNotIn("attr_exam_jia_s2_only_rule_tone_H", rows)
        # 化验单列
        self.assertEqual(rows["lab_jia_kinds_rule_mean"], 1.0)
        self.assertEqual(rows["lab_jia_kinds_ratio"], 1.0)
        self.assertEqual(rows["vital_jia_T_q50_diff"], 0.0)
        self.assertEqual(rows["diagnosis_jia_past_高血压_s2"], 1.0)
        # 既往史拆分："有高血压病史5年"前面同一句没有否定词，S2 也记有
        self.assertEqual(rows["attr_diagnosis_jia_past_高血压_none_pos"], 12)
        self.assertEqual(rows["three_symptom_done_hospital_record"], 12)
        self.assertEqual(rows["three_lab_done_hospital_record"], 12)
        # 药名误标的"化验"（病程里的头孢曲松）照记在三向表，但归不上词表：化验 24 条里归上 12 条
        self.assertEqual(rows["three_lab_done_progress_note"], 12)
        self.assertEqual(rows["norm_lab_jia_inev_mapped"], 12)
        for k in ("verdict_exam_all", "verdict_symptom_all", "verdict_diagnosis_all", "verdict_vital_all"):
            self.assertIn(k, rows)
        # 不出模型名称、引文、诊断名
        names = " ".join(str(k) for k in rows)
        for w in ("社区获得性肺炎", "右下肺", "白细胞", "头孢曲松"):
            self.assertNotIn(w, names)

    def test_rule_pass_matches_stats(self):
        # 规则这一边和统计表同一套读数器：逐住院读数加起来，等于 build_stats 同一层的表（压格子前后一致的项）
        T = self.T
        lines = [fixtures_mc.admission(i) for i in range(12)]
        codes = {f"F{i:03d}": "J18" for i in range(12)}
        pre = T.rule_pass(lines, codes, T.Vocab())
        s = stats.build_stats(lines, codes, k=11)
        syms = collections.Counter(x for *_, rr in pre for x in rr["syms"])
        chief = collections.Counter(x for *_, rr in pre for x in rr["chief"])
        exams = collections.Counter(x for *_, rr in pre for x in rr["exams"])
        self.assertEqual({x: n for x, n in syms.items() if n >= 11}, s["symptoms"]["J18"]["items"])
        self.assertEqual({x: n for x, n in chief.items() if n >= 11}, s["chief_symptoms"]["J18"]["items"])
        self.assertEqual({x: n for x, n in exams.items() if n >= 11}, s["exams"]["J18"]["items"])
        t_rule = [rr["vitals"]["T"] for *_, rr in pre if "T" in rr["vitals"]]
        self.assertEqual(len(t_rule), s["vitals"]["adult"]["T"]["n"])
        notes = [v for *_, rr in pre for v in rr["notes"].values()]
        self.assertEqual(round(sum(notes) / len(notes), 6), round(s["note_vitals"]["rate"], 6))

    def test_main_cli(self):
        T = self.T
        lines, events = self._fixture(12)
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "f.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
            dm = os.path.join(tmp, "dx.csv")
            with open(dm, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["admission_id", "code3"])
                w.writerows([[x["admission_id"], "J18"] for x in lines])
            ev = os.path.join(tmp, "e.jsonl")
            with open(ev, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for es in events.values() for e in es)
            out = os.path.join(tmp, "o.csv")
            argv = sys.argv
            sys.argv = ["x", "--path", p, "--dx-map", dm, "--events", ev, "--out", out]
            try:
                T.main()
            finally:
                sys.argv = argv
            with open(out, encoding="utf-8") as f:
                rows = dict(csv.reader(f))
            self.assertEqual(rows["version"], T.VERSION)
            self.assertEqual(rows["admissions"], "12")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestV020(unittest.TestCase):
    """0.2.0（M2 构建方案 v1）：S2 甲换表与多折 CLI、ANESTHESIA_RULES 重排与空麻醉、既往史全称与儿童体健、
    功能检查去诊断名、"脱出"与整句否定回退、dx_settled_early 闸门与术前诊断问号、诊断计数。全部虚构数据。"""

    def test_anesthesia_rules_and_empty_procedure(self):
        cases = {
            "腹腔镜腹股沟疝修补术": "全身麻醉",
            "腹腔镜胆囊切除术": "全身麻醉",
            "胸腔镜左肺叶切除术": "全身麻醉",
            "宫腔镜下子宫内膜息肉切除术": "静脉麻醉",
            "清宫术": "静脉麻醉",
            "自体动静脉内瘘成形术": "局部麻醉",
            "单根导管的冠状动脉造影术": "局部麻醉",
            "右侧白内障超声乳化摘除联合人工晶体植入术": "局部麻醉",
            "内镜下结肠息肉切除术": "局部麻醉",
            "混合痔外剥内扎术": "腰硬联合麻醉",
            "血栓性外痔切除术": "腰硬联合麻醉",
            "肛瘘切除术": "腰硬联合麻醉",
            "左侧腹股沟疝无张力修补术": "腰硬联合麻醉",
            "经尿道前列腺电切术": "腰硬联合麻醉",
            "左侧股骨粗隆间骨折切开复位内固定术": "腰硬联合麻醉",
            "左侧桡骨远端骨折切开复位内固定术": "全身麻醉",
            "阑尾切除术": "全身麻醉",
            "右侧甲状腺腺叶切除术": "全身麻醉",
            "康复治疗": "",
            "气压治疗": "",
        }
        for name, want in cases.items():
            self.assertEqual(planner.anesthesia_for(name), want, name)
        self.assertFalse(planner.takes_tissue({"name": "康复治疗"}))
        self.assertTrue(planner.takes_tissue({"name": "血栓性外痔切除术"}))
        self.assertTrue(compiler.no_incision({"name": "康复治疗", "anesthesia": ""}))
        j, specs, st, rng = make("AN0", 2)
        j["surgical"] = True
        j["surgery_day"] = 1
        j["procedures"] = [{"id": "a1", "kind": "手术", "name": "康复治疗", "side": "无", "decide_day": 0, "do_day": 1, "anesthesia": "", "source": "key"}]
        specs = planner.schedule(j, st, random.Random(3))
        docs = []
        for s in specs:
            pls = compiler.compile_doc(j, s, specs, st, random.Random(4))
            txt = compiler.assemble(j, s, specs, pls, {p["section"]: "过程顺利。" for p in pls if p["mode"] == "llm"})
            self.assertNotIn("在下行", txt)
            self.assertNotIn("麻醉：，", txt)
            if s["role"] in ("PRE", "POST"):
                self.assertNotIn("麻醉方式", txt)
            docs.append({"role": s["role"], "day_offset": s["day"], "text": txt})
        hr_txt = next(d["text"] for d in docs if d["role"] == "HR")
        self.assertNotIn("anes_conflict", out_diag.qa_flags_v019(j, docs, hr_txt))
        docs_bad = copy.deepcopy(docs)
        docs_bad[-1]["text"] += "在全身麻醉下行康复治疗。\n"
        self.assertIn("anes_conflict", out_diag.qa_flags_v019(j, docs_bad, hr_txt))

    def test_past_synonyms_and_child_past_text(self):
        p_cad = {"age": 62, "primary_dx": {"name": "冠状动脉粥样硬化性心脏病"}, "comorbidities": [], "past_history": []}
        self.assertNotIn("冠心病", compiler.past_text(p_cad))
        st = fixtures_mc.stats()
        p_drawn = planner.draw_patient(st, random.Random(7), "CAD1",
                                       slot={"sex": "男", "age": 62, "dept": "心内科", "dx_name": "冠状动脉粥样硬化性心脏病", "dx_code3": "I25"})
        self.assertIn("冠心病", p_drawn["past_history"])
        self.assertNotIn("否认", compiler.past_text(p_drawn).split("冠心病")[-1] if "否认" not in compiler.past_text(p_drawn) else "")
        self.assertFalse("冠心病" in (compiler.past_text(p_drawn).split("否认", 1)[1] if "否认" in compiler.past_text(p_drawn) else ""))
        p_kid = {"age": 3, "primary_dx": {"name": "社区获得性肺炎"}, "comorbidities": [], "past_history": []}
        self.assertEqual(compiler.past_text(p_kid), "既往体健")
        p_kid_h = {"age": 5, "primary_dx": {"name": "支气管哮喘"}, "comorbidities": [], "past_history": ["哮喘"]}
        self.assertEqual(compiler.past_text(p_kid_h), "有哮喘病史")

    def test_functional_exam_no_dx_name(self):
        st = fixtures_mc.stats()
        st["exams"]["J18"] = {"n": 100, "items": {"肺功能|": 80, "CT|胸部": 90}}
        st["exams"]["*"]["items"].update({"肺功能|": 30, "CT|胸部": 100})
        exams = [{"id": "e1", "key": "肺功能", "region": None, "name": "肺功能", "kind": "功能", "order_day": 0, "result_day": 1, "items": [], "abnormal": True},
                 {"id": "e2", "key": "CT", "region": "胸部", "name": "胸部CT", "kind": "影像", "order_day": 0, "result_day": 1, "items": [], "abnormal": True}]
        planner.mark_exam_findings(st, planner.strata_of("社区获得性肺炎", "J18"), exams, "肺炎", {"肺"})
        self.assertTrue(exams[0]["diagnostic"])
        self.assertEqual(planner.exam_result_text(exams[0]), "可见异常改变")
        self.assertNotIn("符合", planner.exam_result_text(exams[0]))
        self.assertNotIn("相符", planner.exam_result_text(exams[0]))
        self.assertEqual(planner.exam_result_text(exams[1]), "符合肺炎表现")

    def test_chief_prolapse_and_past_sentence_negation(self):
        adms = []
        for i in range(12):
            a = fixtures_mc.admission(i)
            a["docs"][0]["text"] = a["docs"][0]["text"].replace("主诉：咳嗽3天。", "主诉：便后肛门物脱出3天。").replace(
                "既往史：否认糖尿病病史。有高血压病史5年。",
                "既往史：否认高血压、糖尿病、冠心病、脑梗死、乙肝病史。否认外伤，有痛风病史3年。")
            adms.append(a)
        s = stats.build_stats(adms, {f"F{i:03d}": "I84" for i in range(12)}, k=11)
        self.assertIn("脱出", s["chief_symptoms"]["I84"]["items"])
        past_all = s["past"]["*"]["items"]
        for neg_term in ("高血压", "糖尿病", "冠心病", "脑梗死", "乙肝"):
            self.assertNotIn(neg_term, past_all)
        self.assertEqual(past_all.get("痛风"), 12)

    def test_dx_settled_early_gate_and_pre_qmark(self):
        import ast
        with open(os.path.join(ROOT, "medcourse", "gate.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names = [a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
        self.assertFalse(any("oracle2" in x for x in mods + names), mods + names)
        pl = {"section": "鉴别诊断", "target_chars": 5, "facts": [], "allowed": [], "hint": None,
              "unsettled_dx": ["社区获得性肺炎", "肺结核"]}
        for bad in ("现诊断社区获得性肺炎明确。", "目前确诊为社区获得性肺炎。", "患者无低热盗汗，排除肺结核。", "肺结核已排除。"):
            self.assertIn("dx_settled_early", gate.check_section(pl, bad), bad)
        for ok_txt in ("考虑社区获得性肺炎可能。", "主治医师查房分析：社区获得性肺炎诊断尚待进一步明确，需与肺结核鉴别。",
                       "待进一步检查排除肺结核。", "需完善痰培养以除外肺结核。", "拟诊社区获得性肺炎？"):
            self.assertNotIn("dx_settled_early", gate.check_section(pl, ok_txt), ok_txt)
        mixed = "患者咳嗽咳痰，目前排除肺结核，待胸部CT进一步明确。"
        self.assertEqual(gate.repair_section(mixed, ["dx_settled_early"], pl), "患者咳嗽咳痰，待胸部CT进一步明确。")
        pl_sub = dict(pl, unsettled_dx=["肺炎"], settled_dx=["社区获得性肺炎"])
        self.assertNotIn("dx_settled_early", gate.check_section(pl_sub, "结合检查结果，排除社区获得性肺炎。"))
        self.assertIn("dx_settled_early", gate.check_section(pl_sub, "结合检查结果，排除社区获得性肺炎，同时已排除肺炎。"))
        # PRE 术前诊断：主诊断当天疑似带"？"，确诊不带"？"
        j, specs, st, _ = make("PQ1", 11)
        pname = j["patient"]["primary_dx"]["name"]
        j["surgical"], j["surgery_day"] = True, 2
        j["procedures"] = [{"id": "a1", "kind": "手术", "name": "腹腔镜胆囊切除术", "side": "无", "decide_day": 0, "do_day": 2,
                            "anesthesia": "全身麻醉", "source": "key"}]
        j["dx_events"] = [{"id": "d1", "name": pname, "day": 0, "state": "疑似"}, {"id": "d2", "name": pname, "day": 3, "state": "确诊"}]
        pre_spec = {"k": 2, "doc_type": "progress_note", "doc_type_cn": "术前小结", "role": "PRE", "day": 1, "author": "住院医师", "time": "10:00"}
        pls = compiler.compile_doc(j, pre_spec, [pre_spec], st, random.Random(1))
        pre_dx = next(x["program_text"] for x in pls if x["section"] == "术前诊断")
        self.assertIn(pname + "？", pre_dx)
        j["dx_events"] = [{"id": "d1", "name": pname, "day": 0, "state": "确诊"}]
        pls2 = compiler.compile_doc(j, pre_spec, [pre_spec], st, random.Random(1))
        self.assertNotIn("？", next(x["program_text"] for x in pls2 if x["section"] == "术前诊断"))
        # 流程接入：写手越界写定论时被节级修补清掉，模板写手自身 0 次
        tally_t = {}
        for i in range(40):
            pipeline.generate_one(f"DT{i:03d}", st, writer.TemplateWriter(), 20261102, None, tally_t)
        self.assertEqual(tally_t.get("sec_dx_settled_early", 0), 0)

        class EarlyWriter(writer.TemplateWriter):
            def write(self, j, spec, pl, seed):
                t = super().write(j, spec, pl, seed)
                if spec["role"] in ("FP", "SUP") and pl.get("unsettled_dx"):
                    t = t.rstrip("。") + f"。现已确诊为{pl['unsettled_dx'][0]}。"
                return t

        tally_e = {}
        for i in range(40):
            _, docs, _ = pipeline.generate_one(f"DE{i:03d}", st, EarlyWriter(), 20261102, None, tally_e)
            if docs:
                for d in docs:
                    if d["role"] != "DS":
                        self.assertNotIn("现已确诊为", d["text"])
        self.assertGreater(tally_e.get("sec_dx_settled_early_repaired", 0), 0)

    def test_s2_switch_and_multifold_cli(self):
        lines, events = [], {}
        for i in range(12):
            rid = f"S{i:03d}"
            a = fixtures_mc.admission(i)
            a["admission_id"] = rid
            # 原文既往史写长清单使旧规则误抓冠心病；出院诊断多一条 2型糖尿病；辅助检查有血常规（检验）和胸部CT（非检验）
            a["docs"][0]["text"] = (
                "入院记录\n姓名：[患者]　性别：男　年龄：58岁　科室：呼吸内科\n入院时间：2025-03-01 10:00\n"
                "主诉：咳嗽、咳痰3天。\n现病史：患者3天前受凉后咳嗽、咳痰。\n既往史：有高血压病史5年，否认糖尿病、冠心病。\n"
                "体格检查T：37.8℃ P：88次/分 R：20次/分 BP：130/80mmHg，双肺湿啰音。\n"
                "辅助检查：血常规示白细胞12.1×10^9/L，血钾4.1mmol/L。\n入院诊断：\n1.社区获得性肺炎\n"
            )
            a["docs"][2]["text"] = (
                "出院记录\n姓名：[患者]　性别：男　年龄：58岁　科室：呼吸内科\n入院时间：2025-03-01 10:00　出院时间：2025-03-08 09:00　住院天数：7天\n"
                "入院情况：咳嗽3天入院。\n诊疗经过：复查胸部CT吸收好转。\n出院诊断：\n1.社区获得性肺炎\n2.2型糖尿病\n出院情况：好转。\n出院医嘱：随诊。\n"
            )
            hr_t, ds_t = a["docs"][0]["text"], a["docs"][2]["text"]
            c_pos = hr_t.index("咳嗽")
            p_pos = hr_t.index("高血压")
            ct_pos = ds_t.index("胸部CT")
            dm_pos = ds_t.index("2型糖尿病")
            events[rid] = [
                {"admission_id": rid, "doc_idx": 0, "section": "主诉", "event_type": "symptom", "assertion": "done",
                 "name": "咳嗽", "evidence": "咳嗽", "ev_start": c_pos, "ev_end": c_pos + 2},
                {"admission_id": rid, "doc_idx": 0, "section": "既往史", "event_type": "diagnosis", "assertion": "done",
                 "name": "高血压", "evidence": "有高血压病史", "ev_start": p_pos - 1, "ev_end": p_pos + 5},
                {"admission_id": rid, "doc_idx": 2, "section": "诊疗经过", "event_type": "exam", "assertion": "done",
                 "name": "胸部CT", "evidence": "胸部CT", "ev_start": ct_pos, "ev_end": ct_pos + 4},
                {"admission_id": rid, "doc_idx": 2, "section": "出院诊断", "event_type": "diagnosis", "assertion": "done",
                 "name": "社区获得性肺炎", "evidence": "1.社区获得性肺炎", "ev_start": ds_t.index("1.社区获得性肺炎"), "ev_end": ds_t.index("1.社区获得性肺炎") + 9},
                {"admission_id": rid, "doc_idx": 2, "section": "出院诊断", "event_type": "diagnosis", "assertion": "done",
                 "name": "2型糖尿病", "evidence": "2.2型糖尿病", "ev_start": dm_pos - 2, "ev_end": dm_pos + 5},
            ]
            lines.append(a)
        codes = {f"S{i:03d}": "J18" for i in range(12)}
        s = stats.build_stats(lines, codes, k=11, events_by=events)
        self.assertEqual(s["version"], "mc-stats-0.21")
        # 检验类（血常规）照走规则，非检验类（CT|胸）走 S2 甲（胸部CT 没写结果口吻，规则读不到 R，S2 读到 done）
        self.assertEqual(s["exams"]["J18"]["items"].get("血常规|"), 12)
        self.assertEqual(s["exams"]["J18"]["items"].get("CT|胸"), 12)
        # 主诉走 S2 甲（只给了咳嗽，没给咳痰）
        self.assertEqual(s["chief_symptoms"]["J18"]["items"], {"咳嗽": 12})
        self.assertEqual(s["past"]["*"]["items"], {"高血压": 12})
        self.assertEqual(s["comorbid"]["J18"]["items"], {"2型糖尿病": 12})
        self.assertIn("s2_switch_meta", s)
        self.assertEqual(s["s2_switch_meta"]["exams_nonlab_s2_cells"], s["s2_switch_meta"]["exams_nonlab_s2_cells"])
        self.assertGreater(s["s2_switch_meta"]["exams_nonlab_s2_cells"], 0)
        # 多折 CLI：拆成两折输入
        tmp = tempfile.mkdtemp()
        try:
            p1, p2 = os.path.join(tmp, "f1.jsonl"), os.path.join(tmp, "f2.jsonl")
            d1, d2 = os.path.join(tmp, "d1.csv"), os.path.join(tmp, "d2.csv")
            e1, e2 = os.path.join(tmp, "e1.jsonl"), os.path.join(tmp, "e2.jsonl")
            cp = os.path.join(tmp, "copy.csv")
            out = os.path.join(tmp, "stats.json")
            for path_, sub in ((p1, lines[:6]), (p2, lines[6:])):
                with open(path_, "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in sub)
            for path_, sub in ((d1, lines[:6]), (d2, lines[6:])):
                with open(path_, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["admission_id", "code3"])
                    w.writerows([[x["admission_id"], "J18"] for x in sub])
            for path_, sub in ((e1, lines[:6]), (e2, lines[6:])):
                with open(path_, "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(ev, ensure_ascii=False) + "\n" for x in sub for ev in events[x["admission_id"]])
            with open(cp, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row", "group", "value", "exact_share", "numeric_changed_share", "near_share", "pairs"])
                w.writerow(["R", "全部", "全部", "0.2", "0.03", "0.02", "100"])
            r = run_mod("medcourse.stats", "--path", f"{p1},{p2}", "--dx-map", f"{d1},{d2}", "--copyops", cp, "--events", f"{e1},{e2}", "--out", out)
            self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
            with open(out, encoding="utf-8") as f:
                s_cli = json.load(f)
            self.assertEqual(s_cli["n_admissions"], 12)
            self.assertEqual(s_cli["exams"]["J18"]["items"].get("CT|胸"), 12)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_plan_diag_v020_and_copy_diag(self):
        from medcourse import plan_diag
        st = fixtures_mc.stats()
        rows = dict(plan_diag.diagnose(st, 120, 20261102))
        for k in ("child_chronic_denial", "func_exam_has_dx", "past_denies_listed_dx",
                  "home_no_oral", "tx_primary_suspected", "lab_part_gt_total"):
            self.assertEqual(rows.get(k, 0), 0, k)
        self.assertGreater(rows.get("plan_hgb_admissions", 0), 0)
        self.assertGreater(rows.get("plan_plt_admissions", 0), 0)
        self.assertTrue(any(k.startswith("med_core_cls_layer_") for k in rows))
        self.assertTrue(any(k.startswith("med_core_pool_") for k in rows))
        self.assertIn("plan_ds_dx_items", rows)
        # apply_copy 过滤拆分计数相加等于候选总句数
        diag = {}
        compiler.apply_copy("今日患者精神尚可。继续抗感染治疗。", "患者昨日体温38.5℃。今日拟查胸部CT。主治医师查房示病情好转。患者咳嗽减轻。注意休息预防受凉，监测反应。",
                            0.4, random.Random(1), banned=["咳嗽"], diag=diag, banned_meds=["阿司匹林"])
        drops = sum(diag.get(k, 0) for k in ("copy_drop_short", "copy_drop_digit", "copy_drop_plan_word", "copy_drop_round",
                                             "copy_drop_trend", "copy_drop_action", "copy_drop_banned_med", "copy_drop_banned_sym",
                                             "copy_drop_already_in", "copy_drop_trend_clash", "copy_pool_ok"))
        self.assertEqual(drops, diag["copy_cand_sents"])
        self.assertEqual(diag.get("copy_kept_sents", 0) + diag.get("copy_drop_budget", 0), diag.get("copy_pool_ok", 0))


class TestV0201(unittest.TestCase):
    """0.2.0.1：S1-A Z 章 ds_admission_facts_leak 与 exam_invasive_dup 修补（虚构数据）。"""

    def test_z_treatment_no_ds_admission_leak(self):
        from medcourse import plan_diag
        st = fixtures_mc.stats()
        st = json.loads(json.dumps(st))
        # 构造 Z 章三种典型治疗性/随访住院，包含带 "术后" 的肿瘤合并症与第 0 天诊断相关影像检查
        st["joint"]["items"].update({"男|45-64|Z": 40, "女|45-64|Z": 40})
        st["joint"]["n"] += 80
        st["code3"]["Z"] = {"n": 90, "items": {"Z51": 30, "Z47": 30, "Z08": 30}}
        st["dx_names"]["Z51"] = {"n": 30, "items": {"恶性肿瘤维持性化学治疗": 30}}
        st["dx_names"]["Z47"] = {"n": 30, "items": {"骨折内固定取出": 30}}
        st["dx_names"]["Z08"] = {"n": 30, "items": {"恶性肿瘤治疗后随访检查": 30}}
        for code in ("Z51", "Z47", "Z08"):
            st["los"][code] = {"n": 30, "items": {"4": 15, "5": 15}}
            st["dept"][code] = {"n": 30, "items": {"肿瘤科": 30} if code != "Z47" else {"骨科": 30}}
            st["exams"][code] = {"n": 30, "items": {"CT|腹": 30, "血常规|": 30}}
        st["comorbid"]["Z51"] = {"n": 30, "items": {"直肠恶性肿瘤术后": 30}}
        st["comorbid"]["Z08"] = {"n": 30, "items": {"结肠癌术后": 30}}
        st["exams"]["*"]["items"].setdefault("CT|腹", 60)
        st["exam_days"]["CT|腹"] = {"0": 100}
        rows = dict(plan_diag.diagnose(st, 120, 20261102))
        self.assertEqual(rows.get("ds_admission_facts_leak", 0), 0)
        # 直接校验所有 Z 章出院记录 "入院情况" 的事实列表均不触发 gate.admission_leaks
        z_hits = 0
        for i in range(120):
            jid = f"Z0201-{i:04d}"
            rng = random.Random(planner.seed_of(20261102, jid))
            j = planner.plan_journey(st, rng, jid, None)
            if j and j["patient"]["primary_dx"]["code3"].startswith("Z"):
                z_hits += 1
                specs = planner.schedule(j, st, rng)
                for s in specs:
                    for pl in compiler.compile_doc(j, s, specs, st, rng):
                        if pl.get("admission_only"):
                            self.assertEqual(gate.admission_leaks("。".join(f["text"] for f in pl["facts"])), [],
                                             pl["facts"])
        self.assertGreater(z_hits, 5)

    def test_ensure_diagnostic_no_invasive_dup(self):
        from medcourse import plan_diag
        st = fixtures_mc.stats()
        st = json.loads(json.dumps(st))
        # Z51 存在骨髓穿刺（高 lift，在 Z51 内）和腰椎穿刺（中等 lift）：若已抽到一种穿刺，ensure_diagnostic 不得再补另一种穿刺
        st["code3"]["Z"] = {"n": 40, "items": {"Z51": 40}}
        st["joint"]["items"]["男|45-64|Z"] = 40
        st["joint"]["n"] += 40
        st["dx_names"]["Z51"] = {"n": 40, "items": {"恶性肿瘤维持性化学治疗": 40}}
        st["comorbid"]["Z51"] = {"n": 40, "items": {"白血病": 40}}
        st["los"]["Z51"] = {"n": 40, "items": {"4": 40}}
        st["dept"]["Z51"] = {"n": 40, "items": {"血液内科": 40}}
        # 白血病 -> C91/全身 + 血液，KEY_ORGAN["骨髓穿刺"]="血液", KEY_ORGAN["腰椎穿刺"]="脑"；换成肺+全身以便对比同类穿刺：
        st["comorbid"]["Z51"] = {"n": 40, "items": {"急性白血病": 40}}
        # 在全局中放低两种穿刺和 PET 频率，使在 Z51 均有较高 lift
        st["exams"]["*"]["items"]["腰椎穿刺|"] = 20
        st["exams"]["*"]["items"]["骨髓穿刺|"] = 20
        st["exams"]["*"]["items"]["PET|"] = 10
        st["exams"]["Z51"] = {"n": 40, "items": {"腰椎穿刺|": 8, "骨髓穿刺|": 30, "PET|": 6, "血常规|": 40}}
        strata = planner.strata_of("恶性肿瘤维持性化学治疗", "Z51")
        # 1) 若已抽到腰椎穿刺，且它仅在脑不在 {血液, 全身}（仅作已有有创占位），ensure_diagnostic 不得再补骨髓穿刺（同属穿刺大类），
        #    只能选非有创的 PET|（lift >= 1.0）
        exams_with_lp = [{"id": "e1", "key": "腰椎穿刺", "region": None, "name": "腰椎穿刺", "kind": "穿刺", "items": []}]
        got = planner.ensure_diagnostic(st, strata, exams_with_lp, "急性白血病既往治疗改变", {"血液", "全身"})
        self.assertEqual(got, ("PET", ""))
        # 2) 若连 PET 也没有，只能返回 None，绝不重复补同类有创检查
        st_no_pet = json.loads(json.dumps(st))
        st_no_pet["exams"]["Z51"]["items"].pop("PET|")
        self.assertIsNone(planner.ensure_diagnostic(st_no_pet, strata, exams_with_lp, "急性白血病既往治疗改变", {"血液", "全身"}))
        # 3) 若已抽到的是 骨髓穿刺| 本身（in have 且 lift >= 1.0），仍可被选中升级为 guaranteed=True（不新增条目）
        st_promote = json.loads(json.dumps(st_no_pet))
        st_promote["exams"]["*"]["items"]["骨髓穿刺|"] = 200
        st_promote["exams"]["*"]["n"] = 300
        st_promote["exams"]["Z51"]["items"]["骨髓穿刺|"] = 28  # lift = (28/40)/(200/300) = 1.05 (>= 1.0 且 < 1.2)
        exams_with_bm = [{"id": "e1", "key": "骨髓穿刺", "region": None, "name": "骨髓穿刺", "kind": "穿刺", "items": []}]
        self.assertEqual(planner.ensure_diagnostic(st_promote, strata, exams_with_bm, "急性白血病既往治疗改变", {"血液", "全身"}),
                         ("骨髓穿刺", ""))
        # 4) 端到端规划 diag 上 exam_invasive_dup 必为 0
        rows = dict(plan_diag.diagnose(st, 100, 20261102))
        self.assertEqual(rows.get("exam_invasive_dup", 0), 0)


class TestV0202(unittest.TestCase):
    """0.2.0.2：drug_class.SUBCLASS 词干亚类覆盖全词表子串配对，清零 O 章缩宫素并开触发的 qa_adm_med_conflict（虚构数据）。"""

    def test_subclass_covers_all_lexicon_substring_pairs(self):
        from medcourse import drug_class, rules
        rules.setup(("BENCH_G3",))
        lex = os.path.join(os.environ["BENCH_G3"], "ruler_g3", "data", "drug_lexicon.csv")
        with open(lex, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 280)
        self.assertEqual(drug_class.subclass("oxytocin"), "tocin")
        self.assertEqual(drug_class.subclass("carbetocin"), "tocin")
        aliases = []
        for r in rows:
            did = r["drug_id"]
            hits = [n for n, rx in drug_class._SUB_RE if rx.search(did)]
            self.assertLessEqual(len(hits), 1, (did, hits))
            sub = drug_class.subclass(did)
            for a in r["aliases"].split("|"):
                if a:
                    aliases.append((did, sub, a))
        unblocked = [
            (a1, d1, a2, d2)
            for d1, s1, a1 in aliases
            for d2, s2, a2 in aliases
            if d1 != d2 and a1 in a2 and not (s1 and s1 == s2)
        ]
        self.assertEqual(unblocked, [])

    def test_obstetric_oxytocic_no_med_conflict(self):
        from medcourse import out_diag, plan_diag
        st = json.loads(json.dumps(fixtures_mc.stats()))
        st["joint"]["items"] = {"女|18-40|O": 100}
        st["joint"]["n"] = 100
        st["code3"]["O"] = {"n": 100, "items": {"O80": 100}}
        st["dx_names"]["O80"] = {"n": 100, "items": {"孕39周头位顺产": 100}}
        st["dept"]["O80"] = {"n": 100, "items": {"产科": 100}}
        st["los"]["O80"] = {"n": 100, "items": {"3": 50, "4": 50}}
        st["chief_symptoms"]["O80"] = {"n": 100, "items": {"停经": 100}}
        st["med_class"]["O80"] = {"n": 100, "items": {"子宫收缩药": 90}}
        st["med_class"]["*"]["items"]["子宫收缩药"] = 90
        st["meds"]["O80"] = {"n": 100, "items": {"oxytocin": 50, "carbetocin": 50}}
        st["med_class_drug"]["子宫收缩药"] = {"n": 100, "items": {"oxytocin": 50, "carbetocin": 50}}
        st["med_class_multi"]["子宫收缩药"] = {"n": 100, "items": {"2+": 80}}
        st["med_alias"]["oxytocin"] = {"n": 50, "items": {"缩宫素": 50}}
        st["med_dose"]["oxytocin"] = {"n": 50, "items": {"10U": 50}}
        st["med_freq"]["oxytocin"] = {"n": 50, "items": {"qd": 50}}
        st["med_route"]["oxytocin"] = {"n": 50, "items": {"静滴": 50}}
        st["med_alias"]["carbetocin"] = {"n": 50, "items": {"卡贝缩宫素": 50}}
        st["med_dose"]["carbetocin"] = {"n": 50, "items": {"100ug": 50}}
        st["med_freq"]["carbetocin"] = {"n": 50, "items": {"st": 50}}
        st["med_route"]["carbetocin"] = {"n": 50, "items": {"静推": 50}}
        w = writer.TemplateWriter()
        med_conflict_hits, kept = 0, 0
        for i in range(60):
            j, docs, _ = pipeline.generate_one(f"OB0202-{i:04d}", st, w, 20261102, None, {})
            if docs is None:
                continue
            kept += 1
            if "med_conflict" in out_diag.qa_flags(j, docs):
                med_conflict_hits += 1
        self.assertGreaterEqual(kept, 55)
        self.assertEqual(med_conflict_hits, 0)
        rows = dict(plan_diag.diagnose(st, 60, 20261102))
        self.assertEqual(rows.get("same_subclass_pair", 0), 0)


class TestV021Prep(unittest.TestCase):
    """v0.2.1：S2C4 第二段合入前的四项本地先修（B_B3、B_copy5、血常规极值校准、写手端 qa_* 闸门）。"""

    def test_b3_ds_only_never_in_past_and_calibrated(self):
        from medcourse import plan_diag
        from medcourse.stats import PAST_TERMS, past_matches
        from ruler_g2 import build_real, evaluate_admission

        st = json.loads(json.dumps(fixtures_mc.stats()))
        st["real_rates"] = {
            "ds_dx_admissions": 8964, "ds_dx_items": 54993,
            "b3_admissions": 8964, "b3_permille_sum": 2312375,
        }
        st["comorbid"] = {
            "J": {"n": 130, "items": {"高血压病": 40, "2型糖尿病": 30, "低钾血症": 35, "肺部感染": 30, "电解质紊乱": 45}},
            "K": {"n": 130, "items": {"脂肪肝": 35, "胆囊息肉": 30, "腹腔积液": 25, "低蛋白血症": 30, "高血压病": 25}},
            "*": {"n": 300, "items": {}},
        }
        rows = dict(plan_diag.diagnose(st, 120, 20261117))
        self.assertEqual(rows.get("comorb_ds_only_in_past", -1), 0)
        self.assertEqual(rows.get("dup_graded_dx", -1), 0)
        w = writer.TemplateWriter()
        b3_vals = []
        for i in range(60):
            j, docs, _ = pipeline.generate_one(f"B3T_{i:04d}", st, w, 20261117, None, {})
            if docs is None:
                continue
            p = j["patient"]
            for c in p["comorbidities"]:
                if any(past_matches(pt, c["name"]) for pt in PAST_TERMS):
                    self.assertTrue(c["at_admission"], c)
            val = evaluate_admission(build_real({"admission_id": f"B3T_{i:04d}", "docs": docs}))["readings"]["B3"]["value"]
            if val is not None:
                b3_vals.append(val)
        self.assertGreaterEqual(len(b3_vals), 50)
        self.assertGreaterEqual(sum(b3_vals) / len(b3_vals), 0.24)

    def test_copy5_bridges_fp_pre_post_and_caps_cross_repeat(self):
        from ruler_g2 import build_real, evaluate_admission

        st = json.loads(json.dumps(fixtures_mc.stats()))
        st["copy"]["文书细类"].update({
            "术前小结": {"reuse": 0.214, "exact": 0.101, "pairs": 100},
            "术后首次病程": {"reuse": 0.127, "exact": 0.073, "pairs": 100},
        })
        meds = ([{"name": "阿司匹林", "dose": "0.1g", "freq": "qd", "start_day": 0, "stop_day": 5}],)
        j = {
            "patient": {"primary_dx": {"name": "急性阑尾炎"}, "comorbidities": []},
            "chief": {"symptoms": ["腹痛"]},
            "daily": [{"symptoms": ["腹痛"]}] * 4,
            "meds": [
                {"name": "阿司匹林", "dose": "0.1g", "freq": "qd", "start_day": 0, "stop_day": 5},
                {"name": "头孢呋辛", "dose": "1.5g", "freq": "bid", "start_day": 0, "stop_day": 1},
            ],
        }
        self.assertEqual(meds[0][0]["name"], "阿司匹林")
        prev_txt = ("予阿司匹林0.1g口服以改善微循环。予头孢呋辛1.5g抗感染。患者腹痛明显。症状较前改善。"
                    "体温37.8℃正常处理。继续现有抗血小板方案治疗。")
        allowed = compiler.active_med_nums(j, 3)
        kept = compiler.stable_sentences(prev_txt, compiler.copy_banned(j, 3), allowed_nums=allowed)
        self.assertIn("予阿司匹林0.1g口服以改善微循环。", kept)
        self.assertIn("继续现有抗血小板方案治疗。", kept)
        self.assertFalse(any("头孢呋辛" in s for s in kept))
        self.assertFalse(any("较前改善" in s for s in kept))
        self.assertFalse(any("37.8" in s for s in kept))
        # 验证过载长句封顶逻辑（同一句 >=20 字出现 2 次即进入 overused）
        bodies = {
            1: "患者一般情况可，无发热及呼吸困难，生命体征平稳。",
            2: "患者一般情况可，无发热及呼吸困难，生命体征平稳。今日查体无新增阳性体征。",
        }
        specs = [{"k": 0, "day": 0, "role": "FP"}, {"k": 1, "day": 1, "role": "SUP"},
                 {"k": 2, "day": 2, "role": "PRE"}, {"k": 3, "day": 2, "role": "POST"}, {"k": 4, "day": 3, "role": "DAILY"}]
        self.assertEqual(pipeline.prev_note_for(2, specs, {0: "首次病程病例特点"}), {"day": 0, "text": "首次病程病例特点"})
        self.assertEqual(pipeline.prev_note_for(3, specs, {2: "术前小结简要病情"}), {"day": 2, "text": "术前小结简要病情"})
        overused = pipeline._overused_sents(bodies, [{"role": "FP"}, {"role": "PRE"}, {"role": "DAILY"}], 3)
        self.assertEqual(overused, [])
        overused2 = pipeline._overused_sents(bodies, [{"role": "FP"}, {"role": "SUP"}, {"role": "DAILY"}], 3)
        self.assertIn("患者一般情况可，无发热及呼吸困难，生命体征平稳", overused2)
        w = writer.TemplateWriter()
        tally, c1a = {}, []
        for i in range(60):
            _, docs, _ = pipeline.generate_one(f"C1A_{i:04d}", st, w, 20261117, None, tally)
            if docs is None:
                continue
            v = evaluate_admission(build_real({"admission_id": f"C1A_{i:04d}", "docs": docs}))["readings"]["C1a"]["value"]
            if v is not None:
                c1a.append(v)
        drops = sum(tally.get(k, 0) for k in ("copy_drop_short", "copy_drop_digit", "copy_drop_plan_word", "copy_drop_round",
                                              "copy_drop_trend", "copy_drop_action", "copy_drop_banned_med", "copy_drop_banned_sym",
                                              "copy_drop_already_in", "copy_drop_trend_clash", "copy_pool_ok"))
        self.assertEqual(drops, tally.get("copy_cand_sents", 0))
        self.assertEqual(tally.get("copy_kept_sents", 0) + tally.get("copy_drop_budget", 0), tally.get("copy_pool_ok", 0))
        self.assertGreaterEqual(sum(c1a) / len(c1a), 0.30)

    def test_cbc_extremes_calibrated_to_real_rates(self):
        from medcourse import plan_diag

        st = json.loads(json.dumps(fixtures_mc.stats()))
        st["real_rates"] = {
            "wbc_admissions": 4104, "wbc_gt12": 1222,
            "neut_admissions": 3303, "neut_gt85": 814,
            "hgb_admissions": 4514, "hgb_lt90": 679,
            "plt_admissions": 3170, "plt_lt100": 681,
        }
        st["lab_state"]["*"] = {"n": 300, "items": {
            "WBC|abn": 195, "WBC|norm": 105, "NEUT|abn": 180, "NEUT|norm": 120,
            "HGB|abn": 150, "HGB|norm": 150, "PLT|abn": 105, "PLT|norm": 195,
        }}
        st["lab_state"]["J"] = {"n": 130, "items": {
            "WBC|abn": 85, "WBC|norm": 45, "NEUT|abn": 78, "NEUT|norm": 52,
            "HGB|abn": 50, "HGB|norm": 80, "PLT|abn": 40, "PLT|norm": 90,
        }}
        st["labs"] = {
            "WBC": {"q": [3.1, 6.2, 10.5, 14.5, 22.0], "n": 900, "unit": "×10^9/L"},
            "NEUT": {"q": [42.0, 62.0, 79.0, 88.5, 95.0], "n": 900, "unit": "%"},
            "HGB": {"q": [68, 92, 118, 134, 152], "n": 900, "unit": "g/L"},
            "PLT": {"q": [45, 95, 175, 240, 380], "n": 900, "unit": "×10^9/L"},
        }
        rows = dict(plan_diag.diagnose(st, 300, 20261117))
        for pk, ak, rek, rak in [("plan_wbc_gt12", "plan_wbc_admissions", "wbc_gt12", "wbc_admissions"),
                                 ("plan_neut_gt85", "plan_neut_admissions", "neut_gt85", "neut_admissions"),
                                 ("plan_hgb_lt90", "plan_hgb_admissions", "hgb_lt90", "hgb_admissions"),
                                 ("plan_plt_lt100", "plan_plt_admissions", "plt_lt100", "plt_admissions")]:
            self.assertLessEqual(abs(rows[pk] / rows[ak] - st["real_rates"][rek] / st["real_rates"][rak]), 0.05, pk)

    def test_writer_qa_section_gates_and_repairs(self):
        cases = [
            ({"section": "正文", "before_op": True, "target_chars": 20, "facts": ()},
             "患者诉无明显胸闷，今日在全麻下行胆囊切除术，手术顺利，术后第1天恢复可。", "before_op_done"),
            ({"section": "专科情况", "target_chars": 12, "facts": ()},
             "神志清，瞳孔等大（neurological examination）未见异常，qd口服继续。", "english_prose"),
            ({"section": "既往史", "child_past": True, "known_past": [], "target_chars": 15, "facts": ()},
             "既往体健，否认高血压、糖尿病等慢性病史，否认药物过敏史。", "past_contradict"),
            ({"section": "现病史", "known_past": ["高血压"], "target_chars": 15, "facts": ()},
             "患者既往体健，否认高血压，3天前开始发热咳嗽。", "past_contradict"),
            ({"section": "体格检查", "target_chars": 15, "facts": ()},
             "体温未测，双肺呼吸音粗，未闻及湿啰音。", "pe_untested"),
            ({"section": "术后应当特别注意观察的事项", "no_incision": True, "target_chars": 15, "facts": ()},
             "观察穿刺点渗血，注意切口愈合及引流。", "no_incision_leak"),
            ({"section": "正文", "target_chars": 12, "facts": ()},
             "患者今日一般情况良好。本次查房记录完整，符合病历书写规范。", "meta_text"),
        ]
        for plan, bad, code in cases:
            fails = gate.check_section(plan, bad)
            self.assertIn(code, fails, (code, bad, fails))
            fixed = gate.repair_section(bad, [c for c in fails if c in gate.FIXABLE], plan)
            self.assertEqual(gate.check_section(plan, fixed), [], (code, fixed))
        # 验证计划口吻与陈旧史豁免，以及医学缩写不被 english_prose 误杀
        ok_plan = {"section": "正文", "before_op": True, "target_chars": 15, "facts": ()}
        self.assertEqual(gate.check_section(ok_plan, "拟今日在全麻下行腹腔镜胆囊切除术，5年前曾行阑尾切除术，查CT结果可。"), [])
        self.assertFalse(gate.copy_safe({"before_op": True}, "今日在全麻下行手术，手术顺利。"))


class TestV021DynamicTables(unittest.TestCase):
    def test_multi_sup_by_los_and_fallback(self):
        st_empty = copy.deepcopy(fixtures_mc.stats())
        j_base = planner.plan_journey(st_empty, random.Random(101), "SUP0")
        for los in (2, 6, 12):
            j_los = copy.deepcopy(j_base)
            j_los["los"] = los
            specs = planner.schedule(j_los, st_empty, random.Random(101 + los))
            self.assertEqual(sum(1 for s in specs if s["role"] == "SUP"), 1)

        st = copy.deepcopy(fixtures_mc.stats())
        st["pb_subtypes"] = {
            "admissions": 500,
            "notes_q": [1, 2, 3, 5, 8],
            "by_subtype": {},
            "by_los": {
                "1-3": {"admissions": 100, "notes": 300, "sub": {"主治医师查房": {"admissions": 90, "notes": 120}, "上级医师查房": {"admissions": 80, "notes": 80}}},
                "4-7": {"admissions": 150, "notes": 700, "sub": {"主治医师查房": {"admissions": 140, "notes": 240}, "主任医师查房": {"admissions": 130, "notes": 240}}},
                "8-14": {"admissions": 150, "notes": 950, "sub": {"主治医师查房": {"admissions": 140, "notes": 340}, "主任医师查房": {"admissions": 140, "notes": 335}}},
                "15+": {"admissions": 100, "notes": 1100, "sub": {"主治医师查房": {"admissions": 100, "notes": 300}, "主任医师查房": {"admissions": 100, "notes": 300}}},
            },
        }
        means = {}
        for los in (2, 6, 11, 18):
            cnts = []
            for seed in range(25):
                j_los = copy.deepcopy(j_base)
                j_los["los"] = los
                if los >= 6 and seed % 3 == 0:
                    j_los["surgical"] = True
                    j_los["surgery_day"] = los if seed % 2 else 3
                specs = planner.schedule(j_los, st, random.Random(seed))
                sup_days = [s["day"] for s in specs if s["role"] == "SUP"]
                self.assertEqual(len(sup_days), len(set(sup_days)))
                self.assertTrue(all(1 <= d <= max(1, los - 1) for d in sup_days))
                self.assertEqual(contract.validate_docspecs(specs, los), [])
                cnts.append(len(sup_days))
            means[los] = sum(cnts) / len(cnts)
        self.assertLessEqual(means[2], means[6])
        self.assertLess(means[6], means[11])
        self.assertLess(means[11], means[18])

        tw = writer.TemplateWriter()
        for seed in range(10):
            j, docs, fails = pipeline.generate_one(f"V021S{seed}", st, tw, 500 + seed, None, {})
            self.assertEqual(fails, [])
            self.assertIsNotNone(docs)
            self.assertGreaterEqual(sum(1 for d in docs if d["role"] == "SUP"), 1)

    def test_s2_exam_days_and_lags_and_fallback(self):
        lines, events = [], {}
        for i in range(12):
            a = fixtures_mc.admission(i)
            rid = f"E{i:03d}"
            a["admission_id"] = rid
            fp = "2025-03-01 12:00 首次病程记录\n拟行腹部CT检查明确诊断。\n"
            daily = "2025-03-03 09:00 日常病程记录\n查腹部CT示脂肪肝。\n"
            ds_t = (
                "出院记录\n姓名：[患者]　性别：男　年龄：58岁　科室：呼吸内科\n"
                "入院时间：2025-03-01 10:00　出院时间：2025-03-07 09:00　住院天数：6天\n"
                "入院情况：咳嗽3天入院。\n诊疗经过：完善腹部CT。\n出院诊断：\n1.社区获得性肺炎\n出院情况：好转。\n出院医嘱：随诊。\n"
            )
            a["docs"] = [
                a["docs"][0],
                {"doc_type": "progress_note", "day_offset": 0, "text": fp},
                {"doc_type": "progress_note", "day_offset": 2, "text": daily},
                {"doc_type": "discharge_summary", "day_offset": 6, "text": ds_t},
            ]
            events[rid] = [
                {"admission_id": rid, "doc_idx": 1, "section": "正文", "event_type": "exam", "assertion": "planned",
                 "name": "腹部CT", "evidence": "拟行腹部CT", "ev_start": fp.index("拟行"), "ev_end": fp.index("拟行") + 6},
                {"admission_id": rid, "doc_idx": 2, "section": "正文", "event_type": "exam", "assertion": "done",
                 "name": "腹部CT", "evidence": "腹部CT", "ev_start": daily.index("腹部CT"), "ev_end": daily.index("腹部CT") + 4},
                {"admission_id": rid, "doc_idx": 3, "section": "诊疗经过", "event_type": "exam", "assertion": "done",
                 "name": "腹部CT", "evidence": "腹部CT", "ev_start": ds_t.index("腹部CT"), "ev_end": ds_t.index("腹部CT") + 4},
            ]
            lines.append(a)
        codes = {f"E{i:03d}": "J18" for i in range(12)}
        s = stats.build_stats(lines, codes, k=11, events_by=events)
        self.assertEqual(s["version"], "mc-stats-0.21")
        self.assertEqual(s["exam_days"].get("CT|腹"), {"2": 12})
        self.assertEqual(s["exam_lags"].get("CT|腹"), {"2": 12})
        self.assertEqual(s["exam_lags"].get("CT"), {"2": 12})
        self.assertEqual(s["exam_lags"].get("kind:影像"), {"2": 12})
        self.assertEqual(s["s2_switch_meta"]["s2_course_exam_stays"], 12)
        self.assertEqual(s["s2_switch_meta"]["s2_exam_lag_pairs"], 12)
        self.assertGreater(s["s2_switch_meta"]["exam_days_s2_cells"], 0)
        self.assertGreater(s["s2_switch_meta"]["exam_lags_s2_cells"], 0)

        s_small = stats.build_stats(lines[:5], codes, k=11, events_by=events)
        self.assertEqual(s_small["exam_lags"], {})
        self.assertGreater(s_small["suppressed"], 0)

        ex = planner._exam(s, random.Random(7), 0, "CT", "腹", los=6)
        self.assertEqual((ex["result_day"], ex["order_day"]), (2, 0))
        s_nolag = dict(s, exam_lags={})
        for seed in range(10):
            ex_fb = planner._exam(s_nolag, random.Random(seed), 0, "CT", "腹", los=6)
            self.assertIn(ex_fb["result_day"] - ex_fb["order_day"], (0, 1))

    def test_s2_symptom_course_and_fallback(self):
        lines, events = [], {}
        hr_t = (
            "入院记录\n姓名：[患者]　性别：男　年龄：58岁　科室：呼吸内科\n入院时间：2025-03-01 10:00\n"
            "主诉：咳嗽5天。\n现病史：患者5天前受凉后出现咳嗽，伴发热。\n"
            "体格检查T：37.8℃ P：88次/分 R：20次/分 BP：130/80mmHg。\n入院诊断：\n1.社区获得性肺炎\n"
        )
        n2_t = "2025-03-03 09:00 主治医师查房记录\n患者现无发热，仍有咳嗽。\n"
        n4_t = "2025-03-05 09:00 日常病程记录\n患者无发热，咳嗽较前减轻。\n"
        ds_t = (
            "出院记录\n姓名：[患者]　性别：男　年龄：58岁　科室：呼吸内科\n"
            "入院时间：2025-03-01 10:00　出院时间：2025-03-07 09:00　住院天数：6天\n"
            "入院情况：咳嗽5天。\n诊疗经过：抗感染治疗。\n出院诊断：\n1.社区获得性肺炎\n出院情况：好转。\n出院医嘱：随诊。\n"
        )
        for i in range(12):
            rid = f"SC{i:03d}"
            lines.append({
                "admission_id": rid,
                "docs": [
                    {"doc_type": "hospital_record", "day_offset": 0, "text": hr_t},
                    {"doc_type": "progress_note", "day_offset": 2, "text": n2_t},
                    {"doc_type": "progress_note", "day_offset": 4, "text": n4_t},
                    {"doc_type": "discharge_summary", "day_offset": 6, "text": ds_t},
                ],
            })
            events[rid] = [
                {"admission_id": rid, "doc_idx": 0, "section": "主诉", "event_type": "symptom", "assertion": "done",
                 "name": "咳嗽", "evidence": "咳嗽", "ev_start": hr_t.index("咳嗽"), "ev_end": hr_t.index("咳嗽") + 2},
                {"admission_id": rid, "doc_idx": 0, "section": "现病史", "event_type": "symptom", "assertion": "done",
                 "name": "发热", "evidence": "发热", "ev_start": hr_t.index("发热"), "ev_end": hr_t.index("发热") + 2},
                {"admission_id": rid, "doc_idx": 1, "section": "正文", "event_type": "symptom", "assertion": "negated",
                 "name": "发热", "evidence": "无发热", "ev_start": n2_t.index("无发热"), "ev_end": n2_t.index("无发热") + 3},
                {"admission_id": rid, "doc_idx": 1, "section": "正文", "event_type": "symptom", "assertion": "done",
                 "name": "咳嗽", "evidence": "咳嗽", "ev_start": n2_t.index("咳嗽"), "ev_end": n2_t.index("咳嗽") + 2},
                {"admission_id": rid, "doc_idx": 2, "section": "正文", "event_type": "symptom", "assertion": "negated",
                 "name": "发热", "evidence": "无发热", "ev_start": n4_t.index("无发热"), "ev_end": n4_t.index("无发热") + 3},
                {"admission_id": rid, "doc_idx": 2, "section": "正文", "event_type": "symptom", "assertion": "done",
                 "name": "咳嗽", "evidence": "咳嗽", "ev_start": n4_t.index("咳嗽"), "ev_end": n4_t.index("咳嗽") + 2},
            ]
        codes = {f"SC{i:03d}": "J18" for i in range(12)}
        s = stats.build_stats(lines, codes, k=11, events_by=events)
        sc_all = s["symptom_course"]["*"]["items"]
        self.assertEqual(sc_all.get("sym:发热|1-2|gone"), 12)
        self.assertEqual(sc_all.get("sym:咳嗽|1-2|keep"), 12)
        self.assertEqual(sc_all.get("sym:发热|3-5|gone"), 12)
        self.assertEqual(sc_all.get("sym:咳嗽|3-5|keep"), 12)
        self.assertEqual(s["s2_switch_meta"]["s2_prog_sym_stays"], 12)
        self.assertGreater(s["s2_switch_meta"]["symptom_course_s2_cells"], 0)
        self.assertGreater(s["s2_switch_meta"]["symptom_course_rule_cells"], 0)

        st_sc = copy.deepcopy(fixtures_mc.stats())
        st_sc["los"]["J"] = {"n": 130, "items": {"6": 130}}
        st_sc["chief_symptoms"]["J18"] = {"n": 100, "items": {"咳嗽": 100}}
        st_sc["symptoms"]["J18"] = {"n": 100, "items": {"咳嗽": 100, "发热": 100}}
        st_sc["symptom_course"] = s["symptom_course"]
        cough_d3, fever_d3, total_both = 0, 0, 0
        for seed in range(45):
            j = planner.plan_journey(st_sc, random.Random(800 + seed), f"SCJ{seed}")
            if "发热" in j["daily"][0]["symptoms"] and "咳嗽" in j["daily"][0]["symptoms"]:
                total_both += 1
                cough_d3 += "咳嗽" in j["daily"][3]["symptoms"]
                fever_d3 += "发热" in j["daily"][3]["symptoms"]
        self.assertGreater(total_both, 20)
        self.assertGreater(cough_d3 / total_both, 0.70)
        self.assertLess(fever_d3 / total_both, 0.25)

        st_fb1 = copy.deepcopy(fixtures_mc.stats())
        st_fb2 = copy.deepcopy(fixtures_mc.stats())
        st_fb2["symptom_course"] = {}
        j1 = planner.plan_journey(st_fb1, random.Random(1234), "FB0")
        j2 = planner.plan_journey(st_fb2, random.Random(1234), "FB0")
        self.assertEqual(j1["daily"], j2["daily"])


class TestV0211(unittest.TestCase):
    def test_fp_copy_source_and_post_excludes_intraop(self):
        fp = pipeline._note_copy_source("FP", {"病例特点": "特征A。", "拟诊讨论": "鉴别B。", "诊疗计划": "计划C。"})
        self.assertEqual(fp, "特征A。鉴别B。计划C。")
        post = pipeline._note_copy_source("POST", {"手术简要经过": "术中经过X。", "术后处理措施": "术后护理Y。"})
        self.assertEqual(post, "术后护理Y。")
        self.assertNotIn("术中经过X", post)

    def test_sup_to_sup_copy_cap_and_budget(self):
        st = fixtures_mc.stats()
        sup_sp = {"role": "SUP", "doc_type_cn": "主治医师查房记录", "day": 3}
        daily_sp = {"role": "DAILY", "doc_type_cn": "日常病程记录", "day": 3}
        pre_sp = {"role": "PRE", "doc_type_cn": "术前小结", "day": 3}
        self.assertIsNone(compiler.copy_max_sents(sup_sp, "FP"))
        self.assertEqual(compiler.copy_max_sents(sup_sp, "SUP"), 1)
        self.assertEqual(compiler.copy_max_sents(daily_sp, "SUP"), 2)
        self.assertEqual(compiler.copy_max_sents(pre_sp, "SUP"), 2)
        self.assertLessEqual(compiler.copy_budget(st, sup_sp, 1, prev_role="SUP"), 0.22)
        self.assertLessEqual(compiler.copy_budget(st, daily_sp, 1, prev_role="SUP"), 0.28)
        diag = {}
        out, added = compiler.apply_copy(
            "今日查房患者生命体征平稳，继续目前治疗方案并观察症状变化。",
            "患者一般情况可，神志清楚，精神尚可，饮食睡眠正常。双肺呼吸音清，未闻及明显干湿性啰音。腹平软，全腹无压痛及反跳痛。双下肢无水肿，四肢活动自如。",
            0.90,
            random.Random(7),
            diag=diag,
            max_sents=1,
        )
        self.assertGreater(added, 0)
        self.assertNotEqual(out, "")
        self.assertEqual(diag["copy_kept_sents"], 1)
        self.assertEqual(diag["copy_kept_sents"] + diag["copy_drop_budget"], diag["copy_pool_ok"])

    def test_followup_sup_focus_rotates_and_drops_stale_diff(self):
        j, _, _, _ = make("V0211_SUP", 1)
        while len(j["daily"]) < 4:
            nxt = len(j["daily"])
            j["daily"].append(dict(j["daily"][-1], day=nxt))
        d0, d1, d2 = 1, 2, 3
        base_m = copy.deepcopy(j["meds"][0]) if j["meds"] else {
            "id": "m1", "drug_id": "d1", "name": "甲药", "dose": "1g", "freq": "qd",
            "route": "静滴", "start_day": 0, "stop_day": d2 + 2, "discharge": False,
        }
        j["meds"] = [
            dict(base_m, id="m1", name="轮换甲药", start_day=0, stop_day=d2 + 2),
            dict(base_m, id="m2", name="轮换乙药", start_day=0, stop_day=d2 + 2),
        ]
        dx = j["patient"]["primary_dx"]["name"]
        j["differentials"] = ["鉴别甲病"]
        j["dx_events"] = [
            {"id": "d1", "name": dx, "day": d0, "state": "确诊"},
            {"id": "d2", "name": "鉴别甲病", "day": d0, "state": "疑似"},
        ]
        specs = [
            {"k": 0, "day": d0, "role": "FP", "doc_type_cn": "首次病程记录"},
            {"k": 1, "day": d0, "role": "SUP", "doc_type_cn": "上级医师查房记录"},
            {"k": 2, "day": d1, "role": "SUP", "doc_type_cn": "主治医师查房记录"},
            {"k": 3, "day": d2, "role": "SUP", "doc_type_cn": "主任医师查房记录"},
        ]
        f1 = dict(compiler.course_focus(j, specs[1], specs, random.Random(1)))["round"]
        f2 = dict(compiler.course_focus(j, specs[2], specs, random.Random(2)))["round"]
        f3 = dict(compiler.course_focus(j, specs[3], specs, random.Random(3)))["round"]
        self.assertIn(f"{dx}诊断明确", f1)
        self.assertIn("需与鉴别甲病鉴别", f1)
        self.assertNotIn("鉴别甲病", f2)
        self.assertNotIn("鉴别甲病", f3)
        self.assertTrue(any(t in f2 for t in compiler.SUP_FOLLOWUP_THEMES))
        self.assertTrue(any(t in f3 for t in compiler.SUP_FOLLOWUP_THEMES))
        self.assertNotEqual(f2.split("：")[1].split("，")[0], f3.split("：")[1].split("，")[0])
        self.assertIn("轮换乙药", f1)
        self.assertIn("轮换甲药", f2)
        self.assertIn("轮换乙药", f3)


class TestV0212Fixes(unittest.TestCase):
    def test_age_dept_proc_chief_coupling(self):
        self.assertFalse(planner.sym_ok("哭闹", {"age": 80, "sex": "男"}))
        self.assertTrue(planner.sym_ok("哭闹", {"age": 2, "sex": "男"}))
        self.assertFalse(planner.sym_ok("停经", {"age": 35, "sex": "男"}))
        self.assertFalse(planner.sym_ok("停经", {"age": 62, "sex": "女"}))
        self.assertTrue(planner.sym_ok("停经", {"age": 32, "sex": "女"}))
        self.assertFalse(planner.sym_ok("语言落后", {"age": 57, "sex": "女"}))
        self.assertFalse(planner.dx_ok("言语和语言发育障碍", "女", 57))
        self.assertFalse(planner.dept_ok("风湿免疫科(肾内二区)", "男", 80, False, "N43"))
        self.assertTrue(planner.dept_ok("泌尿外科", "男", 80, False, "N43"))
        self.assertTrue(planner.dept_ok("风湿免疫科", "女", 40, False, "M05"))

        st = fixtures_mc.stats()
        st["procedures"]["N43"] = {"n": 20, "items": {"右侧腹腔镜鞘状突高位结扎术": 15}}
        self.assertEqual(planner.proc_for(st, ["n:鞘膜积液", "N43", "N"], "鞘膜积液", age=80), ({}, None))
        items, src = planner.proc_for(st, ["n:鞘膜积液", "N43", "N"], "鞘膜积液", age=5)
        self.assertEqual(src, "key")
        self.assertIn("右侧腹腔镜鞘状突高位结扎术", items)

        # 产科妊娠主诉带"停经"时时长与 gest_weeks 一致
        st_preg = fixtures_mc.stats()
        st_preg["joint"]["items"] = {"女|18-40|O": 50}
        st_preg["code3"]["O"] = {"n": 50, "items": {"O80": 50}}
        st_preg["dx_names"]["O80"] = {"n": 50, "items": {"孕39周待产": 50}}
        st_preg["chief_symptoms"]["O80"] = {"n": 50, "items": {"停经": 50}}
        st_preg["durations"]["O80"] = {"n": 50, "items": {"1周": 50}}
        j = planner.plan_journey(st_preg, random.Random(1), "PREG1")
        self.assertEqual(j["patient"]["gest_weeks"], 39)
        self.assertEqual(j["chief"]["duration"], "39周")

    def test_natural_orifice_no_incision_and_copy_plan_guard(self):
        self.assertTrue(compiler.no_incision({"name": "经尿道右侧输尿管软镜钬激光碎石取石术", "anesthesia": "腰硬联合麻醉"}))
        self.assertFalse(compiler.no_incision({"name": "腹腔镜下胆囊切除术", "anesthesia": "全身麻醉"}))
        kept = compiler.stable_sentences("泌尿系X线检查结果示未见明显异常，尚未实施手术。尚未实施手术。患者配合治疗，饮食睡眠可。")
        self.assertEqual(kept, ["患者配合治疗，饮食睡眠可。"])

        j, specs, st, _ = make("FP_TENT", 2)
        dx = j["patient"]["primary_dx"]["name"]
        j["dx_events"] = [{"id": "d1", "name": dx, "day": 0, "state": "疑似"}]
        fp = next(s for s in specs if s["role"] == "FP")
        plans = compiler.compile_doc(j, fp, specs, st, random.Random(2))
        plan_sec = next(x for x in plans if x["section"] == "诊疗计划")
        self.assertIn("不写“诊断明确”", plan_sec.get("hint", ""))

    def test_counterfactual_prefix_non_anticipation(self):
        j, specs, st, _ = make("CF1", 3)
        j["los"] = 5
        while len(j["daily"]) < 6:
            j["daily"].append(dict(j["daily"][-1], day=len(j["daily"]), T=36.6, P=78, R=18, SBP=122, DBP=76, symptoms=["咳嗽"]))
        j["surgical"] = False
        j["surgery_day"] = None
        j["procedures"] = []
        j["exams"] = [e for e in j["exams"] if e["result_day"] <= 1]
        j["dx_events"] = [{"id": "d1", "name": j["patient"]["primary_dx"]["name"], "day": 0, "state": "疑似"},
                          {"id": "d2", "name": j["patient"]["primary_dx"]["name"], "day": 2, "state": "确诊"}]
        early_specs = [
            {"k": 0, "day": 0, "role": "HR", "doc_type": "hospital_record", "doc_type_cn": "入院记录", "time": "09:00"},
            {"k": 1, "day": 0, "role": "FP", "doc_type": "progress_note", "doc_type_cn": "首次病程记录", "time": "11:00"},
            {"k": 2, "day": 1, "role": "SUP", "doc_type": "progress_note", "doc_type_cn": "主治医师查房记录", "time": "09:30"},
            {"k": 3, "day": 2, "role": "DAILY", "doc_type": "progress_note", "doc_type_cn": "日常病程记录", "time": "10:00"},
        ]
        specs_a = early_specs + [{"k": 4, "day": 5, "role": "DS", "doc_type": "discharge_summary", "doc_type_cn": "出院记录", "time": "09:00"}]
        j_cf = copy.deepcopy(j)
        j_cf["surgical"] = True
        j_cf["surgery_day"] = 4
        j_cf["procedures"] = [{"id": "a1", "kind": "手术", "name": "腹腔镜胆囊切除术", "side": "无", "decide_day": 3, "do_day": 4, "anesthesia": "全身麻醉", "source": "key"}]
        j_cf["exams"].append({"id": "e99", "key": "腹部超声", "region": "腹", "name": "腹部超声", "kind": "影像", "order_day": 3, "result_day": 4, "items": [], "abnormal": True, "finding": "符合胆囊结石表现"})
        specs_b = early_specs + [
            {"k": 4, "day": 3, "role": "PRE", "doc_type": "progress_note", "doc_type_cn": "术前小结", "time": "16:00"},
            {"k": 5, "day": 4, "role": "POST", "doc_type": "progress_note", "doc_type_cn": "术后首次病程记录", "time": "15:00"},
            {"k": 6, "day": 5, "role": "DS", "doc_type": "discharge_summary", "doc_type_cn": "出院记录", "time": "09:00"},
        ]
        for s_a, s_b in zip(early_specs, early_specs):
            pa = compiler.compile_doc(j, s_a, specs_a, st, random.Random(99))
            pb = compiler.compile_doc(j_cf, s_b, specs_b, st, random.Random(99))
            self.assertEqual(pa, pb, s_a["role"])

    def test_ds_course_calendar_dates_no_day0_and_order(self):
        j, specs, st, _ = make("DSDT", 4)
        ds = next(s for s in specs if s["role"] == "DS")
        plans = compiler.compile_doc(j, ds, specs, st, random.Random(4))
        course = next(x for x in plans if x["section"] == "诊疗经过")
        texts = [f["text"] for f in course["facts"]]
        self.assertFalse(any("第0天" in t for t in texts), texts)
        self.assertTrue(any(j["admit_date"] + "查" in t for t in texts), texts)

    def test_placeholder_mask_unmask_and_gate(self):
        src = "白细胞8.61×10^9/L，中性粒细胞86.5%，2型糖尿病，血压148/81mmHg，2025-01-12行手术"
        dmap, nmap = {}, {}
        masked = writer.mask_text_with_map(src, dmap, nmap, allow_new=True)
        self.assertEqual(masked, "白细胞<N1>×10^9/L，中性粒细胞<N2>%，2型糖尿病，血压<N3>/<N4>mmHg，<D1>行手术")
        self.assertEqual(writer.unmask_text(masked, dmap, nmap), src)
        self.assertEqual(writer.unmask_text("白细胞<N1>×10^9/L<N99>", dmap, nmap, strip_unknown=False), "白细胞8.61×10^9/L<N99>")
        self.assertIn("prompt_leak", gate.check_section({"section": "正文", "target_chars": 5, "facts": [], "allowed": ["8.61"]}, "白细胞8.61×10^9/L<N99>"))
        self.assertEqual(writer.unmask_text("白细胞<N1>×10^9/L<N99>", dmap, nmap, strip_unknown=True), "白细胞8.61×10^9/L")

        j, specs, st, _ = make("PH1", 5)
        hr = next(s for s in specs if s["role"] == "HR")
        pl = next(x for x in compiler.compile_doc(j, hr, specs, st, random.Random(5)) if x["section"] == "体格检查")
        mpl, dp, np_ = writer.mask_plan(pl)
        prompt = writer.section_prompt(j, hr, mpl)
        if np_ or dp:
            self.assertIn("占位符", prompt)

    def test_v0213_date_clean_future_gate_and_leak_tightening(self):
        # 1. Extended _DATE_TOK (Chinese/dot/slash/month-day) & writer.clean period+semi cleanup
        dmap, nmap = {}, {}
        s_in = "2025年5月3日行手术，2025.05.04及5月6日复查，3个月前起病，2型糖尿病。"
        masked = writer.mask_text_with_map(s_in, dmap, nmap, allow_new=True)
        self.assertEqual(masked, "<D1>行手术，<D2>及<D3>复查，<N1>个月前起病，2型糖尿病。")
        self.assertEqual(writer.unmask_text(masked, dmap, nmap), s_in)
        self.assertEqual(writer.clean("出院医嘱：按时服药。；出院后不适随诊，。", "出院医嘱"), "按时服药。出院后不适随诊。")

        # 2. future_date check & repair (including PRE surgical exemption & fallback to 近期)
        pl = {"section": "正文", "role": "DAILY", "doc_date": "2025-05-02", "target_chars": 5, "facts": [], "allowed": []}
        self.assertEqual(gate.check_section(pl, "患者今日病情平稳，拟5月5日复查血常规。"), [])
        bad_txt = "患者今日病情平稳，2025-05-09复查白细胞正常。"
        fails = gate.check_section(pl, bad_txt)
        self.assertIn("future_date", fails)
        self.assertEqual(gate.repair_section(bad_txt, fails, pl), "患者今日病情平稳。")
        only_bad = "2025-05-09复查胸部CT。"
        self.assertEqual(gate.repair_section(only_bad, ["future_date"], pl), "近期复查胸部CT。")
        pre_pl = {"section": "手术指征", "role": "PRE", "doc_date": "2025-05-02", "target_chars": 5, "facts": [], "allowed": []}
        self.assertEqual(gate.check_section(pre_pl, "定于2025年5月3日行手术。"), [])

        # 3. Tightened O/B checks: "进一步" no longer exempts result_unsourced; earlier-clause "需" no longer exempts excl
        rpl = {"section": "正文", "target_chars": 5, "facts": [], "allowed": [], "result_keys": ["血常规"]}
        self.assertIn("result_unsourced", gate.check_section(rpl, "胸部CT示右下肺占位需进一步明确。"))
        self.assertIn("result_unsourced", gate.check_section(rpl, "完善胸部CT示右下肺占位。"))
        dx_pl = {"section": "鉴别诊断", "target_chars": 5, "facts": [], "allowed": [], "unsettled_dx": ["肺结核"]}
        self.assertIn("dx_settled_early", gate.check_section(dx_pl, "患者咳嗽需治疗，目前已排除肺结核。"))

        # 4. future_leak catches future imaging exam even without lab numerics
        j, specs, st, _ = make("FLK", 7)
        j["exams"] = [{"id": "e1", "key": "CT", "region": "胸", "name": "胸部CT", "kind": "影像", "order_day": 0, "result_day": 4, "items": []}]
        docs = [{"k": 0, "role": "HR", "doc_type": "hospital_record", "day_offset": 0, "text": "入院记录\n体格检查：胸部CT示右肺大片影。"}]
        self.assertIn(0, gate.future_leak(j, docs))

        # 5. Pre-op FP before_op_done & future_leak procedure catch + valid plan preservation
        fp_pl = {"section": "诊疗计划", "role": "FP", "before_op": True, "target_chars": 5, "facts": [], "allowed": []}
        fp_bad = "拟明日行左腹股沟疝修补术，术后6小时去枕平卧，安返病房后监测体征。"
        fp_fails = gate.check_section(fp_pl, fp_bad)
        self.assertIn("before_op_done", fp_fails)
        self.assertEqual(gate.repair_section(fp_bad, fp_fails, fp_pl), "拟明日行左腹股沟疝修补术。")
        j["procedures"] = [{"id": "a1", "kind": "手术", "name": "左腹股沟疝修补术", "do_day": 1}]
        fp_doc = [{"k": 1, "role": "FP", "doc_type": "progress_note", "day_offset": 0,
                   "text": "首次病程记录\n2025-05-01 11:00\n诊疗计划：已完成腹股沟疝修补，术后6小时监测生命体征。"}]
        self.assertIn(1, gate.future_leak(j, fp_doc))


class TestV030(unittest.TestCase):
    """v0.3.0：路线B融合、四类主诉状态机、精确术后天数与查体极性、禁顿号切句、LLM 上游延续与重试纠正。"""

    def test_kb_v22_load_match_and_b_only_pipeline(self):
        from medcourse import kb, plan_diag

        kb_data = kb.load_default_kb()
        self.assertIsNotNone(kb_data)
        self.assertEqual(len(kb_data["entries"]), 17)
        self.assertEqual(kb.match_kb(kb_data, "K80", "胆囊结石合并急性胆囊炎")["icd10"], "K80.0_acute")
        self.assertEqual(kb.match_kb(kb_data, "Z51", "结肠癌术后化疗", tkind="化疗")["icd10"], "C18_chemo")
        st = kb.build_b_only_stats(kb_data)
        planner.require_stats(st)
        rows = dict(plan_diag.diagnose(st, 60, 20261121))
        self.assertEqual(rows.get("plan_check_failed", -1), 0)
        self.assertEqual(rows.get("no_patient", 0), 0)
        self.assertGreater(rows.get("kb_matched", 0), 20)
        for k in ("results_never_must", "ds_admission_facts_leak", "past_denies_listed_dx", "comorb_wrong_sex_or_age", "before_op_done_fact", "interv_incision_fact"):
            self.assertEqual(rows.get(k, 0), 0, (k, rows.get(k)))
        w = writer.TemplateWriter()
        kept = 0
        for i in range(30):
            _, docs, fails = pipeline.generate_one(f"KB0_{i:03d}", st, w, 20261121 + i, None, {})
            if docs is not None and fails == []:
                kept += 1
        self.assertEqual(kept, 30)

    def test_chief_4class_no_amenorrhea_gone_and_lesion_persistence(self):
        st_preg = copy.deepcopy(fixtures_mc.stats())
        st_preg["joint"]["items"] = {"女|18-40|O": 50}
        st_preg["code3"]["O"] = {"n": 50, "items": {"O80": 50}}
        st_preg["dx_names"]["O80"] = {"n": 50, "items": {"孕39周待产": 50}}
        st_preg["chief_symptoms"]["O80"] = {"n": 50, "items": {"停经": 50}}
        st_preg["durations"]["O80"] = {"n": 50, "items": {"1周": 50}}
        st_preg["symptoms"]["O80"] = {"n": 50, "items": {"停经": 50, "腹痛": 20}}
        j = planner.plan_journey(st_preg, random.Random(11), "OB_V03")
        self.assertEqual(j["chief"]["category"], "obstetric")
        self.assertTrue(all("停经" not in d["symptoms"] for d in j["daily"]))
        specs = planner.schedule(j, st_preg, random.Random(11))
        for s in specs:
            if s["role"] in compiler.COURSE:
                for pl in compiler.compile_doc(j, s, specs, st_preg, random.Random(11)):
                    self.assertFalse(any("停经已消失" in f["text"] for f in pl["facts"]))

        # 足月妊娠禁开米非司酮等流产药
        st_m = copy.deepcopy(st_preg)
        st_m["med_class"]["O80"] = {"n": 50, "items": {"子宫收缩药": 50}}
        st_m["meds"]["O80"] = {"n": 50, "items": {"mifepristone": 50, "oxytocin": 50}}
        st_m["med_alias"]["mifepristone"] = {"n": 50, "items": {"米非司酮": 50}}
        st_m["med_dose"]["mifepristone"] = {"n": 50, "items": {"25mg": 50}}
        st_m["med_freq"]["mifepristone"] = {"n": 50, "items": {"bid": 50}}
        st_m["med_route"]["mifepristone"] = {"n": 50, "items": {"口服": 50}}
        st_m["med_alias"]["oxytocin"] = {"n": 50, "items": {"缩宫素": 50}}
        st_m["med_dose"]["oxytocin"] = {"n": 50, "items": {"10U": 50}}
        st_m["med_freq"]["oxytocin"] = {"n": 50, "items": {"qd": 50}}
        st_m["med_route"]["oxytocin"] = {"n": 50, "items": {"静滴": 50}}
        j_m = planner.plan_journey(st_m, random.Random(3), "OB_MED")
        self.assertTrue(all(m["drug_id"] != "mifepristone" for m in j_m["meds"]))

        # scheduled 化疗住院 HR 现病史明确标明本次化疗尚未实施
        st_c = copy.deepcopy(fixtures_mc.stats())
        slot_c = {"slot_id": "Z51", "sex": "女", "age": 55, "dept": "肿瘤内科", "dx_name": "结肠癌术后辅助化疗", "dx_code3": "Z51", "los": 4, "surgical": False}
        jc, sc, stc, rc = make("Z51C", 5, slot_c, st_c)
        self.assertEqual(jc["chief"]["category"], "scheduled")
        hpi_c = next(x for x in compiler.compile_doc(jc, sc[0], sc, stc, rc) if x["section"] == "现病史")
        self.assertIn("尚未实施", hpi_c["facts"][0]["text"])

    def test_postop_day_exact_and_preop_wound_drain_gate(self):
        st = fixtures_mc.stats()
        slot_s = {"slot_id": "SG3", "sex": "男", "age": 48, "dept": "肝胆外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80", "los": 6, "surgical": True, "surgery_day": 1, "procedure": "腹腔镜胆囊切除术"}
        js, specs, sts, rngs = make("SG3", 7, slot_s, st)
        d4 = next((s for s in specs if s["role"] in compiler.COURSE and s["day"] == 4), None)
        if d4 is None:
            d4 = {"k": 4, "doc_type": "progress_note", "doc_type_cn": "日常病程记录", "role": "DAILY", "day": 4, "time": "09:00"}
            specs = [s for s in specs if s["day"] < 4] + [d4] + [s for s in specs if s["day"] >= 4]
        pl = compiler.compile_doc(js, d4, specs, sts, rngs)[0]
        self.assertEqual(pl["postop_day"], 3)
        self.assertIn("术后第3天", "".join(f["text"] for f in pl["facts"]))
        self.assertEqual(gate.check_section({"section": "正文", "role": "DAILY", "postop_day": 3, "target_chars": 5, "facts": []}, "术后第3天，患者切口愈合可。"), [])
        self.assertEqual(gate.check_section({"section": "正文", "role": "DAILY", "postop_day": 3, "target_chars": 5, "facts": []}, "术后第三天，患者切口愈合可。"), [])
        self.assertIn("postop_day_mismatch", gate.check_section({"section": "正文", "role": "DAILY", "postop_day": 3, "target_chars": 5, "facts": []}, "术后第一天，患者一般情况可。"))
        self.assertIn("postop_day_mismatch", gate.check_section({"section": "手术简要经过", "role": "POST", "postop_day": 0, "target_chars": 5, "facts": []}, "手术顺利，术后第1天安返病房。"))

        # 术前伤口敷料/引流管隐性未来泄露（节级门禁 + future_leak）
        pre_pl = {"section": "正文", "role": "SUP", "before_op": True, "target_chars": 5, "facts": [], "allowed": []}
        self.assertIn("before_op_done", gate.check_section(pre_pl, "拟明日行腹腔镜阑尾切除术，目前伤口敷料干洁，腹腔引流管引流通畅。"))
        js["procedures"] = [{"id": "a1", "kind": "手术", "name": "腹腔镜阑尾切除术", "do_day": 2}]
        pre_doc = [{"k": 2, "role": "SUP", "doc_type": "progress_note", "day_offset": 1, "text": "2025-05-02 09:00\n今日拟行手术。切口敷料干洁，腹腔引流管引流通畅。"}]
        self.assertIn(2, gate.future_leak(js, pre_doc))

        # 异常化验写“未见明显异常”拦截
        self.assertIn("lab_abn_called_normal", gate.check_section({"section": "现病史", "has_abn_labs": True, "target_chars": 5, "facts": []}, "入院前查血常规未见明显异常。"))

    def test_clause_split_preserves_dunhao_negation_and_ped_filter(self):
        pl = {"section": "既往史", "known_past": ["高血压"], "child_past": False, "target_chars": 5, "facts": []}
        txt = "既往有高血压病史，否认高血压、冠心病、糖尿病病史。"
        fixed = gate.repair_section(txt, ["past_contradict"], pl)
        self.assertNotIn("冠心病", fixed)
        self.assertNotIn("糖尿病", fixed)
        self.assertFalse(planner.dx_ok("黏膜皮肤淋巴结综合征", "男", 82))
        self.assertFalse(planner.dx_ok("手足口病", "女", 75))

    def test_llm_writer_carry_over_no_tail_splice_and_retry_guidance(self):
        seen_prompts = []
        seen_facts = []

        class FakeLLM(writer.LLMWriter):
            def __init__(self):
                self.raw = False
                self.num_placeholder = False
                self.splice_copy = False
                self.calls = self.prompt_tokens = self.completion_tokens = 0
                self.seconds = 0.0
                self.no_filler = False
                self.no_filler2 = True
                self.prev_note = False

            def write(self, j, spec, plan, attempt):
                self.calls += 1
                seen_facts.append((spec["role"], [f["kind"] for f in plan["facts"]]))
                p_txt = writer.section_prompt(j, spec, plan)
                seen_prompts.append((spec["role"], attempt, p_txt))
                if spec["role"] == "DAILY" and attempt == 0:
                    return "太短"
                return "。".join(f["text"].rstrip("。") for f in plan["facts"]) + "。SENT_END。"

        st = fixtures_mc.stats()
        j, docs, fails = pipeline.generate_one("LLM_V03", st, FakeLLM(), 20261121, None, {})
        self.assertEqual(fails, [])
        self.assertIsNotNone(docs)
        # 无 post-hoc 尾句拼接：DAILY 正文仅包含 FakeLLM 自身返回结尾
        for d in docs:
            if d["role"] == "DAILY":
                self.assertTrue(d["text"].rstrip().endswith("SENT_END。"), d["text"])
        # DAILY 首次 too_short 后，第二次写入收到了 retry guidance
        daily_prompts = [(a, p) for r, a, p in seen_prompts if r == "DAILY"]
        self.assertTrue(any("上一次生成校验未通过" in p and "长度" in p for a, p in daily_prompts if a >= 1), daily_prompts)


class TestV031(unittest.TestCase):
    """0.3.1：产科破水孕周与急症病程、儿科肾贫血化验与单次极量、术前屏蔽术后病理、术后日期去歧义、面板化验矛盾与新闸门。全部虚构数据。"""

    def test_obstetric_prom_weeks_acute_dur_and_no_unperformed_delivery_comorb(self):
        self.assertEqual(icd.week_range("胎膜早破"), (37, 41))
        self.assertEqual(icd.week_range("前置胎盘"), (28, 39))
        self.assertEqual(icd.week_range("先兆流产"), (6, 13))
        st = copy.deepcopy(fixtures_mc.stats())
        st["chief_symptoms"]["O42"] = {"n": 50, "items": {"停经": 50, "阴道流液": 50}}
        st["symptoms"]["O42"] = {"n": 50, "items": {"停经": 50, "阴道流液": 50}}
        st["durations"]["O"] = {"n": 100, "items": {"38周": 100}}
        st["comorbid"]["O42"] = {"n": 100, "items": {"单胎活产": 90, "头位顺产": 90, "剖宫产术后": 60, "轻度贫血": 50}}
        st["real_rates"] = {"ds_dx_admissions": 100, "ds_dx_items": 300, "b3_admissions": 100, "b3_permille_sum": 40000}
        slot = {"slot_id": "O42", "sex": "女", "age": 29, "dept": "产科", "dx_name": "胎膜早破", "dx_code3": "O42", "los": 4, "surgical": False}
        saw_combo = False
        for s in range(25):
            j = planner.plan_journey(st, random.Random(s), f"PROM{s}", slot)
            self.assertGreaterEqual(j["patient"]["gest_weeks"], 37)
            ct = planner.chief_text(j)
            if "阴道流液" in ct:
                saw_combo = True
                self.assertRegex(ct, r"^(?:停经\d+周，)?阴道流液\d+(?:小时|天)$")
                self.assertNotIn("阴道流液38周", ct)
            names = [c["name"] for c in j["patient"]["comorbidities"]]
            self.assertNotIn("单胎活产", names)
            self.assertNotIn("头位顺产", names)
            for c in j["patient"]["comorbidities"]:
                if "术后" in c["name"]:
                    self.assertFalse(c["at_admission"])
        self.assertTrue(saw_combo)

    def test_pediatric_ckd_anemia_labs_and_child_dose_ceiling(self):
        self.assertFalse(planner._child_dose_ok({"age": 2, "weight_kg": 12.0}, "pip_tazo", "哌拉西林他唑巴坦", "4.5g"))
        self.assertFalse(planner._child_dose_ok({"age": 2, "weight_kg": 12.0}, "vancomycin", "万古霉素", "1000mg"))
        self.assertTrue(planner._child_dose_ok({"age": 2, "weight_kg": 12.0}, "pip_tazo", "哌拉西林他唑巴坦", "1.2g"))
        st = copy.deepcopy(fixtures_mc.stats())
        st["lab_state"]["kid:*"] = {"n": 100, "items": {"CREA|norm": 95, "CREA|abn": 5, "BUN|norm": 95, "BUN|abn": 5, "HGB|norm": 95, "HGB|abn": 5}}
        st["med_class"]["N18"] = {"n": 50, "items": {"抗菌药": 50}}
        st["meds"]["N18"] = {"n": 50, "items": {"pip_tazo": 30, "vancomycin": 25}}
        st["meds"]["kid:*"] = {"n": 50, "items": {"pip_tazo": 30, "vancomycin": 25}}
        st["med_alias"].update({"pip_tazo": {"n": 50, "items": {"哌拉西林他唑巴坦": 50}}, "vancomycin": {"n": 50, "items": {"万古霉素": 50}}})
        st["med_dose"].update({"pip_tazo": {"n": 50, "items": {"4.5g": 50}}, "vancomycin": {"n": 50, "items": {"1000mg": 50}}})
        st["med_freq"].update({"pip_tazo": {"n": 50, "items": {"q8h": 50}}, "vancomycin": {"n": 50, "items": {"q12h": 50}}})
        st["med_route"].update({"pip_tazo": {"n": 50, "items": {"静滴": 50}}, "vancomycin": {"n": 50, "items": {"静滴": 50}}})
        slot = {"slot_id": "CKD2", "sex": "男", "age": 2, "dept": "儿科", "dx_name": "慢性肾脏病5期", "dx_code3": "N18", "los": 5, "surgical": False}
        j = planner.plan_journey(st, random.Random(11), "CKD2", slot)
        j["patient"]["comorbidities"].append({"name": "肾性贫血", "at_admission": True})
        lctx = planner.lab_ctx(random.Random(11), st, planner.strata_of("慢性肾脏病5期", "N18"), j["patient"])
        crea = planner.draw_lab(random.Random(12), "CREA", lctx)
        bun = planner.draw_lab(random.Random(13), "BUN", lctx)
        hgb = planner.draw_lab(random.Random(14), "HGB", lctx)
        self.assertEqual(crea["flag"], "偏高")
        self.assertGreaterEqual(float(crea["value"]), 180.0)
        self.assertEqual(bun["flag"], "偏高")
        self.assertEqual(hgb["flag"], "偏低")
        for m in j["meds"]:
            self.assertNotIn(m["dose"], ("4.5g", "1000mg"))
        self.assertTrue(any(m["drug_id"] in ("pip_tazo", "vancomycin") for m in j["meds"]))

    def test_preop_excludes_postop_pathology_and_postop_date_unambiguous(self):
        st = fixtures_mc.stats()
        slot = {"slot_id": "OP0", "sex": "男", "age": 48, "dept": "普外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80",
                "los": 5, "surgical": True, "surgery_day": 0, "procedure": "腹腔镜胆囊切除术"}
        j, specs, _, rng = make("OP0", 9, slot, st)
        j["exams"] = [e for e in j["exams"] if e["name"] != "术后病理"] + [
            {"id": "epath", "key": "病理", "region": None, "name": "术后病理", "kind": "病理", "order_day": 0, "result_day": 3, "items": []}
        ]
        fp = next(s for s in specs if s["role"] == "FP")
        fp_plans = compiler.compile_doc(j, fp, specs, st, rng)
        fp_text = "".join(f["text"] for pl in fp_plans for f in pl["facts"])
        self.assertNotIn("术后病理", fp_text)

        d2 = next((s for s in specs if s["role"] in compiler.COURSE and s["day"] == 2), None)
        if d2 is None:
            d2 = {"k": max(s["k"] for s in specs) + 1, "doc_type": "progress_note", "doc_type_cn": "日常病程记录", "role": "DAILY", "day": 2, "time": "09:00"}
            specs = [s for s in specs if s["day"] < 2] + [d2] + [s for s in specs if s["day"] >= 2]
        pl2 = compiler.compile_doc(j, d2, specs, st, rng)[0]
        proc_f = [f["text"] for f in pl2["facts"] if f["kind"] == "procedure"][0]
        self.assertIn("今日为术后第2天", proc_f)
        self.assertIn(compiler.doc_date(j, 0), proc_f)
        self.assertIsNone(compiler.pick_carry_fact("建议进一步完善胸部CT检查。", j, d2, 1, st, rng))

        post_s = next(s for s in specs if s["role"] == "POST")
        post_pls = compiler.compile_doc(j, post_s, specs, st, rng)
        for pl in post_pls:
            if pl["mode"] == "llm":
                self.assertIn("术后第1天", pl.get("hint") or "")

    def test_panel_scoped_lab_abn_called_normal(self):
        st = fixtures_mc.stats()
        j, specs, _, rng = make("LABP", 4, None, st)
        j["exams"] = [
            {"id": "e1", "key": "血常规", "region": None, "name": "血常规", "kind": "检验", "order_day": 0, "result_day": 0,
             "items": [{"item": "白细胞", "key": "WBC", "value": "6.5", "unit": "×10^9/L", "flag": "正常"}]},
            {"id": "e2", "key": "肝功能", "region": None, "name": "肝功能", "kind": "检验", "order_day": 0, "result_day": 0,
             "items": [{"item": "谷丙转氨酶", "key": "ALT", "value": "120", "unit": "U/L", "flag": "偏高"}]},
        ]
        hr_pl = next(x for x in compiler.compile_doc(j, specs[0], specs, st, rng) if x["section"] == "现病史")
        self.assertEqual(hr_pl["abn_lab_keys"], ["肝功能"])
        self.assertEqual(gate.check_section({"section": "现病史", "abn_lab_keys": ["肝功能"], "has_abn_labs": True, "target_chars": 5, "facts": []},
                                            "血常规未见异常，肝功能示谷丙转氨酶偏高。"), [])
        self.assertEqual(gate.check_section({"section": "现病史", "abn_lab_keys": ["肝功能"], "has_abn_labs": True, "target_chars": 5, "facts": [], "allowed": ["120"]},
                                            "肝功能示谷丙转氨酶120U/L，血常规未见异常。"), [])
        self.assertIn("lab_abn_called_normal", gate.check_section({"section": "现病史", "abn_lab_keys": ["肝功能"], "has_abn_labs": True, "target_chars": 5, "facts": []},
                                                                   "肝功能未见明显异常。"))
        self.assertIn("lab_abn_called_normal", gate.check_section({"section": "现病史", "abn_lab_keys": ["肝功能"], "has_abn_labs": True, "target_chars": 5, "facts": []},
                                                                   "上述指标在正常范围内。"))
        self.assertIn("lab_abn_called_normal", gate.check_section({"section": "现病史", "abn_lab_keys": ["心肌酶"], "has_abn_labs": True, "target_chars": 5, "facts": [], "allowed": ["0.923"]},
                                                                   "肌钙蛋白I 0.923ng/mL，未见明显升高。"))

    def test_v031_gates_wound_senior_pe_vitals_loop_and_adm_postop_day(self):
        self.assertIn("nonsurg_postop", gate.check_section({"target_chars": 5, "facts": [], "nonsurgical": True}, "出院后注意保持伤口清洁干燥。"))
        self.assertEqual(gate.check_section({"target_chars": 5, "facts": [], "nonsurgical": True}, "门诊随诊，适度休息。"), [])
        self.assertIn("senior_in_daily", gate.check_section({"section": "正文", "role": "DAILY", "target_chars": 5, "facts": []}, "今日主任医师查房：患者平稳。"))
        self.assertEqual(gate.check_section({"section": "正文", "role": "SUP", "target_chars": 5, "facts": []}, "今日主任医师查房：患者平稳。"), [])
        self.assertIn("pe_dup_vitals", gate.check_section({"section": "体格检查", "program_prefix": "体温36.5℃...", "target_chars": 5, "facts": []},
                                                           "体温、脉搏、呼吸、血压已记录。双肺呼吸音清。"))
        self.assertIn("pe_dup_vitals", gate.check_section({"section": "体格检查", "program_prefix": "体温36.5℃...", "target_chars": 5, "facts": []},
                                                           "T 36.7℃，P 78次/分，双肺清。"))
        self.assertIn("prompt_leak", gate.check_section({"target_chars": 5, "facts": []}, "延续情况：患者无发热。"))

        pfx_loop = "".join(f"患者无特殊症状第{i}项表现。" for i in range(10))
        self.assertIn("repeat", gate.check_section({"target_chars": 10, "facts": []}, pfx_loop))
        trimmed = gate.repair_section(pfx_loop, ["repeat"])
        self.assertNotIn("repeat", gate.check_section({"target_chars": 10, "facts": []}, trimmed))

        st = fixtures_mc.stats()
        slot = {"slot_id": "SG_ADM", "sex": "女", "age": 52, "dept": "普外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80",
                "los": 5, "surgical": True, "surgery_day": 1, "procedure": "腹腔镜胆囊切除术"}
        j, specs, _, rng = make("SG_ADM", 1, slot, st)
        w, docs = writer.TemplateWriter(), []
        for s in specs:
            txt, _ = pipeline.write_doc(j, s, specs, st, w, rng, {}, None)
            docs.append({"doc_type": s["doc_type"], "day_offset": s["day"], "text": txt, "doc_type_cn": s["doc_type_cn"], "k": s["k"], "role": s["role"]})
        self.assertEqual(gate.check_admission(j, docs), [])
        bad_post = copy.deepcopy(docs)
        pk = next(d["k"] for d in bad_post if d["role"] == "POST")
        bad_post[pk]["text"] = bad_post[pk]["text"].replace("手术顺利", "术后第1天手术顺利")
        self.assertIn((pk, "postop_day_mismatch"), gate.check_admission(j, bad_post))


class TestV032(unittest.TestCase):
    def test_v032_diag_imaging_coupling_and_preop_clamp(self):
        st_b = kb.build_b_only_stats()
        cases = [
            {"slot_id": "D_K35", "sex": "男", "age": 26, "dept": "普外科", "dx_name": "急性化脓性阑尾炎", "dx_code3": "K35",
             "los": 6, "surgical": True, "surgery_day": 1, "procedure": "腹腔镜阑尾切除术"},
            {"slot_id": "D_I21", "sex": "男", "age": 62, "dept": "心内科", "dx_name": "急性心肌梗死", "dx_code3": "I21",
             "los": 9, "surgical": True, "surgery_day": 1, "procedure": "经皮冠状动脉介入术"},
            {"slot_id": "D_N20", "sex": "男", "age": 42, "dept": "泌尿外科", "dx_name": "右侧输尿管结石", "dx_code3": "N20",
             "los": 5, "surgical": True, "surgery_day": 1, "procedure": "经尿道右侧输尿管软镜钬激光碎石取石术"},
            {"slot_id": "D_J15", "sex": "女", "age": 54, "dept": "呼吸内科", "dx_name": "社区获得性肺炎", "dx_code3": "J15",
             "los": 7, "surgical": False},
        ]
        for idx, slot in enumerate(cases):
            j, _, _, _ = make(f"V32_{idx}", 10 + idx, slot, st_b)
            diag_imgs = [e for e in j["exams"] if e.get("diagnostic") and e.get("kind") in ("影像", "造影", "内镜")]
            self.assertTrue(diag_imgs, f"{slot['slot_id']} missing diagnostic imaging: {j['exams']}")
            self.assertTrue(all(e.get("abnormal") and e.get("finding") for e in diag_imgs))
            if slot["surgical"]:
                do_d = j["procedures"][0]["do_day"]
                for e in diag_imgs:
                    if e["kind"] == "影像" and not e.get("with_proc"):
                        self.assertLessEqual(e["result_day"], do_d)

        exams = [
            {"id": "e1", "key": "超声", "region": "腹", "name": "腹部超声", "kind": "影像", "order_day": 0, "result_day": 0, "items": [], "abnormal": False},
            {"id": "e2", "key": "心脏超声", "region": None, "name": "心脏超声", "kind": "影像", "order_day": 0, "result_day": 0, "items": [], "abnormal": False},
        ]
        planner.mark_exam_findings({}, ["*"], exams, "急性阑尾炎", {"腹"})
        self.assertTrue(exams[0]["diagnostic"] and exams[0]["abnormal"])
        self.assertIn("急性阑尾炎", exams[0]["finding"])
        self.assertFalse(exams[1]["diagnostic"] or exams[1]["abnormal"])

    def test_v032_same_day_preop_and_no_incision_adm(self):
        st = fixtures_mc.stats()
        slot_sg = {"slot_id": "SG32", "sex": "女", "age": 52, "dept": "普外科", "dx_name": "胆囊结石伴急性胆囊炎", "dx_code3": "K80",
                   "los": 5, "surgical": True, "surgery_day": 1, "procedure": "腹腔镜胆囊切除术"}
        j, specs, _, rng = make("SG32", 1, slot_sg, st)
        w, docs = writer.TemplateWriter(), []
        for s in specs:
            txt, _ = pipeline.write_doc(j, s, specs, st, w, rng, {}, None)
            docs.append({"doc_type": s["doc_type"], "day_offset": s["day"], "text": txt, "doc_type_cn": s["doc_type_cn"], "k": s["k"], "role": s["role"]})
        self.assertEqual(gate.check_admission(j, docs), [])

        bad_same = copy.deepcopy(docs)
        pre_idx = next(i for i, d in enumerate(bad_same) if d["role"] == "PRE")
        bad_same[pre_idx]["day_offset"] = j["procedures"][0]["do_day"]
        bad_same[pre_idx]["text"] += "\n术后第2天患者可进流食，胆囊切除已完成。"
        self.assertIn(bad_same[pre_idx]["k"], gate.future_leak(j, bad_same))
        self.assertIn((bad_same[pre_idx]["k"], "postop_day_mismatch"), gate.check_admission(j, bad_same))

        self.assertIn("no_incision_leak", writer.RETRY_GUIDANCE)
        st_b = kb.build_b_only_stats()
        slot_ni = {"slot_id": "NI32", "sex": "男", "age": 42, "dept": "泌尿外科", "dx_name": "右侧输尿管结石", "dx_code3": "N20",
                   "los": 5, "surgical": True, "surgery_day": 1, "procedure": "经尿道右侧输尿管软镜钬激光碎石取石术"}
        j2, specs2, _, rng2 = make("NI32", 2, slot_ni, st_b)
        self.assertTrue(compiler.no_incision(j2["procedures"][0]))
        post_spec2 = next(s for s in specs2 if s["role"] == "POST")
        post_pls2 = compiler.compile_doc(j2, post_spec2, specs2, st_b, rng2)
        for pl in post_pls2:
            if pl["mode"] == "llm":
                self.assertTrue(pl.get("no_incision"))
                self.assertIn("无切口", pl.get("hint") or "")
        docs2 = []
        for s in specs2:
            txt, _ = pipeline.write_doc(j2, s, specs2, st_b, w, rng2, {}, None)
            docs2.append({"doc_type": s["doc_type"], "day_offset": s["day"], "text": txt, "doc_type_cn": s["doc_type_cn"], "k": s["k"], "role": s["role"]})
        bad_ni = copy.deepcopy(docs2)
        pk2 = next(d["k"] for d in bad_ni if d["role"] == "POST")
        bad_ni[pk2]["text"] += " 切口敷料干洁，无渗血。"
        self.assertIn((pk2, "no_incision_leak"), gate.check_admission(j2, bad_ni))

    def test_v032_pe_dup_vitals_ds_admit_date_and_sym_age(self):
        self.assertIn("pe_dup_vitals", gate.check_section(
            {"section": "体格检查", "program_prefix": "体温36.5℃...", "target_chars": 5, "facts": []},
            "体温、脉搏、呼吸、血压已另录。双肺呼吸音清。"))
        self.assertIn("pe_dup_vitals", gate.check_section(
            {"section": "体格检查", "program_prefix": "体温36.5℃...", "target_chars": 5, "facts": []},
            "体温、呼吸、血压见前录，腹部平坦。"))
        self.assertIn("pe_dup_vitals", gate.check_section(
            {"section": "体格检查", "program_prefix": "体温36.5℃...", "target_chars": 5, "facts": []},
            "脉搏、血压如前所述，腹部平坦。"))

        st = fixtures_mc.stats()
        j, specs, _, rng = make("DS32", 3, None, st)
        ds_spec = next(s for s in specs if s["role"] == "DS")
        ds_pls = compiler.compile_doc(j, ds_spec, specs, st, rng)
        course_pl = next(x for x in ds_pls if x["section"] == "诊疗经过")
        self.assertTrue(any(f"患者于{j['admit_date']}入院后完善相关检查" in str(f) for f in course_pl.get("facts", [])))

        self.assertFalse(planner.sym_ok("胸骨后压榨性疼痛", {"age": 4, "sex": "男"}))
        self.assertFalse(planner.sym_ok("绝经后阴道流血", {"age": 5, "sex": "女"}))
        self.assertFalse(planner.sym_ok("前列腺增生伴排尿困难", {"age": 6, "sex": "男"}))
        self.assertTrue(planner.sym_ok("胸骨后压榨性疼痛", {"age": 45, "sex": "男"}))

    def test_v032_repair_drop_clauses_and_kb_b_only_lifts(self):
        repaired = gate.repair_section("患者发热，切口敷料干洁，生命体征平稳。", ["no_incision_leak"])
        self.assertIn("患者发热", repaired)
        self.assertIn("生命体征平稳", repaired)
        self.assertNotIn("切口", repaired)

        st_b = kb.build_b_only_stats()
        global_items = st_b["exams"]["*"]["items"]
        self.assertIn("CT|胸", global_items)
        self.assertIn("超声|腹", global_items)
        lifts = planner.exam_lifts(st_b, ["J15", "J", "*"], {"肺"})
        self.assertGreater(lifts.get("CT|胸", 0.0), 1.0)


class TestV033(unittest.TestCase):
    def test_s4_counterfactual_prefix_non_anticipation(self):
        pools = [(fixtures_mc.stats(), "FIX"), (kb.build_b_only_stats(), "KBB")]
        checked_specs = 0
        for st, tag in pools:
            for i in range(20):
                rng = random.Random(20261125 + i)
                j = planner.plan_journey(st, rng, f"{tag}{i:02d}", None)
                specs = planner.schedule(j, st, rng)
                for s in specs:
                    if s["role"] == "DS":
                        continue
                    checked_specs += 1
                    day = s["day"]
                    j2 = copy.deepcopy(j)
                    j2["los"] = j["los"] + 4
                    for t in range(day + 1, len(j2["daily"])):
                        j2["daily"][t]["T"] = 38.9
                        j2["daily"][t]["symptoms"] = ["新发未来高热"]
                    for ed in range(len(j2["daily"]), j2["los"] + 1):
                        j2["daily"].append({"day": ed, "T": 38.9, "P": 108, "R": 24, "SBP": 140, "DBP": 88, "trend": "平稳", "symptoms": ["新发未来高热"]})
                    for e in j2["exams"]:
                        if e["result_day"] > day:
                            e["result_day"] += 2
                            e["finding"] = "未来篡改结果"
                            e["abnormal"] = True
                            for it in e["items"]:
                                it["value"] = "999.9"
                                it["flag"] = "偏高"
                    j2["exams"].append({"id": "efut", "key": "CT", "region": "头", "name": "头颅CT", "kind": "影像",
                                        "order_day": day + 1, "result_day": day + 2, "items": [], "abnormal": True, "diagnostic": True, "finding": "未来梗死灶"})
                    for m in j2["meds"]:
                        if m["stop_day"] >= day:
                            m["stop_day"] = m["stop_day"] + 3
                    j2["meds"].append({"id": "mfut", "drug_id": "vancomycin", "name": "万古霉素", "class": "抗菌药", "dose": "0.5g",
                                       "freq": "q12h", "route": "静滴", "start_day": day + 1, "stop_day": day + 4, "discharge": False})
                    if not any(a["decide_day"] <= day for a in j2["procedures"]):
                        j2["surgical"] = True
                        j2["surgery_day"] = day + 2
                        j2["procedures"].append({"id": "afut", "kind": "手术", "name": "未来急诊手术", "side": None,
                                                 "decide_day": day + 1, "do_day": day + 2, "anesthesia": "全身麻醉"})
                    j2["dx_events"].append({"id": "dfut", "name": "未来新发诊断", "day": day + 1, "state": "确诊"})
                    j2["patient"]["comorbidities"].append({"name": "未来出院新增", "at_admission": False})
                    specs2 = [copy.deepcopy(x) for x in specs if x["k"] <= s["k"]] + [
                        {"k": s["k"] + 1, "day": day + 2, "role": "POST", "doc_type": "progress_note", "doc_type_cn": "术后首次病程记录", "time": "18:00"}
                    ]
                    for ep in (True, False):
                        p1 = compiler.compile_doc(j, s, specs, st, random.Random(77), epistemic=ep)
                        p2 = compiler.compile_doc(j2, s, specs2, st, random.Random(77), epistemic=ep)
                        self.assertEqual(p1, p2, (tag, i, s["role"], s["day"], ep))
                    self.assertEqual(compiler.copy_banned(j, day), compiler.copy_banned(j2, day))
        self.assertGreater(checked_specs, 150)

    def test_ablate_epistemic_history_parity_and_state_stripping(self):
        st = fixtures_mc.stats()
        j, specs, _, _ = make("EPI33", 5, None, st)
        dx_name = j["patient"]["primary_dx"]["name"]
        j["los"] = max(j["los"], 3)
        j["daily"] = j["daily"][:4]
        while len(j["daily"]) < 4:
            j["daily"].append(dict(j["daily"][-1], day=len(j["daily"])))
        for idx, d in enumerate(j["daily"]):
            d["day"] = idx
        j["exams"] = [
            {"id": "e1", "key": "血常规", "region": None, "name": "血常规", "kind": "检验", "order_day": 0, "result_day": 0,
             "items": [{"item": "白细胞", "key": "WBC", "value": "12.5", "unit": "×10^9/L", "flag": "偏高"}]},
            {"id": "e2", "key": "CT", "region": "胸", "name": "胸部CT", "kind": "影像", "order_day": 0, "result_day": 2,
             "items": [], "abnormal": True, "diagnostic": True, "finding": f"符合{dx_name}表现"},
            {"id": "e3", "key": "痰培养", "region": None, "name": "痰培养", "kind": "培养", "order_day": 1, "result_day": 2,
             "items": [], "abnormal": False, "diagnostic": False, "finding": "未培养出致病菌"},
        ]
        j["dx_events"] = [
            {"id": "d1", "name": dx_name, "day": 0, "state": "疑似"},
            {"id": "d2", "name": dx_name, "day": 2, "state": "确诊"},
            {"id": "d3", "name": "肺栓塞", "day": 0, "state": "疑似"},
            {"id": "d4", "name": "肺栓塞", "day": 2, "state": "排除"},
        ]
        j["differentials"] = ["肺栓塞"]
        specs = [
            {"k": 0, "day": 0, "role": "HR", "doc_type": "hospital_record", "doc_type_cn": "入院记录", "time": "09:00"},
            {"k": 1, "day": 0, "role": "FP", "doc_type": "progress_note", "doc_type_cn": "首次病程记录", "time": "10:00"},
            {"k": 2, "day": 1, "role": "SUP", "doc_type": "progress_note", "doc_type_cn": "主治医师查房记录", "time": "09:30"},
            {"k": 3, "day": 2, "role": "SUP", "doc_type": "progress_note", "doc_type_cn": "主任医师查房记录", "time": "09:30"},
            {"k": 4, "day": 3, "role": "DS", "doc_type": "discharge_summary", "doc_type_cn": "出院记录", "time": "10:00"},
        ]
        for sp in specs:
            p_full = compiler.compile_doc(j, sp, specs, st, random.Random(11), epistemic=True)
            p_vis = compiler.compile_doc(j, sp, specs, st, random.Random(11), epistemic=False)
            self.assertEqual([x["section"] for x in p_full], [x["section"] for x in p_vis])
            self.assertEqual([x.get("target_chars") for x in p_full], [x.get("target_chars") for x in p_vis])
            if sp["role"] == "FP":
                vis_disc = next(x for x in p_vis if x["section"] == "拟诊讨论")
                vis_plan = next(x for x in p_vis if x["section"] == "诊疗计划")
                self.assertTrue(any("胸部CT" in f["text"] and "结果待回报" in f["text"] for f in vis_disc["facts"]))
                self.assertTrue(any("胸部CT" in f["text"] for f in vis_plan["facts"]))
                self.assertIsNone(vis_plan.get("hint"))
                vis_adm = next(x for x in p_vis if x["section"] == "初步诊断")
                self.assertNotIn("？", vis_adm["program_text"])
            if sp["role"] == "SUP" and sp["day"] == 1:
                vis_txt = next(x for x in p_vis if x["section"] == "正文")["facts"]
                self.assertFalse(any("诊断尚待进一步明确" in f["text"] for f in vis_txt))
                self.assertTrue(any("已开具痰培养，待回报" in f["text"] for f in vis_txt))
            if sp["role"] == "SUP" and sp["day"] == 2:
                vis_txt = next(x for x in p_vis if x["section"] == "正文")["facts"]
                self.assertFalse(any("结合检查结果" in f["text"] for f in vis_txt))
                self.assertTrue(any("胸部CT示：" in f["text"] for f in vis_txt))

        tmp = tempfile.mkdtemp(prefix="epi33_")
        try:
            d_full = os.path.join(tmp, "full")
            d_vis = os.path.join(tmp, "vis")
            r1 = run_mod("medcourse.pipeline", "--kb-only", "--mode", "free", "--n", "3", "--seed", "20261125",
                         "--writer", "template", "--save-first-try", "--concurrency", "1", "--out", d_full)
            r2 = run_mod("medcourse.pipeline", "--kb-only", "--mode", "free", "--n", "3", "--seed", "20261125",
                         "--writer", "template", "--ablate-epistemic", "--save-first-try", "--concurrency", "1", "--out", d_vis)
            self.assertEqual(r1.returncode, 0, r1.stderr)
            self.assertEqual(r2.returncode, 0, r2.stderr)
            self.assertEqual(open(os.path.join(d_full, "journeys.jsonl"), encoding="utf-8").read(),
                             open(os.path.join(d_vis, "journeys.jsonl"), encoding="utf-8").read())
            self.assertTrue(os.path.isfile(os.path.join(d_full, "first_try_admissions.jsonl")))
            self.assertTrue(json.load(open(os.path.join(d_vis, "counts.json"), encoding="utf-8"))["ablate_epistemic"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_save_first_try_records_attempt0_before_gate_retry(self):
        class RetryWriter(writer.TemplateWriter):
            def write(self, j, spec, plan, attempt):
                if spec["role"] == "FP" and plan["section"] == "拟诊讨论":
                    dx = j["patient"]["primary_dx"]["name"]
                    if attempt == 0:
                        return f"根据当前表现，{dx}诊断明确。"
                    return f"目前考虑{dx}可能，待检查回报后进一步明确。"
                return super().write(j, spec, plan, attempt)

        from unittest import mock
        st = fixtures_mc.stats()
        with mock.patch.object(pipeline.planner, "P_DX_CHANGE", 1.0):
            first_docs = []
            j, docs, fails = pipeline.generate_one("FT33", st, RetryWriter(), 42, None, {}, first_docs_out=first_docs)
        self.assertIsNotNone(docs, fails)
        self.assertEqual(len(first_docs), len(docs))
        fp_first = next(d for d in first_docs if d["role"] == "FP")
        fp_final = next(d for d in docs if d["role"] == "FP")
        self.assertIn("诊断明确", fp_first["text"])
        self.assertNotIn("诊断明确", fp_final["text"])
        self.assertIn("进一步明确", fp_final["text"])


class TestV034ClinicalAntiHallucination(unittest.TestCase):
    """0.3.4：非手术外科/未分娩产科防呆、器官影像部位与不取组织病理过滤、儿科/新生儿/肿瘤化疗一致性及前缀不变性。"""

    def test_v034_nonsurg_no_incision_and_s4_invariance(self):
        st = copy.deepcopy(fixtures_mc.stats())
        p = {
            "patient_id": "V34INC", "sex": "男", "age": 42, "age_text": "42岁", "weight_kg": 68,
            "pregnant": False, "gest_weeks": None, "dept": "口腔颌面外科一区",
            "primary_dx": {"name": "腮腺良性肿物", "code3": "D11", "source": "name", "organs": ["咽喉"]},
            "comorbidities": [], "past_history": [], "allergies": [], "source": "free",
        }
        j_nosurg = {
            "journey_id": "V34INC", "contract": planner.CONTRACT_VERSION, "patient": p,
            "admit_date": "2025-05-01", "los": 3, "surgical": False, "surgery_day": None,
            "chief": {"symptoms": ["耳垂下肿物"], "duration": "6个月", "source": "name", "category": "lesion"},
            "exams": [], "meds": [], "procedures": [],
            "dx_events": [{"id": "d1", "name": "腮腺良性肿物", "day": 0, "state": "确诊"}],
            "differentials": [],
            "daily": [{"day": d, "T": 36.5, "P": 76, "R": 18, "SBP": 120, "DBP": 78, "trend": "同前" if d == 0 else "平稳", "symptoms": ["耳垂下肿物"]} for d in range(4)],
        }
        j_fut_surg = copy.deepcopy(j_nosurg)
        j_fut_surg.update(surgical=True, surgery_day=3, procedures=[
            {"id": "a1", "kind": "手术", "name": "右侧腮腺肿物切除术", "side": "右", "decide_day": 2, "do_day": 3, "anesthesia": "全身麻醉", "source": "key"}
        ])
        specs = [
            {"k": 0, "day": 0, "role": "HR", "doc_type": "hospital_record", "doc_type_cn": "入院记录", "time": "09:00"},
            {"k": 1, "day": 0, "role": "FP", "doc_type": "progress_note", "doc_type_cn": "首次病程记录", "time": "11:00"},
            {"k": 2, "day": 1, "role": "SUP", "doc_type": "progress_note", "doc_type_cn": "主治医师查房记录", "time": "09:00"},
            {"k": 3, "day": 3, "role": "DS", "doc_type": "discharge_summary", "doc_type_cn": "出院记录", "time": "10:00"},
        ]
        for sp in specs[:3]:
            p1 = compiler.compile_doc(j_nosurg, sp, specs, st, random.Random(7))
            p2 = compiler.compile_doc(j_fut_surg, sp, specs, st, random.Random(7))
            self.assertEqual(p1, p2)
        hr_pe = next(pl for pl in compiler.compile_doc(j_nosurg, specs[0], specs, st, random.Random(7)) if pl["section"] == "体格检查")
        self.assertTrue(hr_pe.get("no_incision"))
        self.assertIn("手术刀口", hr_pe.get("hint") or "")
        docs = []
        for sp in specs:
            txt, _ = pipeline.write_doc(j_nosurg, sp, specs, st, writer.TemplateWriter(), random.Random(7), {}, None)
            docs.append({"doc_type": sp["doc_type"], "day_offset": sp["day"], "text": txt, "doc_type_cn": sp["doc_type_cn"], "k": sp["k"], "role": sp["role"]})
        self.assertEqual(gate.check_admission(j_nosurg, docs), [])
        bad_docs = copy.deepcopy(docs)
        bad_docs[0]["text"] = bad_docs[0]["text"].replace("专科情况：", "专科情况：颌面部见手术刀口已缝合，")
        self.assertIn((0, "no_incision_leak"), gate.check_admission(j_nosurg, bad_docs))
        bad_sup = copy.deepcopy(docs)
        bad_sup[2]["text"] += "切口敷料干洁，引流管在位。"
        self.assertIn((2, "no_incision_leak"), gate.check_admission(j_nosurg, bad_sup))

    def test_v034_obstetric_weeks_pe_and_postpartum_guard(self):
        self.assertEqual(icd.week_range("晚期先兆流产"), (13, 27))
        self.assertEqual(icd.week_range("先兆流产"), (6, 13))
        st = copy.deepcopy(fixtures_mc.stats())
        p_early = {
            "patient_id": "V34OB", "sex": "女", "age": 27, "age_text": "27岁", "weight_kg": 54,
            "pregnant": True, "gest_weeks": 8, "dept": "产科",
            "primary_dx": {"name": "先兆流产", "code3": "O20", "source": "name", "organs": ["盆"]},
            "comorbidities": [], "past_history": [], "allergies": [], "source": "free",
        }
        self.assertFalse(planner.sym_ok("胎动正常", p_early))
        self.assertTrue(planner.sym_ok("阴道流血", p_early))
        j_ob = {
            "journey_id": "V34OB", "contract": planner.CONTRACT_VERSION, "patient": p_early,
            "admit_date": "2025-05-01", "los": 3, "surgical": False, "surgery_day": None,
            "chief": {"symptoms": ["停经", "阴道流血"], "duration": "2天", "source": "name", "category": "obstetric"},
            "exams": [], "meds": [], "procedures": [],
            "dx_events": [{"id": "d1", "name": "先兆流产", "day": 0, "state": "确诊"}],
            "differentials": [],
            "daily": [{"day": d, "T": 36.6, "P": 82, "R": 18, "SBP": 112, "DBP": 70, "trend": "同前" if d == 0 else "平稳", "symptoms": []} for d in range(4)],
        }
        pe_txt, spec_txt = kb.organ_pe_facts(j_ob, j_ob["daily"][0])
        self.assertNotIn("腹隆", pe_txt)
        self.assertNotIn("胎心音", pe_txt)
        self.assertNotIn("宫高腹围", spec_txt)
        specs = [
            {"k": 0, "day": 0, "role": "HR", "doc_type": "hospital_record", "doc_type_cn": "入院记录", "time": "09:00"},
            {"k": 1, "day": 1, "role": "SUP", "doc_type": "progress_note", "doc_type_cn": "主治医师查房记录", "time": "09:00"},
            {"k": 2, "day": 3, "role": "DS", "doc_type": "discharge_summary", "doc_type_cn": "出院记录", "time": "10:00"},
        ]
        docs = []
        for sp in specs:
            pls = compiler.compile_doc(j_ob, sp, specs, st, random.Random(5))
            if sp["role"] in ("SUP", "DS"):
                for pl in pls:
                    if pl["mode"] == "llm":
                        self.assertTrue(pl.get("undelivered_preg"))
            txt, _ = pipeline.write_doc(j_ob, sp, specs, st, writer.TemplateWriter(), random.Random(5), {}, None)
            docs.append({"doc_type": sp["doc_type"], "day_offset": sp["day"], "text": txt, "doc_type_cn": sp["doc_type_cn"], "k": sp["k"], "role": sp["role"]})
        self.assertEqual(gate.check_admission(j_ob, docs), [])
        bad_ds = copy.deepcopy(docs)
        bad_ds[-1]["text"] += "出院后注意观察恶露及乳汁分泌。"
        self.assertIn((2, "undelivered_postpartum"), gate.check_admission(j_ob, bad_ds))
        bad_fm = copy.deepcopy(docs)
        bad_fm[1]["text"] += "患者诉无胎动异常。"
        self.assertIn((1, "early_preg_fetal"), gate.check_admission(j_ob, bad_fm))
        self.assertFalse(planner.dx_ok("高龄经产妇妊娠监督", "女", 25))
        self.assertTrue(planner.dx_ok("高龄经产妇妊娠监督", "女", 36))

    def test_v034_organ_region_completion_and_takes_tissue(self):
        self.assertEqual(planner.dx_organs("慢性鼻窦炎", "J32"), {"咽喉"})
        self.assertEqual(planner.dx_organs("腰椎间盘突出症", "M51"), {"脊柱"})
        got_sinus = planner.dedupe_bare([{"key": "CT", "region": None, "name": "CT"}, {"key": "超声", "region": None, "name": "超声"}], {"咽喉"}, age=0.8, dx_name="慢性鼻窦炎")
        self.assertEqual([e["name"] for e in got_sinus], ["头颅CT", "颈部超声"])
        got_spine = planner.dedupe_bare([{"key": "磁共振", "region": None, "name": "磁共振"}, {"key": "超声", "region": None, "name": "超声"}], {"脊柱"}, age=45, dx_name="腰椎间盘突出症")
        self.assertEqual([e["name"] for e in got_spine], ["脊柱磁共振", "腹部超声"])
        self.assertFalse(planner.takes_tissue({"name": "经尿道右侧输尿管软镜钬激光碎石取石术"}))
        self.assertFalse(planner.takes_tissue({"name": "子宫下段剖宫产术"}))
        self.assertFalse(planner.takes_tissue({"name": "右侧桡骨骨折切开复位内固定术"}))
        self.assertTrue(planner.takes_tissue({"name": "腹腔镜胆囊切除术"}))
        self.assertNotIn("血管", planner.BARE_REGION)

    def test_v034_pediatric_steroid_and_neonatal_age(self):
        self.assertFalse(planner._child_dose_ok({"age": 0.8, "weight_kg": 9.0}, "methylprednisolone", "甲泼尼龙", "80mg"))
        self.assertTrue(planner._child_dose_ok({"age": 0.8, "weight_kg": 9.0}, "methylprednisolone", "甲泼尼龙", "15mg"))
        age_neo = planner._band_age(random.Random(1), "0-1", neonatal=True)
        self.assertIsNotNone(age_neo)
        self.assertLess(age_neo, 1 / 12)
        self.assertRegex(planner.age_text(age_neo), r"^\d+天$")
        self.assertIsNone(planner._band_age(random.Random(1), "1-6", neonatal=True))
        self.assertFalse(planner.dept_ok("新生儿科", "男", 0.67, False, "P59"))
        self.assertTrue(planner.dept_ok("新生儿科", "男", age_neo, False, "P59"))
        self.assertFalse(planner.dx_ok("新生儿高胆红素血症", "男", 0.67))
        self.assertFalse(planner.dept_ok("乳腺甲状腺外科", "男", 15, False, "Z51"))

    def test_v034_z51_oncology_rules(self):
        self.assertFalse(planner._is_specific_cancer("恶性肿瘤维持性化学治疗"))
        self.assertTrue(planner._is_specific_cancer("急性髓系白血病"))
        self.assertFalse(planner._dept_tumor_ok("妇瘤科一区", "急性髓系白血病"))
        self.assertTrue(planner._dept_tumor_ok("血液内科一区", "急性髓系白血病"))
        self.assertTrue(planner._dept_tumor_ok("血液内科一区", "恶性肿瘤维持性化学治疗"))
        self.assertFalse(planner.antitumor_ok("paclitaxel", "急性髓系白血病"))
        self.assertTrue(planner.antitumor_ok("oxaliplatin", "结肠恶性肿瘤术后辅助化疗"))
        self.assertFalse(planner.antitumor_ok("oxaliplatin", "乳腺恶性肿瘤"))

    def test_v034_course_dx_event_must(self):
        st = copy.deepcopy(fixtures_mc.stats())
        j, specs, _, _ = make("DX34", 11, None, st)
        j["surgical"] = False
        j["procedures"] = []
        j["exams"] = [{"id": "e1", "key": "CT", "region": "胸", "name": "胸部CT", "kind": "影像", "order_day": 0, "result_day": 1, "items": [], "abnormal": True, "diagnostic": True, "finding": "符合肺炎表现"}]
        j["dx_events"] = [
            {"id": "d1", "name": j["patient"]["primary_dx"]["name"], "day": 0, "state": "疑似"},
            {"id": "d2", "name": j["patient"]["primary_dx"]["name"], "day": 1, "state": "确诊"},
        ]
        sp_sup = {"k": 2, "day": 1, "role": "SUP", "doc_type": "progress_note", "doc_type_cn": "主治医师查房记录", "time": "09:30"}
        pls = compiler.compile_doc(j, sp_sup, [
            {"k": 0, "day": 0, "role": "HR", "doc_type": "hospital_record", "doc_type_cn": "入院记录", "time": "08:00"},
            {"k": 1, "day": 0, "role": "FP", "doc_type": "progress_note", "doc_type_cn": "首次病程记录", "time": "09:00"},
            sp_sup,
        ], st, random.Random(3), epistemic=True)
        body = next(x for x in pls if x["section"] == "正文")
        dx_facts = [f for f in body["facts"] if f.get("key") == j["patient"]["primary_dx"]["name"] and "诊断明确" in f["text"]]
        self.assertTrue(dx_facts and dx_facts[0]["must"])


if __name__ == "__main__":
    unittest.main()
