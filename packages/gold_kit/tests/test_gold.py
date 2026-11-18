"""gold_kit 自检。全部虚构数据（借 medcourse_core 自检的虚构文书）。环境变量同 medcourse_core；PYTHONPATH 要有 medcourse_core。"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MC = os.environ.get("MC_CORE") or next(p for p in sys.path if os.path.isdir(os.path.join(p, "medcourse")))
sys.path.insert(0, os.path.join(MC, "tests"))

import fixtures_mc  # noqa: E402
from gold import sample, serve, summarize  # noqa: E402
from medcourse.extract.segments import segments  # noqa: E402

TYPES = list(serve.TYPES)


def synthetic(n_adm=12):
    """虚构住院和虚构抽取结果：每片按位置取几段当引文，类型轮着来。"""
    source, events, done = {}, [], set()
    k = 0
    for i in range(n_adm):
        adm = fixtures_mc.admission(i)
        source[adm["admission_id"]] = adm
        for di, doc in enumerate(adm["docs"]):
            for seg in segments(doc):
                done.add(f"{adm['admission_id']}|{di}|{seg['start']}")
                for off in range(0, max(1, len(seg["text"]) - 4), 6):
                    t = TYPES[k % len(TYPES)]
                    k += 1
                    a = seg["start"] + off
                    events.append({"admission_id": adm["admission_id"], "doc_idx": di, "doc_type": doc["doc_type"], "event_type": t, "name": doc["text"][a:a + 3],
                                   "value": "12.1" if t == "lab" else None, "unit": "g/L" if t == "lab" else None, "evidence": doc["text"][a:a + 4],
                                   "ev_start": a, "ev_end": a + 4, "section": seg["section"], "seg_start": seg["start"], "seg_end": seg["end"],
                                   "assertion": "done", "time_text": None})
    return source, events, done


def env():
    return dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, MC]), PYTHONIOENCODING="utf-8")


class TestSample(unittest.TestCase):
    def test_allocate(self):
        a = sample.allocate({"lab": 1000, "symptom": 300, "vital": 20, "procedure": 5}, 220, 15)
        self.assertEqual(sum(a.values()), 220)
        self.assertEqual(a["procedure"], 5)
        self.assertGreaterEqual(a["vital"], 15)
        self.assertGreater(a["lab"], a["symptom"])
        self.assertEqual(sum(sample.allocate({"a": 10, "b": 10}, 40, 15).values()), 20)

    def test_build_items(self):
        source, events, done = synthetic()
        items, counts = sample.build(events, done, source, seed=1, n_a=140, n_b=10)
        a = [x for x in items if x["part"] == "A"]
        b = [x for x in items if x["part"] == "B"]
        self.assertEqual((len(a), len(b)), (140, 10))
        self.assertTrue(all(counts[f"A_sampled:{t}"] >= 15 for t in TYPES))
        for x in a:
            s, e = x["hl"][0]
            self.assertEqual(len(x["text"][s:e]), 4)
            self.assertIn(x["event"]["event_type"], TYPES)
        for x in b:
            self.assertEqual(len(x["hl"]), len(x["events"]))
            self.assertTrue(all(0 <= s < e <= len(x["text"]) for s, e in x["hl"]))
        self.assertEqual(len({x["item_id"] for x in items}), len(items))
        again, _ = sample.build(events, done, source, seed=1, n_a=140, n_b=10)
        self.assertEqual(items, again)                                          # 同种子同题

    def test_cli_guards(self):
        tmp = tempfile.mkdtemp(prefix="g_")
        try:
            source, events, done = synthetic(3)
            ev, dn, src = (os.path.join(tmp, x) for x in ("e.jsonl", "d.txt", "s.jsonl"))
            with open(ev, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
            with open(dn, "w", encoding="utf-8") as f:
                f.writelines(k + "\n" for k in done)
            with open(src, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(a, ensure_ascii=False) + "\n" for a in source.values())
            base = [sys.executable, "-B", "-m", "gold.sample", "--events", ev, "--done", dn, "--source", src, "--out", os.path.join(tmp, "gold")]
            p = subprocess.run(base, env=env(), cwd=ROOT, capture_output=True)
            self.assertNotEqual(p.returncode, 0)                                # 不在派生目录下，又没加 --synthetic
            p = subprocess.run(base + ["--synthetic"], env=env(), cwd=ROOT, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr[-300:])
            self.assertNotIn("咳嗽".encode(), p.stderr)
            p = subprocess.run(base + ["--synthetic"], env=env(), cwd=ROOT, capture_output=True)
            self.assertIn(b"items_exist", p.stderr)                             # 不许重抽覆盖
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestServe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="g_")
        source, events, done = synthetic()
        items, _ = sample.build(events, done, source, seed=1, n_a=140, n_b=10)
        with open(os.path.join(self.tmp, "items.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
        self.items = items
        self.srv = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(self.tmp, "t0ken"))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _req(self, path, body=None, cookie=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **({"Cookie": cookie} if cookie else {})})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers

    def test_token_required_and_labels_saved(self):
        self.assertEqual(self._req("/")[0], 403)
        self.assertEqual(self._req("/?t=wrong")[0], 403)
        code, page, headers = self._req("/?t=t0ken")
        self.assertEqual(code, 200)
        self.assertNotIn(b"http://", page.replace(b"http://127.0.0.1", b""))   # 页面不引外部资源
        self.assertNotIn(b"https://", page)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        code, body, _ = self._req("/api/state", cookie="gold_t=t0ken")
        self.assertEqual((code, len(json.loads(body)["items"])), (200, 150))
        iid = self.items[0]["item_id"]
        self.assertEqual(self._req("/api/label", {"item_id": iid, "labels": {"valid": "yes", "note": "很长的一段自由文本不该被收下来" * 3}}, "gold_t=t0ken")[0], 200)
        self.assertEqual(self._req("/api/label", {"item_id": "nope", "labels": {}}, "gold_t=t0ken")[0], 400)
        self.assertEqual(self._req("/api/label", {"item_id": iid, "labels": {"valid": "no"}})[0], 403)
        labs = serve.load_labels(os.path.join(self.tmp, "labels.jsonl"))
        self.assertEqual(labs[iid]["valid"], "yes")
        self.assertLessEqual(len(labs[iid]["note"]), 16)


class TestSummarize(unittest.TestCase):
    def test_rates_and_verdict(self):
        source, events, done = synthetic()
        items, _ = sample.build(events, done, source, seed=1, n_a=140, n_b=10)
        rng = random.Random(0)
        labels = {}
        for x in items:
            if x["part"] == "A":
                ok = rng.random() < 0.95
                lab = {"valid": "yes" if ok else "no"}
                if ok:
                    lab.update({"name": "ok", "assertion": "ok" if rng.random() < 0.9 else "bad"})
                    if lab["assertion"] == "bad":
                        lab["assertion_true"] = "planned"
                    if x["event"]["value"] is not None:
                        lab["value"] = "ok"
                labels[x["item_id"]] = lab
            else:
                labels[x["item_id"]] = {f"miss_{t}": "0" for t in TYPES}
                labels[x["item_id"]]["miss_symptom"] = "1"
        rows = {r["metric"]: r for r in summarize.summarize(items, labels)}
        self.assertEqual(rows["valid:all"]["n"], 140)
        self.assertGreater(rows["valid:all"]["rate"], 0.85)
        self.assertEqual(rows["valid:all"]["line"], "")                         # 0.2：总成立率只作描述
        self.assertTrue(rows["verdict"]["k"].startswith("准入："))
        self.assertEqual(rows["B_n"]["k"], 10)
        self.assertLess(rows["recall:symptom"]["rate"], rows["recall:lab"]["rate"])
        self.assertTrue(any(k.startswith("assert_confusion:done->planned") for k in rows))
        bad = {k: ({"valid": "no", "reason": "absent"} if k.startswith("A") else v) for k, v in labels.items()}
        rows_bad = {r["metric"]: r for r in summarize.summarize(items, bad)}
        self.assertEqual(rows_bad["verdict"]["k"], "七类都不准入")
        self.assertGreater(rows_bad["invalid_reason:lab|absent"]["k"], 0)
        half = {k: v for k, v in list(labels.items())[:20]}
        rows = {r["metric"]: r for r in summarize.summarize(items, half)}
        self.assertGreater(rows["A_unlabeled"]["k"], 0)
        self.assertEqual(rows["verdict"]["k"], "未标完")

    def test_per_type_admission_without_floor(self):
        """0.2：症状全不成立、总成立率很低时，过线的化验照样准入（v1 的总成立率底线把化验、诊断一起挡住了）。"""
        source, events, done = synthetic()
        items, _ = sample.build(events, done, source, seed=1, n_a=140, n_b=10)
        labels = {}
        for x in items:
            if x["part"] == "B":
                labels[x["item_id"]] = {f"miss_{t}": "0" for t in TYPES}
            elif x["event"]["event_type"] in ("lab", "diagnosis"):
                labels[x["item_id"]] = dict({"valid": "yes", "name": "ok", "assertion": "ok"}, **({"value": "ok"} if x["event"]["value"] is not None else {}))
            else:
                labels[x["item_id"]] = {"valid": "no", "reason": "wrong_type", "type_true": "sign"}
        rows = {r["metric"]: r for r in summarize.summarize(items, labels)}
        self.assertLess(rows["valid:all"]["rate"], 0.5)
        self.assertEqual((rows["admit:lab"]["k"], rows["admit:diagnosis"]["k"], rows["admit:symptom"]["k"]), (1, 1, 0))
        self.assertEqual(rows["verdict"]["k"], "准入：诊断、化验")
        self.assertGreater(rows["type_confusion:symptom->sign"]["k"], 0)


class TestV02(unittest.TestCase):
    """0.2：只抽核心事件；标注页的定义和抽取提示词同一份；纠偏的开发检查。"""

    def test_sample_skips_exits(self):
        source, events, done = synthetic()
        for i, e in enumerate(events):
            if i % 5 == 0:
                e["event_type"] = "sign"
            elif i % 7 == 0:
                e["assertion"] = "not_patient"
        items, counts = sample.build(events, done, source, seed=1, n_a=140, n_b=10)
        for x in items:
            for e in ([x["event"]] if x["part"] == "A" else x["events"]):
                self.assertIn(e["event_type"], TYPES)
                self.assertNotEqual(e["assertion"], "not_patient")
        self.assertGreater(counts["events_before_core_filter"], counts["events_total"])
        self.assertTrue(any(k.startswith("routed_out:sign") for k in counts))

    def test_page_uses_extraction_definitions(self):
        from medcourse.extract.schema import ASSERT_DEFS, TYPE_DEFS
        page = serve.PAGE.replace("__DEFS__", serve.defs_html())
        for cn, d in list(TYPE_DEFS.values()) + list(ASSERT_DEFS.values()):
            self.assertIn(cn, page)
            self.assertIn(d.replace("&", "&amp;"), page)
        self.assertNotIn("not_patient", serve.ASSERT)
        self.assertEqual(set(serve.TYPES), set(TYPES))

    def test_rescore_false_drop(self):
        from gold import rescore
        text = "查体：腹软，无压痛。患者咳嗽3天。"
        def item(i, name, typ, ev):
            s = text.index(ev)
            return {"part": "A", "item_id": f"A{i:03d}", "doc_type": "入院记录", "section": "现病史", "text": text, "hl": [[s, s + len(ev)]],
                    "event": {"event_type": typ, "name": name, "value": None, "unit": None, "assertion": "done", "time_text": None}}
        items = [item(0, "腹软", "symptom", "腹软"), item(1, "咳嗽", "symptom", "咳嗽")]
        labels = {"A000": {"valid": "no", "reason": "wrong_type", "type_true": "sign"}, "A001": {"valid": "yes", "name": "ok", "assertion": "ok"}}
        rows = dict(rescore.rescore(items, labels))
        self.assertEqual(rows["routed_out_invalid:symptom"], 1)                # 查体所见被分到体征
        self.assertEqual(rows.get("routed_out_valid:symptom", 0), 0)
        self.assertEqual(rows["false_drop_pass"], 1)
        self.assertEqual(rows["assert_rate_after:all"], 1.0)


class TestGoldV03(unittest.TestCase):
    """0.3（PREREG S2G v4）：名称回退层、开发检查的类型改对和逐条规则翻转。全部虚构。"""

    def test_fallback_stratum_sample_and_summary(self):
        source, events, done = synthetic()
        for n, e in enumerate(events):
            if n % 7 == 0:
                e["ground"] = "name"
        items, counts = sample.build(events, done, source, seed=3, n_a=140, n_b=10, n_fb=20)
        a = [x for x in items if x["part"] == "A"]
        fb = [x for x in a if x["stratum"] == "fallback"]
        self.assertEqual((len(a), len(fb), counts["A_sampled_fallback"]), (160, 20, 20))
        self.assertTrue(all(x["stratum"] in ("type", "fallback") for x in a))
        keys = [(x["admission_id"], x["doc_idx"], tuple(x["hl"][0]), x["event"]["name"]) for x in a]
        self.assertEqual(len(keys), len(set(keys)))                                     # 两层不重复
        labels = {}
        for x in items:
            if x["part"] == "A":
                labels[x["item_id"]] = {"valid": "yes", "name": "ok", "assertion": "ok", **({"value": "ok"} if x["event"].get("value") else {})}
            else:
                labels[x["item_id"]] = {f"miss_{t}": "0" for t in TYPES}
        rows = {r["metric"]: r for r in summarize.summarize(items, labels)}
        self.assertEqual((rows["fallback:valid"]["k"], rows["fallback:valid"]["n"]), (20, 20))
        self.assertEqual(rows["valid:all"]["n"], 140)                                   # 回退层不进总体和按类型
        self.assertEqual(sum(rows[f"valid:{t}"]["n"] for t in TYPES), 140)

    def test_rescore_v03_type_fix_and_flips(self):
        from gold import rescore
        text = "有高血压病史5年。体重65kg。行第2次化疗。"

        def item(i, name, typ, ev, sec, st="done"):
            s = text.index(ev)
            return {"part": "A", "item_id": f"A{i:03d}", "doc_type": "入院记录", "section": sec, "text": text, "hl": [[s, s + len(ev)]], "stratum": "type",
                    "event": {"event_type": typ, "name": name, "value": None, "unit": None, "assertion": st, "time_text": None}}
        items = [item(0, "高血压病史", "diagnosis", "高血压病史", "既往史", "history"), item(1, "体重", "vital", "体重65kg", "体格检查"),
                 item(2, "化疗", "procedure", "化疗", "现病史")]
        labels = {"A000": {"valid": "yes", "name": "ok", "assertion": "bad", "assertion_true": "done"},
                  "A001": {"valid": "no", "reason": "not_event"},
                  "A002": {"valid": "no", "reason": "wrong_type", "type_true": "medication"}}
        rows = dict(rescore.rescore(items, labels))
        self.assertEqual(rows["routed_out_invalid:vital"], 1)                          # D4：体重归七类以外
        self.assertEqual((rows["type_fixed"], rows["after_valid:medication"]), (1, 1))  # 化疗改用药，正是老板选的
        self.assertEqual(rows["up:dx_history_suffix|diagnosis"], 1)                     # D1：慢性病改已发生，错变对
        self.assertEqual((rows["false_drop_pass"], rows["assert_rate_after:all"]), (1, 1.0))


if __name__ == "__main__":
    unittest.main()
