# -*- coding: utf-8 -*-
"""gm_tools 自检。全部虚构数据，假模型客户端，不连服务。需要 PYTHONPATH 带 medcourse_core（v0.1.13.1 或同 S1 的版本）和本包，
BENCH_G2 等评测台环境变量同 medcourse 自检（本机用 本机探针/runtests.sh 的设置）。"""
import csv
import json
import os
import sys
import tempfile
import unittest

from gmtools import m1_direct, m1_patients, m1_table, s2_stage1
from medcourse.extract import run as R

HR = ("入院记录\n姓名：[患者]　性别：男　年龄：61岁　科室：神经内科五区\n入院时间：2025-06-16 08:45\n"
      "主诉：乏力1天。\n现病史：患者乏力1天，阿司匹林100mg口服后无缓解。\n既往史：有高血压病史。\n个人史：生于原籍，无特殊嗜好。\n"
      "体格检查：体温36.6℃，脉搏85次/分。神志清楚。\n辅助检查：头颅CT：符合脑梗死表现。\n入院诊断：\n1.脑梗死\n2.高血压病\n")
FP = ("首次病程记录\n2025-06-16 11:50\n姓名：[患者]　年龄：61岁　科室：神经内科五区\n"
      "病例特点：患者男，61岁，因乏力1天入院。\n初步诊断：\n1.脑梗死\n拟诊讨论：头颅CT示符合脑梗死表现。\n诊疗计划：阿托伐他汀20mg每晚一次。\n")
NOTE = "2025-06-17 09:20　日常病程记录\n体温37.0℃。患者乏力较前减轻，继续阿司匹林100mg治疗。\n"
DS = ("出院记录\n姓名：[患者]　性别：男　年龄：61岁　科室：神经内科五区\n入院时间：2025-06-16 08:45　出院时间：2025-06-22 09:10　住院天数：6天\n"
      "入院情况：患者因乏力1天入院。\n入院诊断：\n1.脑梗死\n诊疗经过：住院期间给予阿托伐他汀及阿司匹林治疗。\n出院诊断：\n1.脑梗死\n2.高血压病3级\n"
      "出院情况：一般情况良好。\n出院医嘱：继续口服阿司匹林100mg。\n")
SECRET = ("阿司匹林100mg", "阿托伐他汀20mg", "符合脑梗死表现", "高血压病3级")


def adm(aid="MC20261111-00000"):
    docs = [
        {"doc_type": "hospital_record", "day_offset": 0, "text": HR, "doc_type_cn": "入院记录", "k": 0, "role": "HR"},
        {"doc_type": "progress_note", "day_offset": 0, "text": FP, "doc_type_cn": "首次病程记录", "k": 1, "role": "FP"},
        {"doc_type": "progress_note", "day_offset": 1, "text": NOTE, "doc_type_cn": "日常病程记录", "k": 2, "role": "DAILY"},
        {"doc_type": "discharge_summary", "day_offset": 6, "text": DS, "doc_type_cn": "出院记录", "k": 3, "role": "DS"},
    ]
    return {"admission_id": aid, "docs": docs, "meta": {"admit_date": "2025-06-16", "contract": "mc-contract-0.2"}}


def journey(aid="MC20261111-00000"):
    return {"journey_id": aid, "los": 6, "patient": {"primary_dx": {"name": "脑梗死"}, "dept": "神经内科五区", "sex": "男", "age": 61}}


class FakeClient:
    def __init__(self, *a, **kw):
        self.prompts = []

    def generate(self, *, sampling, seed, messages=None, prompt=None, max_tokens=512, schema=None, stop=None):
        u = messages[-1]["content"]
        self.prompts.append(u)
        if schema is not None and "events" in json.dumps(schema, ensure_ascii=False):
            return {"text": json.dumps({"events": []}), "finish": "stop", "prompt_tokens": 10, "completion_tokens": 5, "seconds": 0.0}
        if schema is not None:
            i = len(self.prompts)
            x = {"dept": "心内科", "sex": "男" if i % 3 else "女", "age": 60 + i % 4, "dx": "冠心病" if i % 2 else "高血压", "los": 7}
            return {"text": json.dumps(x, ensure_ascii=False), "finish": "stop", "prompt_tokens": 10, "completion_tokens": 5, "seconds": 0.0}
        sec = u.split("的“")[1].split("”一节")[0]
        text = "1.脑血管病" if sec.endswith("诊断") else f"{sec}写好了第{len(self.prompts)}段。"
        return {"text": text, "finish": "stop", "prompt_tokens": 100, "completion_tokens": 20, "seconds": 0.0}


class TestDirect(unittest.TestCase):
    def test_split_hr(self):
        head, secs = m1_direct.split_doc(HR, "入院记录")
        self.assertEqual(head.count("\n"), 3)
        self.assertTrue(head.startswith("入院记录\n姓名"))
        self.assertEqual([t for t, _ in secs], ["主诉", "现病史", "既往史", "个人史", "体格检查", "辅助检查", "入院诊断"])
        self.assertEqual(dict(secs)["入院诊断"], "1.脑梗死\n2.高血压病")

    def test_split_note(self):
        head, secs = m1_direct.split_doc(NOTE, "日常病程记录")
        self.assertEqual(head, "2025-06-17 09:20　日常病程记录\n")
        self.assertEqual(len(secs), 1)
        self.assertEqual(secs[0][0], "")

    def test_split_titles_cover_compiler(self):
        # 第一组用到的节标题都要认得（compiler 里 llm()/prog() 的全部节名）
        names = {"专科情况", "入院情况", "出院医嘱", "出院情况", "手术指征", "手术简要经过", "拟诊讨论", "既往史", "术后处理措施",
                 "术后应当特别注意观察的事项", "注意事项", "现病史", "病例特点", "简要病情", "诊疗经过", "诊疗计划", "鉴别诊断",
                 "个人史", "主诉", "入院诊断", "出院诊断", "初步诊断", "婚育史", "家族史", "手术方式", "手术时间", "拟施手术名称和方式",
                 "拟施麻醉方式", "月经史", "术中诊断", "术前诊断", "辅助检查", "麻醉方式", "体格检查"}
        self.assertEqual(names - set(m1_direct.TITLES), set())

    def test_admission_keeps_structure_drops_facts(self):
        fc = FakeClient()
        d = m1_direct.Direct(fc)
        out = d.admission(adm(), journey())
        self.assertEqual(out["admission_id"], "DMC20261111-00000")
        self.assertEqual([x["doc_type_cn"] for x in out["docs"]], ["入院记录", "首次病程记录", "日常病程记录", "出院记录"])
        self.assertEqual([x["day_offset"] for x in out["docs"]], [0, 0, 1, 6])
        for new, old in zip(out["docs"], adm()["docs"]):
            n = m1_direct.HEAD_LINES.get(old["doc_type_cn"], 1)
            self.assertEqual(new["text"].split("\n")[:n], old["text"].split("\n")[:n])
        self.assertIn("入院诊断：\n1.脑血管病\n", out["docs"][0]["text"])
        self.assertEqual(d.t["calls"], 7 + 4 + 1 + 6)
        # 第一组正文里的事实一条都不进提示词（抬头里的性别、年龄、科室和主要诊断、住院天数除外）
        for p in fc.prompts:
            for s in SECRET:
                self.assertNotIn(s, p)
        self.assertIn("主要诊断：脑梗死；共住院6天", fc.prompts[0])
        self.assertIn("【本次住院已经写好的病历】\n入院记录", fc.prompts[0])
        # 后面的节看得到前面写好的
        self.assertIn("主诉写好了第1段", fc.prompts[1])
        self.assertIn("现病史写好了第2段", fc.prompts[-1])

    def test_context_capped(self):
        class Long(FakeClient):
            def generate(self, **kw):
                return dict(super().generate(**kw), text="长" * 2000)
        fc = Long()
        d = m1_direct.Direct(fc)
        d.admission(adm(), journey())
        self.assertGreater(max(len(p) for p in fc.prompts), m1_direct.CONTEXT_CHARS)
        self.assertTrue(all(len(p.split("【本次住院已经写好的病历】\n")[1].split("\n【要求】")[0]) <= m1_direct.CONTEXT_CHARS for p in fc.prompts))

    def test_list_prompt_and_target(self):
        info = {"sex": "男", "age_text": "61岁", "dept": "神经内科", "dx": "脑梗死", "los": 6}
        p = m1_direct.section_prompt(info, "出院记录", 6, "出院诊断", 30, "")
        self.assertIn("逐条列出诊断", p)
        self.assertIn("已经写好的病历】\n无", p)
        p = m1_direct.section_prompt(info, "日常病程记录", 2, "", 220, "x")
        self.assertIn("“正文”一节", p)
        self.assertIn("约 220 字", p)

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as t:
            run = os.path.join(t, "run")
            os.makedirs(run)
            with open(os.path.join(run, "admissions.jsonl"), "w", encoding="utf-8") as f:
                for i in range(3):
                    f.write(json.dumps(adm(f"MC20261111-0000{i}"), ensure_ascii=False) + "\n")
            with open(os.path.join(run, "journeys.jsonl"), "w", encoding="utf-8") as f:
                for i in range(3):
                    f.write(json.dumps(journey(f"MC20261111-0000{i}"), ensure_ascii=False) + "\n")
            with open(os.path.join(run, "counts.json"), "w", encoding="utf-8") as f:
                json.dump({"mode": "free", "writer": "llm", "seed": 20261111, "version": "s1-pipeline-0.11"}, f)
            out = os.path.join(t, "out")
            old = m1_direct.Client
            m1_direct.Client = FakeClient
            argv = sys.argv
            try:
                sys.argv = ["x", "--run", run, "--base-url", "http://127.0.0.1:1/v1", "--model", "Qwen3-8B", "--out", out, "--concurrency", "2"]
                m1_direct.main()
                with self.assertRaises(Exception):
                    m1_direct.main()
            finally:
                m1_direct.Client, sys.argv = old, argv
            with open(os.path.join(out, "counts.json"), encoding="utf-8") as f:
                c = json.load(f)
            self.assertEqual((c["planned"], c["kept"], c["discarded"], c["mode"], c["writer"]), (3, 3, 0, "free", "llm"))
            self.assertEqual(c["calls"], 3 * 18)
            self.assertEqual(c["version"], "m1-direct-0.1")
            with open(os.path.join(out, "admissions.jsonl"), encoding="utf-8") as f:
                ids = sorted(json.loads(x)["admission_id"] for x in f)
            self.assertEqual(ids, ["DMC20261111-00000", "DMC20261111-00001", "DMC20261111-00002"])

    def test_refuses_template_source(self):
        with tempfile.TemporaryDirectory() as t:
            with open(os.path.join(t, "counts.json"), "w", encoding="utf-8") as f:
                json.dump({"mode": "free", "writer": "template"}, f)
            argv = sys.argv
            try:
                sys.argv = ["x", "--run", t, "--base-url", "http://127.0.0.1:1/v1", "--model", "m", "--out", os.path.join(t, "o")]
                with self.assertRaises(Exception):
                    m1_direct.main()
            finally:
                sys.argv = argv


class TestPatients(unittest.TestCase):
    def test_gen_and_stats(self):
        rows, t = m1_patients.gen(FakeClient(), 12, concurrency=1)
        self.assertEqual(len(rows), 12)
        self.assertEqual(t["calls"], 12)
        s = m1_patients.header_stats(rows)
        self.assertEqual(s["n"], 12)
        self.assertEqual(s["distinct_age_256"], 4)
        self.assertEqual(s["distinct_dx"], 2)
        self.assertEqual(s["top1_dx_share"], 0.5)
        self.assertEqual(s["los_p50"], 7)

    def test_journey_rows(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "j.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps(journey(), ensure_ascii=False) + "\n")
            self.assertEqual(m1_patients.journey_rows(p), [{"dept": "神经内科五区", "sex": "男", "age": 61, "dx": "脑梗死", "los": 6}])


class TestTable(unittest.TestCase):
    def board(self, mc, direct, real):
        b = {}
        for row, vals in (("MC-8B", mc), ("DIRECT-8B", direct), ("R", real)):
            for k, v in vals.items():
                b[(row, k)] = v
        return b

    def test_m1_pass_and_verdicts(self):
        mc = {"B_tfidf": (0.33, 0.32, 0.34), "B_exam": (7.7, 7.5, 7.9), "B_dxchg": (0.5, 0.45, 0.55), "B_mentions": (11.4, 11, 11.8),
              "B_dose": (0.27, 0.25, 0.29), "A_g3_hard": (0, 0, 0.003), "A_g3_n1": (0, 0, 0.003), "B_ages": (70, None, None), "B_chars": (880, 870, 890)}
        dr = {"B_tfidf": (0.52, 0.50, 0.54), "B_exam": (4.0, 3.8, 4.2), "B_dxchg": (0.1, 0.08, 0.12), "B_mentions": (6.0, 5.8, 6.2),
              "B_dose": (0.4, 0.38, 0.42), "A_g3_hard": (0.02, 0.012, 0.03), "A_g3_n1": (0, 0, 0.003), "B_ages": (70, None, None), "B_chars": (700, 690, 710)}
        real = {"B_tfidf": (0.23, 0.22, 0.24), "B_exam": (8.3, 8.1, 8.5), "B_dxchg": (0.37, 0.33, 0.41), "B_mentions": (8.4, 8.2, 8.6),
                "B_dose": (0.24, 0.23, 0.25), "A_g3_hard": (0.01, 0.005, 0.02), "A_g3_n1": (0, 0, 0.002), "B_ages": (57, None, None), "B_chars": (1100, 1080, 1120)}
        rows, s = m1_table.build(self.board(mc, dr, real), {"calls_per_kchar": 4.8, "discard_rate": 0.0},
                                 {"calls_per_kchar": 1.2, "discard_rate": 0.0}, {"leak_O": 0, "leak_A": 0, "leak_B": 0})
        self.assertTrue(s["m1_pass"])
        v = {(r["rule"], r["metric"]): r["MC_vs_DIRECT"] for r in rows}
        self.assertEqual(v[(1, "B_tfidf")], "第一组胜")
        self.assertEqual(v[(2, "B_exam")], "第一组胜")
        self.assertEqual(v[(3, "B_dxchg")], "第一组胜")          # 0.5 离真实 0.37 近于 0.1
        self.assertEqual(v[(4, "B_dose")], "第一组胜")
        self.assertEqual(v[(5, "A_g3_hard")], "第一组胜")
        self.assertEqual(v[(6, "A_g3_n1")], "不分（区间重叠）")
        self.assertTrue(v[(7, "B_ages")].startswith("不比"))
        self.assertTrue(v[(10, "leak_O")].startswith("不适用"))
        self.assertEqual(s["direct_rules_passed"], [3, 6])

    def test_m1_fails_on_leak(self):
        rows, s = m1_table.build({}, {"calls_per_kchar": 4.8, "discard_rate": 0.0}, {}, {"leak_O": 1, "leak_A": 0, "leak_B": 0})
        self.assertFalse(s["m1_pass"])
        self.assertIn(10, s["rules_failed"])

    def test_read_files(self):
        with tempfile.TemporaryDirectory() as t:
            b = os.path.join(t, "b.csv")
            with open(b, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row", "metric", "value", "ci_lo", "ci_hi", "cp_lo", "cp_hi"])
                w.writerow(["MC-8B", "B_exam", "7.7", "7.5", "7.9", "", ""])
                w.writerow(["MC-8B", "A_g3_hard", "0", "", "", "0", "0.003"])
                w.writerow(["R", "B_ages", "57", "", "", "", ""])
            got = m1_table.read_board(b)
            self.assertEqual(got[("MC-8B", "B_exam")], (7.7, 7.5, 7.9))
            self.assertEqual(got[("MC-8B", "A_g3_hard")], (0.0, 0.0, 0.003))
            self.assertEqual(got[("R", "B_ages")], (57.0, None, None))
            l = os.path.join(t, "l.csv")
            with open(l, "w", encoding="utf-8", newline="") as f:
                f.write("scope,metric,doc_type,role,day,count,denominator\nadmission,O,,,,0,1024\nadmission,A,,,,0,1024\n"
                        "admission,B,,,,0,1024\nadmission,B_loose,,,,9,1024\n")
            self.assertEqual(m1_table.read_leak(l), {"leak_O": 0.0, "leak_A": 0.0, "leak_B": 0.0})


class TestStage1(unittest.TestCase):
    def test_two_stages_share_done_and_doc_idx(self):
        with tempfile.TemporaryDirectory() as t:
            src = os.path.join(t, "fold.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                for i in range(2):
                    a = adm(f"R{i}")
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
            out = os.path.join(t, "out")
            old_client, argv = R.Client, sys.argv
            R.Client = FakeClient
            try:
                sys.argv = ["x", "--path", src, "--base-url", "http://127.0.0.1:1/v1", "--model", "m", "--out", out, "--synthetic", "--concurrency", "2"]
                s2_stage1.main()
                with open(os.path.join(out, "done_segments.txt"), encoding="utf-8") as f:
                    stage1 = {x.strip() for x in f if x.strip()}
                self.assertTrue(stage1)
                self.assertEqual({k.split("|")[1] for k in stage1}, {"0", "3"})      # 原文件里的文书序号：入院记录 0、出院记录 3
                with open(os.path.join(out, "counts.json"), encoding="utf-8") as f:
                    c1 = json.load(f)
                self.assertEqual(c1["stage1_docs_hospital_record"], 2)
                self.assertEqual(c1["stage1_docs_discharge_summary"], 2)
                self.assertEqual(c1["stage1_segments_hospital_record"] + c1["stage1_segments_discharge_summary"], len(stage1))
                self.assertGreater(c1["stage1_skipped_segments_progress_note"], 0)
                # 第二段：原程序、原文件、同一目录
                R.work_items = s2_stage1._ORIG
                sys.argv = ["x", "--path", src, "--base-url", "http://127.0.0.1:1/v1", "--model", "m", "--out", out, "--synthetic", "--concurrency", "2"]
                R.main()
                with open(os.path.join(out, "done_segments.txt"), encoding="utf-8") as f:
                    allk = {x.strip() for x in f if x.strip()}
                with open(os.path.join(out, "counts.json"), encoding="utf-8") as f:
                    c2 = json.load(f)
                self.assertEqual(c2["skipped_done"], len(stage1))
                self.assertEqual({k.split("|")[1] for k in allk - stage1}, {"1", "2"})
            finally:
                R.Client, sys.argv = old_client, argv
                R.work_items = s2_stage1._ORIG


if __name__ == "__main__":
    unittest.main()
