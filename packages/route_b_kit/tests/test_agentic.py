# -*- coding: utf-8 -*-
"""8B 提问检索流程（0.4 协议：决定 → 检索 → 按句编号作答）的自检。假模型按脚本出 JSON，不连任何服务。示意文本是编的。"""
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from route_b import agentic, chunks, retrieve  # noqa: E402

PATH = ("示意病临床路径\n一、示意病临床路径标准住院流程\n（一）适用对象\n第一诊断为示意病。\n（二）诊断依据\n"
        "1.症状：右上腹痛，可伴发热。\n2.体征：局部压痛，墨菲征阳性。\n（三）治疗方案的选择\n行示意切除术。\n（四）标准住院日为≤７天。\n"
        "（六）术前准备１－２天\n必需的检查项目：血常规、肝功能、腹部超声。\n（八）手术日为入院第３天\n麻醉方式：全身麻醉。\n"
        "（十）出院标准\n1.体温正常。\n2.切口愈合良好。\n二、示意病临床路径表单\n住院第1天 示意病 示意病 示意病 主要诊疗工作 □询问病史 □症状评估\n"
        "住院第2天 示意病 示意病 □复查血常规\n")
GUIDE = "第一部分 基本原则\n（五）疗程\n抗菌药物疗程因感染不同而异，一般宜用至体温正常、症状消退后72～96小时。\n"
KEYS = ("血常规", "肝功能", "超声", "CT")
CLS = {"血常规": "检验", "肝功能": "检验", "超声": "影像", "CT": "影像"}
SID = re.compile(r"^(S\d+) (.*)$", re.M)


class Scripted:
    """scripts：{需求关键词: [步骤]}，步骤是 ("search", 检索词) 或 ("answer", 要引的句子片段或 None, value)。
    决定调用按脚本走；作答调用按"句子片段"去已展示的句子里找编号。脚本用完：先检索，满了就不引句作答。"""
    def __init__(self, scripts):
        self.scripts, self.pos, self.schemas = scripts, {}, []

    def _step(self, first):
        for key, seq in self.scripts.items():
            if key in first:
                i = self.pos.get(key, 0)
                return key, (seq[i] if i < len(seq) else None)
        return None, None

    def generate(self, *, sampling, seed, messages, max_tokens, schema):
        assert sampling["temperature"] == 0.0
        props = schema["properties"]
        first = messages[1]["content"]
        if "questions" in props:
            return {"text": json.dumps({"questions": ["术后第几天复查血常规"]}, ensure_ascii=False)}
        key, step = self._step(first)
        if "query" in props:                      # 决定调用
            if step and step[0] == "answer":
                return {"text": json.dumps({"action": "answer", "query": ""}, ensure_ascii=False)}
            if step and step[0] == "search":
                self.pos[key] = self.pos.get(key, 0) + 1
                return {"text": json.dumps({"action": "search", "query": step[1]}, ensure_ascii=False)}
            return {"text": json.dumps({"action": "search", "query": ""}, ensure_ascii=False)}
        self.schemas.append(schema)               # 作答调用
        shown = {sid: s for m in messages for sid, s in SID.findall(m["content"])}
        if step and step[0] == "answer":
            self.pos[key] = self.pos.get(key, 0) + 1
            ids = [sid for sid, s in shown.items() if step[1] and step[1] in s][:1]
            if not ids and props["cite_ids"].get("minItems"):
                ids = list(shown)[:1]
            return {"text": json.dumps({"cite_ids": ids, "value": step[2]}, ensure_ascii=False)}
        val = _default(props["value"])
        ids = list(shown)[:1] if props["cite_ids"].get("minItems") else []
        return {"text": json.dumps({"cite_ids": ids, "value": val}, ensure_ascii=False)}


def _default(v):
    t = v.get("type")
    if t == "object" and "max" in v.get("properties", {}):
        return {"min": None, "max": 1}
    return {"integer": 1, "array": [], "boolean": True, "object": {"incision": True, "drain": False, "watch": []}}.get(t, "无")


def _index():
    return retrieve.BM25(chunks.make_chunks("P", PATH) + chunks.make_chunks("guideline_abx2015", GUIDE))


class TestChunksRetrieve(unittest.TestCase):
    def test_chunks(self):
        cs = chunks.make_chunks("P", PATH, max_chars=60)
        self.assertTrue(all(len(c["text"]) <= 100 for c in cs))
        self.assertTrue(any(c["kind"] == "form" for c in cs))
        self.assertEqual(len({c["id"] for c in cs}), len(cs))

    def test_toc_lines_dropped(self):
        cs = chunks.make_chunks("G", "目录\n第一部分 基本原则 .............. 3\n第一部分 基本原则\n正文内容")
        self.assertNotIn("......", "".join(c["text"] for c in cs))

    def test_sentences(self):
        self.assertEqual(agentic.sentences("1.症状：腹痛。2.体征：压痛；\n□询问病史□症状评估"),
                         ["1.症状：腹痛。", "2.体征：压痛；", "□询问病史", "□症状评估"])

    def test_drop_disease_name_and_form_weight(self):
        idx = _index()
        q = "示意病 典型症状"
        plain = idx.search(q, k=1, docs={"P"})[0]
        fixed = idx.search(q, k=1, docs={"P"}, drop="示意病", form_weight=0.5)[0]
        self.assertEqual(plain["kind"], "form")                 # 病名吸走得分，表单段排第一
        self.assertIn("症状：右上腹痛", fixed["text"])          # 去掉病名、表单降权后，诊断依据段排第一
        self.assertEqual(idx.search("示意病", k=1, docs={"P"}, drop="示意病")[0]["doc"], "P")   # 只剩病名时用原词

    def test_bm25_docs_and_exclude(self):
        idx = _index()
        self.assertEqual(idx.search("疗程 停药", k=1, docs={"guideline_abx2015"})[0]["doc"], "guideline_abx2015")
        first = idx.search("麻醉", k=1)[0]["id"]
        rest = idx.search("麻醉", k=1, exclude={first})
        self.assertTrue(not rest or rest[0]["id"] != first)


class TestAgentic(unittest.TestCase):
    def run_build(self, scripts, rounds=2):
        client = Scripted(scripts)
        facts, log, missing = agentic.build_disease(client, _index(), "示意病", "P", {"P": PATH, "guideline_abx2015": GUIDE}, KEYS, CLS,
                                                    rounds=rounds, k=2)
        return facts, log, missing, client

    def test_grounded_by_cite(self):
        facts, log, _, _ = self.run_build({
            "标准住院日": [("search", "标准住院日"), ("answer", "标准住院日", {"min": None, "max": 7})],
            "典型症状": [("search", "症状"), ("answer", "症状：右上腹痛", ["右上腹痛", "发热"])],
            "必需做的检查": [("search", "必需的检查项目"), ("answer", "必需的检查项目",
                                                       [{"name": "血常规", "key": "血常规"}, {"name": "腹部超声", "key": "超声"}])],
            "抗菌药物用到什么时候停": [("search", "疗程 停药"), ("answer", "72～96小时", "症状消退后72～96小时")],
            "手术、操作或主要治疗": [("search", "治疗方案"), ("answer", "行示意切除术", "示意切除术")],
        })
        los = facts["los_max"][0]
        self.assertEqual((los["grounded"], los["source"], los["quote_failed"]), (True, "pathway", False))
        self.assertIn("标准住院日为≤７天", los["quote"])                      # 引文是程序照原文取的
        self.assertEqual(los["value_max"], 7)
        self.assertEqual(facts["chief_symptoms"][0]["items_in_quote"], [2, 2])
        self.assertEqual(facts["exams_required"][0]["value"][1]["key"], "超声")
        abx = facts["antibiotic_rule"][0]
        self.assertEqual((abx["grounded"], abx["source"]), (True, "guideline"))
        # 第二阶段：两项检查都问回报天数，只有超声（影像）问客观所见；没引句 → 模型先验
        self.assertEqual(sorted(f["exam_key"] for f in facts["exam_turnaround_days"]), ["血常规", "超声"])
        self.assertEqual([f["exam_key"] for f in facts["exam_findings"]], ["超声"])
        self.assertEqual(facts["postop_care"][0]["procedure"], "示意切除术")
        self.assertEqual(facts["extra_qa"][0]["question"], "术后第几天复查血常规")

    def test_value_not_in_cited(self):
        facts, _, _, _ = self.run_build({"标准住院日": [("search", "标准住院日"), ("answer", "标准住院日", {"min": 3, "max": 10})]})
        f = facts["los_max"][0]
        self.assertEqual((f["grounded"], f["quote_failed"], f["source"]), (False, True, "model_prior"))

    def test_partial_items(self):
        facts, _, _, _ = self.run_build({"典型症状": [("search", "症状"), ("answer", "症状：右上腹痛", ["右上腹痛", "发热", "黄疸"])]})
        f = facts["chief_symptoms"][0]
        self.assertEqual((f["grounded"], f["quote_failed"], f["items_in_quote"]), (False, True, [2, 3]))

    def test_must_cite_before_rounds_used_up(self):
        # 第一轮检索后就想不引句作答：格式要求至少一个编号；检索满以后才允许不引
        facts, log, _, client = self.run_build({"标准住院日": [("search", "标准住院日"), ("answer", None, {"min": None, "max": 7})]})
        first_answer = next(s for s in client.schemas if s["properties"]["cite_ids"].get("minItems") is not None)
        self.assertEqual(first_answer["properties"]["cite_ids"]["minItems"], 1)
        self.assertTrue(facts["los_max"][0]["cite_ids"])          # 被迫引了句
        tr = next(t for t in log if t.get("field") == "signs")      # 没有脚本：检索满 2 次后不引句作答
        self.assertEqual((tr["searches"], tr["answered_after_searches"]), (2, 2))
        self.assertEqual(facts["signs"][0]["source"], "model_prior")

    def test_cite_enum_only_shown(self):
        s = agentic.answer_schema("int", (), ["S1", "S2"], True)
        self.assertEqual(s["properties"]["cite_ids"]["items"]["enum"], ["S1", "S2"])
        self.assertEqual(list(s["properties"]), ["cite_ids", "value"])     # 出处句编号排在 value 前面
        none = agentic.answer_schema("int", (), [], False)
        self.assertEqual(none["properties"]["cite_ids"]["maxItems"], 0)

    def test_answer_before_any_search_becomes_search(self):
        facts, log, _, _ = self.run_build({"标准住院日": [("answer", "标准住院日", {"min": None, "max": 7})]})
        tr = next(t for t in log if t.get("field") == "los_max")
        self.assertGreaterEqual(tr["searches"], 1)

    def test_missing_recorded(self):
        class Broken(Scripted):
            def generate(self, **kw):
                if "cite_ids" in kw["schema"]["properties"] and "标准住院日" in kw["messages"][1]["content"]:
                    return {"text": "不是JSON"}
                return super().generate(**kw)
        client = Broken({})
        facts, log, missing = agentic.build_disease(client, _index(), "示意病", "P", {"P": PATH, "guideline_abx2015": GUIDE}, KEYS, CLS, rounds=2, k=2)
        self.assertIn("los_max", [m["field"] for m in missing])
        self.assertNotIn("los_max", facts)

    def test_surgery_gate_false(self):
        facts, log, missing, client = self.run_build({"主要治疗是不是外科手术": [("search", "治疗方案"), ("answer", "行示意切除术", False)]})
        g = facts["surgery_gate"][0]
        self.assertEqual((g["value"], g["grounded"], g["check"]), (False, True, "cite_only"))
        for fld in ("anesthesia", "surgery_day_max", "postop_days", "postop_care"):
            f = facts[fld][0]
            self.assertEqual((f["value"], f["source"], f["gated_by"]), ("不适用", "not_applicable", "surgery_gate"))
        asked = {t["field"] for t in log if not t.get("gated")}
        self.assertNotIn("anesthesia", asked)                 # 不做手术就不再追问
        self.assertEqual(agentic.__dict__["NA"], "不适用")

    def test_surgery_gate_true_asks_as_usual(self):
        facts, log, _, _ = self.run_build({"主要治疗是不是外科手术": [("search", "治疗方案"), ("answer", "行示意切除术", True)],
                                           "麻醉方式": [("search", "麻醉方式"), ("answer", "麻醉方式", "全身麻醉")]})
        self.assertTrue(facts["anesthesia"][0]["grounded"])

    def test_los_range_min_null(self):
        facts, _, _, _ = self.run_build({"标准住院日": [("search", "标准住院日"), ("answer", "标准住院日", {"min": None, "max": 7})]})
        f = facts["los_max"][0]
        self.assertEqual((f["grounded"], f["value_max"]), (True, 7))

    def test_incremental_only_fields_and_merge(self):
        from route_b import build_kb_agentic as B
        client = Scripted({"主要治疗是不是外科手术": [("search", "治疗方案"), ("answer", "行示意切除术", False)]})
        facts, log, missing = agentic.build_disease(client, _index(), "示意病", "P", {"P": PATH, "guideline_abx2015": GUIDE}, KEYS, CLS,
                                                    rounds=2, k=2, only_fields=["los_max", "anesthesia", "postop_care"],
                                                    base_facts={"procedure": [{"value": "示意切除术"}]})
        self.assertEqual(set(facts), {"los_max", "surgery_gate", "anesthesia", "postop_care"})   # 门控自动带上，别的不跑
        base = {"facts": {"chief_symptoms": [{"value": ["腹痛"], "grounded": True, "source": "pathway", "quote": "q"}],
                          "los_max": [{"value": 3, "grounded": True, "source": "pathway", "quote": "q"}]},
                "missing": [{"field": "anesthesia", "need": "x"}], "kit": "route-b-kit-0.4",
                "meta": {k: "x" for k in ("pathway", "version", "icd10", "issuer", "url", "retrieved", "text_sha256")}}
        new = {"facts": facts, "missing": missing, "seconds": 1.0}
        out = B.merge(base, new, ["los_max", "anesthesia", "postop_care", "surgery_gate"], "route-b-kit-0.4.1")
        self.assertEqual(out["facts"]["chief_symptoms"], base["facts"]["chief_symptoms"])   # 没重跑的原样沿用
        self.assertIsInstance(out["facts"]["los_max"][0]["value"], dict)
        self.assertEqual(out["missing"], [])                                                 # 旧的 anesthesia 缺失被新结果替掉
        self.assertEqual(out["merged"]["base_kit"], "route-b-kit-0.4")
        self.assertEqual(out["validate"], [])

    def test_compare_groups(self):
        from route_b import compare_kb
        v0 = {"X": {"facts": {"chief_symptoms": [{"value": "右上腹痛，可伴发热", "grounded": True}],
                              "exams_required": [{"value": ["血常规", "需在10~30分钟后复查"], "bench_key": ["血常规", None], "grounded": True}]}}}
        v1 = {"X": {"facts": {"chief_symptoms": [{"value": ["右上腹痛", "发热", "黄疸"], "grounded": False, "quote_failed": True,
                                                  "source": "model_prior", "items_in_quote": [2, 3]}],
                              "exams_required": [{"value": [{"name": "血常规", "key": "血常规"}], "grounded": True, "source": "pathway",
                                                  "items_in_quote": [1, 1]}],
                              "exam_turnaround_days": [{"value": 1, "grounded": False, "source": "model_prior"}]}}}
        rows = {r["field"]: r for r in compare_kb.compare(v0, v1)}
        sym = rows["chief_symptoms"]
        self.assertEqual((sym["v1_items"], sym["v1_items_in_quote"], sym["v1_item_rate"]), (3, 2, 0.667))
        a = rows["GROUP_A_路径应有项（格子）"]
        self.assertEqual((a["diseases"], a["v0_grounded"], a["v1_grounded"]), (11, 2, 1))
        b = rows["GROUP_B_追问补全项（格子）"]
        self.assertEqual((b["diseases"], b["v0_with_fact"], b["v1_with_fact"], b["v1_model_prior_facts"]), (6, 0, 1, 1))


GUIDE_FULL = (
    "第一部分 抗菌药物临床应用的基本原则\n（五）疗程\n抗菌药物疗程因感染不同而异，一般宜用至体温正常、症状消退后72～96小时。\n"
    "抗菌药物预防性应用的基本原则\n二、围手术期抗菌药物的预防性应用\n（四）给药方案\n清洁手术的预防用药时间不超过 24 小时，心脏手术可视情况延长至 48 小时。\n"
    "三、侵入性诊疗操作患者的抗菌药物的预防应用\n放射介入预防用药。\n"
    "附录 2 抗菌药物在围手术期预防应用的品种选择\n结肠手术 24 小时。\n"
    "附录 3 特殊诊疗操作抗菌药物预防应用的建议\n冠状动脉造影 不推荐预防用药。\n"
)


class TestV05(unittest.TestCase):
    def test_hierarchical_chunk_titles(self):
        cs = chunks.make_chunks("G", GUIDE_FULL, max_chars=200)
        c24 = next(c for c in cs if "清洁手术的预防用药时间不超过 24 小时" in c["text"])
        self.assertIn("围手术期", c24["title"])
        self.assertIn("给药方案", c24["title"])
        c_app3 = next(c for c in cs if "冠状动脉造影" in c["text"])
        self.assertIn("附录 3", c_app3["title"])

    def test_symptoms_signs_exclude_form(self):
        idx = retrieve.BM25(chunks.make_chunks("P", PATH) + chunks.make_chunks("guideline_abx2015", GUIDE_FULL))
        client = Scripted({
            "典型症状": [("search", "询问病史 症状评估"), ("answer", "症状：右上腹痛", ["右上腹痛"])],
            "典型体征": [("search", "询问病史 症状评估"), ("answer", "体征：局部压痛", ["局部压痛"])],
        })
        facts, log, _ = agentic.build_disease(client, idx, "示意病", "P", {"P": PATH, "guideline_abx2015": GUIDE_FULL},
                                              KEYS, CLS, rounds=2, k=3, only_fields=["chief_symptoms", "signs"])
        chunk_kind = {c["id"]: c["kind"] for c in idx.chunks}
        for fld in ("chief_symptoms", "signs"):
            tr = next(t for t in log if t.get("field") == fld)
            self.assertEqual(tr.get("exclude_kinds"), ["form"])
            self.assertTrue(all(chunk_kind[cid] != "form" for cid in tr["chunks"]))

    def test_optional_exams_dedupe_required(self):
        req = [{"name": "血常规", "key": "血常规"}, {"name": "尿常规（包括酮体）", "key": "无对应"}, {"name": "心电图", "key": "CT"}]
        opt = [{"name": "血常规", "key": "血常规"}, {"name": "尿常规", "key": "无对应"},
               {"name": "18导联心电图", "key": "CT"}, {"name": "胸部CT", "key": "CT"}]
        kept, dropped = agentic.dedupe_optional(opt, req)
        self.assertEqual(dropped, 2)
        self.assertEqual([e["name"] for e in kept], ["18导联心电图", "胸部CT"])

        idx = _index()
        client = Scripted({
            "必需做的检查": [("search", "必需的检查项目"), ("answer", "必需的检查项目",
                                                       [{"name": "血常规", "key": "血常规"}, {"name": "腹部超声", "key": "超声"}])],
            "可选做的检查": [("search", "必需的检查项目"), ("answer", "必需的检查项目",
                                                       [{"name": "血常规", "key": "血常规"}, {"name": "肝功能", "key": "肝功能"}])],
        })
        facts, log, _ = agentic.build_disease(client, idx, "示意病", "P", {"P": PATH, "guideline_abx2015": GUIDE},
                                              KEYS, CLS, rounds=2, k=2, only_fields=["exams_required", "exams_optional"])
        tr_opt = next(t for t in log if t.get("field") == "exams_optional")
        self.assertIn("血常规、腹部超声", tr_opt["need"])
        f_opt = facts["exams_optional"][0]
        self.assertEqual(f_opt["deduped_from_required"], 1)
        self.assertEqual([e["name"] for e in f_opt["value"]], ["肝功能"])
        self.assertTrue(f_opt["grounded"])

    def test_nonsurgical_abx_excludes_prophylaxis(self):
        idx = retrieve.BM25(chunks.make_chunks("P", PATH) + chunks.make_chunks("guideline_abx2015", GUIDE_FULL, max_chars=200))
        client = Scripted({
            "主要治疗是不是外科手术": [("search", "治疗方案"), ("answer", "行示意切除术", False)],
            "抗菌药物的治疗疗程": [("search", "清洁手术 预防用药 24 小时 疗程"), ("answer", "72～96小时", "72～96小时")],
        })
        facts, log, _ = agentic.build_disease(client, idx, "示意病", "P", {"P": PATH, "guideline_abx2015": GUIDE_FULL},
                                              KEYS, CLS, rounds=2, k=3, only_fields=["surgery_gate", "antibiotic_rule"])
        tr_abx = next(t for t in log if t.get("field") == "antibiotic_rule")
        by_id = {c["id"]: c for c in idx.chunks}
        for cid in tr_abx["chunks"]:
            self.assertNotIn("清洁手术的预防用药时间不超过 24 小时", by_id[cid]["text"])
            self.assertNotIn("附录 2", by_id[cid]["title"])
            self.assertNotIn("附录 3", by_id[cid]["title"])
        self.assertTrue(facts["antibiotic_rule"][0]["grounded"])
        self.assertEqual(facts["antibiotic_rule"][0]["value"], "72～96小时")

        # 增量只跑 antibiotic_rule 时自动继承 base_facts 里的 surgery_gate=False
        client2 = Scripted({"抗菌药物的治疗疗程": [("search", "清洁手术 24 小时 疗程"), ("answer", "72～96小时", "72～96小时")]})
        facts2, log2, _ = agentic.build_disease(client2, idx, "示意病", "P", {"P": PATH, "guideline_abx2015": GUIDE_FULL},
                                                KEYS, CLS, rounds=2, k=3, only_fields=["antibiotic_rule"],
                                                base_facts={"surgery_gate": [{"value": False, "grounded": True}]})
        self.assertEqual(set(facts2), {"antibiotic_rule"})
        tr2 = next(t for t in log2 if t.get("field") == "antibiotic_rule")
        self.assertIsNotNone(tr2.get("exclude_title"))


if __name__ == "__main__":
    unittest.main()
