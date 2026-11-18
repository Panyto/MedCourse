"""关2 核心代码的自检（校1）。只用虚构病历，不碰真实数据。

运行：在 g2_core 目录下  python -B -m unittest tests.test_g2 -v
（或  python -B tests/test_g2.py）
伙计第 0 步必须全部通过；任何一条不过，停下来写进 GAPS，不要自己改代码。
"""
import copy
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
sys.dont_write_bytecode = True

import fixtures  # noqa: E402
from g2_kit import floors as F  # noqa: E402
from g2_kit import gen as G  # noqa: E402
from g2_kit import judge as J  # noqa: E402
from g2_kit import mines as M  # noqa: E402
from g2_kit import stats as S  # noqa: E402
from ruler_g2 import (READINGS, build_real, build_synth, classify_progress, core_set,  # noqa: E402
                      evaluate_admission, hit_span, k_any)
from ruler_g2 import diagnoses as dxm  # noqa: E402
from ruler_g2.copyrate import containment, cross_patient_similarity, lcs_ratio  # noqa: E402
from ruler_g2.events import tone_of  # noqa: E402


def real(name):
    return build_real(fixtures.ALL[name]())


def hard_conflicts(result):
    return sorted(k for k, v in result["readings"].items() if v["type"] == "hard" and v["adm_conflict"])


# 每份虚构病历上应该埋得进去的雷（埋不进去的是因为病历里没有对应的结构，例如内科没有手术）
APPLICABLE = {"GB": {"S1", "S4a", "S4b", "K1", "K2", "K3", "K4", "K5", "B2"},
              "KNEE": {"S1", "S3", "S4b", "K1", "K3", "K4", "K5"},   # 0.3：K3 有了模板方式
              "PNA": {"S1", "K3", "K4", "K5"}}


class TestRulerClean(unittest.TestCase):
    def test_clean_fixtures_have_no_hard_conflict(self):
        for name in fixtures.ALL:
            r = evaluate_admission(real(name))
            self.assertEqual(hard_conflicts(r), [], name)
            self.assertEqual(set(r["readings"]), set(READINGS))

    def test_checkability(self):
        gb = evaluate_admission(real("GB"))["readings"]
        pna = evaluate_admission(real("PNA"))["readings"]
        for rid in ("K1", "K2", "K3", "K4", "S4a", "B2"):
            self.assertTrue(gb[rid]["adm_checkable"], rid)
        for rid in ("K1", "K2", "S4a"):
            self.assertFalse(pna[rid]["adm_checkable"], rid)       # 内科没有手术
        self.assertTrue(evaluate_admission(real("KNEE"))["readings"]["S3"]["adm_checkable"])
        chk, hit = k_any(evaluate_admission(real("GB")))
        self.assertTrue(chk)
        self.assertFalse(hit)


class TestMines(unittest.TestCase):
    def test_every_mine_hits_where_applicable(self):
        covered = set()
        for name, want in APPLICABLE.items():
            for mt in M.MINE_TYPES:
                for seed in range(3):
                    out = M.plant(real(name), mt, random.Random(seed))
                    if mt not in want:
                        self.assertIsNone(out, f"{name} {mt} 本不该埋得进去")
                        break
                    self.assertIsNotNone(out, f"{name} {mt}")
                    new, rec = out
                    r = evaluate_admission(new)
                    self.assertTrue(hit_span(r, M.MINE_READING[mt], rec.doc_idx, rec.start, rec.end),
                                    f"{name} {mt} seed={seed} 没命中")
                    covered.add(mt)
        self.assertEqual(covered, set(M.MINE_TYPES))

    def test_mine_does_not_touch_original(self):
        adm = real("GB")
        before = [d.text for d in adm.docs]
        M.plant(adm, "K1", random.Random(0))
        self.assertEqual(before, [d.text for d in adm.docs])

    def test_plant_many(self):
        for combo in (("K1", "S1"), ("K1", "S1", "K3", "S4b")):
            out = M.plant_many(real("GB"), combo, random.Random(1))
            self.assertIsNotNone(out, combo)
            new, recs = out
            self.assertEqual(len(recs), len(combo))
            r = evaluate_admission(new)
            # 后埋的雷可能挪动先埋的雷的位置，所以只要求每种读数都报冲突、最后一颗位置对
            for rec in recs:
                self.assertTrue(r["readings"][M.MINE_READING[rec.mine]]["adm_conflict"], rec.mine)
            last = recs[-1]
            self.assertTrue(hit_span(r, M.MINE_READING[last.mine], last.doc_idx, last.start, last.end))

    def test_legit_differences_raise_nothing(self):
        n = 0
        for name in fixtures.ALL:
            for kind in M.LEGIT_TYPES:
                out = M.legit(real(name), kind, random.Random(0))
                if out is None:
                    continue
                new, rec = out
                r = evaluate_admission(new)
                self.assertEqual(hard_conflicts(r), [], f"{name} {kind}")
                n += 1
        self.assertGreaterEqual(n, 12)


class TestPieces(unittest.TestCase):
    def tone(self, text, word):
        s = text.index(word)
        return tone_of(text, s, s + len(word))[0]

    def test_tone(self):
        self.assertEqual(self.tone("腹部CT示胆囊结石。", "CT"), "R")
        self.assertEqual(self.tone("完善腹部CT、MRCP、血淀粉酶等检查。", "MRCP"), "P")
        self.assertEqual(self.tone("血淀粉酶正常，急性胰腺炎可排除。", "淀粉酶"), "R")
        self.assertEqual(self.tone("MRCP结果待回报。", "MRCP"), "P")
        self.assertEqual(self.tone("标本送病理，待病理结果回报。", "病理结果"), "P")   # 0.2
        self.assertEqual(self.tone("待MRCP结果回报。", "MRCP"), "P")                   # 0.2
        self.assertEqual(self.tone("于外院行胃镜示慢性胃炎。", "胃镜"), "H")
        self.assertEqual(self.tone("今日已行胃镜检查。", "胃镜"), "D")
        self.assertEqual(self.tone("拟明日行冠脉造影。", "冠脉造影"), "P")

    def test_dx_items(self):
        hr = real("PNA").hr
        names = [it.norm for it in dxm.list_items(hr) if it.section == "入院诊断"]
        self.assertIn("2型糖尿病", names)                     # "2.2型糖尿病" 不能被拆错
        fp = real("GB").fp
        diffs = [it.norm for it in dxm.differential_items(fp)]
        self.assertEqual(diffs, ["急性胰腺炎", "胆总管结石", "右肾结石"])
        hedged = [it for it in dxm.list_items(real("GB").hr) if it.section == "入院诊断"]
        self.assertTrue(hedged[0].hedged)
        self.assertFalse(hedged[1].hedged)

    def test_dx_context(self):
        t = "血淀粉酶正常，急性胰腺炎可排除。急性胆囊炎诊断明确。考虑胆总管结石可能。"
        for name, st in (("急性胰腺炎", "排除"), ("急性胆囊炎", "确诊"), ("胆总管结石", "疑似")):
            s = t.index(name)
            self.assertEqual(dxm._context_state(t, s, s + len(name)), st, name)
        self.assertTrue(dxm.dx_match("急性胆囊炎", "胆囊炎"))
        self.assertFalse(dxm.dx_match("胆囊炎", "胰腺炎"))

    def test_core_set_roles(self):
        c = core_set(real("GB"))
        self.assertEqual([d.role for d in c.docs], ["HR", "FP", "SUP", "PRE", "POST", "DAILY", "DAILY", "DS"])
        c = core_set(real("PNA"))
        self.assertEqual([d.role for d in c.docs][:3], ["HR", "FP", "SUP"])
        self.assertNotIn("POST", [d.role for d in c.docs])
        self.assertEqual(c.docs[-1].role, "DS")

    def test_null_and_post_discharge(self):
        rec = fixtures.ALL["PNA"]()
        extra = copy.deepcopy(rec["docs"][3])
        extra["day_offset"] = None
        late = copy.deepcopy(rec["docs"][3])
        late["day_offset"] = 9
        rec["docs"].insert(2, extra)
        rec["docs"].insert(4, late)
        adm = build_real(rec)
        self.assertEqual(len(adm.off_docs), 2)
        self.assertEqual({f for d in adm.off_docs for f in d.flags}, {"null_day", "post_discharge"})
        self.assertEqual([d.pos for d in adm.docs], list(range(len(adm.docs))))
        self.assertEqual(adm.docs[-1].doc_type, "discharge_summary")
        self.assertEqual(hard_conflicts(evaluate_admission(adm)), [])

    def test_classify_progress(self):
        self.assertEqual(classify_progress("首次病程记录\n2025-03-01"), "首次病程")
        self.assertEqual(classify_progress("2025-03-03 10:00 术前小结\n简要病情"), "术前小结")
        self.assertEqual(classify_progress("2025-03-04 术后首次病程记录\n手术时间"), "术后首次病程")
        self.assertEqual(classify_progress("2025-03-07 09:00 主任医师查房记录\n今日"), "主任医师查房")
        self.assertEqual(classify_progress("2025-03-05 09:00 日常病程记录\n患者"), "日常查房")

    def test_copyrate(self):
        a = "患者今日无发热，咳嗽较前减轻，双肺呼吸音粗，继续抗感染治疗。"
        self.assertAlmostEqual(containment(a, a), 1.0)
        self.assertAlmostEqual(containment("完全不同的另一段文字内容在这里", a), 0.0)
        self.assertAlmostEqual(lcs_ratio(a, a), 1.0)
        self.assertLess(lcs_ratio(a, "完全不同的另一段文字内容在这里"), 0.2)
        pool = [(n, "外科" if n != "PNA" else "内科", real(n).hr.text) for n in fixtures.ALL]
        pool.append(("GB2", "外科", real("GB").hr.text))
        sims = cross_patient_similarity(pool)
        self.assertAlmostEqual(sims["GB2"], 1.0)
        self.assertEqual(sims["PNA"], 0.0)


# ---------- 生成 ----------

def gb_slot():
    return G.make_slot("T-GB", core_set(real("GB")), "男", 58, "普外科", "急性胆囊炎", "K81")


def good_draft():
    return {
        "final_diagnoses": ["急性胆囊炎", "胆囊结石"],
        "presentation": {"chief_complaint": "右上腹痛3天", "history": "进食油腻后右上腹持续性胀痛，伴恶心",
                         "past_history": "否认高血压、糖尿病", "exam": "右上腹压痛，Murphy征阳性"},
        "observations": [
            {"id": "o1", "name": "血常规", "kind": "检验", "order_day": 0, "result_day": 0,
             "result": "白细胞12.1×10^9/L，中性粒细胞比例升高"},
            {"id": "o2", "name": "腹部CT", "kind": "影像", "order_day": 0, "result_day": 1,
             "result": "胆囊增大，胆囊壁增厚，胆囊内多发结石"},
            {"id": "o3", "name": "血淀粉酶", "kind": "检验", "order_day": 0, "result_day": 1, "result": "血淀粉酶正常"},
            {"id": "o4", "name": "MRCP", "kind": "影像", "order_day": 1, "result_day": 2,
             "result": "胆总管未见结石，肝内外胆管无扩张"},
            {"id": "o5", "name": "术后病理", "kind": "病理", "order_day": 3, "result_day": 6,
             "result": "慢性胆囊炎急性发作，胆囊结石"}],
        "judgments": [
            {"day": 0, "diagnosis": "急性胆囊炎", "state": "疑似", "basis": []},
            {"day": 0, "diagnosis": "急性胰腺炎", "state": "疑似", "basis": []},
            {"day": 0, "diagnosis": "胆总管结石", "state": "疑似", "basis": []},
            {"day": 1, "diagnosis": "急性胰腺炎", "state": "排除", "basis": ["o3"]},
            {"day": 1, "diagnosis": "急性胆囊炎", "state": "确诊", "basis": ["o1", "o2"]},
            {"day": 2, "diagnosis": "胆总管结石", "state": "排除", "basis": ["o4"]}],
        "actions": [
            {"id": "a1", "kind": "治疗", "name": "抗感染治疗", "side": "无", "plan_day": 0, "do_day": 0},
            {"id": "a2", "kind": "手术", "name": "腹腔镜胆囊切除术", "side": "无", "plan_day": 1, "do_day": 3}],
    }


class TestGen(unittest.TestCase):
    def test_slot(self):
        s = gb_slot()
        self.assertEqual(s["los"], 7)
        self.assertTrue(s["surgical"])
        self.assertEqual(s["surgery_day"], 3)
        self.assertEqual(s["admit_date"], G.fictional_admit_date("T-GB"))
        self.assertEqual([x["doc_type_cn"] for x in s["skeleton"]][:2], ["入院记录", "首次病程记录"])
        self.assertEqual(s["skeleton"][-1]["doc_type_cn"], "出院记录")

    def test_validate_good(self):
        self.assertEqual(G.validate_draft(good_draft(), gb_slot()), [])

    def test_validate_bad(self):
        slot = gb_slot()
        cases = []
        d = good_draft(); d["observations"][4]["order_day"] = 2; cases.append((d, "病理"))
        d = good_draft(); d["judgments"][4]["basis"] = ["o4"]; cases.append((d, "第2天才出结果"))
        d = good_draft(); d["judgments"][4]["state"] = "疑似"; cases.append((d, "要有一条"))
        d = good_draft(); d["actions"].append(dict(d["actions"][1], id="a3")); cases.append((d, "有且只有一项手术"))
        d = good_draft(); d["observations"][1]["result_day"] = 9; cases.append((d, "天数不对"))
        d = good_draft(); d["final_diagnoses"] = ["胆囊结石"]; cases.append((d, "第一条"))
        d = good_draft(); d["judgments"] = d["judgments"][3:] + [d["judgments"][0]]; cases.append((d, "第0天"))
        d = good_draft(); del d["actions"]; cases.append((d, "缺少字段"))
        for draft, want in cases:
            errs = G.validate_draft(draft, slot)
            self.assertTrue(any(want in e for e in errs), (want, errs))
        self.assertEqual(G.validate_draft(G.parse_json_object("```json\n[]\n```"), slot), ["不是 JSON 对象"])

    def test_parse_json(self):
        import json
        s = "```json\n" + json.dumps(good_draft(), ensure_ascii=False) + "\n```"
        self.assertEqual(G.parse_json_object(s), good_draft())
        self.assertIsNone(G.parse_json_object("没有 JSON"))

    def test_visibility(self):
        dr = good_draft()
        v0 = G.render_visible(dr, 0)
        v1 = G.render_visible(dr, 1)
        v1e = G.render_visible(dr, 1, with_judgments=False)
        vds = G.render_visible(dr, 7, final=True)
        self.assertNotIn("胆囊壁增厚", v0)                      # 第1天才回报
        self.assertIn("胆囊壁增厚", v1)
        self.assertNotIn("胆总管未见结石", v1)                  # MRCP 第2天回报
        self.assertIn("MRCP（第1天安排）", v1)
        self.assertNotIn("慢性胆囊炎急性发作", v1)
        self.assertNotIn("急性胆囊炎=确诊", v0)
        self.assertIn("急性胆囊炎=确诊", v1)
        self.assertNotIn("诊断判断", v1e)
        self.assertNotIn("出院诊断", v1)
        self.assertIn("出院诊断：急性胆囊炎", vds)
        self.assertIn("腹腔镜胆囊切除术（拟第3天实施）", v1)
        self.assertIn("慢性胆囊炎急性发作", G.render_full(dr))

    def test_doc_messages_arms(self):
        slot, dr = gb_slot(), good_draft()
        sup_k = next(x["k"] for x in slot["skeleton"] if x["role"] == "SUP")
        prior = [{"k": 0, "doc_type_cn": "入院记录", "day": 0, "text": "入院记录……"},
                 {"k": 1, "doc_type_cn": "首次病程记录", "day": 0, "text": "首次病程记录……"}]
        txt = {}
        for arm in G.ARMS:
            msgs, mt, info = G.doc_messages(arm, slot, sup_k, prior, dr)
            txt[arm] = msgs[1]["content"]
            self.assertEqual(info["dropped_prior"], 0)
            self.assertEqual(mt, 900)
        self.assertIn("急性胆囊炎", txt["G-A"])
        self.assertNotIn("住院天数", txt["G-A"])
        self.assertIn("手术：第3天", txt["G-B"])
        self.assertIn("慢性胆囊炎急性发作", txt["G-C"])
        for arm in ("G-D", "G-E"):
            self.assertNotIn("慢性胆囊炎急性发作", txt[arm])
            self.assertNotIn("胆总管未见结石", txt[arm])
            self.assertNotIn("出院主要诊断", txt[arm])
        self.assertIn("诊断判断", txt["G-D"])
        self.assertNotIn("诊断判断", txt["G-E"])
        self.assertNotIn("…（男性写", txt["G-A"])        # 查房模板里没有入院记录的格式

    def test_doc_messages_budget(self):
        slot, dr = gb_slot(), good_draft()
        ds_k = len(slot["skeleton"]) - 1
        big = "患者病情平稳" * 300                       # 每篇约 1,800 字，7 篇放不下
        prior = [{"k": i, "doc_type_cn": slot["skeleton"][i]["doc_type_cn"], "day": slot["skeleton"][i]["day"],
                  "text": ("入院记录\n" if i == 0 else "") + big} for i in range(ds_k)]
        msgs, mt, info = G.doc_messages("G-C", slot, ds_k, prior, dr)
        self.assertGreater(info["dropped_prior"], 0)
        content = msgs[1]["content"]
        self.assertLessEqual(G.estimate_tokens(msgs[0]["content"] + content), G.MAX_MODEL_LEN - mt - 200)
        self.assertIn("没有列出", content)
        self.assertIn("【第1篇｜住院第0天｜入院记录】", content)     # 入院记录保留

    def test_oracle_leaks(self):
        slot, dr = gb_slot(), good_draft()
        sk = slot["skeleton"]

        def doc(role, text):
            spec = next(x for x in sk if x["role"] == role)
            return {"slot_id": "T-GB", "k": spec["k"], "role": role, "doc_type_cn": spec["doc_type_cn"],
                    "day": spec["day"], "date": G.doc_date(slot, spec["day"]), "text": text}
        clean = [doc("FP", "初步诊断：急性胆囊炎？胆总管结石待排。完善MRCP。"),
                 doc("SUP", "腹部CT示胆囊壁增厚。急性胆囊炎诊断明确。拟行腹腔镜胆囊切除术。MRCP结果待回报。"),
                 doc("PRE", "拟施手术名称和方式：腹腔镜胆囊切除术。术后可能出现出血等并发症。")]
        per, summ = G.oracle_leaks(dr, slot, clean)
        self.assertEqual((summ["O"], summ["A"], summ["B"]), (0, 0, 0), per)
        leak = [doc("FP", "胆总管结石已排除。"),
                doc("SUP", "MRCP示胆总管未见结石。"),
                doc("PRE", "患者已行腹腔镜胆囊切除术，手术顺利。")]
        per, summ = G.oracle_leaks(dr, slot, leak)
        self.assertEqual([(h["O"], h["A"], h["B"]) for h in per], [(0, 0, 1), (1, 0, 0), (0, 1, 0)])
        self.assertEqual(summ["any"], 1)

    def test_format_checks(self):
        slot = gb_slot()
        ds_spec = slot["skeleton"][-1]
        date = G.doc_date(slot, ds_spec["day"])
        ok = {"k": ds_spec["k"], "role": "DS", "date": date,
              "text": f"出院记录\n入院时间：{slot['admit_date']} 10:00　住院天数：7天\n出院医嘱：于{G.doc_date(slot, 21)}复查。"}
        bad = {"k": 1, "role": "FP", "date": G.doc_date(slot, 0), "text": f"{G.doc_date(slot, 2)}腹部CT示胆囊结石。"}
        out = G.format_checks(slot, [ok, bad])
        self.assertEqual(out[0], {"k": ds_spec["k"], "future_dates": 0, "header_ok": True})
        self.assertEqual(out[1]["future_dates"], 1)

    def test_build_synth_runs(self):
        slot = gb_slot()
        slot["admit_date"] = "2025-03-01"          # 虚构病历正文里写的是这个入院日期
        core = core_set(real("GB"))
        docs = [{"k": k, "doc_type_cn": x["doc_type_cn"], "day": x["day"], "role": x["role"], "text": d.text}
                for k, (x, d) in enumerate(zip(slot["skeleton"], core.docs))]
        adm = build_synth(slot, docs, "synth:G-D")
        self.assertEqual(hard_conflicts(evaluate_admission(adm)), [])
        self.assertEqual(adm.source, "synth:G-D")


# ---------- 评委 ----------

class TestJudge(unittest.TestCase):
    def test_render(self):
        s = J.render_for_judge(real("GB"))
        self.assertTrue(s.startswith("【第1篇｜住院第0天｜入院记录】"))
        self.assertIn("【第8篇｜住院第7天｜出院记录】", s)
        w = J.render_for_judge(real("GB"), only_idx={0, 4})
        self.assertIn("【第2篇｜住院第3天｜术后首次病程记录】", w)
        self.assertNotIn("第3篇", w)
        for q in ("QK", "QS", "QB"):
            m = J.q_messages(q, real("PNA"))
            self.assertIn("只输出一个 JSON 对象", m[0]["content"])
            self.assertNotIn("{docs}", m[0]["content"])
        self.assertIn("1 到 5", J.h_messages(real("PNA"))[0]["content"])

    def test_parse_q(self):
        self.assertEqual(J.parse_q('{"结论": "有", "第几篇": 3, "引文": "术中见胆囊肿大"}'),
                         {"verdict": True, "doc": 3, "quote": "术中见胆囊肿大"})
        self.assertFalse(J.parse_q('```json\n{"结论":"无","第几篇":null,"引文":""}\n```')["verdict"])
        self.assertIsNone(J.parse_q("我认为这些文书没有问题。")["verdict"])
        self.assertTrue(J.parse_q('结论："有"，第几篇：2')["verdict"])
        self.assertIsNone(J.parse_q('{"结论": "不确定"}')["verdict"])
        self.assertIsNone(J.parse_q('{"结论": "有", "第几篇": true}')["doc"])

    def test_parse_h(self):
        self.assertEqual(J.parse_h('{"分数": 4, "理由": "基本一致"}'), 4)
        self.assertEqual(J.parse_h('{"分数": "5"}'), 5)
        self.assertEqual(J.parse_h('分数: 3'), 3)
        self.assertIsNone(J.parse_h('{"分数": 7}'))
        self.assertIsNone(J.parse_h("很好"))

    def test_paired(self):
        self.assertEqual(J.paired(False, True), {"host_unclean": 0, "upper": 1, "lower": 1, "point": 1})
        self.assertEqual(J.paired(True, True), {"host_unclean": 1, "upper": 0, "lower": 0, "point": 0})
        self.assertEqual(J.paired(None, True), {"host_unclean": 0, "upper": 1, "lower": 0, "point": None})
        self.assertEqual(J.paired(False, None), {"host_unclean": 0, "upper": 1, "lower": 0, "point": None})
        self.assertEqual(J.paired(False, False)["point"], 0)

    def test_locate_quote(self):
        t = "患者术后第1天，诉切口轻度疼痛，无发热。"
        self.assertEqual(J.locate_quote("诉切口轻度疼痛", t), (t.index("诉"), t.index("痛，") + 1))
        self.assertIsNotNone(J.locate_quote("患者术后第 1 天，诉切口轻度疼痛（原文）", t))
        self.assertIsNone(J.locate_quote("短", t))
        self.assertIsNone(J.locate_quote("完全没有出现过的一句话", t))


# ---------- 地板、退化解 ----------

class TestFloors(unittest.TestCase):
    def test_easy_and_hard(self):
        pool = [real(n) for n in fixtures.ALL]
        gb = pool[0]
        new = F.floor_easy(gb, pool, random.Random(0))
        self.assertEqual(new.hr.text, gb.hr.text)
        self.assertNotEqual([d.text for d in new.docs], [d.text for d in gb.docs])
        self.assertEqual(new.source, "floor:easy")
        meta = {"FAKE-GB": {"dept": "普外科", "chapter": "K", "los": 7},
                "FAKE-KNEE": {"dept": "骨科", "chapter": "M", "los": 7},
                "FAKE-PNA": {"dept": "呼吸科", "chapter": "J", "los": 6}}
        self.assertIsNone(F.floor_hard(gb, pool, meta, random.Random(0)))     # 没有同科室供体

    def test_shuffle(self):
        gb = real("GB")
        new = F.floor_shuffle(gb, random.Random(0))
        self.assertEqual(new.docs[0].doc_type, "hospital_record")
        self.assertEqual(new.docs[-1].doc_type, "discharge_summary")
        self.assertEqual(sorted(d.text for d in new.docs), sorted(d.text for d in gb.docs))
        self.assertNotEqual([d.text for d in new.docs], [d.text for d in gb.docs])
        self.assertEqual([d.day for d in new.docs], [d.day for d in gb.docs])
        self.assertTrue(evaluate_admission(new))

    def test_degenerates(self):
        gb = real("GB")
        c = F.degen_copy(gb)
        progs = [d for d in c.docs if d.doc_type == "progress_note" and d.subtype != "首次病程"]
        self.assertEqual(len({d.text for d in progs}), 1)
        p = F.degen_plan_only(gb)
        self.assertNotIn("CT示", p.ds.text)
        a = F.degen_all_excluded(gb)
        self.assertIn("均已排除", a.ds.text)
        pool = [real("GB"), real("PNA"), real("KNEE")]
        tpl = F.degen_template(pool, {"FAKE-GB": "外科", "FAKE-PNA": "内科", "FAKE-KNEE": "外科"})
        self.assertEqual(len(tpl), 3)
        knee_prog = [d.text for d in tpl[2].docs if d.doc_type == "progress_note"]
        gb_prog = [d.text for d in pool[0].docs if d.doc_type == "progress_note"]
        self.assertTrue(set(knee_prog) <= set(gb_prog))
        for x in (c, p, a):
            self.assertTrue(evaluate_admission(x))


# ---------- 统计 ----------

class TestStats(unittest.TestCase):
    def test_clopper_pearson(self):
        lo, hi = S.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** 0.1, places=4)
        lo, hi = S.clopper_pearson(10, 10)
        self.assertAlmostEqual(lo, 0.025 ** 0.1, places=4)
        self.assertEqual(hi, 1.0)
        lo, hi = S.clopper_pearson(5, 10)
        self.assertAlmostEqual(lo, 0.1871, places=3)
        self.assertAlmostEqual(hi, 0.8129, places=3)
        lo, hi = S.clopper_pearson(48, 60)
        self.assertAlmostEqual(lo, 0.6767, places=3)
        self.assertAlmostEqual(hi, 0.8922, places=3)

    def test_auc_tau_band(self):
        self.assertEqual(S.auc([3, 4], [1, 2]), 1.0)
        self.assertEqual(S.auc([1, 2], [3, 4]), 0.0)
        self.assertEqual(S.auc([1, 1], [1, 1]), 0.5)
        self.assertEqual(S.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(S.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(S.band_verdict(-0.01, 0.01, 0.10), "带内")
        self.assertEqual(S.band_verdict(0.05, 0.10, 0.10), "带外偏高")
        self.assertEqual(S.band_verdict(-0.10, -0.05, 0.10), "带外偏低")
        self.assertEqual(S.band_verdict(-0.01, 0.05, 0.10), "未定")
        self.assertEqual(S.band_verdict(-0.01, 0.01, 0.0, abs_min=0.02), "带内")

    def test_bootstrap(self):
        units = {f"s{i}": (i % 3 == 0, 1) for i in range(60)}
        units = {k: (int(v[0]), v[1]) for k, v in units.items()}
        a = S.cluster_ratio(units)
        self.assertEqual(a, S.cluster_ratio(units))                 # 同一种子结果相同
        self.assertAlmostEqual(a[0], 20 / 60)
        self.assertLess(a[1], a[0])
        self.assertLess(a[0], a[2])
        ub = {k: (1, 1) for k in units}
        d = S.paired_diff(ub, units)
        self.assertAlmostEqual(d[0], 40 / 60)
        self.assertEqual(d[3], 60)
        m = S.paired_mean_diff({k: 1.0 for k in units}, {k: 0.5 for k in units})
        self.assertAlmostEqual(m[0], 0.5)
        rows_a = {"x": {k: (1, 2) for k in units}, "y": {k: (1, 4) for k in units}, "z": {k: (3, 4) for k in units}}
        t = S.tau_ci(rows_a, rows_a)
        self.assertEqual(t[0], 1.0)


# ---------- 0.2：照 G2 CPU 包 1 查出来的毛病，各造一份更像真实写法的虚构病历 ----------

LONG_HEAD = ("[本院]病程记录\n姓名：[患者] 性别：男 年龄：58岁 科室：普外科 病区：[病区] 床号：[床号] "
             "住院号：[住院号] 主管医师：[医师] 费别：[费别] 记录类型：病程\n")


def variant(name, edits):
    """edits：{文书序号: 函数(原文) -> 新文}。"""
    rec = fixtures.ALL[name]()
    for i, fn in edits.items():
        rec["docs"][i]["text"] = fn(rec["docs"][i]["text"])
    return build_real(rec)


def long_head_gb():
    return variant("GB", {i: (lambda t: LONG_HEAD + t) for i in (1, 2, 3, 4, 5, 6)})


class TestV02(unittest.TestCase):
    def test_title_beyond_80_chars(self):
        self.assertGreater(len(LONG_HEAD), 80)
        adm = long_head_gb()
        subs = [d.subtype for d in adm.docs]
        self.assertEqual(subs, ["入院记录", "首次病程", "主治医师查房", "术前小结", "术后首次病程", "日常查房",
                                "主任医师查房", "出院记录"])
        from g2_kit.diag import _classify_v01
        self.assertEqual(_classify_v01(adm.post.text), "其他病程")      # 0.1 认不出
        core = core_set(adm)
        self.assertEqual([d.role for d in core.docs], ["HR", "FP", "SUP", "PRE", "POST", "DAILY", "DAILY", "DS"])
        self.assertEqual(hard_conflicts(evaluate_admission(adm)), [])

    def test_title_earliest_wins(self):
        self.assertEqual(classify_progress("2025-03-05 09:00 日常病程记录\n患者术后第1天，已完成术前讨论，会诊意见同前。"),
                         "日常查房")
        self.assertEqual(classify_progress(LONG_HEAD + "2025-03-04 15:00 术后首次病程记录\n会诊意见：同意手术。"),
                         "术后首次病程")

    def test_k2_mine_skips_pending_and_keeps_clause(self):
        adm = variant("GB", {0: lambda t: t.replace("急诊血常规示白细胞12.1×10^9/L。", "急诊血常规示白细胞12.1×10^9/L"),
                             4: lambda t: t.replace("标本送病理检查。", "标本送病理，待病理结果回报。")})
        new, rec = M.plant(adm, "K2", random.Random(0))
        piece = new.doc(rec.doc_idx).text[rec.start:rec.end]
        self.assertNotIn("待", piece)
        self.assertTrue(hit_span(evaluate_admission(new), "K2", rec.doc_idx, rec.start, rec.end))

    def test_b2_mine_strips_colon_results(self):
        adm = variant("GB", {2: lambda t: t.replace("急性胆囊炎诊断明确。", "血常规：WBC 10.2×10^9/L。急性胆囊炎诊断明确。")})
        self.assertEqual(hard_conflicts(evaluate_admission(adm)), [])
        new, rec = M.plant(adm, "B2", random.Random(0))
        self.assertTrue(hit_span(evaluate_admission(new), "B2", rec.doc_idx, rec.start, rec.end))

    def test_s3_mine_flips_every_mention(self):
        adm = variant("KNEE", {6: lambda t: t.replace("出院诊断：1.右膝骨关节炎 2.高血压病",
                                                     "出院诊断：1.右膝骨关节炎 2.高血压病 3.右膝关节内翻畸形")})
        self.assertEqual(hard_conflicts(evaluate_admission(adm)), [])
        new, rec = M.plant(adm, "S3", random.Random(0))
        self.assertTrue(rec.note.endswith("×2"))
        ds = new.ds
        a, b = ds.section("出院诊断")
        self.assertNotIn("右膝", ds.text[a:b])
        self.assertTrue(hit_span(evaluate_admission(new), "S3", rec.doc_idx, rec.start, rec.end))

    def test_k5_stamp_line_and_mine(self):
        adm = variant("GB", {3: lambda t: t + "上级医师签名：[医师] 2025-03-06 08:00\n"})
        self.assertFalse(evaluate_admission(adm)["readings"]["K5"]["adm_conflict"])
        new, rec = M.plant(adm, "K5", random.Random(0))
        t = new.doc(rec.doc_idx).text
        self.assertLess(rec.start, t.index("上级医师签名"))
        self.assertTrue(hit_span(evaluate_admission(new), "K5", rec.doc_idx, rec.start, rec.end))

    def test_l3_clinic_word_next_to_exam(self):
        adm = variant("PNA", {3: lambda t: t.replace("咳嗽减轻。痰培养示肺炎链球菌", "咳嗽减轻，痰培养示肺炎链球菌")})
        new, rec = M.legit(adm, "L3", random.Random(0))
        self.assertFalse(evaluate_admission(new)["readings"]["K3"]["adm_conflict"])
        mined, mrec = M.plant(adm, "K3", random.Random(0))
        self.assertTrue(hit_span(evaluate_admission(mined), "K3", mrec.doc_idx, mrec.start, mrec.end))

    def test_diag_outputs_only_categories(self):
        from g2_kit import diag
        adms = [real(n) for n in fixtures.ALL] + [long_head_gb()]
        acc = diag.classifier_acceptance(adms)
        self.assertEqual(acc["post_admissions"], 3)
        self.assertEqual(acc["pre_before_post"], acc["pre_and_post_admissions"])
        self.assertEqual(acc["fp_multi_admissions"], 0)
        rows = []
        for adm in adms:
            r = evaluate_admission(adm)
            rows += diag.classifier_profile(adm) + diag.k5_profile(adm) + diag.k4_profile(adm) + diag.k3_profile(r, adm)
        self.assertTrue(rows)
        for row in rows:
            for v in row.values():
                self.assertTrue(isinstance(v, bool) or (isinstance(v, str) and len(v) <= 12), (row, v))
        adm = real("GB")
        clean = evaluate_admission(adm)
        new, rec = M.plant(adm, "K1", random.Random(0))
        self.assertEqual(diag.mine_miss("K1", clean, new, rec, evaluate_admission(new)), "hit")
        # 0.1 的 K2 雷会抄"待病理回报"：照那样手工造一个，诊断应该说口吻不是结果
        rec2 = M.MineRecord("K2", 0, 0, 0, 4, "own")
        bad = adm.clone("mine:K2")
        t = bad.doc(0)
        at = t.text.index("入院诊断")
        piece = "标本送病理，待病理结果回报。"
        bad.replace_text(0, t.text[:at] + piece + t.text[at:])
        rec2 = M.MineRecord("K2", 0, at, at + len(piece), 4, "own")
        self.assertTrue(diag.mine_miss("K2", clean, bad, rec2, evaluate_admission(bad)).startswith("k2_tone_"))




# ---------- 0.3（依据 G2_CPU_pkg2 的 fold 0 计数）----------

def k5c(adm):
    return evaluate_admission(adm)["readings"]["K5"]["adm_conflict"]


class TestV03(unittest.TestCase):
    def test_k5_correction_section_in_hr(self):
        adm = variant("GB", {0: lambda t: t + "修正诊断：\n1.慢性胆囊炎急性发作\n2.胆囊结石\n医师：[医师] 2025-03-05\n"})
        self.assertFalse(k5c(adm))

    def test_k5_hr_header_field(self):
        adm = variant("GB", {0: lambda t: t.replace("入院时间：2025-03-01 10:00\n",
                                                    "入院时间：2025-03-01 10:00\n病史采集时间：2025-03-02 09:00\n")})
        self.assertFalse(k5c(adm))

    def test_k5_preop_surgery_date(self):
        adm = variant("GB", {3: lambda t: t + "手术时间：2025-03-04\n"})
        self.assertFalse(k5c(adm))
        # 术前小结以外的病程里写了以后的检查结果，照样报
        adm2 = variant("GB", {2: lambda t: t + "2025-03-05复查血常规示白细胞正常。\n"})
        self.assertTrue(k5c(adm2))

    def test_k5_tail_date_line_vs_content(self):
        adm = variant("GB", {2: lambda t: t + "[医师]\n2025-03-05\n"})
        self.assertFalse(k5c(adm))
        adm2 = variant("GB", {2: lambda t: t + "2025-03-05 血常规示白细胞正常\n"})
        self.assertTrue(k5c(adm2))

    def test_k4_merged_course_section_cut(self):
        merged = lambda t: t.replace("Murphy征阳性。\n入院诊断：1.急性胆囊炎？2.胆囊结石\n诊疗经过：",
                                     "Murphy征阳性。入院后诊疗经过：")
        adm = variant("GB", {7: merged})
        ds = adm.ds
        self.assertIn("术后病理", ds.text[slice(*ds.section("入院情况"))])   # 并段确实发生（标题没认出来）
        self.assertFalse(evaluate_admission(adm)["readings"]["K4"]["adm_conflict"])
        new, rec = M.plant(real("GB"), "K4", random.Random(0))
        self.assertTrue(hit_span(evaluate_admission(new), "K4", rec.doc_idx, rec.start, rec.end))

    def test_k4_prior_clause(self):
        adm = variant("GB", {7: lambda t: t.replace("Murphy征阳性。\n入院诊断", "Murphy征阳性。阑尾切除术后5年。\n入院诊断")})
        self.assertFalse(evaluate_admission(adm)["readings"]["K4"]["adm_conflict"])
        adm2 = variant("GB", {7: lambda t: t.replace("Murphy征阳性。\n入院诊断", "Murphy征阳性。术后病理示慢性胆囊炎。\n入院诊断")})
        self.assertTrue(evaluate_admission(adm2)["readings"]["K4"]["adm_conflict"])

    def test_k3_outpatient_result_in_hr(self):
        adm = variant("GB", {0: lambda t: t.replace("辅助检查：急诊血常规示白细胞12.1×10^9/L。",
                                                    "辅助检查：急诊血常规示白细胞12.1×10^9/L。MRCP示胆总管未见结石。")})
        self.assertFalse(evaluate_admission(adm)["readings"]["K3"]["adm_conflict"])

    def test_k3_template_mine(self):
        new, rec = M.plant(real("PNA"), "K3", random.Random(0))
        self.assertTrue(hit_span(evaluate_admission(new), "K3", rec.doc_idx, rec.start, rec.end))
        self.assertNotEqual(new.doc(rec.doc_idx).doc_type, "hospital_record")

    def test_b2_mine_strips_named_evidence(self):
        new, rec = M.plant(real("GB"), "B2", random.Random(0))
        r = evaluate_admission(new)
        self.assertTrue(hit_span(r, "B2", rec.doc_idx, rec.start, rec.end))
        name = new.doc(rec.doc_idx).text[rec.start:rec.end]
        from ruler_g2 import diagnoses as dxm
        self.assertTrue(any(x.state in dxm.SUSPECT_STATES and dxm.dx_match(x.norm, name)
                            for x in dxm.mentions(new) if x.doc_idx == new.hr.idx))

    def test_acceptance_ds_breakdown_keys(self):
        from g2_kit import diag
        acc = diag.classifier_acceptance([real(n) for n in fixtures.ALL])
        self.assertEqual(acc["post_with_ds_surgery_words"] + acc["post_ds_missing"] + acc["post_ds_wide_words_only"]
                         <= acc["post_admissions"], True)
        self.assertEqual(acc["post_ds_missing"], 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
