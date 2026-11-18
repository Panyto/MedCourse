"""g6_core 0.1 自检。全部用编造的病历和编造的参考集，不碰真实病历。

G45_CORE=<g45_core 0.2 解包目录> 设了，另核对 ruler_g3/、ruler_g4/、q45_kit/ 与 g45_core 0.2 逐字节相同。
"""
import ast
import hashlib
import os
import random
import re
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from q6_kit import b6, care_quota as Q, diversity as D, entities as E, kcov as K, mask as M, split as SP, tail as T  # noqa: E402

OFFICIAL = ROOT / "q6_kit" / "official" / "llmcare.py"
OFFICIAL_SHA = "7d0c10e52c123843138ac02e4515aff1538df0faa32bb81fb470595b63f5f743"
WORDS = ["发热", "咳嗽", "咳痰", "胸痛", "头晕", "乏力", "血常规", "胸部CT", "心电图", "C反应蛋白", "阿司匹林", "头孢曲松",
         "二甲双胍", "胰岛素", "胆囊切除术", "雾化吸入", "吸氧", "腹痛", "呕吐", "肝功能", "肾功能", "降钙素原", "痰培养",
         "布地奈德", "氨溴索", "心脏彩超", "冠脉造影", "氯吡格雷", "阿托伐他汀", "呋塞米"]


def rec(sex="男", age="65", dx="甲病", cc="", hpi="", hc="", di=""):
    return {"Gender": sex, "Age": age, "Primary Diagnosis": dx, "Chief Complaint": cc,
            "History of Present Illness": hpi, "Hospital Course": hc, "Discharge Instructions": di}


def _official_class():
    """从公开代码里按 AST 取出五个方法，装进一个不加载模型的壳子里（原文一个字都不改）。"""
    src = OFFICIAL.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMCARe")
    want = {"_corpus_critic", "allocate_samples", "_get_corpus_statistic", "_extract_feature", "_load_ref"}
    funcs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in want]
    assert {f.name for f in funcs} == want
    body = "\n\n".join(ast.get_source_segment(src, f) for f in funcs)
    body = "\n".join("    " + line if line.strip() else line for line in body.splitlines())
    code = "class Official:\n" + body + "\n"
    ns = {"random": random, "re": re, "Counter": Counter, "defaultdict": defaultdict, "json": __import__("json")}
    exec(compile(code, str(OFFICIAL), "exec"), ns)
    return ns["Official"]


def _fake_ref_lines(rng, icd, n):
    lines = []
    for i in range(n):
        ent = {}
        for f in Q.FIELDS:
            k = rng.randint(0, 5)
            ws = rng.sample(WORDS, k)
            ent[f] = {"symptom": ws[: k // 2], "medication": ws[k // 2:]} if ws else {}
        lines.append({"icd": icd, "sex": rng.choice(["male", "female"]), "age": str(rng.randint(1, 95)), "entity": ent})
    return lines


def _fake_drafts(rng, n, english=True):
    out = []
    for i in range(n):
        sex = rng.choice(["male", "female", None]) if english else rng.choice(["男", "女", None])
        age = rng.choice([str(rng.randint(1, 95)), None])
        pi = "; ".join(x for x in ([f"Gender: {sex}"] if sex else []) + ([f"Age: {age}"] if age else []))
        d = {"patient_info": pi, "discharge_diagnosis": "甲病"}
        for f in Q.FIELDS:
            ws = rng.sample(WORDS, rng.randint(0, 6))
            d[f] = "，".join(ws) + "。" if ws else "无特殊。"
        out.append(d)
    return out


class TestOfficialFidelity(unittest.TestCase):
    def test_sha(self):
        self.assertEqual(hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(), OFFICIAL_SHA)

    def test_allocate(self):
        O = _official_class()()
        rng = random.Random(1)
        for t in range(200):
            dist = {f"k{i}": rng.random() for i in range(rng.randint(1, 12))}
            s = sum(dist.values())
            dist = {k: v / s * rng.uniform(0.2, 3) for k, v in dist.items()}
            n = rng.randint(1, 60)
            self.assertEqual(O.allocate_samples(dist, n), Q.allocate_samples(dist, n))

    def test_load_ref_values(self):
        import json
        import tempfile
        O = _official_class()()
        O.fields = list(Q.FIELDS)
        rng = random.Random(2)
        lines = _fake_ref_lines(rng, "X01", 25) + _fake_ref_lines(rng, "X02", 7)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
            for l in lines:
                fh.write(json.dumps(l, ensure_ascii=False) + "\n")
        O._load_ref(fh.name)
        os.unlink(fh.name)
        mine = Q.load_ref(lines)
        for icd in ("X01", "X02"):
            for k in ["sex", "age"] + Q.FIELDS:
                self.assertEqual(dict(O.ref_dist[icd][k]), dict(mine[icd][k]), (icd, k))

    def test_corpus_critic_same_as_official(self):
        Off = _official_class()
        rng = random.Random(3)
        checked = 0
        for t in range(300):
            lines = _fake_ref_lines(rng, "X01", rng.randint(1, 30))
            ref = Q.load_ref(lines)
            drafts = _fake_drafts(rng, rng.randint(1, 30))
            o = Off()
            o.fields = list(Q.FIELDS)
            o.ref_dist = ref
            o.icd = "X01"
            seed = 1000 + t
            random.seed(seed)
            try:
                fb_off = o._corpus_critic([(i, d) for i, d in enumerate(drafts)])
            except IndexError:
                fb_off = "IndexError"
            try:
                fb_mine, _ = Q.corpus_critic(drafts, ref["X01"], random.Random(seed))
            except IndexError:
                fb_mine = "IndexError"
            if fb_off == "IndexError" or fb_mine == "IndexError":
                self.assertEqual(fb_off, fb_mine)
                continue
            self.assertEqual(len(fb_off), len(fb_mine))
            for a, b in zip(fb_off, fb_mine):
                self.assertEqual(a.get("patient_info"), b.get("patient_info"))
                self.assertEqual(set(a) - {"patient_info"}, set(b) - {"patient_info"})
                for f in Q.FIELDS:
                    pa = sorted((Q.parse_keywords(s)[0], tuple(sorted(Q.parse_keywords(s)[1]))) for s in a.get(f, []))
                    pb = sorted((Q.parse_keywords(s)[0], tuple(sorted(Q.parse_keywords(s)[1]))) for s in b.get(f, []))
                    self.assertEqual(pa, pb)
            checked += 1
        self.assertGreater(checked, 250)

    def test_chinese_sex(self):
        ref = Q.load_ref([{"icd": "X", "sex": "male", "age": "60", "entity": {f: {} for f in Q.FIELDS}}])
        f = Q.extract_feature(Q.to_internal(rec(sex="男", age="5")), ref["X"])
        self.assertEqual((f["sex"], f["age"]), ("male", 1))
        f = Q.extract_feature(Q.to_internal(rec(sex="女", age="8个月")), ref["X"])
        self.assertEqual((f["sex"], f["age"]), ("female", 1))   # 原代码把"8个月"读成 8 岁，照旧

    def test_inflation_official_example(self):
        """公开 example-ref.jsonl：每个诊断 1 份参考病历，出院医嘱一段 7 个药，3 份初稿时每个药"应该"出现 21 份。"""
        import json
        ex = OFFICIAL.parent / "example-ref.jsonl"
        if not ex.exists():
            self.skipTest("example-ref.jsonl 不在包里")
        ref = Q.load_ref([json.loads(l) for l in ex.read_text(encoding="utf-8").splitlines() if l.strip()])
        e = Q.expected_per_entity(ref["I63.900"], 3)["discharge_instructions"]
        self.assertEqual(set(e.values()), {21})
        e = Q.expected_per_entity(ref["I63.900"], 3, corrected=True)["discharge_instructions"]
        self.assertEqual(set(e.values()), {3})

    def test_corrected_differs_only_in_fields(self):
        rng = random.Random(6)
        lines = _fake_ref_lines(rng, "X01", 20)
        ref = Q.load_ref(lines)["X01"]
        drafts = _fake_drafts(rng, 20, english=False)
        a = Q.corpus_critic(drafts, ref, random.Random(7))[0]
        b = Q.corpus_critic(drafts, ref, random.Random(7), corrected=True)[0]
        self.assertEqual([x.get("patient_info") for x in a], [x.get("patient_info") for x in b])

    def test_deterministic(self):
        rng = random.Random(4)
        lines = _fake_ref_lines(rng, "X01", 20)
        drafts = _fake_drafts(rng, 20, english=False)
        a = Q.corpus_critic(drafts, Q.load_ref(lines)["X01"], Q.round_rng("S6", "X01", 0))[0]
        b = Q.corpus_critic(drafts, Q.load_ref(list(lines))["X01"], Q.round_rng("S6", "X01", 0))[0]
        self.assertEqual(a, b)


class TestIdeal(unittest.TestCase):
    def test_obedience(self):
        rng = random.Random(5)
        lines = _fake_ref_lines(rng, "X01", 15)
        ref = Q.load_ref(lines)["X01"]
        drafts = [(f"S{i:02d}", rec(sex=rng.choice("男女"), age=str(rng.randint(1, 90)),
                                    cc="咳嗽3天。", hpi="患者发热。", hc="予吸氧。", di="出院。")) for i in range(30)]
        out, plans = Q.run_ideal(drafts, ref, "FQOK", "X01", rounds=1)
        for (sid, after), p in zip(out, plans[0]):
            for f in Q.FIELDS:
                txt = after[Q.CARE7_OF[f]]
                for w in p[f]["add"]:
                    self.assertIn(w, txt)
        # 第二轮的计划基于第一轮的结果
        out2, plans2 = Q.run_ideal(drafts, ref, "FQOK", "X01", rounds=2)
        self.assertEqual(len(plans2), 2)

    def test_single_reference_clones(self):
        """参考集只有 1 个人：每个实体"应该"出现在每一份里，性别、年龄档全部改成那个人的。"""
        ent = {f: {} for f in Q.FIELDS}
        ent["history_of_present_illness"] = {"symptom": ["甲症", "乙症", "丙症"]}
        ent["hospital_course"] = {"medication": ["丁药", "戊药"]}
        ref = Q.load_ref([{"icd": "T", "sex": "female", "age": "37", "entity": ent}])["T"]
        exp = Q.expected_per_entity(ref, 10)
        # 原代码的算式：期望份数 = n × S × p = 10 × 3 × 1 = 30（比 10 份还多）；修正版 = n × p = 10
        self.assertTrue(all(v == 30 for v in exp["history_of_present_illness"].values()))
        expc = Q.expected_per_entity(ref, 10, corrected=True)
        self.assertTrue(all(v == 10 for v in expc["history_of_present_illness"].values()))
        drafts = [(f"D{i}", rec(sex="男", age="60", hpi="患者不适。", hc="予治疗。")) for i in range(10)]
        out, plans = Q.run_ideal(drafts, ref, "T6", "T", rounds=2)
        self.assertTrue(all(r["Gender"] == "女" and 35 <= int(r["Age"]) <= 39 for _, r in out))
        n_hpi = sum(len(p["history_of_present_illness"]["add"]) for pl in plans for p in pl)
        self.assertGreater(n_hpi, 10)


class TestEntities(unittest.TestCase):
    def test_basic(self):
        ms = E.mentions("患者3天前受凉后出现发热，无胸痛，行腹腔镜胆囊切除术，予阿司匹林肠溶片100mg口服，查胸部CT示斑片影。")
        got = {(m["cat"], m["surface"], m["neg"]) for m in ms}
        self.assertIn(("SYM", "发热", 0), got)
        self.assertIn(("SYM", "胸痛", 1), got)
        self.assertIn(("SURG", "腹腔镜胆囊切除术", 0), got)
        self.assertIn(("MED", "阿司匹林肠溶片", 0), got)
        self.assertIn(("AUX", "胸部CT", 0), got)
        med = [m for m in ms if m["cat"] == "MED"][0]
        self.assertTrue(med["key"].startswith("drug:"))

    def test_noise(self):
        surfs = {m["surface"] for m in E.mentions("建议患者接受手术，复查胸片，甲县人民医院彩超，每次2片。")}
        self.assertNotIn("建议患者接受手术", surfs)
        self.assertNotIn("胸片", {s for s in surfs if s.endswith("片") and s != "胸片"})
        self.assertFalse(any("医院" in s or "县" in s for s in surfs))

    def test_ref_entity_affirmed_only(self):
        r = rec(hpi="患者发热，无头痛、头晕。", di="出院带药：二甲双胍0.5g tid。")
        ent = E.ref_entity(r)
        self.assertEqual(ent["history_of_present_illness"], {"SYM": ["发热"]})
        self.assertIn("MED", ent["discharge_instructions"])


class TestKcovDiversity(unittest.TestCase):
    def test_clone_full(self):
        refs = [rec(hpi="患者发热、咳嗽，查血常规示白细胞升高。", hc="予头孢曲松抗感染。"),
                rec(hpi="患者胸痛。", hc="予阿司匹林。")]
        vocab = K.ref_vocab(refs)
        line = "、".join(w for _, w in vocab)
        clone = [rec(hpi=line)] * 3
        self.assertEqual(K.kcov(clone, vocab)["all"], 1.0)
        self.assertEqual(K.kcov([rec(hpi="无特殊。")], vocab)["all"], 0.0)
        self.assertEqual(K.kcov_per_record(clone, vocab)["all"], 1.0)
        half = [rec(hpi=line), rec(hpi="无特殊。")]
        self.assertAlmostEqual(K.kcov_per_record(half, vocab)["all"], 0.5)
        sets = {f"c{i}": E.key_set(r) for i, r in enumerate(clone)}
        self.assertEqual(D.within_code_jaccard(sets, {"X": list(sets)}), 1.0)

    def test_self_bleu_vendi(self):
        t = D.record_text(rec(hpi="患者发热咳嗽三天，予头孢曲松抗感染治疗后好转。"))
        self.assertGreater(D.self_bleu([("a", t), ("b", t), ("c", t)]), 0.99)
        vn, v = D.vendi([t, t, t])
        self.assertAlmostEqual(v, 1.0, places=6)
        u = D.record_text(rec(hpi="右膝关节疼痛半年，行关节镜检查。"))
        self.assertGreater(D.vendi([t, u])[1], 1.2)

    def test_truncate_pair(self):
        a, b = D.truncate_pair(rec(hpi="短句。"), rec(hpi="[地址]一段很长很长的现病史文字。"))
        self.assertEqual(len(a["History of Present Illness"]), len(b["History of Present Illness"]))
        self.assertNotIn("[", b["History of Present Illness"])

    def test_profile(self):
        p = D.profile([5, 5, 5, 68])
        self.assertEqual((p["distinct"], p["mode"]), (2, 5))
        self.assertAlmostEqual(p["modal_share"], 0.75)
        self.assertEqual(D.max_temp(rec(hpi="最高体温38.5℃，后39.2℃。")), 39.2)


class TestTail(unittest.TestCase):
    def test_unique_combo(self):
        all_sets = [("A", {"a", "b", "c", "d"}), ("B", {"a", "b", "x"}), ("C", {"a", "b", "c", "y"})]
        idx = T.build_index(all_sets)
        self.assertEqual(T.unique_combo({"a", "b", "c", "d", "z"}, [0], all_sets, idx)["hit"], 1)
        self.assertEqual(T.unique_combo({"a", "b", "c"}, [0], all_sets, idx)["hit"], 0)   # C 也有 a b c
        self.assertAlmostEqual(T.copy_share({"a", "b"}, [all_sets[0][1]]), 0.5)

    def test_atypical(self):
        code_df = Counter({"x": 10, "r": 1})
        all_df = Counter({"x": 50, "r": 4, "o": 1})
        a, o = T.atypical_items({"x", "r", "o"}, code_df, all_df, n_code_other=29, self_in_ref=False)
        self.assertEqual((a, o), (["o", "r"], []))   # 合成病历："o"别的病人写过、本病种没人写过，算不典型
        a, o = T.atypical_items({"x", "r", "o"}, code_df, all_df, n_code_other=29, self_in_ref=True)
        self.assertEqual((a, o), (["r"], ["o"]))


class TestB6(unittest.TestCase):
    def test_inserted_conflicts(self):
        before = rec(sex="女", age="6", hpi="患儿发热2天。", hc="予补液治疗。")
        after = rec(sex="女", age="6", hpi="患儿发热2天，前列腺彩超未见异常。", hc="予补液治疗，左氧氟沙星0.2g静滴。")
        ms = b6.conflicts(after, b6.label_mentions(before, after))
        ins = {m["surface"]: m["rules"] for m in ms if m["inserted"]}
        self.assertIn("SEX-ENT", ins.get("前列腺彩超", []))
        self.assertIn("DRUG", ins.get("左氧氟沙星", []))
        s = b6.summarize(ms)
        self.assertEqual(s["inserted"]["conflict"], 2)
        self.assertEqual(s["native"]["conflict"], 0)


class TestMaskSplit(unittest.TestCase):
    def test_mask(self):
        r = rec(dx="某罕见病", hpi="某罕见病患者，予奇怪药甲片治疗，发热。")
        prot = M.protected({"奇怪药甲片": "MED", "发热": "SYM"}, {"发热": 900, "奇怪药甲片": 1})
        out = M.mask_record(r, prot, ["某罕见病"])
        self.assertEqual(M.leak_count(out["History of Present Illness"], prot, ["某罕见病"]), 0)
        self.assertIn("发热", out["History of Present Illness"])
        self.assertIn("【词:药物:5字】", out["History of Present Illness"])

    def test_split(self):
        ids = {"A01": [f"a{i:03d}" for i in range(40)], "B02": [f"b{i}" for i in range(12)]}
        s1, s2 = SP.split_hold_ref(ids), SP.split_hold_ref({k: list(reversed(v)) for k, v in ids.items()})
        self.assertEqual(s1, s2)
        h, r = s1["A01"]
        self.assertEqual((len(h), len(r), len(set(h) & set(r))), (15, 25, 0))
        self.assertEqual(s1["B02"][0], [])
        a, b = SP.halves([f"s{i}" for i in range(30)], "A01")
        self.assertEqual((len(a), len(b), len(set(a) & set(b))), (15, 15, 0))
        t = SP.pick_tail({"Z1": 3, "Z2": 11, "Z3": 1, "Z4": 2}, {"Z1": 1, "Z2": 2, "Z3": 0, "Z4": 1}, exclude=("Z4",))
        self.assertEqual(t, ["Z1"])
        self.assertEqual(SP.tail_alias(["Q9", "A1"]), {"A1": "TAIL01", "Q9": "TAIL02"})


@unittest.skipUnless(os.environ.get("G45_CORE"), "G45_CORE 没设：跳过与 g45_core 0.2 的逐字节核对")
class TestVerbatim(unittest.TestCase):
    def test_copies(self):
        src = Path(os.environ["G45_CORE"])
        n = 0
        for sub in ("ruler_g3", "ruler_g4", "q45_kit"):
            for p in sorted((ROOT / sub).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    q = src / p.relative_to(ROOT)
                    self.assertTrue(q.exists(), str(q))
                    self.assertEqual(p.read_bytes(), q.read_bytes(), str(p))
                    n += 1
        self.assertGreater(n, 20)


if __name__ == "__main__":
    unittest.main()
