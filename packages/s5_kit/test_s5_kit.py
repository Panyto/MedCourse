# -*- coding: utf-8 -*-
"""Offline unit tests for s5_kit (does not modify frozen medcourse_core)."""
import json
import os
import pathlib
import random
import re
import tempfile
import unittest

import s5_common
from medcourse import kb, pipeline, planner, writer
import audit_s5_2_17d
import audit_s5_intrinsic
import build_s6_blind_pack
import gen_direct_32b
import gen_s5_shard
import plan_s5_corpora
import s6_mia_audit
import t123_bench
import t12_neural_bench


class TestS5Kit(unittest.TestCase):
    def test_no_middle_dot_in_s5_kit(self):
        kit_dir = pathlib.Path(__file__).resolve().parent
        banned_unblind = "脱" + "盲"
        banned_epist = "认" + "识论"
        for p in kit_dir.glob("*"):
            if p.suffix in (".py", ".sh", ".md"):
                txt = p.read_text(encoding="utf-8")
                self.assertNotIn("\u00b7", txt, f"U+00B7 found in {p.name}")
                self.assertNotIn(banned_unblind, txt, f"banned term found in {p.name}")
                if p.suffix != ".py" or p.name != "test_s5_kit.py":
                    self.assertNotIn(banned_epist, txt, f"epist term found in {p.name}")

    def test_all_17_route_b_pathways_plan_cleanly(self):
        kb_data = kb.load_default_kb()
        self.assertEqual(len(s5_common.PATHWAY_SPECS), 17)
        for row in s5_common.PATHWAY_SPECS:
            kb_key = row[0]
            st = s5_common.build_pathway_stats("route_b", kb_key, kb_data=kb_data)
            planner.require_stats(st)
            jid = f"UT-{kb_key}"
            rng = random.Random(planner.seed_of(20261220, jid))
            j = planner.plan_journey(st, rng, jid, None)
            self.assertIsNotNone(j, kb_key)
            specs = planner.schedule(j, st, rng)
            self.assertEqual(planner.check_plan(j, specs), [], kb_key)

    def test_template_full_vis_oracle_parity(self):
        kb_data = kb.load_default_kb()
        st = s5_common.build_pathway_stats("route_b", "I20.0", kb_data=kb_data)
        jid = "UT-PARITY-01"
        rng = random.Random(planner.seed_of(20261210, jid))
        j = planner.plan_journey(st, rng, jid, None)
        specs = planner.schedule(j, st, rng)

        tw = writer.TemplateWriter()
        tw.splice_copy = False
        stats_map = {"_default": st}
        tally_f, tally_v, tally_o = {}, {}, {}

        tw.ablate_epistemic = False
        docs_f, fails_f, fdocs_f = gen_s5_shard.generate_from_plan(
            j, specs, stats_map, "full", tw, 20261210, tally_f, save_first_try=True
        )
        tw.ablate_epistemic = True
        docs_v, fails_v, fdocs_v = gen_s5_shard.generate_from_plan(
            j, specs, stats_map, "vis_only", tw, 20261210, tally_v, save_first_try=True
        )
        tw.ablate_epistemic = False
        docs_o, fails_o, _ = gen_s5_shard.generate_from_plan(
            j, specs, stats_map, "oracle_leak", tw, 20261210, tally_o, save_first_try=False
        )

        self.assertEqual(len(docs_f), len(specs))
        self.assertEqual(len(docs_v), len(specs))
        self.assertEqual(len(docs_o), len(specs))
        self.assertEqual(len(fdocs_f), len(specs))
        prompt = s5_common.build_oracle_prompt(j, specs[0], "现病史", 120)
        self.assertIn("全周期所有检查及已出结果", prompt)

    def test_t123_bench_filters_and_masking(self):
        with tempfile.TemporaryDirectory() as tmp:
            fold_p = os.path.join(tmp, "f.jsonl")
            adms = [
                # Case 1: None day_offset progress_note with LATE_OP_RE must NOT enter d01 or count;
                # only DS mentions surgery -> excluded under non-DS rule.
                {
                    "admission_id": "R1",
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": "入院诊断：\n1.急性胰腺炎\n2.急性胆囊炎？"},
                        {"doc_type": "progress_note", "day_offset": None, "text": "在全身麻醉下行腹腔镜胆囊切除术"},
                        {"doc_type": "progress_note", "day_offset": 0, "text": "首次病程：暂与急性胃炎鉴别。"},
                        {"doc_type": "progress_note", "day_offset": 1, "text": "第1天复查淀粉酶。"},
                        {"doc_type": "progress_note", "day_offset": 3, "text": "第3天保守好转。"},
                        {"doc_type": "discharge_summary", "day_offset": 4, "text": "出院记录：在全身麻醉下行内镜逆行胰胆管造影术。"},
                    ],
                },
                # Case 2: Day 2 non-DS surgery -> positive at ge2, excluded/non-pos under gt2_only=True.
                {
                    "admission_id": "R2",
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": "入院诊断：\n1.急性阑尾炎？"},
                        {"doc_type": "progress_note", "day_offset": 0, "text": "第0天拟保守观察。"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": "术后首次病程记录：今日在全麻下行腹腔镜阑尾切除术。急性阑尾炎诊断明确。"},
                        {"doc_type": "progress_note", "day_offset": 3, "text": "术后第1天恢复可。"},
                        {"doc_type": "discharge_summary", "day_offset": 4, "text": "出院诊断：急性阑尾炎"},
                    ],
                },
                # Case 3: Day 3 non-DS surgery -> positive under BOTH gt2_only=False and gt2_only=True.
                {
                    "admission_id": "R3",
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": "入院诊断：\n1.急性胆囊炎？"},
                        {"doc_type": "progress_note", "day_offset": 1, "text": "第1天继续抗感染，排除心肌梗死可能。"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": "第2天复查超声。"},
                        {"doc_type": "progress_note", "day_offset": 3, "text": "术后首次病程记录：今日在全麻下行胆囊切除术。"},
                        {"doc_type": "discharge_summary", "day_offset": 5, "text": "出院诊断：急性胆囊炎"},
                    ],
                },
            ]
            with open(fold_p, "w", encoding="utf-8") as f:
                for row in adms:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            audit = t123_bench.audit_fold_docs(fold_p)
            self.assertEqual(audit["missing_day_offset_docs"], 1)
            self.assertEqual(audit["t1_pos_non_ds_d2p"], 2)
            self.assertEqual(audit["t1_pos_only_in_ds"], 1)
            self.assertEqual(audit["t1_pos_gt2_non_ds"], 1)

            rows_ge2 = t123_bench.build_t1_dataset([fold_p], gt2_only=False)
            ymap_ge2 = {r[3]: r[1] for r in rows_ge2}
            self.assertEqual(ymap_ge2, {"R1": 0, "R2": 1, "R3": 1})
            self.assertNotIn("腹腔镜胆囊切除术", rows_ge2[0][0])

            rows_gt2 = t123_bench.build_t1_dataset([fold_p], gt2_only=True)
            ymap_gt2 = {r[3]: r[1] for r in rows_gt2}
            self.assertEqual(ymap_gt2, {"R1": 0, "R3": 1})

            masked = t123_bench._mask_dx_clauses("高血压诊断明确。第2天复查确认急性阑尾炎诊断明确。", "急性阑尾炎", window=12)
            self.assertIn("[TARGET_DX]", masked)
            self.assertNotIn("急性阑尾炎", masked)
            self.assertNotIn("高血压诊断明确", masked)
            self.assertNotIn("[SILENT_NO_MENTION]", masked)
            self.assertNotIn(" || ", masked)
            silent_clean = t123_bench._mask_dx_clauses("今日生命体征平稳。", "急性阑尾炎", replace_token="该病")
            self.assertNotIn("[SILENT_NO_MENTION]", silent_clean)
            self.assertNotIn("[TARGET_DX]", silent_clean)

            t2a_rows = t123_bench.build_t2a_clause_dataset([fold_p])
            self.assertTrue(any(r[1] == 2 and r[3] == "急性阑尾炎" for r in t2a_rows))
            t2b_rows = t123_bench.build_t2b_update_dataset([fold_p])
            # R1 day 1/3 should preserve silent rows with y=0; R2 day 2 hits confirmed -> y=1.
            self.assertTrue(any(r[2] is True and r[1] == 0 for r in t2b_rows))
            self.assertTrue(any(r[2] is False and r[1] == 1 and r[3] == "R2" for r in t2b_rows))
            self.assertTrue(all("[SILENT_NO_MENTION]" not in r[0] and "[TARGET_DX]" not in r[0] for r in t2b_rows))

            pool_p = os.path.join(tmp, "pool.jsonl")
            pool_res = t123_bench.export_t2_blind_pool(fold_p, pool_p)
            self.assertGreaterEqual(pool_res["total_exported"], 2)
            self.assertTrue(os.path.exists(pool_p))
            sheet_txt = pathlib.Path(pool_p).read_text(encoding="utf-8")
            self.assertNotIn("auto_stratum", sheet_txt)
            self.assertTrue(os.path.exists(pool_res["key_path"]))

            t1_dir = os.path.join(tmp, "t1_audit")
            t1_pool = t123_bench.export_t1_audit_pool(fold_p, t1_dir)
            self.assertGreaterEqual(t1_pool["total_exported"], 2)
            self.assertNotIn("auto_stratum", pathlib.Path(t1_dir, "t1_clinician_audit_sheet.jsonl").read_text(encoding="utf-8"))

            care_p = os.path.join(tmp, "care.jsonl")
            with open(care_p, "w", encoding="utf-8") as cf:
                cf.write(json.dumps({
                    "slot_id": "C1",
                    "record": {"Discharge Diagnosis": "肺炎", "Hospital Course": "肺炎诊断明确。给予静脉滴注头孢曲松2g抗感染。"},
                }, ensure_ascii=False) + "\n")
            care_rows = t123_bench.build_t3_llmcare_notes(care_p)
            self.assertEqual(len(care_rows), 1)
            self.assertEqual(care_rows[0][1][0], 1)
            self.assertEqual(care_rows[0][2], 1)
            care_t2a = t123_bench.build_t2a_llmcare_clauses(care_p)
            self.assertEqual(len(care_t2a), 1)
            self.assertEqual(care_t2a[0][1], 2)

            if t123_bench.HAS_SKLEARN:
                train_t2b = [
                    ("今日一般情况平稳，饮食睡眠可。", 0, True, "S1", "dx", 1, 1),
                    ("今日复查考虑该病待排。", 0, False, "S2", "dx", 1, 1),
                    ("结合检查结果，该病诊断明确。", 1, False, "S3", "dx", 2, 2),
                ] * 6
                res_t2b = t123_bench.run_t2b_eval(train_t2b, train_t2b, n_boot=20)
                self.assertIn("mentioned_only_auroc", res_t2b)
                self.assertIn("trivial_mention_only_full_auroc", res_t2b)
                self.assertIn("calibrated_threshold", res_t2b)
                self.assertIn("f1_cal", res_t2b)

                train_t3 = [
                    ("静脉滴注头孢曲松2g抗感染。", [1] + [0] * (len(t123_bench.DRUG_GROUP_NAMES) - 1), 1, f"T{i}")
                    for i in range(6)
                ] + [
                    ("今日复查血常规，饮食可。", [0] * len(t123_bench.DRUG_GROUP_NAMES), 0, f"N{i}")
                    for i in range(6)
                ]
                res_t3 = t123_bench.run_t3_eval(train_t3, train_t3[:4])
                self.assertIn("drug_micro_f1", res_t3)
                self.assertIn("has_dose_acc", res_t3)
                self.assertIn("gold_dose_positives", res_t3)
                self.assertIn("gold_dose_majority_baseline_acc", res_t3)

    def test_run_t1_augment_eval_resists_dilution(self):
        if not t123_bench.HAS_SKLEARN:
            self.skipTest("sklearn missing")
        real_tr = [
            ("入院急性腹痛发热白细胞升高需外科会诊", 1, 1, f"R{i}") for i in range(25)
        ] + [
            ("慢性糖尿病复查血糖平稳口服药维持", 0, 0, f"RN{i}") for i in range(25)
        ]
        # 5x more synthetic rows with adversarial reversed noise
        noisy_synth = [
            ("慢性糖尿病复查血糖平稳口服药维持", 1, 1, f"S{i}") for i in range(150)
        ] + [
            ("入院急性腹痛发热白细胞升高需外科会诊", 0, 0, f"SN{i}") for i in range(150)
        ]
        tune = real_tr[:15] + real_tr[25:40]
        ev = real_tr[15:25] + real_tr[40:50]
        base = t123_bench.run_t1_eval(real_tr, tune, ev, n_boot=20)
        aug = t123_bench.run_t1_augment_eval(real_tr, noisy_synth, tune, ev, n_boot=20)
        self.assertGreaterEqual(aug["T1a"]["auroc"], base["T1a"]["auroc"] - 0.02)
        self.assertEqual(aug["T1a"]["selected_weight"], 0.0)

    def test_gen_direct_32b_mock_and_intrinsic_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = gen_direct_32b.sample_direct_cases(10, 20261230, shard_idx=0)
            self.assertEqual(len(cases), 10)
            txt = gen_direct_32b._mock_generate_text(cases[0])
            docs, err = gen_direct_32b.parse_direct_stay(txt, cases[0]["los"])
            self.assertIsNone(err)
            self.assertGreaterEqual(len(docs), 3)
            self.assertEqual(docs[0]["doc_type"], "hospital_record")

            adm_p = os.path.join(tmp, "admissions.jsonl")
            adms = []
            for c in cases:
                d_list, _ = gen_direct_32b.parse_direct_stay(gen_direct_32b._mock_generate_text(c), c["los"])
                rec = gen_direct_32b.make_admission_record(c, d_list)
                adms.append(rec)
            with open(adm_p, "w", encoding="utf-8") as f:
                for r in adms:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            t1_rows = t123_bench.build_t1_dataset([adm_p])
            self.assertGreaterEqual(len(t1_rows), 5)
            rep = audit_s5_intrinsic.audit_corpus("DIRECT-MOCK", [adm_p], max_stays=10)
            self.assertEqual(rep["n_stays"], 10)
            self.assertIsNone(rep["oracle_any_rate"])
            self.assertIn("g3_hard_stay_rate", rep)

    def test_s6_mia_fold9_guard_and_canary(self):
        with self.assertRaises(ValueError):
            s6_mia_audit.assert_no_fold9(["/home/weify/pyt/data/deid_run9_fold9.jsonl"], "stage1_cal")
        s6_mia_audit.assert_no_fold9(["/home/weify/pyt/data/deid_run9_fold9.jsonl"], "stage2_lock")
        if not s6_mia_audit.HAS_SKLEARN:
            self.skipTest("sklearn missing")

        members = [
            {
                "admission_id": f"M{i}",
                "docs": [
                    {"doc_type": "hospital_record", "day_offset": 0, "text": f"科室：心内科一区 性别：男 年龄:{50+i}岁\n入院诊断：独特稀有病例标签甲乙丙{i}号"},
                    {"doc_type": "progress_note", "day_offset": 2, "text": f"第2天查房：复查独特稀有短语信号{i}号连续三十二个汉字记录明确好转出院评估。"},
                ],
            }
            for i in range(12)
        ]
        non_members = [
            {
                "admission_id": f"NM{i}",
                "docs": [
                    {"doc_type": "hospital_record", "day_offset": 0, "text": f"科室：骨科一区 性别：女 年龄:{30+i}岁\n入院诊断：完全无关对照胫骨骨折{i}号"},
                    {"doc_type": "progress_note", "day_offset": 2, "text": f"第2天查房：摄片复查骨折对位对线可，继续消肿止痛对症处理观察随访{i}。"},
                ],
            }
            for i in range(12)
        ]
        synth_canary = members[:10]
        rep = s6_mia_audit.evaluate_mia_arm("Canary", synth_canary, members, non_members, n_boot=30)
        self.assertGreaterEqual(rep["nn_mia"]["combined"]["roc_auc"], 0.90)

    def test_build_s6_blind_pack_and_score(self):
        with self.assertRaises(ValueError):
            build_s6_blind_pack.assert_blind_fold_allowed("/tmp/deid_run9_fold9.jsonl", allow_stage2_fold9=False)
        with tempfile.TemporaryDirectory() as tmp:
            r, f_adms, v_adms, d_adms = build_s6_blind_pack._make_dry_run_fixture()
            key_obj = build_s6_blind_pack.build_blind_package(r, f_adms, v_adms, d_adms, tmp, seed=20261240)
            self.assertEqual(key_obj["total_cases"], 60)
            packets_txt = pathlib.Path(tmp, "blind_packets.md").read_text(encoding="utf-8")
            self.assertNotIn("[患者]", packets_txt)
            self.assertNotIn("[本院]", packets_txt)
            self.assertIn("【盲评病历编号：B01】", packets_txt)

            # Fill synthetic rater sheets and verify scoring pipeline
            import csv
            with open(os.path.join(tmp, "SEALED_KEY_DO_NOT_OPEN.json"), encoding="utf-8") as kf:
                arm_of = {m["blind_id"]: m["arm"] for m in json.load(kf)["mapping"]}
            for rater_fn, delta in (("rater_A_sheet.csv", 0), ("rater_B_sheet.csv", 0)):
                p = os.path.join(tmp, rater_fn)
                with open(p, encoding="utf-8") as rf:
                    rows = list(csv.DictReader(rf))
                for row in rows:
                    arm = arm_of[row["blind_id"]]
                    base_score = 5 if arm == "MedCourse-Full" else (4 if arm == "Real" else 2)
                    for q in build_s6_blind_pack.Q_COLS:
                        row[q] = str(base_score + delta)
                    row["hard_defect_0or1"] = "0" if arm in ("MedCourse-Full", "Real") else "1"
                with open(p, "w", encoding="utf-8", newline="") as wf:
                    w = csv.DictWriter(wf, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            scored = build_s6_blind_pack.score_blind_sheets(
                os.path.join(tmp, "rater_A_sheet.csv"),
                os.path.join(tmp, "rater_B_sheet.csv"),
                os.path.join(tmp, "SEALED_KEY_DO_NOT_OPEN.json"),
                out_json=os.path.join(tmp, "scored.json"),
            )
            self.assertEqual(scored["agreement"]["cohens_kappa_hard_defect"], 1.0)
            comps = scored["comparisons_holm"][build_s6_blind_pack.Q_COLS[0]]
            self.assertEqual(comps["MedCourse-VisOnly"]["test"], "paired_wilcoxon")
            self.assertEqual(comps["Real"]["test"], "stratified_van_elteren")
            self.assertEqual(comps["DIRECT-32B"]["test"], "stratified_van_elteren")
            self.assertIsNotNone(scored["arms"]["MedCourse-Full"]["hard_defect_any_rater_ci95"])

    def test_v15_t1_counterexamples_and_freeze_roundtrip(self):
        # Reviewer B counterexamples
        self.assertTrue(t123_bench.has_early_op_plan("患者症状较重，拟行腹腔镜胆囊切除术。"))
        self.assertTrue(t123_bench.has_early_op_plan("已向家属交代病情，已签署手术知情同意书。"))
        self.assertTrue(t123_bench.has_early_op_plan("术前检查大致正常，明日行胆囊切除术。"))
        self.assertFalse(t123_bench.has_early_op_plan("既往1年前于外院拟行胆囊切除术未果。"))
        self.assertFalse(t123_bench.has_late_op_event("患者既往在全身麻醉下行胆囊切除术，恢复可。"))
        self.assertTrue(t123_bench.has_late_op_event("今日行经皮冠状动脉介入治疗，过程顺利。"))
        self.assertFalse(t123_bench.has_late_op_event("今日在局麻下行胸腔穿刺置管术。", major_only=True))

        # Relative date scrubbing and None day_offset preservation in S6-2 blind pack
        adm_rel = {
            "admission_id": "REL-1",
            "docs": [
                {"doc_type": "hospital_record", "role": "HR", "day_offset": 0, "k": 0, "text": "2024-05-10入院，主诉腹痛。"},
                {"doc_type": "progress_note", "role": "DAILY", "day_offset": None, "k": 1, "text": "2024-05-13复查B超。"},
            ],
        }
        blind_txt = build_s6_blind_pack.normalize_for_blind(adm_rel, "B99")
        self.assertIn("日期未记载", blind_txt)
        self.assertNotIn("住院第 0 天 | 日常病程", blind_txt)
        self.assertIn("[入院第+0天]", blind_txt)
        self.assertIn("[入院第+3天]", blind_txt)

        if t123_bench.HAS_SKLEARN:
            tr = [("急性右上腹痛发热白细胞升高需手术评估", 1, 1, f"A{i}") for i in range(15)] + [
                ("慢性血糖控制可口服二甲双胍维持", 0, 0, f"B{i}") for i in range(15)
            ]
            art = t123_bench.fit_t1_bundle(tr, tr[:10])
            preds1 = t123_bench.predict_t1_bundle(art, tr[10:])
            res_direct, preds2 = t123_bench.run_t1_eval(tr, tr[:10], tr[10:], n_boot=20, return_preds=True)
            self.assertAlmostEqual(float(preds1["T1a"][0][0]), float(preds2["T1a"][0][0]), places=10)
            self.assertIn("T1a", res_direct)

    def test_audit_s5_2_17d_unified_text_probe(self):
        adms = []
        jmap = {}
        for i in range(12):
            aid = f"PW-{i}"
            is_pos = i % 2 == 0
            ex_txt = "第0天拟行血常规、腹部CT检查。" if is_pos else "第0天查血常规。"
            op_txt = "术后首次病程记录：今日在全身麻醉下行腹腔镜胆囊切除术，术程顺利。" if is_pos else "第3天继续保守抗感染治疗，症状好转。"
            adms.append({
                "admission_id": aid,
                "docs": [
                    {"doc_type": "hospital_record", "role": "HR", "day_offset": 0, "text": "年龄:45岁 性别：男\n入院诊断：急性胆囊炎"},
                    {"doc_type": "progress_note", "role": "FP", "day_offset": 0, "text": ex_txt},
                    {"doc_type": "progress_note", "role": "DAILY", "day_offset": 3, "text": op_txt},
                    {"doc_type": "discharge_summary", "role": "DS", "day_offset": 5, "text": "出院诊断：急性胆囊炎"},
                ],
            })
            jmap[aid] = {"journey_id": aid, "kb_pathway_key": "K80.0_acute"}
        rows = audit_s5_2_17d.extract_unified_cohort(adms, jmap=jmap, is_real=False, gt2_only=False)
        self.assertEqual(len(rows), 12)
        rep = audit_s5_2_17d.evaluate_unified_probe("TEST-17D", rows)
        if audit_s5_2_17d.HAS_SKLEARN:
            self.assertEqual(rep["n_estimable_pathways"], 1)
            self.assertIsNotNone(rep["within_pathway_weighted_exams_auc"])

    def test_t2a_inhomogeneous_classes_and_llmcare_primary_dx(self):
        with tempfile.TemporaryDirectory() as tmp:
            care_p = os.path.join(tmp, "care.jsonl")
            rows = [
                {
                    "slot_id": "CARE-0",
                    "record": {
                        "Primary Diagnosis": "急性阑尾炎",
                        "Hospital Course": "入院后完善检查，明确诊断为急性阑尾炎，予手术治疗。",
                    },
                },
                {
                    "slot_id": "CARE-1",
                    "record": {
                        "Discharge Diagnosis": "胆囊结石",
                        "Hospital Course": "初步考虑胆囊结石可能，待进一步复查。",
                    },
                },
            ]
            with open(care_p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            care_cls = t123_bench.build_t2a_llmcare_clauses(care_p)
            self.assertGreaterEqual(len(care_cls), 2)
            for txt, lbl, _, dx in care_cls:
                self.assertIn("[TARGET_DX]", txt)
                self.assertNotIn("Primary Diagnosis", txt)
                self.assertNotIn("Discharge Diagnosis", txt)

        if not s6_mia_audit.HAS_SKLEARN:
            self.skipTest("sklearn missing")

        # 1) Rare class 0 (only 1 sample) across 3 classes {0, 1, 2} must not fail predict_t2a_bundle
        tr_rare = [("排除阑尾炎，考虑肠炎。", 0)]
        for i in range(15):
            tr_rare.append((f"考虑待查病变{i}，待排。", 1))
            tr_rare.append((f"明确诊断病变{i}，确诊。", 2))
        art_rare = t123_bench.fit_t2a_bundle(tr_rare, seeds=(1, 2, 3))
        pr_rare = t123_bench.predict_t2a_bundle(art_rare, tr_rare[:6])
        self.assertIsNotNone(pr_rare)
        self.assertEqual(pr_rare[1].shape, (6, 3))
        sc_rare = t123_bench.score_t2a_bundle(art_rare, pr_rare, tr_rare[:6])
        self.assertIsNotNone(sc_rare["macro_f1_3class"])

        # 2) Only 2 classes {1, 2} in train must also return canonical (n, 3) probs cleanly
        tr_two = [(f"待查病例{i}", 1) for i in range(10)] + [(f"确诊病例{i}", 2) for i in range(10)]
        art_two = t123_bench.fit_t2a_bundle(tr_two, seeds=(4, 5))
        pr_two = t123_bench.predict_t2a_bundle(art_two, tr_rare[:4])
        self.assertEqual(pr_two[1].shape, (4, 3))
        sc_two = t123_bench.score_t2a_bundle(art_two, pr_two, tr_rare[:4])
        self.assertIsNotNone(sc_two["macro_f1_3class"])

    def test_t12_neural_bench_metrics_guard_and_roundtrip(self):
        import numpy as np

        with self.assertRaises(ValueError):
            t12_neural_bench.assert_stage1_fit_allowed("/home/weify/pyt/data/deid_run9_fold9.jsonl")
        with self.assertRaises(ValueError):
            t12_neural_bench.assert_stage1_fit_allowed("/tmp/f1.jsonl", "/tmp/deid_run9_fold9.jsonl")

        y = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=int)
        p = np.array([0.1, 0.9, 0.4, 0.4, 0.8, 0.3, 0.7, 0.4], dtype=float)
        w = np.array([1.0, 2.5, 0.5, 1.2, 3.0, 1.0, 2.0, 0.8], dtype=float)
        gids = ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"]

        auc_np = t12_neural_bench.np_roc_auc(y, p)
        ap_np = t12_neural_bench.np_average_precision(y, p)
        auc_w_np = t12_neural_bench.np_roc_auc(y, p, sample_weight=w)
        ap_w_np = t12_neural_bench.np_average_precision(y, p, sample_weight=w)
        prec_np, rec_np, f1_np = t12_neural_bench.np_prf1(y, (p >= 0.4).astype(int))
        f1_mc_np = t12_neural_bench.np_f1_per_class(np.array([0, 1, 2, 1, 2, 0]), np.array([0, 2, 2, 1, 2, 1]))

        if t123_bench.HAS_SKLEARN:
            from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

            self.assertLess(abs(auc_np - float(roc_auc_score(y, p))), 1e-12)
            self.assertLess(abs(ap_np - float(average_precision_score(y, p))), 1e-12)
            self.assertLess(abs(auc_w_np - float(roc_auc_score(y, p, sample_weight=w))), 1e-12)
            self.assertLess(abs(ap_w_np - float(average_precision_score(y, p, sample_weight=w))), 1e-12)
            self.assertLess(abs(prec_np - float(precision_score(y, (p >= 0.4).astype(int)))), 1e-12)
            self.assertLess(abs(rec_np - float(recall_score(y, (p >= 0.4).astype(int)))), 1e-12)
            self.assertLess(abs(f1_np - float(f1_score(y, (p >= 0.4).astype(int)))), 1e-12)
            sk_mc = f1_score(np.array([0, 1, 2, 1, 2, 0]), np.array([0, 2, 2, 1, 2, 1]), labels=[0, 1, 2], average=None)
            self.assertLess(float(np.max(np.abs(f1_mc_np - sk_mc))), 1e-12)

        cb = t12_neural_bench.cluster_bootstrap_binary(y, p, gids, n_boot=20)
        self.assertGreater(cb["auroc"], 0.5)
        pb = t12_neural_bench.paired_bootstrap_diff(y, p, np.flip(p), gids, n_boot=20)
        self.assertIsNotNone(pb["delta_auroc"])

        if not t12_neural_bench.HAS_TORCH_HF:
            self.skipTest("torch/transformers missing locally")

        import sys
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

        with tempfile.TemporaryDirectory() as tmp:
            enc_dir = os.path.join(tmp, "tiny_bert")
            os.makedirs(enc_dir, exist_ok=True)
            base_chars = list(dict.fromkeys(
                "入院记录诊断急性腹痛发热白细胞升高需手术评估慢性糖尿病复查血糖平稳口服药维持头尾标记病程确诊排除待排第天检查术后首次在全身麻醉下行腹腔镜阑尾切除术可能心肌梗死"
            ))
            vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + base_chars
            vocab_tokens.extend([f"t{i}" for i in range(max(0, 256 - len(vocab_tokens)))])
            vocab_p = os.path.join(enc_dir, "vocab.txt")
            with open(vocab_p, "w", encoding="utf-8") as vf:
                vf.write("\n".join(vocab_tokens[:256]) + "\n")
            tok = BertTokenizer(vocab_p, tokenize_chinese_chars=True)
            tok.save_pretrained(enc_dir)
            cfg = BertConfig(
                vocab_size=len(tok),
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=512,
                id2label={"0": "LABEL_0"},
                label2id={"LABEL_0": 0},
            )
            BertForSequenceClassification(cfg).save_pretrained(enc_dir)

            # 1) Verify head+tail 512 truncation keeps both start and end characters
            long_txt = ("急" * 350) + ("稳" * 350) + "尾"
            enc_ids, enc_mask = t12_neural_bench.encode_head_tail_batch([long_txt], tok, 512)
            self.assertEqual(tuple(enc_ids.shape), (1, 512))
            self.assertEqual(int(enc_mask.sum().item()), 512)
            decoded_tail = tok.convert_ids_to_tokens(enc_ids[0, -5:].tolist())
            self.assertIn("尾", decoded_tail)

            # 2) Verify anti-dilution resistance under adversarial reversed synthetic data
            real_tr = [
                ("急性腹痛发热白细胞升高需手术评估", 1, 1, f"R{i}") for i in range(10)
            ] + [
                ("慢性糖尿病复查血糖平稳口服药维持", 0, 0, f"RN{i}") for i in range(10)
            ]
            noisy_synth = [
                ("慢性糖尿病复查血糖平稳口服药维持", 1, 1, f"S{i}") for i in range(30)
            ] + [
                ("急性腹痛发热白细胞升高需手术评估", 0, 0, f"SN{i}") for i in range(30)
            ]
            tune_tr = real_tr[:6] + real_tr[10:16]
            aug_dir = os.path.join(tmp, "aug_test")
            aug_art, _, _ = t12_neural_bench.fit_t1_neural_augment(
                aug_dir, real_tr, noisy_synth, tune_tr, encoder_path=enc_dir, max_epochs_override=1
            )
            self.assertIn(aug_art["targets"]["T1a"]["selected_weight"], (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.25, 0.75, 1.0))
            if aug_art["targets"]["T1a"]["real_only_tune_auroc"] >= 0.99:
                self.assertEqual(aug_art["targets"]["T1a"]["selected_weight"], 0.0)

            # 3) Full CLI fit-and-freeze -> eval-frozen roundtrip + tamper detection
            def _make_adm(aid, idx):
                pos = (idx % 2 == 0)
                if pos:
                    d0_txt = "急性腹痛发热白细胞升高\n入院诊断：\n1.急性阑尾炎？"
                    d2_txt = "术后首次病程记录：今日在全身麻醉下行腹腔镜阑尾切除术。急性阑尾炎诊断明确。"
                elif idx % 4 == 1:
                    d0_txt = "慢性糖尿病复查血糖平稳\n入院诊断：\n1.急性阑尾炎？"
                    d2_txt = "第2天查房：复查平稳，急性阑尾炎待排。"
                else:
                    d0_txt = "慢性糖尿病复查血糖平稳\n入院诊断：\n1.急性阑尾炎？"
                    d2_txt = "第2天查房：复查正常，排除急性阑尾炎。"
                return {
                    "admission_id": aid,
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": d0_txt},
                        {"doc_type": "progress_note", "day_offset": 0, "text": "首次病程：急性阑尾炎待排。"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": d2_txt},
                        {"doc_type": "progress_note", "day_offset": 3, "text": "第3天查房：一般情况平稳，无特殊主诉。"},
                        {"doc_type": "discharge_summary", "day_offset": 4, "text": "出院诊断：急性阑尾炎"},
                    ],
                }

            val_p = os.path.join(tmp, "val_fold1.jsonl")
            real_p = os.path.join(tmp, "real_folds27.jsonl")
            with open(val_p, "w", encoding="utf-8") as vf:
                for i in range(16):
                    vf.write(json.dumps(_make_adm(f"V{i}", i), ensure_ascii=False) + "\n")
            with open(real_p, "w", encoding="utf-8") as rf:
                for i in range(20):
                    rf.write(json.dumps(_make_adm(f"TR{i}", i), ensure_ascii=False) + "\n")

            s5_tmp = os.path.join(tmp, "s5_tmp")
            bundle_dir = os.path.join(tmp, "frozen_neural")
            out_json = os.path.join(tmp, "rehearsal_neural.json")

            old_argv = sys.argv
            try:
                sys.argv = [
                    "t12_neural_bench.py",
                    "--s5-root", s5_tmp,
                    "--encoder-path", enc_dir,
                    "--val-fold", val_p,
                    "--real-folds", real_p,
                    "--fit-and-freeze", bundle_dir,
                    "--rehearsal-split",
                    "--max-epochs-override", "1",
                ]
                t12_neural_bench.main()
                sys.argv = [
                    "t12_neural_bench.py",
                    "--s5-root", s5_tmp,
                    "--encoder-path", enc_dir,
                    "--val-fold", val_p,
                    "--real-folds", real_p,
                    "--eval-frozen", bundle_dir,
                    "--n-boot", "20",
                    "--out", out_json,
                ]
                t12_neural_bench.main()
            finally:
                sys.argv = old_argv

            self.assertTrue(os.path.exists(out_json))
            res_obj = json.loads(pathlib.Path(out_json).read_text(encoding="utf-8"))
            self.assertIn("Real-Small (250)", res_obj["T1"])
            self.assertIsNotNone(res_obj["T1"]["Real-Small (250)"]["T1a"]["auroc"])
            self.assertGreater(res_obj["T2a"]["Real-Small (250)"]["n_val"], 0)
            self.assertGreater(res_obj["T2b"]["Real-Small (250)"]["n_val"], 0)

            # Tamper one checkpoint file -> eval-frozen must raise RuntimeError
            man = json.loads(pathlib.Path(bundle_dir, "freeze_manifest.json").read_text(encoding="utf-8"))
            self.assertGreater(len(man["checkpoint_hashes"]), 0)
            first_ckpt = next(iter(man["checkpoint_hashes"]))
            ckpt_abs = pathlib.Path(bundle_dir, first_ckpt)
            ckpt_abs.write_bytes(ckpt_abs.read_bytes() + b"TAMPER")
            try:
                sys.argv = [
                    "t12_neural_bench.py",
                    "--s5-root", s5_tmp,
                    "--encoder-path", enc_dir,
                    "--eval-frozen", bundle_dir,
                    "--n-boot", "5",
                    "--out", out_json,
                ]
                with self.assertRaises(RuntimeError):
                    t12_neural_bench.main()
            finally:
                sys.argv = old_argv

    def test_v16_reviewer_b_round2_regressions(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            # 1) Note-level (t, k) labels & strictly shared Fixed-Frame 2x2 observation window
            j_rows = [
                {
                    "journey_id": "J1",
                    "los": 4,
                    "patient": {"primary_dx": {"name": "急性胆囊炎", "code3": "K80"}},
                    "dx_events": [
                        {"day": 0, "name": "急性胆囊炎", "state": "疑似"},
                        {"day": 2, "name": "急性胆囊炎", "state": "确诊"},
                    ],
                },
                {
                    "journey_id": "J2",
                    "los": 4,
                    "patient": {"primary_dx": {"name": "急性胰腺炎", "code3": "K85"}},
                    "dx_events": [
                        {"day": 0, "name": "急性胰腺炎", "state": "疑似"},
                        {"day": 2, "name": "急性胰腺炎", "state": "确诊"},
                    ],
                },
            ]
            full_adms = [
                {
                    "admission_id": "J1",
                    "docs": [
                        {"doc_type": "hospital_record", "role": "HR", "day_offset": 0, "text": "入院诊断：\n1.急性胆囊炎？"},
                        {"doc_type": "progress_note", "role": "FP", "day_offset": 0, "text": "首次病程：急性胆囊炎待排。"},
                        {"doc_type": "progress_note", "role": "PRE", "day_offset": 2, "text": "术前小结：考虑急性胆囊炎？拟行手术。"},
                        {"doc_type": "progress_note", "role": "POST", "day_offset": 2, "text": "术后首次病程记录：术中诊断：急性胆囊炎。急性胆囊炎诊断明确。"},
                        {"doc_type": "progress_note", "role": "DAILY", "day_offset": 2, "text": "第2天日常病程：术后病情平稳。"},
                        {"doc_type": "discharge_summary", "role": "DS", "day_offset": 3, "text": "出院诊断：急性胆囊炎"},
                    ],
                },
                {
                    "admission_id": "J2",
                    "docs": [
                        {"doc_type": "hospital_record", "role": "HR", "day_offset": 0, "text": "入院诊断：\n1.急性胰腺炎？"},
                        {"doc_type": "progress_note", "role": "FP", "day_offset": 0, "text": "首次病程：急性胰腺炎待排。"},
                        {"doc_type": "progress_note", "role": "SUP", "day_offset": 2, "text": "第2天上级查房：结合检查结果，急性胰腺炎诊断明确。"},
                        {"doc_type": "progress_note", "role": "DAILY", "day_offset": 2, "text": "第2天日常病程：一般情况尚可。"},
                        {"doc_type": "progress_note", "role": "SUP", "day_offset": 3, "text": "第3天上级查房：复查淀粉酶正常。"},
                        {"doc_type": "discharge_summary", "role": "DS", "day_offset": 3, "text": "出院诊断：急性胰腺炎"},
                    ],
                },
            ]
            vis_adms = [
                {
                    "admission_id": "J1",
                    "docs": [
                        {"doc_type": "hospital_record", "role": "HR", "day_offset": 0, "text": "入院诊断：\n1.急性胆囊炎"},
                        {"doc_type": "progress_note", "role": "FP", "day_offset": 0, "text": "首次病程：给予解痉抗炎治疗。"},
                        {"doc_type": "progress_note", "role": "PRE", "day_offset": 2, "text": "术前小结：患者腹痛拟行手术。"},
                        {"doc_type": "progress_note", "role": "POST", "day_offset": 2, "text": "术后首次病程记录：术中过程顺利。"},
                        {"doc_type": "progress_note", "role": "DAILY", "day_offset": 2, "text": "第2天日常病程：一般情况尚可。"},
                        {"doc_type": "discharge_summary", "role": "DS", "day_offset": 3, "text": "出院诊断：急性胆囊炎"},
                    ],
                },
                {
                    "admission_id": "J2",
                    "docs": [
                        {"doc_type": "hospital_record", "role": "HR", "day_offset": 0, "text": "入院诊断：\n1.急性胰腺炎"},
                        {"doc_type": "progress_note", "role": "FP", "day_offset": 0, "text": "首次病程：抑酸禁食。"},
                        {"doc_type": "progress_note", "role": "SUP", "day_offset": 2, "text": "第2天上级查房：继续抗感染治疗。"},
                        {"doc_type": "progress_note", "role": "DAILY", "day_offset": 2, "text": "第2天日常病程：一般情况尚可。"},
                        {"doc_type": "progress_note", "role": "SUP", "day_offset": 3, "text": "第3天上级查房：今日一般情况平稳。"},
                        {"doc_type": "discharge_summary", "role": "DS", "day_offset": 3, "text": "出院诊断：急性胰腺炎"},
                    ],
                },
            ]
            fa_p = os.path.join(tmp, "fa.jsonl")
            fj_p = os.path.join(tmp, "fj.jsonl")
            va_p = os.path.join(tmp, "va.jsonl")
            for p, rows in ((fa_p, full_adms), (fj_p, j_rows), (va_p, vis_adms)):
                with open(p, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            t2b_auto = t123_bench.build_t2b_update_dataset([fa_p], [fj_p], label_source="auto")
            obs_keys = {(r[3], r[4], r[6]): r[1] for r in t2b_auto}
            # J1: PRE(idx=2)->y=0, POST(idx=3)->y=1, DAILY(idx=4) excluded
            self.assertEqual(obs_keys.get(("J1", "急性胆囊炎", 2)), 0)
            self.assertEqual(obs_keys.get(("J1", "急性胆囊炎", 3)), 1)
            self.assertNotIn(("J1", "急性胆囊炎", 4), obs_keys)
            # J2: SUP(idx=2)->y=1, DAILY(idx=3) and SUP(idx=4) excluded
            self.assertEqual(obs_keys.get(("J2", "急性胰腺炎", 2)), 1)
            self.assertNotIn(("J2", "急性胰腺炎", 3), obs_keys)
            self.assertNotIn(("J2", "急性胰腺炎", 4), obs_keys)

            cells, ffaudit = t123_bench.build_t2b_fixed_frame_2x2([fa_p], [fj_p], [va_p], [fj_p])
            self.assertEqual(ffaudit["n_common_windows"], 3)
            self.assertEqual(ffaudit["misaligned_stays_n"], 0)
            frame_ref = [(r[3], r[4], r[6]) for r in cells["Synth-Full | fixed_planner_tk"]]
            for cname, crows in cells.items():
                self.assertEqual([(r[3], r[4], r[6]) for r in crows], frame_ref, cname)
            self.assertEqual([r[1] for r in cells["Synth-Full | fixed_text_rule"]], [0, 1, 1])
            self.assertEqual([r[1] for r in cells["Synth-VisibilityOnly | fixed_text_rule"]], [0, 0, 0])

            # 2) 4-bucket event accounting eliminates single-arm denominator mismatches
            pj_asym = [
                {
                    "journey_id": "A1",
                    "dx_events": [
                        {"day": 1, "name": "冠心病", "state": "排除"},
                        {"day": 2, "name": "急性胆囊炎", "state": "确诊"},
                    ],
                },
                {
                    "journey_id": "A2",
                    "dx_events": [{"day": 1, "name": "脑梗死", "state": "排除"}],
                },
            ]
            f_asym = [
                {
                    "admission_id": "A1",
                    "docs": [
                        {"doc_type": "progress_note", "day_offset": 1, "text": "排除冠心病。"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": "急性胆囊炎诊断明确。"},
                    ],
                }
            ]
            v_asym = [
                {
                    "admission_id": "A1",
                    "docs": [{"doc_type": "progress_note", "day_offset": 1, "text": "今日一般情况可。"}],
                }
            ]
            pj_p = os.path.join(tmp, "pj_asym.jsonl")
            fa_a = os.path.join(tmp, "fa_asym.jsonl")
            va_a = os.path.join(tmp, "va_asym.jsonl")
            for p, rows in ((pj_p, pj_asym), (fa_a, f_asym), (va_a, v_asym)):
                with open(p, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            asym = t123_bench.audit_epistemic_asymmetry([va_a], [pj_p], [fa_a], [pj_p], planned_j_paths=[pj_p], n_boot=20)
            self.assertEqual(asym["Explicit-State"]["confirmed_total"], asym["Visibility-Only"]["confirmed_total"])
            self.assertEqual(asym["Explicit-State"]["excluded_total"], asym["Visibility-Only"]["excluded_total"])
            self.assertEqual(asym["event_flow_accounting"]["confirmed"]["full_only_evaluable"], 1)
            self.assertEqual(asym["event_flow_accounting"]["excluded"]["neither_stay_discarded_any"], 1)
            self.assertTrue(asym["event_flow_accounting"]["confirmed"]["partition_sum_ok"])
            self.assertTrue(asym["event_flow_accounting"]["excluded"]["partition_sum_ok"])

            # 3) Bijective blind-pack prediction coverage including post_silver_settled + indeterminate handling
            real_fold = os.path.join(tmp, "real_blind.jsonl")
            real_adms = [
                {
                    "admission_id": "RB1",
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": "入院诊断：\n1.急性阑尾炎？\n2.急性胆囊炎？"},
                        {"doc_type": "progress_note", "day_offset": 0, "text": "首次病程：急性阑尾炎待排，与急性肠炎相鉴别。"},
                        {"doc_type": "progress_note", "day_offset": 1, "text": "第1天查房：生命体征平稳。"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": "第2天查房：超声回报，急性阑尾炎诊断明确。"},
                        {"doc_type": "progress_note", "day_offset": 3, "text": "第3天查房：继续关注急性阑尾炎恢复。"},
                        {"doc_type": "discharge_summary", "day_offset": 4, "text": "出院诊断：急性阑尾炎"},
                    ],
                },
                {
                    "admission_id": "RB2",
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": "入院诊断：\n1.社区获得性肺炎？"},
                        {"doc_type": "progress_note", "day_offset": 0, "text": "首次病程：社区获得性肺炎待查。"},
                        {"doc_type": "progress_note", "day_offset": 1, "text": "第1天查房：社区获得性肺炎待排，复查胸片。"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": "第2天查房：排除社区获得性肺炎。"},
                        {"doc_type": "progress_note", "day_offset": 3, "text": "第3天查房：体温正常。"},
                        {"doc_type": "discharge_summary", "day_offset": 4, "text": "出院诊断：上呼吸道感染"},
                    ],
                },
            ]
            with open(real_fold, "w", encoding="utf-8") as rf:
                for r in real_adms:
                    rf.write(json.dumps(r, ensure_ascii=False) + "\n")
            cons = t123_bench.audit_fold_docs(real_fold)
            self.assertTrue(cons["t1_consort_flow"]["arithmetic_checks"]["los_partition_ok"])
            self.assertTrue(cons["t1_consort_flow"]["arithmetic_checks"]["v13_to_v16_delta_ok"])

            bdir = os.path.join(tmp, "bpool")
            binfo = t123_bench.export_t2_blind_pool(real_fold, bdir, seed=20261201)
            self.assertGreaterEqual(binfo["strata"]["post_silver_settled"]["n_h"], 1)
            blind_rows, key_rows = t123_bench.build_blind_window_eval_rows(real_fold, binfo["key_path"])
            self.assertEqual(len(blind_rows), binfo["total_exported"])
            wkeys = [km["window_key"] for km in key_rows]
            arm1 = ({wk: (0.8 if i == 0 else 0.1) for i, wk in enumerate(wkeys)}, {wk: (1 if i == 0 else 0) for i, wk in enumerate(wkeys)})
            arm2 = ({wk: 0.2 for wk in wkeys}, {wk: 0 for wk in wkeys})

            with self.assertRaises(ValueError):
                bad_p = dict(arm1[0])
                bad_p.pop(wkeys[-1])
                t123_bench.score_t2_human_gold_multiarm({"A1": (bad_p, arm1[1])}, binfo["sheet_path"], binfo["key_path"])

            with open(binfo["sheet_path"], encoding="utf-8") as sf:
                gold_rows = [json.loads(l) for l in sf if l.strip()]
            for idx, gr in enumerate(gold_rows):
                if idx == 0:
                    gr["q_expressed_state"] = "confirmed"
                    gr["q_should_update_evidence"] = "confirm"
                elif idx == 1:
                    gr["q_expressed_state"] = "timing_unclear"
                    gr["q_should_update_evidence"] = "timing_unclear"
                elif idx == 2:
                    gr["q_expressed_state"] = None
                    gr["q_should_update_evidence"] = None
                else:
                    gr["q_expressed_state"] = "differential"
                    gr["q_should_update_evidence"] = "no_update"
            gold_p = os.path.join(tmp, "gold_filled.jsonl")
            with open(gold_p, "w", encoding="utf-8") as gf:
                for gr in gold_rows:
                    gf.write(json.dumps(gr, ensure_ascii=False) + "\n")
            mrep = t123_bench.score_t2_human_gold_multiarm({"A1": arm1, "A2": arm2}, gold_p, binfo["key_path"])
            t_exp = mrep["targets"]["q_expressed_state"]
            self.assertEqual(t_exp["n_indeterminate_excluded_from_point"], 1)
            self.assertEqual(t_exp["n_missing"], 1)
            self.assertEqual(t_exp["common_valid_n"], len(gold_rows) - 2)
            self.assertEqual(t_exp["arms"]["A1"]["n_scored"], t_exp["arms"]["A2"]["n_scored"])
            self.assertIn("all_indet_as_neg", t_exp["arms"]["A1"]["indeterminate_bounds"])
            self.assertIn("all_indet_as_pos", t_exp["arms"]["A1"]["indeterminate_bounds"])

        # 4) Frozen MIA zero-drift across non-member pools + bidirectional 1% FPR thresholds
        t_pos = s6_mia_audit._freeze_upper_1pct_thr(np.array([0.1, 0.2, 0.8, 0.8, 0.9] + [0.0] * 195))
        self.assertLessEqual(int(np.sum(np.array([0.1, 0.2, 0.8, 0.8, 0.9] + [0.0] * 195) >= t_pos)), 2)
        t_neg = s6_mia_audit._freeze_lower_1pct_thr(np.array([0.05, 0.05, 0.1] + [0.5] * 197))
        self.assertLessEqual(int(np.sum(np.array([0.05, 0.05, 0.1] + [0.5] * 197) <= t_neg)), 2)
        bi = s6_mia_audit.eval_frozen_bidirectional_thresholds(
            y_true=np.array([1, 1, 1, 0, 0, 0]),
            scores=np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9]),
            thr_pos=0.95,
            thr_neg=0.2,
        )
        self.assertGreater(bi["neg_direction_lower_is_member"]["realized_tpr"], bi["pos_direction_higher_is_member"]["realized_tpr"])

        if s6_mia_audit.HAS_SKLEARN:
            mem = [
                {
                    "admission_id": f"MM{i}",
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": f"科室：心内科 年龄:{40+i}岁\n入院诊断：心肌梗死"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": f"第2天查房：独特稀有序列样本甲{i:02d}号连续三十个字符描述复查恢复情况良好。"},
                    ],
                }
                for i in range(8)
            ]
            nm8 = [
                {
                    "admission_id": f"NM8_{i}",
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": f"科室：骨科 年龄:{30+i}岁\n入院诊断：骨折"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": f"第2天查房：折八对照组无关短语编号{i:02d}继续石膏固定观察消肿随访。"},
                    ],
                }
                for i in range(8)
            ]
            nm9 = [
                {
                    "admission_id": f"NM9_{i}",
                    "docs": [
                        {"doc_type": "hospital_record", "day_offset": 0, "text": f"科室：眼科 年龄:{20+i}岁\n入院诊断：白内障"},
                        {"doc_type": "progress_note", "day_offset": 2, "text": f"第2天查房：独特稀有序列样本甲{0:02d}号连续三十个字符描述复查恢复情况良好。"},
                    ],
                }
                for i in range(8)
            ]
            syn_p = [s6_mia_audit._extract_profile(a) for a in mem[:5]]
            mem_p = [s6_mia_audit._extract_profile(a) for a in mem]
            nm8_p = [s6_mia_audit._extract_profile(a) for a in nm8]
            nm9_p = [s6_mia_audit._extract_profile(a) for a in nm9]
            fscorer = s6_mia_audit.fit_frozen_arm_scorer(syn_p, mem_p, nm8_p)
            sc8 = s6_mia_audit.score_profiles_with_frozen_scorer(fscorer, mem_p + nm8_p)
            sc9 = s6_mia_audit.score_profiles_with_frozen_scorer(fscorer, mem_p + nm9_p)
            for atk in ("combined", "stay_1nn_cos", "stay_nn_ratio", "line_1nn_cos", "rare_30gram_attack"):
                self.assertLess(float(np.max(np.abs(sc8[atk][:len(mem_p)] - sc9[atk][:len(mem_p)]))), 1e-12, atk)

    def test_v17_intraday_kstar_and_gen_hard_gate(self):
        # 1) s5_common precedence respects MEDCOURSE_CORE_DIR at sys.path[0]
        import sys
        old_env = os.environ.get("MEDCOURSE_CORE_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["MEDCOURSE_CORE_DIR"] = tmp
            try:
                s5_common.setup_paths()
                self.assertEqual(sys.path[0], tmp)
            finally:
                if tmp in sys.path:
                    sys.path.remove(tmp)
                if old_env is None:
                    os.environ.pop("MEDCOURSE_CORE_DIR", None)
                else:
                    os.environ["MEDCOURSE_CORE_DIR"] = old_env

        # 2) Intra-day (day, hm) alignment of _find_compiler_update_doc_idx and _dx_state_at_doc_idx
        j_intraday = {
            "journey_id": "UT-INTRA-01",
            "patient": {"primary_dx": {"name": "急性阑尾炎", "code3": "K35"}},
            "dx_events": [
                {"id": "d1", "name": "急性阑尾炎", "day": 0, "hm": "07:30", "state": "确诊"},
                {"id": "d2", "name": "右侧输尿管结石", "day": 0, "hm": "07:30", "state": "疑似"},
                {"id": "d3", "name": "右侧输尿管结石", "day": 1, "hm": "14:20", "state": "排除", "evidence_ids": ["e1"], "basis_text": "结合泌尿系超声未见明显异常"},
            ],
        }
        docs_intra = [
            {"doc_type": "hospital_record", "role": "HR", "day_offset": 0, "time": "08:00", "text": "入院记录"},
            {"doc_type": "progress_note", "role": "FP", "day_offset": 0, "time": "10:00", "text": "首次病程记录 考虑右侧输尿管结石"},
            {"doc_type": "progress_note", "role": "SUP", "day_offset": 1, "time": "09:00", "text": "2024-05-02 09:00 上级医师查房 右侧输尿管结石待排"},
            {"doc_type": "progress_note", "role": "DAILY", "day_offset": 1, "time": "16:30", "text": "2024-05-02 16:30 病程记录 结合泌尿系超声未见明显异常，已排除右侧输尿管结石。"},
            {"doc_type": "discharge_summary", "role": "DS", "day_offset": 3, "time": "10:00", "text": "出院记录"},
        ]
        k_star, obs_idx = t123_bench._find_compiler_update_doc_idx(j_intraday, docs_intra, "右侧输尿管结石", 1, "排除")
        self.assertEqual(k_star, 3)
        self.assertEqual(obs_idx, [2, 3])
        self.assertEqual(t123_bench._dx_state_at_doc_idx(j_intraday, docs_intra, "右侧输尿管结石", 2, [(0, "疑似"), (1, "排除")]), "疑似")
        self.assertEqual(t123_bench._dx_state_at_doc_idx(j_intraday, docs_intra, "右侧输尿管结石", 3, [(0, "疑似"), (1, "排除")]), "排除")

        # 3) gen_s5_shard hard-gate rejection & cascade rewrite parity
        kb_data = kb.load_default_kb()
        st = s5_common.build_pathway_stats("route_b", "I20.0", kb_data=kb_data)
        rng = random.Random(planner.seed_of(20261210, "UT-HARDGATE-01"))
        j_g = planner.plan_journey(st, rng, "UT-HARDGATE-01", None)
        j_g["dx_events"] = [
            {"id": "d1", "name": j_g["patient"]["primary_dx"]["name"], "day": 0, "hm": "07:30", "state": "疑似", "evidence_ids": []},
            {"id": "d2", "name": j_g["patient"]["primary_dx"]["name"], "day": 1, "hm": "07:45", "state": "确诊", "evidence_ids": [j_g["exams"][0]["id"]], "basis_text": f"结合{j_g['exams'][0]['name']}结果"},
        ]
        if j_g["los"] < 2:
            j_g["los"] = 2
        specs_g = planner.schedule(j_g, st, rng)
        sup_spec = next((s for s in specs_g if s["role"] == "SUP" and s["day"] >= 1), None)
        if sup_spec is None:
            specs_g.insert(2, {"k": 2, "doc_type": "progress_note", "doc_type_cn": "上级医师查房记录", "role": "SUP", "day": 1, "author": "主治医师", "time": "09:00"})
            for idx_s, s in enumerate(specs_g):
                s["k"] = idx_s

        class BadPolWriter(writer.TemplateWriter):
            def write(self, j_in, spec_in, pl_in, s_in):
                txt = super().write(j_in, spec_in, pl_in, s_in)
                if spec_in["role"] == "SUP" and any(f.get("req_dx_polarity") == "确诊" for f in pl_in.get("facts") or ()):
                    return f"继续关注{j_in['patient']['primary_dx']['name']}变化，复查指标后再评估。"
                return txt

        bw = BadPolWriter()
        bw.splice_copy = False
        docs_out, fails_out, _ = gen_s5_shard.generate_from_plan(j_g, specs_g, {"_default": st}, "full", bw, 20261210, {})
        self.assertIsNone(docs_out)
        self.assertTrue(any(c.startswith("hard_sec_dx_polarity_missing") for c in fails_out), fails_out)

    def test_v18_smoke10_failed_restart_does_not_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            core_dir = os.path.join(tmp, "core")
            os.makedirs(core_dir, exist_ok=True)
            with open(os.path.join(core_dir, "VERSION"), "w", encoding="utf-8") as f:
                f.write("0.3.6\n")
            with open(os.path.join(core_dir, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
                f.write("deadbeef  medcourse/planner.py\n")
            plan_p = os.path.join(tmp, "journeys.jsonl")
            with open(plan_p, "w", encoding="utf-8") as f:
                f.write('{"journey_id": "S1"}\n')
            attempt_p = os.path.join(tmp, "v036_live_smoke10_last_attempt.json")
            stamp_p = os.path.join(tmp, "v036_live_smoke10_passed.stamp.json")
            fp = s5_common.compute_smoke10_fingerprint(core_dir, plan_p, "qwen", "tp1")

            # Run 1: fail on llm_errors -> writes last_attempt, never creates passed stamp
            bad_llm = {"kept": 10, "discard_rate": 0.0, "llm_errors": 1, "delivered_adm_gate_fails": 0, "delivered_oracle_any_hits": 0}
            with self.assertRaises(AssertionError):
                s5_common.record_smoke10_attempt_and_stamp(attempt_p, stamp_p, bad_llm, fp)
            self.assertTrue(os.path.isfile(attempt_p))
            self.assertFalse(os.path.exists(stamp_p))
            self.assertFalse(s5_common.smoke10_stamp_valid(stamp_p, fp))

            # Run 1b: fail on oracle hits even with 0 llm_errors -> still no stamp
            bad_ora = {"kept": 10, "discard_rate": 0.0, "llm_errors": 0, "delivered_adm_gate_fails": 0, "delivered_oracle_any_hits": 1}
            with self.assertRaises(AssertionError):
                s5_common.record_smoke10_attempt_and_stamp(attempt_p, stamp_p, bad_ora, fp)
            self.assertFalse(os.path.exists(stamp_p))
            self.assertFalse(s5_common.smoke10_stamp_valid(stamp_p, fp))

            # Run 2/3: clean pass -> atomic passed stamp created and validates
            good = {"kept": 10, "discard_rate": 0.0, "llm_errors": 0, "delivered_adm_gate_fails": 0, "delivered_oracle_any_hits": 0}
            st_rec = s5_common.record_smoke10_attempt_and_stamp(attempt_p, stamp_p, good, fp)
            self.assertTrue(st_rec["passed"])
            self.assertTrue(os.path.isfile(stamp_p))
            self.assertTrue(s5_common.smoke10_stamp_valid(stamp_p, fp))

            # Run 4: changing plan journeys invalidates fingerprint
            with open(plan_p, "w", encoding="utf-8") as f:
                f.write('{"journey_id": "S2"}\n')
            fp2 = s5_common.compute_smoke10_fingerprint(core_dir, plan_p, "qwen", "tp1")
            self.assertNotEqual(fp, fp2)
            self.assertFalse(s5_common.smoke10_stamp_valid(stamp_p, fp2))

    def test_v19_ledgers_accounting_and_parser_parity(self):
        import build_v037_ledgers as bl
        from medcourse import gate

        with tempfile.TemporaryDirectory() as tmp:
            st = s5_common.build_pathway_stats("route_b", "J15", kb_data=kb.load_default_kb())
            rng = random.Random(20261210)
            j = planner.plan_journey(st, rng, "UT-LEDGER-01", None)
            specs = planner.schedule(j, st, rng)
            pdir = os.path.join(tmp, "planned", "s5_1_shards", "shard_0")
            os.makedirs(pdir, exist_ok=True)
            with open(os.path.join(pdir, "journeys.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
            with open(os.path.join(pdir, "specs.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"slot_id": j["journey_id"], "specs": specs}, ensure_ascii=False) + "\n")
            pre = bl.build_pre_gpu_ledger(tmp)
            s51 = pre["cohorts"]["S5-1"]
            self.assertEqual(s51["contract_violations_total"], 0, s51["contract_violations"])
            self.assertEqual(s51["semantic_rule_violations_total"], 0, s51["semantic_rule_violations"])
            self.assertIn("by_icd_chapter", s51["comorbidity_ledger"])
            self.assertIn("top_40_primary_dx", s51["comorbidity_ledger"])

            # Verify parser ledger pulls unparsed_30_samples strictly from gate.audit_item_numbers
            # and does NOT put numeric strings like "血糖10.59mmol/L" into unparsed samples
            docs = []
            for s in specs:
                txt, _ = pipeline.write_doc(j, s, specs, st, writer.TemplateWriter(), random.Random(1), {}, None)
                if s["role"] == "FP":
                    txt = txt.replace("拟诊讨论：", "拟诊讨论：复查白细胞需结合临床，")
                docs.append({"doc_type": s["doc_type"], "day_offset": s["day"], "text": txt, "doc_type_cn": s["doc_type_cn"], "k": s["k"], "role": s["role"]})
            cdir = os.path.join(tmp, "gen")
            os.makedirs(cdir, exist_ok=True)
            with open(os.path.join(cdir, "admissions.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"admission_id": j["journey_id"], "docs": docs}, ensure_ascii=False) + "\n")
            with open(os.path.join(cdir, "journeys.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
            sp_file = os.path.join(tmp, "stats.json")
            with open(sp_file, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False)
            par = bl.audit_generated_parser(cdir, sp_file)
            self.assertGreaterEqual(par["llm_sections_with_lab_roles"], len([d for d in docs if d["role"] in ("HR", "FP", "SUP", "DAILY", "DS")]))
            self.assertGreaterEqual(par["parsed_value_tokens"], 1)
            for smp in par["unparsed_30_samples"]:
                if smp["reason"] != "parallel_ambiguous":
                    self.assertEqual(gate._extract_val_matches(smp["seg"], ()), [], smp)
                    self.assertFalse(re.search(r"\d+(?:\.\d+)?\s*(?:mmol/L|μmol/L|umol/L|g/L|mg/L|U/L|%|×10)", smp["seg"]), smp)

    def test_v20_ledger_semantic_fault_injection(self):
        import build_v037_ledgers as bl

        def _base_stay(pdx_name="肺炎", code3="J18", organs=None, sex="男", age=55):
            return {
                "journey_id": "UT-FAULT-1",
                "los": 5,
                "patient": {
                    "sex": sex,
                    "age": age,
                    "primary_dx": {"name": pdx_name, "code3": code3, "organs": organs or ["肺"]},
                    "comorbidities": [],
                },
                "procedures": [],
                "exams": [],
                "dx_events": [],
            }

        # 1) Reviewer B Fault 1: normal head CT cited to exclude pneumonia (even with forged organs=["肺", "脑"])
        j1 = _base_stay(pdx_name="肺炎", code3="J18", organs=["肺", "脑"])
        j1["exams"] = [{
            "id": "e1", "key": "CT", "region": "头", "name": "头颅CT", "kind": "影像",
            "order_day": 1, "result_day": 1, "result_hm": "10:00",
            "items": [], "abnormal": False, "diagnostic": False, "can_confirm": False,
        }]
        j1["dx_events"] = [{
            "id": "d1", "name": "肺炎", "state": "排除", "day": 1, "hm": "11:00", "evidence_ids": ["e1"],
        }]
        v1, _ = bl._check_stay_semantics(j1)
        self.assertIn("dx_event_unsupported_evidence:肺炎:排除", v1)

        # 2) Reviewer B Fault 2: head CT incidental_findings=["低钾血症"] confirming 低钾血症
        j2 = _base_stay(pdx_name="肺炎", code3="J18", organs=["肺"])
        j2["patient"]["comorbidities"] = [{
            "name": "低钾血症", "at_admission": False, "discover_day": 1, "discover_hm": "11:00", "evidence_ids": ["e1"],
        }]
        j2["exams"] = [{
            "id": "e1", "key": "CT", "region": "头", "name": "头颅CT", "kind": "影像",
            "order_day": 1, "result_day": 1, "result_hm": "10:00",
            "items": [], "abnormal": True, "diagnostic": False, "incidental_findings": ["低钾血症"],
        }]
        j2["dx_events"] = [{
            "id": "d1", "name": "低钾血症", "state": "确诊", "day": 1, "hm": "11:00",
            "evidence_ids": ["e1"], "newly_discovered": True,
        }]
        v2, _ = bl._check_stay_semantics(j2)
        self.assertIn("invalid_incidental_finding:CT:头:低钾血症", v2)
        self.assertIn("inpatient_comorb_unsupported_evidence:低钾血症", v2)
        self.assertIn("dx_event_unsupported_evidence:低钾血症:确诊", v2)

        # 3) Wrong-procedure obstetric comorbidity: female + cholecystectomy cited for 产后状态
        j3 = _base_stay(pdx_name="胆囊结石", code3="K80", organs=["腹"], sex="女", age=30)
        j3["patient"]["comorbidities"] = [{
            "name": "产后状态", "at_admission": False, "discover_day": 1, "discover_hm": "13:00", "evidence_ids": ["pr1"],
        }]
        j3["procedures"] = [{"id": "pr1", "name": "腹腔镜胆囊切除术", "decide_day": 0, "do_day": 1, "do_hm": "13:00"}]
        j3["dx_events"] = [{
            "id": "d1", "name": "产后状态", "state": "确诊", "day": 1, "hm": "13:00",
            "evidence_ids": ["pr1"], "newly_discovered": True,
        }]
        v3, _ = bl._check_stay_semantics(j3)
        self.assertIn("inpatient_comorb_unsupported_evidence:产后状态", v3)
        self.assertIn("dx_event_unsupported_evidence:产后状态:确诊", v3)

        # 4) Forged can_confirm=True on gastroscopy cited to confirm acute appendicitis
        j4 = _base_stay(pdx_name="急性阑尾炎", code3="K35", organs=["腹"])
        j4["exams"] = [{
            "id": "e1", "key": "胃镜", "region": "腹", "name": "胃镜", "kind": "内镜",
            "order_day": 1, "result_day": 1, "result_hm": "11:00",
            "items": [], "abnormal": True, "diagnostic": True, "can_confirm": True, "finding": "符合急性阑尾炎表现",
        }]
        j4["dx_events"] = [{
            "id": "d1", "name": "急性阑尾炎", "state": "确诊", "day": 1, "hm": "11:00", "evidence_ids": ["e1"],
        }]
        v4, _ = bl._check_stay_semantics(j4)
        self.assertIn("endoscopy_subsite_mismatch:胃镜:急性阑尾炎", v4)
        self.assertIn("dx_event_unsupported_evidence:急性阑尾炎:确诊", v4)

        # 5) Wrong-direction lab (high potassium K=5.8 H) cited to confirm 低钾血症
        j5 = _base_stay(pdx_name="肺炎", code3="J18", organs=["肺"])
        j5["patient"]["comorbidities"] = [{
            "name": "低钾血症", "at_admission": False, "discover_day": 1, "discover_hm": "11:20", "evidence_ids": ["e1"],
        }]
        j5["exams"] = [{
            "id": "e1", "key": "电解质", "region": None, "name": "电解质", "kind": "检验",
            "order_day": 1, "result_day": 1, "result_hm": "11:20",
            "abnormal": True, "diagnostic": False,
            "items": [{"key": "K", "name": "钾", "val": 5.8, "unit": "mmol/L", "flag": "H"}],
        }]
        j5["dx_events"] = [{
            "id": "d1", "name": "低钾血症", "state": "确诊", "day": 1, "hm": "11:20",
            "evidence_ids": ["e1"], "newly_discovered": True,
        }]
        v5, _ = bl._check_stay_semantics(j5)
        self.assertIn("inpatient_comorb_unsupported_evidence:低钾血症", v5)
        self.assertIn("dx_event_unsupported_evidence:低钾血症:确诊", v5)

    def test_v21_mechanism_contribution_decomposition(self):
        import analyze_mechanism_contribution as amc

        with self.assertRaises(ValueError):
            amc.build_mechanism_report("/tmp/dummy_root", real_fold="/home/weify/pyt/data/deid_run9_fold9.jsonl")

        kb_data = kb.load_default_kb()
        st = s5_common.build_pathway_stats("route_b", "J15", kb_data=kb_data)
        stats_map = {"_default": st}
        tw = writer.TemplateWriter()
        tw.splice_copy = False

        with tempfile.TemporaryDirectory() as tmp:
            pdir = os.path.join(tmp, "planned", "s5_1_shards", "shard_0")
            os.makedirs(pdir, exist_ok=True)
            planned = []
            for idx in range(3):
                jid = f"UT-MECH-{idx}"
                rng = random.Random(planner.seed_of(20261210 + idx, jid))
                j = planner.plan_journey(st, rng, jid, None)
                if j["los"] < 3:
                    j["los"] = 3
                # Guarantee at least one confirmed and one excluded day>=1 transition
                j["dx_events"] = [
                    {"id": "d1", "name": j["patient"]["primary_dx"]["name"], "day": 0, "hm": "07:30", "state": "疑似", "evidence_ids": []},
                    {"id": "d2", "name": j["patient"]["primary_dx"]["name"], "day": 1, "hm": "11:30", "state": "确诊", "evidence_ids": ["e1"], "basis_text": "结合影像结果"},
                    {"id": "d3", "name": "肺脓肿", "day": 0, "hm": "07:30", "state": "疑似", "evidence_ids": []},
                    {"id": "d4", "name": "肺脓肿", "day": 1, "hm": "11:30", "state": "排除", "evidence_ids": ["e2"], "basis_text": "结合影像未见明显异常"},
                ]
                specs = planner.schedule(j, st, rng)
                planned.append((j, specs))

            with open(os.path.join(pdir, "journeys.jsonl"), "w", encoding="utf-8") as jf, \
                 open(os.path.join(pdir, "specs.jsonl"), "w", encoding="utf-8") as sf:
                for j, specs in planned:
                    jf.write(json.dumps(j, ensure_ascii=False) + "\n")
                    sf.write(json.dumps({"slot_id": j["journey_id"], "specs": specs}, ensure_ascii=False) + "\n")

            for arm, prefix, save_ft in (("full", "full_shard", True), ("vis_only", "vis_only_shard", True), ("oracle_leak", "oracle_leak_shard", False)):
                sdir = os.path.join(tmp, "generated", "s5_1", f"{prefix}0")
                os.makedirs(sdir, exist_ok=True)
                tw.ablate_epistemic = (arm == "vis_only")
                adm_lines, ft_lines, j_lines = [], [], []
                for j, specs in planned:
                    tally = {}
                    docs, fails, fdocs = gen_s5_shard.generate_from_plan(j, specs, stats_map, arm, tw, 20261210, tally, save_first_try=save_ft)
                    self.assertIsNotNone(docs, (arm, fails))
                    adm_lines.append(pipeline.admission_line(j, docs) if hasattr(pipeline, "admission_line") else {"admission_id": j["journey_id"], "docs": docs})
                    j_lines.append(j)
                    if save_ft and fdocs is not None:
                        ft_lines.append({"admission_id": j["journey_id"], "docs": fdocs})
                with open(os.path.join(sdir, "admissions.jsonl"), "w", encoding="utf-8") as f:
                    for row in adm_lines:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                with open(os.path.join(sdir, "journeys.jsonl"), "w", encoding="utf-8") as f:
                    for row in j_lines:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if ft_lines:
                    with open(os.path.join(sdir, "first_try_admissions.jsonl"), "w", encoding="utf-8") as f:
                        for row in ft_lines:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                with open(os.path.join(sdir, "counts.json"), "w", encoding="utf-8") as f:
                    json.dump({"s5_arm": arm, "planned": len(planned), "kept": len(adm_lines), "discarded": 0, "llm_errors": 0, "sections": 20, "sections_first_try": 20, "first_pass": 3, "chars": 3000, "calls": 0}, f)

            rep = amc.build_mechanism_report(tmp, real_fold="", max_intrinsic_stays=10, n_boot=20)
            b0_id = rep["block0_fixed_rule_scope_and_paired_identity"]["paired_identity"]
            self.assertTrue(b0_id["zero_confounding_verified"])
            self.assertEqual(b0_id["mismatches"], 0)
            self.assertEqual(b0_id["stays_verified_total"], 9)

            b3 = rep["block3_compiler_state_vs_gate_rewrite"]
            self.assertTrue(b3["available"])
            fc_conf = b3["four_cell_paired"]["confirmed"]
            self.assertEqual(set(fc_conf["cells"].keys()), {"VisOnly_FirstTry", "VisOnly_PostGate", "Full_FirstTry", "Full_PostGate"})
            dc_conf = fc_conf["decomposition"]
            self.assertIsNotNone(dc_conf["total_miss_reduction_vis_ft_to_full_pg"])
            if dc_conf["total_miss_reduction_vis_ft_to_full_pg"] > 0:
                self.assertAlmostEqual(
                    dc_conf["compiler_share_of_total_reduction"] + dc_conf["gate_marginal_share_of_total_reduction"],
                    1.0,
                    places=3,
                )

            md_txt = amc.render_markdown_summary(rep)
            self.assertNotIn("\u00b7", md_txt)
            self.assertNotIn("脱" + "盲", md_txt)
            self.assertNotIn("认" + "识论", md_txt)


if __name__ == "__main__":
    unittest.main()
