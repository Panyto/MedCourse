"""记分牌指标表（产品工程计划 v2 第 6、7 节）。门线在这里定死，S0 封存后只能升版改。

每项：id、组、关、中文名、读数来源 src、取法 how、方向 dir、底线 floor、目标 target、状态 status、primary（是否主打）。
- how="mean"：逐住院取均值；"rate"：hit/chk，chk=None 时分母是全部住院；"pooled"：分子列之和/分母列之和；"row"：行级读数。
- floor、target 是 (下限, 上限)，None 表示这一侧不设线。方向只用于说明。
- status：可用 / 只作诊断 / 只报告 / 待接入（S0 不跑，写明在哪一阶段接）。
"""

M = []


def add(mid, group, gate, name, src, how, field, floor=None, target=None, status="可用", primary=False, chk=None, den=None,
        need=("admission", "care7")):
    M.append({"id": mid, "group": group, "gate": gate, "name": name, "src": src, "how": how, "field": field,
              "floor": floor, "target": target, "status": status, "primary": primary, "chk": chk, "den": den, "need": need})


ADM = ("admission",)

# ---------- A 组：产品能力和闸门保证（7.1 全量闸门的读数，合成行应为 0） ----------
# PREREG 第 4 节写的是"主打指标（A、B 组）"，A 组 19 项全是主打。v0.1 只标了 A_date_order，和预登记不一致，v0.2 改回。
add("A_n_docs", "A", 2, "每次住院文书数", "extra", "mean", "n_docs", floor=(4, None), target=(6, None), primary=True, need=ADM)
add("A_doc_types", "A", 2, "文书类型数", "extra", "mean", "n_doc_types", floor=(3, None), target=(5, None), primary=True, need=ADM)
add("A_timeline", "A", 2, "文书带日程的比例", "extra", "mean", "timeline_share", floor=(0.9, None), target=(0.99, None), primary=True, need=ADM)
add("A_date_order", "A", 2, "病程文书头日期倒退（住院率）", "extra", "rate", "date_viol", chk="date_chk", floor=(None, 0.05), target=(None, 0.02), primary=True, need=ADM)
for mid, name, field in (("A_g3_hard", "硬用药错误（g3-rules-0.3）", "any_hard"), ("A_g3_ped", "儿科成人量", "any_pedadult"),
                         ("A_g3_sex", "性别用药", "any_sex"), ("A_g3_warn", "警示级用药", "any_warning"), ("A_g3_n1", "不可能数值 N1", "any_n1")):
    add(mid, "A", 3, name, "g3", "rate", field, floor=(None, 0.02), target=(None, 0.0), primary=True)
for pid, name in (("G1_1", "性别 抬头对正文"), ("G1_2", "性别特有内容"), ("G1_3", "年龄 抬头对正文"), ("G1_5", "侧别 单篇"), ("G1_6", "住院天数")):
    add("A_" + pid, "A", 1, name, "g1", "rate", pid + "_hit", chk=pid + "_chk", floor=(None, 0.03), target=(None, 0.0), primary=True)
add("A_b7", "A", 1, "年龄对病史、儿童对成人内容（b7）", "g1", "rate", "b7_hit", chk="b7_chk", floor=(None, 0.03), target=(None, 0.0), primary=True)
for rid, name in (("S1", "性别 跨文书"), ("S2", "年龄 跨文书"), ("S3", "侧别 跨文书"), ("S4b", "手术日 跨文书")):
    add("A_" + rid, "A", 2, name, "g2", "rate", rid + "_hit", chk=rid + "_chk", floor=(None, 0.03), target=(None, 0.0), primary=True, need=ADM)

# ---------- 7.2 只作诊断，不进闸门 ----------
add("D_G1_4", "诊断", 1, "G1-4 起病时长", "g1", "rate", "G1_4_hit", chk="G1_4_chk", status="只作诊断")
for k in ("K1", "K2", "K3", "K4", "K5"):
    add("D_" + k, "诊断", 2, k + " 时间口吻", "g2", "rate", k + "_hit", chk=k + "_chk", status="只作诊断", need=ADM)
add("D_B2", "诊断", 2, "B2 无据升级", "g2", "rate", "B2_hit", chk="B2_chk", status="只作诊断", need=ADM)
add("D_S4a", "诊断", 2, "S4a 手术日（粗）", "g2", "rate", "S4a_hit", chk="S4a_chk", status="只作诊断", need=ADM)
add("D_S5", "诊断", 2, "S5", "g2", "rate", "S5_hit", chk="S5_chk", status="只作诊断", need=ADM)

# ---------- B 组：闸门外质量（真实模式） ----------
add("B_tfidf", "B", 2, "相邻 tf-idf", "g2", "mean", "tfidf_adj", floor=(None, 0.30), target=(None, 0.27), primary=True, need=ADM)
add("B_copy5", "B", 2, "相邻 5-gram 拷贝（C1a）", "g2", "mean", "C1a_val", floor=(0.295, 0.49), target=(0.35, 0.43), primary=True, need=ADM)
add("B_exam", "B", 2, "检查种类", "g2", "mean", "exam_kinds", floor=(6.25, None), target=(8, None), primary=True, need=ADM)
add("B_dxchg", "B", 2, "诊断变化步数", "g2", "mean", "dx_changes", floor=(None, 0.87), target=(None, 0.5), primary=True, need=ADM)
add("B_B1", "B", 2, "鉴别诊断闭环率", "g2", "mean", "B1_val", floor=(None, 0.38), target=(0.25, 0.36), primary=True, need=ADM)
add("B_B3", "B", 2, "出院新增诊断比例", "g2", "mean", "B3_val", floor=(0.21, None), target=(0.24, 0.32), primary=True, need=ADM)
add("B_course_empty", "B", 2, "病程空句比例", "g4", "mean", "course_empty", floor=(None, 0.40), target=(None, 0.35), primary=True, need=ADM)
add("B_dose", "B", 3, "写了剂量的比例", "g3", "pooled", "n_with_dose", den="n_mentions", floor=(0.18, None), target=(0.20, 0.30), primary=True)
add("B_mentions", "B", 3, "每份药名次数", "g3", "mean", "n_mentions", floor=(6.3, None), target=(8, None), primary=True)
add("B_chars", "B", 4, "四段叙述字数", "g4", "mean", "chars", floor=(768, None), target=(1000, None), primary=True)
add("B_iu", "B", 4, "信息单元", "g4", "mean", "iu_distinct", floor=(29.5, None), target=(40, None), primary=True)
add("B_num", "B", 4, "数值", "g4", "mean", "NUM", floor=(15, None), target=(20, None), primary=True)
add("B_anchor", "B", 4, "时间锚点", "g4", "mean", "anchor_n", floor=(4.5, None), target=(6, None), primary=True)
add("B_auc", "B", 5, "判别器 AUC（逐段等长）", "c2", "row", "auc", floor=(None, 0.95), target=(None, 0.90), primary=True)
add("B_jaccard", "B", 6, "同病实体重合", "g6", "row", "jaccard", floor=(None, 0.25), target=(None, 0.22), primary=True)
add("B_ages", "B", 6, "不同年龄数（每 256 份）", "g6", "row", "distinct_age_256", floor=(43, None), target=(50, None), primary=True)
add("B_vendi", "B", 6, "Vendi（只报告）", "g6", "row", "vendi", status="只报告")
add("B_selfbleu", "B", 6, "Self-BLEU（只报告）", "g6", "row", "self_bleu", status="只报告")
add("B_mauve", "B", 5, "MAUVE", "mauve", "row", "mauve", status="只报告")
add("B_reader_age", "B", 5, "年龄读者", "-", "row", "-", status="待接入 S5")
add("B_blind", "B", 5, "人盲测正确率", "-", "row", "-", status="待接入 S6")
add("B_leak", "B", 2, "真未来泄露", "-", "row", "-", status="待接入 S1（有底稿才能算，oracle2 记 4 条已知假报警）")

# ---------- C 组：它的卷子考我们（规则能算的部分） ----------
add("C_L9", "C", 4, "出院带药都写了剂量（规则）", "g4", "rate", "L9", status="只报告")
add("C_kcov", "C", 6, "留出词表覆盖率（相对 15 个真人）", "g6", "row", "kcov_rel", floor=(-0.05, None), target=(-0.02, None))
add("C_complete9", "C", 4, "完整性 9 条（32B 评委）", "-", "row", "-", status="待接入 S5")

# ---------- F 组：效率 ----------
add("F_calls_adm", "F", 0, "每次住院调用数", "eff", "row", "calls_per_adm", status="只报告")
add("F_calls_kchar", "F", 0, "每千字调用数", "eff", "row", "calls_per_kchar", target=(None, 15), status="可用")
add("F_tok_kchar", "F", 0, "每千字 token", "eff", "row", "tokens_per_kchar", status="只报告")
add("F_sec_kadm", "F", 0, "每千次住院耗时（秒）", "eff", "row", "seconds_per_kadm", status="只报告")

BY_ID = {m["id"]: m for m in M}


def verdict(value, band):
    """值和门线比：True 在线内，False 在线外，None 没有门线或没有值。"""
    if band is None or value is None:
        return None
    lo, hi = band
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True
