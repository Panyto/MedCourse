"""LLM-CARe 语料层（corpus stage）配额的移植：参考分布、最大余数分配、特征抽取、Critic 的改写计划和反馈句。

逐行照抄公开代码 `official/llmcare.py`（sha256 7d0c10e5…，G1-B 的 EVIDENCE 原件）的
`_load_ref`、`allocate_samples`、`_get_corpus_statistic`、`_extract_feature`、`_corpus_critic`，只改四处，每处都是为了
"同样的输入每次给同样的输出"或"认得中文"，不改配额的逻辑：

A1 随机数：原代码用全局 `random`；这里传进一个 `random.Random` 实例 `rng`，调用的函数和顺序与原代码一样
   （自检 `test_official_fidelity` 用同一个种子对原代码和本移植逐条比，见 tests）。
A2 顺序：原代码 `_load_ref` 用 `set(...)` 往 Counter 里加实体，实体的先后取决于 Python 的字符串哈希种子，
   最大余数法打平时选谁、候选词的先后都跟着变。这里每份参考病历的实体按写法排序后再加（参考病历由调用方按住院号排好）。
A3 反馈句：原代码 `"、".join(一个 set)`，词的先后也取决于哈希种子；这里按写法排序后再连。句子原文不变：
   "Add the following keywords：…"（全角冒号，原代码如此）、"Remove the following keywords: …"。
A4 性别：原代码只认英文 male、female（`"male" in patient_info.lower()`）。我们的病历写"男""女"，
   这里 "男" 当 male、"女" 当 female，判断的先后与原代码相同（后判 female，所以两个都出现时算 female）。
   年龄照原代码：patient_info 里第一个整数整除 5（"8个月"会当 8 岁，原代码如此，不改）。

**原代码的一个算式问题（照抄，不修；另给一个修正版只作对照）**：实体字段的期望份数是
`allocate_samples(ref[段], round(n × S))`，ref[段] 是每个实体在参考病历里的出现比例 p，S = Σp（参考病历这一段平均有几个实体）。
allocate_samples 是给"加起来等于 1 的分布"写的，这里没先除以 S，于是每个实体的期望份数 = n × S × p，是"按真实比例"
n × p 的 S 倍（公开 example-ref.jsonl 的出院医嘱一段 S = 7 和 14）。后果：几乎每个参考实体都"不够"，Critic 只会加词、
几乎不会删词。`corpus_critic(..., corrected=True)` 先除以 S（期望份数 = n × p），只给地板 F-QOKc 用，LLM-CARe 行一律用原样。

病历在这里用 LLM-CARe 的内部格式（`to_internal`，与关1 G1-D 伙计代码 `g1d_reviser_replay.to_internal` 相同）。
"""
import random  # noqa: F401  （只为与原代码对照；本模块里不调用全局 random）
import re
from collections import Counter, defaultdict

FIELDS = ["chief_complaint", "history_of_present_illness", "hospital_course", "discharge_instructions"]
SEX_TO_EN = {"男": "male", "女": "female", "male": "male", "female": "female"}
SEX_TO_CN = {"male": "男", "female": "女"}
ADD_PREFIX = "Add the following keywords："
REMOVE_PREFIX = "Remove the following keywords: "


def to_internal(care7):
    """7 字段 → LLM-CARe 内部格式（与 g1d_reviser_replay.to_internal 逐字相同）。"""
    return {
        "patient_info": f"Gender: {care7['Gender']}; Age: {care7['Age']}",
        "chief_complaint": care7["Chief Complaint"],
        "history_of_present_illness": care7["History of Present Illness"],
        "hospital_course": care7["Hospital Course"],
        "discharge_diagnosis": care7["Primary Diagnosis"],
        "discharge_instructions": care7["Discharge Instructions"],
    }


def from_internal(internal, base):
    """内部格式 → 7 字段。性别、年龄取 base（语料层的性别、年龄由 apply_demog 直接写，不经过 Reviser）。"""
    return {
        "Gender": base["Gender"], "Age": base["Age"],
        "Primary Diagnosis": internal["discharge_diagnosis"],
        "Chief Complaint": internal["chief_complaint"],
        "History of Present Illness": internal["history_of_present_illness"],
        "Hospital Course": internal["hospital_course"],
        "Discharge Instructions": internal["discharge_instructions"],
    }


def ref_line(icd, sex, age, entity):
    """一条参考记录，格式同公开的 example-ref.jsonl：{icd, sex(male/female), age(字符串), entity{段:{类别:[写法]}}}。"""
    return {"icd": icd, "sex": SEX_TO_EN.get(str(sex), str(sex)), "age": str(int(age)), "entity": entity}


def load_ref(lines, fields=FIELDS):
    """= 原代码 _load_ref（A2：实体按写法排序后加）。lines 由调用方按住院号排好。返回 ref_dist[icd]。"""
    ref_dist = defaultdict(lambda: ({"sex": Counter(), "age": Counter()} | {i: Counter() for i in fields}))
    for line in lines:
        ref_dist[line["icd"]]["sex"][line["sex"]] += 1
        ref_dist[line["icd"]]["age"][int(line["age"]) // 5] += 1
        for i in fields:
            ref_dist[line["icd"]][i].update(sorted(set(sum(line["entity"][i].values(), []))))
    for icd in ref_dist:
        total = sum(ref_dist[icd]["sex"].values())
        for i in ref_dist[icd]:
            for j in ref_dist[icd][i]:
                ref_dist[icd][i][j] /= total
    return ref_dist


def allocate_samples(prob_dist, n):
    """= 原代码 allocate_samples（最大余数法）。"""
    allocation = Counter()
    residuals = {}
    current_total = 0
    for x_i, p_i in prob_dist.items():
        ideal_count = n * p_i
        allocation[x_i] = int(ideal_count)
        residuals[x_i] = ideal_count - allocation[x_i]
        current_total += allocation[x_i]
    remaining = n - current_total
    for _ in range(remaining):
        x_max_residual = max(residuals, key=residuals.get)
        allocation[x_max_residual] += 1
        residuals[x_max_residual] -= 1
    return allocation


def extract_feature(emr, ref_icd, fields=FIELDS):
    """= 原代码 _extract_feature（A4：认"男""女"）。实体是"写法在该段文字里出现"（子串，否定的也算，原代码如此）。"""
    feature = {}
    feature["sex"] = None
    pi = emr["patient_info"].lower()
    if "male" in pi or "男" in pi:
        feature["sex"] = "male"
    if "female" in pi or "女" in pi:
        feature["sex"] = "female"
    m = re.search(r"\d+", emr["patient_info"])
    if m:
        feature["age"] = int(m.group(0)) // 5
    else:
        feature["age"] = None
    for i in fields:
        feature[i] = []
        for j in ref_icd[i]:
            if j in emr[i]:
                feature[i].append(j)
    return feature


def corpus_statistic(corpus, ref_icd, fields=FIELDS):
    """= 原代码 _get_corpus_statistic。"""
    count = {i: Counter() for i in ref_icd}
    inverse_map = {"sex": defaultdict(list), "age": defaultdict(list)} | {i: defaultdict(list) for i in fields}
    for i, x in enumerate(corpus):
        feature = extract_feature(x, ref_icd, fields)
        count["sex"][feature["sex"]] += 1
        count["age"][feature["age"]] += 1
        inverse_map["sex"][feature["sex"]].append(i)
        inverse_map["age"][feature["age"]].append(i)
        for j in fields:
            count[j].update(feature[j])
            for k in feature[j]:
                inverse_map[j][k].append(i)
    return count, inverse_map


def corpus_critic(draft, ref_icd, rng, fields=FIELDS, corrected=False):
    """= 原代码 _corpus_critic（A1 rng、A3 排序后连词）。
    draft：同一诊断的一批病历（内部格式），顺序由调用方定（按槽位号排好）。
    返回 (feedback, update_plan)：feedback 每份一个 {段: [反馈句]}，与原代码相同；update_plan 每份一个
    {"sex": {"add","remove"}, "age": {"add","remove"}, 段: {"add": set, "remove": set}}，给我们自己计数、构造地板用。"""
    count, inverse_map = corpus_statistic(draft, ref_icd, fields)
    expect_count = {}
    for i in ref_icd:
        tot = sum(ref_icd[i].values())
        dist = ref_icd[i]
        if corrected and i in fields and tot > 0:
            dist = Counter({k: v / tot for k, v in ref_icd[i].items()})
        expect_count[i] = allocate_samples(dist, round(len(draft) * tot))
    update_plan = [
        {"sex": {"add": None, "remove": None}, "age": {"add": None, "remove": None}}
        | {i: {"add": set(), "remove": set()} for i in fields}
        for _ in draft
    ]
    if count["sex"]["male"] > expect_count["sex"]["male"]:
        for i in rng.sample(inverse_map["sex"]["male"], count["sex"]["male"] - expect_count["sex"]["male"]):
            update_plan[i]["sex"]["add"] = "female"
            update_plan[i]["sex"]["remove"] = "male"
        for i in inverse_map["sex"][None]:
            update_plan[i]["sex"]["add"] = "female"
    elif count["sex"]["female"] > expect_count["sex"]["female"]:
        for i in rng.sample(inverse_map["sex"]["female"], count["sex"]["female"] - expect_count["sex"]["female"]):
            update_plan[i]["sex"]["add"] = "male"
            update_plan[i]["sex"]["remove"] = "female"
        for i in inverse_map["sex"][None]:
            update_plan[i]["sex"]["add"] = "male"
    else:
        rng.shuffle(inverse_map["sex"][None])
        for i in inverse_map["sex"][None][: expect_count["sex"]["male"] - count["sex"]["male"]]:
            update_plan[i]["sex"]["add"] = "male"
        for i in inverse_map["sex"][None][expect_count["sex"]["male"] - count["sex"]["male"]:]:
            update_plan[i]["sex"]["add"] = "female"

    candidate_age = []
    for i in expect_count["age"]:
        if expect_count["age"][i] > count["age"][i]:
            for j in range(expect_count["age"][i] - count["age"][i]):
                candidate_age.append(rng.randint(i * 5, i * 5 + 4))
    rng.shuffle(candidate_age)
    for i in count["age"]:
        if i is None:
            for j in inverse_map["age"][i]:
                update_plan[j]["age"]["add"] = candidate_age.pop()
        elif count["age"][i] > expect_count["age"][i]:
            for j in rng.sample(inverse_map["age"][i], count["age"][i] - expect_count["age"][i]):
                update_plan[j]["age"]["add"] = candidate_age.pop()
                update_plan[j]["age"]["remove"] = i

    for field in fields:
        candidate = []
        for i in expect_count[field]:
            if expect_count[field][i] > count[field][i]:
                candidate += [i] * (expect_count[field][i] - count[field][i])
        rng.shuffle(candidate)
        for i in count[field]:
            if count[field][i] > expect_count[field][i] + 1:
                for j in rng.sample(inverse_map[field][i], count[field][i] - expect_count[field][i]):
                    if candidate:
                        for k in candidate:
                            if k not in draft[j][field] and k not in update_plan[j][field]["add"]:
                                update_plan[j][field]["add"].add(k)
                                candidate.remove(k)
                                break
                    update_plan[j][field]["remove"].add(i)
        if candidate:
            index = list(range(len(draft)))
            rng.shuffle(index)
            for i in index:
                for j in candidate:
                    if j not in draft[i][field] and j not in update_plan[i][field]["add"]:
                        update_plan[i][field]["add"].add(j)
                        candidate.remove(j)
                        break
                if not candidate:
                    break

    feedback = []
    for i in update_plan:
        current_feedback = {}
        if i["sex"]["add"]:
            if i["sex"]["remove"]:
                current_feedback.setdefault("patient_info", []).append("Please modify the gender to “{}”".format(i["sex"]["add"]))
            else:
                current_feedback.setdefault("patient_info", []).append("Please add the gender “{}”".format(i["sex"]["add"]))
        if i["age"]["add"]:
            if i["age"]["remove"]:
                current_feedback.setdefault("patient_info", []).append("Please modify the age to “{}”".format(i["age"]["add"]))
            else:
                current_feedback.setdefault("patient_info", []).append("Please add the age “{}”".format(i["age"]["add"]))
        for field in fields:
            if i[field]["add"]:
                current_feedback.setdefault(field, []).append(ADD_PREFIX + "、".join(sorted(i[field]["add"])))
            if i[field]["remove"]:
                current_feedback.setdefault(field, []).append(REMOVE_PREFIX + "、".join(sorted(i[field]["remove"])))
        feedback.append(current_feedback)
    return feedback, update_plan


def round_rng(row, code, round_no):
    """每一行、每个诊断、每一轮一个随机数发生器（字符串种子，与哈希种子无关）。"""
    return random.Random(f"20261028|{row}|{code}|{round_no}")


# ---------- 我们自己用的：性别年龄直接写回、"配额照单全收"地板 ----------

def apply_demog(care7, plan_entry):
    """性别、年龄按计划直接写回（同关1 G1-E 伙计代码 apply_plan：年龄写成整数字符串，没有"岁"）。"""
    after = dict(care7)
    if plan_entry["sex"]["add"]:
        after["Gender"] = SEX_TO_CN[plan_entry["sex"]["add"]]
    if plan_entry["age"]["add"] is not None:
        after["Age"] = str(plan_entry["age"]["add"])
    return after


CARE7_OF = {"chief_complaint": "Chief Complaint", "history_of_present_illness": "History of Present Illness",
            "hospital_course": "Hospital Course", "discharge_instructions": "Discharge Instructions"}


def apply_ideal(care7, plan_entry, fields=FIELDS):
    """F-QOK"配额照单全收"：一个完全听话、但只会照单抄词的 Reviser。
    要删的词：从该段删掉它的每一处；要加的词：在该段末尾加一句"词1、词2。"。性别、年龄同 apply_demog。"""
    after = apply_demog(care7, plan_entry)
    for f in fields:
        k = CARE7_OF[f]
        text = after[k]
        for w in sorted(plan_entry[f]["remove"], key=lambda x: (-len(x), x)):
            text = text.replace(w, "")
        if plan_entry[f]["add"]:
            text = text + "、".join(sorted(plan_entry[f]["add"])) + "。"
        after[k] = text
    return after


def run_ideal(drafts, ref_icd, row, code, rounds=2, fields=FIELDS, corrected=False):
    """F-QOK：对同一诊断的一批 7 字段病历（[(槽位, 病历)]，按槽位排好）跑 rounds 轮语料层，每轮 apply_ideal。
    corrected=True 是 F-QOKc（期望份数按 n × p）。返回 ([(槽位, 病历)], 每轮的 update_plan 列表)。"""
    cur = list(drafts)
    plans = []
    for r in range(rounds):
        fb, plan = corpus_critic([to_internal(x) for _, x in cur], ref_icd, round_rng(row, code, r), fields, corrected)
        plans.append(plan)
        cur = [(sid, apply_ideal(x, p, fields)) for (sid, x), p in zip(cur, plan)]
    return cur, plans


def parse_keywords(sentence):
    """反馈句 → (add/remove, 词集合)。认不出返回 (None, set())。"""
    if sentence.startswith(ADD_PREFIX):
        return "add", set(sentence[len(ADD_PREFIX):].split("、"))
    if sentence.startswith(REMOVE_PREFIX):
        return "remove", set(sentence[len(REMOVE_PREFIX):].split("、"))
    if sentence.startswith("Add the following keywords"):
        return "add", set(re.split(r"[:：]\s*", sentence, 1)[1].split("、"))
    if sentence.startswith("Remove the following keywords"):
        return "remove", set(re.split(r"[:：]\s*", sentence, 1)[1].split("、"))
    return None, set()


def expected_per_entity(ref_icd, n, fields=FIELDS, corrected=False):
    """每段、每个参考实体在 n 份里"应该出现"的份数（原代码的算式；corrected=True 按 n × p）。R6-0 描述用。"""
    out = {}
    for f in fields:
        tot = sum(ref_icd[f].values())
        dist = Counter({k: v / tot for k, v in ref_icd[f].items()}) if corrected and tot > 0 else ref_icd[f]
        out[f] = allocate_samples(dist, round(n * tot))
    return out
