"""关3：ruler_g3 的硬错误和具体度。g3（0.3）和 g3old（0.2）共用这份代码，靠子进程的 sys.path 区分版本。"""
from ruler_g3 import RULES_VERSION, evaluate, from_admission, from_care7, summarize

FIELDS = ("any_hard", "any_severe", "any_pedadult", "any_sex", "any_warning", "any_n1", "n_mentions", "n_with_dose")


def run(rows, config):
    per = []
    for x in rows:
        rec = from_care7(x["record"], x["id"]) if x["kind"] == "care7" else from_admission(
            {"admission_id": x["id"], "docs": x["docs"]}, x["id"])
        s = summarize(evaluate(rec))
        row = {"row": x["row"], "id": x["id"]}
        row.update({k: s[k] for k in FIELDS})
        per.append(row)
    return per, [{"row": "*", "metric": "rules_version_is_03", "value": RULES_VERSION.endswith("0.3")}]
