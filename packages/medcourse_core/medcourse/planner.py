"""L3 规划器 v0.2（S1）：不读真实文本，只读 Stats。抽病人，排逐日事件和文书日程。

属性从联合分布抽；检查集合按诊断和相对日抽；用药先抽药再抽别名、剂量、频次、途径，过全部用药规则；化验按分位数抽；诊断默认锁定。
每张表从最细的非空层往粗的退：主诊断名 → code3 → 章 → 全局。手术只取主诊断名和 code3 两层，取不到就不做手术，不借别的病的术式。
合并症、鉴别诊断、科室都过性别和年龄；手术侧别跟着诊断走；既往史和诊断清单一致。
配对模式把槽位的性别、年龄、科室、诊断、住院天数、手术日当条件，其余照抽，日程从 Journey 推，不借真实住院的骨架。

0.1.8（第七次合成病历审阅、监督员第七轮）：分层多一层 ICD-10 节（主诊断名 → code3 → 节 → 章 → 全局）。
主诉从主诉症状表按出现频率抽 1–2 个（0.1.7 按字码排序取前两个，"发热"总排在"胸痛""腹痛"前面）；伴随症状照旧逐项抽，
按频率排在主诉后面。用药退到章层为止，再没有就不开药（0.1.7 退到全局层，4 岁言语发育障碍开了美托洛尔）。
麻醉按术式关键词定（0.1.7 随机，冠脉造影写了硬膜外麻醉）。没有数值的检查：在本诊断层的出现率至少是全局层两倍的算"诊断相关"，
写"提示＋诊断名"，并且一定异常；其余写"未见明显异常"（0.1.7 写"可见与主要诊断相符的异常改变"，异常与否随机，
确诊胆囊结石的腹部超声却"未见明显异常"）。

0.1.9（第八次合成病历审阅，老板批准的十条）：
- 主诉：主诉症状表最多退到节层（0.1.8 退到章层，痔编码在循环系统章，抽到了"胸闷""头晕"）。化疗、放疗、复查这类住院
  （TREAT_DX）不用症状，写"确诊××N月，为行第N次化疗"；其余节层以内没有主诉表的，用节层以内伴随症状里频率最高的一个，
  再没有照 0.1.8 写"不适"（不按病名补症状）。伴随症状也只到节层。来源记在 chief.source，plan_diag 计 level_chief_*。
  0.1.9.1：写"不适"之前，章和诊断的器官系统一致时可以用章层的主诉表（chief.source = chapter_organ）。
- 检查："诊断相关"排除常规化验和心电图，只在节层以内比出现率（0.1.8 在章层比，"尿常规提示冠心病"）；检查的部位要和诊断所在的
  器官系统对得上（审查员第八轮补充：脑梗死写了"心脏超声提示脑梗死"。器官从 ICD-10 的分类结构推，检查部位用评测台的部位词表）；
  一次住院一项都没抽到时补出现率比值最高的那项。结果按检查类型写：影像、内镜写"符合××表现"，培养写"培养出／未培养出致病菌"，
  病理、活检写"符合××"，化验只给数值（0.1.8 写"提示××"，写手写成"示提示"；病理写"与术前诊断相符"）。
  没写部位的 CT、超声按诊断器官补部位，补完重复的删掉；有创检查只从节层以内抽、同类一次住院一项；退到章层只抽通用检查
  （这一层比全院出现率高不到 1.5 倍的，审查员第八轮补充和采纳的意见）。
- 化验：一次住院抽一次病情轻重，各项按诊断层的异常比例一起偏（高斯 copula，单项的异常比例不变）；异常值在真实分布的尾部里取，
  这一层异常少见的只取轻度；复查血常规以入院值为起点，往正常方向走（0.1.8 每项各抽各的，固定 25% 异常，复查和入院无关）。
- 用药：先按频率抽药物大类，主治的一类一定要（本诊断层用得够多的大类里、出现率比全院高得最多的，Lift；采纳审查员的意见，
  按频率取会取到止吐药），再在大类里抽药；同类两种药按真实比例；长期用药住院期间不停、出院带药；主治的一类至少用到住院六成的日子
  （0.1.8 肺炎的布地奈德第 1 天就停）。
- 出院当天的生命体征回到正常范围（0.1.8 成人出院 85/65 写"生命体征平稳"）。儿童先按统计表本年龄段的体征抽，没有的项
  14 岁以下按儿科正常范围（审查员第八轮补充：3 岁患儿 112/50mmHg）。
- 评测台检查词表里有、这里漏了的"粪常规""血气""肌钙蛋白""分泌物培养"补进 EXAM_CLS。

0.1.11（第十次合成病历审阅、审查员补充，老板批准的 14 条，见 README）：鉴别诊断同器官；检查并进手术、取组织才开病理；
长期药要适应证、同类不同亚类；化验分位数按节 → 章，14 岁以下化验和用药取儿童层，糖化血红蛋白和血糖异常要有糖尿病；
诊断相关检查比值线放宽到 1.2；检查结论用类目名；阳性培养不在出院当天回报；记下写"不适"的原因。各处注释标 0.1.11。

0.1.7：配对模式的槽位写明了术式（procedure，S3 探针用公开知识写的虚构槽位）时照用，不从统计表取，侧别按术式名读。

0.1.6（监督员第五轮）：术式在主诊断名、code3 两层取不到时，再看章层、全局层，但只取诊断关键词对得上的大类（FAMILY_DX，
诊断带"疝"才配疝修补）。分侧术式要配分侧诊断：诊断没写侧别、又和术式是同一个器官时，把术式的侧别加进诊断，S3 才能测。
产科诊断名的孕周、孕产次占位符按名字填，gest_weeks 从名字里读；剖宫产不给 28 周以前，清宫不给 13 周以后。

0.1.5：没有一个 code3 凑够 k 次的章退到章层主干；检查补齐本层不够往粗层取；无数值的检查结果分正常、异常写；第 0 天的病理写成外院病理；
病程时长不超过年龄。

0.1.4：code3 抽到了但没有一个诊断名凑够 k 次时，0.1.3 整章跳过（外科、产科章因此一例没有）；现在依次退到诊断主干、
公开词典的标准名，最后才是章层的诊断名。术式没有凑够 k 次的写法时用大类的常用名；配对模式取不到术式就不做手术，不再拼"手术治疗"。
检查先逐项抽，不够这次住院抽到的种类数（真实分布）就按权重补齐。症状逐日减少，不再前半程天天一样。
"""
import datetime as dt
import hashlib
import math
import random
import re
from statistics import NormalDist

from . import icd, kb
from .contract import CONTRACT_VERSION, validate_docspecs, validate_journey
from .drug_class import antitumor_ok, drug_class, indicated, is_chronic, oral_ok, subclass
from .rules import setup
from .stats import (AGE_BANDS, LAB_NAME, LAB_UNIT, PAST_TERMS, PLAUSIBLE, PROC_FAMILY_NAME, REF, STATS_VERSION,
                    los_bucket, past_matches, proc_family, prog_day_bucket)

setup(("BENCH_G3", "BENCH_G2"))
from g2_kit.gen2 import doc_times, slot_filter  # noqa: E402
from ruler_g2.events import _REGION as BENCH_REGION  # noqa: E402  评测台的部位词表（部位写法 → 部位代码）
from ruler_g3 import evaluate  # noqa: E402
from ruler_g3.adapters import Record  # noqa: E402
from ruler_g3.lexicon import weight_p50  # noqa: E402

PANELS = {"血常规": ("WBC", "NEUT", "HGB", "PLT"), "肝功能": ("ALT", "AST", "TBIL", "ALB"), "肾功能": ("CREA", "BUN", "UA"), "电解质": ("K", "NA", "CL", "CA"),
          "血糖": ("GLU",), "凝血": ("PT", "APTT", "INR", "FIB"), "心肌酶": ("CK", "CKMB", "LDH"), "C反应蛋白": ("CRP",), "降钙素原": ("PCT",), "D二聚体": ("DD",),
          "心肌肌钙蛋白": ("TNI",), "肌钙蛋白": ("TNI",), "利钠肽": ("BNP",), "糖化血红蛋白": ("HBA1C",), "血脂": ("TC", "TG", "LDL"), "淀粉酶": ("AMY",)}
# 文献常用参考范围 REF（公开知识）0.1.9 挪到 stats.py，这里照旧可用 planner.REF
DECIMALS = {"WBC": 2, "HGB": 0, "PLT": 0, "NEUT": 1, "K": 2, "NA": 1, "CL": 1, "CA": 2, "GLU": 2, "CREA": 0, "BUN": 2, "UA": 0, "ALT": 0, "AST": 0, "TBIL": 1, "ALB": 1, "CRP": 1,
            "PCT": 2, "PT": 1, "APTT": 1, "INR": 2, "FIB": 2, "DD": 2, "CK": 0, "CKMB": 0, "LDH": 0, "TNI": 3, "BNP": 0, "HBA1C": 1, "TC": 2, "TG": 2, "LDL": 2, "AMY": 0}
VITAL_DEFAULT = {"adult": {"T": [36.3, 36.5, 36.6, 36.8, 37.5], "P": [66, 74, 80, 88, 104], "R": [16, 18, 19, 20, 22], "SBP": [105, 118, 129, 142, 165], "DBP": [62, 70, 78, 85, 98]},
                 "child": {"T": [36.3, 36.5, 36.7, 37.0, 38.0], "P": [76, 88, 100, 114, 136], "R": [18, 20, 24, 28, 36], "SBP": [85, 92, 100, 108, 120], "DBP": [50, 56, 62, 68, 76]}}
# 0.1.9：补上评测台词表里有、这里漏了的"粪常规""血气""肌钙蛋白""分泌物培养"（0.1.8 归成"其他"，会被当成能"提示诊断"的检查）
EXAM_CLS = {"检验": ("血常规", "尿常规", "便常规", "粪常规", "肝功能", "肾功能", "电解质", "血糖", "血脂", "凝血", "D二聚体", "血气", "利钠肽", "心肌肌钙蛋白", "肌钙蛋白", "心肌酶",
                   "C反应蛋白", "降钙素原", "糖化血红蛋白", "淀粉酶", "血沉", "肿瘤标志物", "甲状腺功能"),
            "影像": ("CT", "磁共振", "超声", "心脏超声", "X线", "MRCP", "PET"), "功能": ("心电图", "动态心电图", "动态血压", "肺功能", "脑电图", "肌电图", "骨密度", "睡眠监测"),
            "内镜": ("胃镜", "肠镜", "支气管镜", "喉镜", "膀胱镜", "超声内镜", "ERCP", "胶囊内镜"), "造影": ("冠脉造影", "脑血管造影"), "穿刺": ("骨髓穿刺", "腰椎穿刺"),
            "培养": ("血培养", "痰培养", "尿培养", "脓液培养", "分泌物培养"), "病理": ("病理",), "取材": ("活检",)}
KIND_OF = {k: c for c, ks in EXAM_CLS.items() for k in ks}
# 0.1.9（审查员第八轮补充）：检查看的是哪个器官，诊断在哪个器官。"诊断相关"要两边对得上（第八次：脑梗死写了"心脏超声提示脑梗死"
# "泌尿系超声提示脑梗死"，慢性肾脏病写了"心脏超声提示慢性肾脏病"）。公开解剖常识，不来自病历。
REGION_ORGAN = {"头": "脑", "颈": "颈", "胸": "肺", "心": "心", "腹": "腹", "盆": "盆", "泌尿": "泌尿", "脊柱": "脊柱", "四肢": "骨", "乳腺": "乳腺", "血管": "血管"}
KEY_ORGAN = {"心脏超声": "心", "动态心电图": "心", "动态血压": "心", "冠脉造影": "心", "MRCP": "腹", "超声内镜": "腹", "ERCP": "腹", "胃镜": "腹", "肠镜": "腹",
             "胶囊内镜": "腹", "肺功能": "肺", "睡眠监测": "肺", "支气管镜": "肺", "痰培养": "肺", "脑电图": "脑", "脑血管造影": "脑", "腰椎穿刺": "脑",
             "肌电图": "神经", "骨密度": "骨", "喉镜": "咽喉", "膀胱镜": "泌尿", "尿培养": "泌尿", "骨髓穿刺": "血液", "血培养": "全身", "PET": "全身"}
# 诊断所在的器官系统从 ICD-10 的分类结构推（公开分类，不按病名写规则）：先查三位码（只有良性肿瘤 D10–D36 这一节按部位分到三位码），
# 再查节，再查章。诊断没有三位码时（章层诊断名、化疗住院的原发病）用公开词典按名字查三位码
ORGAN_ORDER = ("脑", "颈", "肺", "心", "腹", "泌尿", "盆", "乳腺", "脊柱", "骨", "血管", "血液", "咽喉", "神经", "全身")
CHAPTER_ORGAN = {"A": "全身", "B": "全身", "F": "脑", "G": "脑", "I": "心", "J": "肺", "K": "腹", "M": "骨", "N": "泌尿", "O": "盆"}
BLOCK_ORGAN = {"A15-A19": "肺", "C00-C14": "咽喉", "C15-C26": "腹", "C30-C39": "肺", "C40-C41": "骨", "C50-C50": "乳腺", "C51-C58": "盆", "C60-C63": "泌尿",
               "C64-C68": "泌尿", "C69-C72": "脑", "C73-C75": "颈", "C81-C96": "血液", "D50-D53": "血液", "D55-D59": "血液", "D60-D64": "血液",
               "D65-D69": "血液", "D70-D77": "血液", "E00-E07": "颈", "G50-G59": "神经", "G60-G64": "神经", "G70-G73": "神经",
               "H60-H62": "咽喉", "H65-H75": "咽喉", "I60-I69": "脑",
               "I70-I79": "血管", "I80-I89": "血管", "J00-J06": "咽喉", "J30-J39": "咽喉", "K00-K14": "咽喉",
               "M40-M43": "脊柱", "M45-M49": "脊柱", "M50-M54": "脊柱",
               "N60-N64": "乳腺", "N70-N77": "盆", "N80-N98": "盆",
               "S00-S09": "脑", "S10-S19": "颈", "S20-S29": "肺", "S30-S39": "腹"}
BLOCK_ORGAN.update({f"S{a}-S{b}": "骨" for a, b in (("40", "49"), ("50", "59"), ("60", "69"), ("70", "79"), ("80", "89"), ("90", "99"))})
CODE3_ORGAN = {"D10": "咽喉", "D12": "腹", "D13": "腹", "D14": "肺", "D15": "肺", "D16": "骨", "D24": "乳腺", "D25": "盆", "D26": "盆", "D27": "盆",
               "D28": "盆", "D29": "泌尿", "D30": "泌尿", "D32": "脑", "D33": "脑", "D34": "颈"}
# 0.1.9（审查员第八轮补充）：有创检查只从本诊断的节层以内抽，不拿粗层的当凑数项；一次住院同类有创检查最多一项
INVASIVE = ("穿刺", "取材", "病理", "造影", "内镜")   # 评测台检查词表的类别（ruler_g2/config/exams.json 的 cls）
COARSE_LIFT = 1.5
BARE_REGION = {"脑": "头", "颈": "颈", "肺": "胸", "心": "胸", "腹": "腹", "泌尿": "泌尿", "盆": "盆", "乳腺": "乳腺", "脊柱": "脊柱", "骨": "四肢", "咽喉": "颈"}
REGIONAL_KEYS = ("CT", "磁共振", "超声", "X线")
_MODALITY_DEFAULT_REGION = {"CT": "胸", "X线": "胸", "超声": "腹", "磁共振": "头"}
_NAME_ORGAN_KW = (
    ("脑", ("脑", "颅内", "脑膜", "胶质")),
    ("脊柱", ("颈椎", "胸椎", "腰椎", "骶椎", "椎间盘", "脊柱", "椎管")),
    ("咽喉", ("鼻", "咽", "喉", "扁桃体", "腺样体", "口腔", "舌", "颌", "腮腺", "耳")),
    ("肺", ("肺", "支气管", "胸膜", "纵隔")),
    ("腹", ("胃", "结肠", "直肠", "肠", "肝", "胆", "胰", "食管", "腹膜", "阑尾", "脾")),
    ("乳腺", ("乳腺", "乳房")),
    ("颈", ("甲状腺", "颈部")),
    ("盆", ("子宫", "宫颈", "卵巢", "输卵管", "盆腔", "阴道", "外阴")),
    ("泌尿", ("肾", "输尿管", "膀胱", "前列腺", "尿道")),
    ("血液", ("白血病", "淋巴瘤", "骨髓瘤", "骨髓增生")),
    ("骨", ("股骨", "胫骨", "腓骨", "肱骨", "桡骨", "尺骨", "锁骨", "关节", "四肢")),
)


def exam_organ(key, region):
    """检查看的器官。部位按评测台的部位词表归一（"胸部"→"胸"）。CT、超声这类没写部位的返回 None（看不出部位，不当诊断相关）。"""
    if region:
        return REGION_ORGAN.get(region) or REGION_ORGAN.get(BENCH_REGION.get(region))
    return KEY_ORGAN.get(key)


def code_organ(code3):
    if not code3:
        return None
    if code3 in CODE3_ORGAN:
        return CODE3_ORGAN[code3]
    blk = icd.block_of(code3) if len(code3) == 3 else None
    return BLOCK_ORGAN.get(blk[4:] if blk else None) or CHAPTER_ORGAN.get(code3[:1])


def dx_organs(name, code3):
    """诊断所在的器官系统（集合，可能为空）：三位码 → 节 → 章；没有三位码、或三位码是 Z 章（化疗这类住院）时，
    用公开词典按诊断名查三位码再推。若仍缺具体器官，再按显式解剖关键词兜底。恶性肿瘤另加"全身"。"""
    codes = [code3] if code3 and len(code3) == 3 and not code3.startswith("Z") else []
    if not codes:
        got = icd.code3_exact(name) or icd.code3_stem(name) if name else None
        codes = [got] if got else ([code3] if code3 and not code3.startswith("Z") else [])
    out = {o for o in (code_organ(c) for c in codes) if o}
    if not (out - {"全身"}) and name:
        for o, kws in _NAME_ORGAN_KW:
            if _has_any(name, kws):
                out.add(o)
    if any(c[:1] == "C" for c in codes):
        out.add("全身")
    return out


def site_ok(key, region, organs):
    o = exam_organ(key, region)
    return o is not None and (o in organs or (o == "全身" and "全身" in organs))
# 统计表里的部位是读数器归一后的代码（"胸""腹"），写成检查名要换回病历写法；换回的写法读数器仍归到同一代码
REGION_NAME = {"胸": "胸部", "腹": "腹部", "头": "头颅", "心": "心脏", "泌尿": "泌尿系", "盆": "盆腔", "颈": "颈部"}
SCHED = {"影像": 1, "内镜": 2, "造影": 2, "穿刺": 1, "培养": 3, "病理": 4}
SUP_TYPES = (("主治医师查房记录", 0.45), ("主任医师查房记录", 0.35), ("上级医师查房记录", 0.20))
SUP_SUBTYPE_CN = {"主治医师查房": "主治医师查房记录", "主任医师查房": "主任医师查房记录", "上级医师查房": "上级医师查房记录"}
P_DX_CHANGE = 0.30     # 真实诊断变化步数约 0.37，默认锁定，只有约三成走一次疑似到确诊
P_DIFF = 0.5
P_DIFF_CLOSE = 0.3
P_DS_ONLY = 0.55       # 0.21：无 real_rates 兜底时非慢病合并症出院新增比例（真实 fold 2–7 次要诊断约 56% 为入院未列）
MAX_COMORB = 5
_DX_GRADE_TAIL = re.compile(r"(?:[（(][^）)]{0,12}[）)])?(?:NYHA\s*)?(?:心功能)?[IⅠⅡⅢⅣV1-5一二三四五]+(?:级|期)|(?:极高|很高|高|中|低)危(?:组)?")


def _dx_grade_stem(dx_name):
    s = re.sub(r"[\s，,、；;（）()]", "", dx_name or "")
    if not _DX_GRADE_TAIL.search(s):
        return None
    base = _DX_GRADE_TAIL.sub("", s)
    return base if len(base) >= 2 else ("心功能分级" if "心功能" in s else None)


def _ds_only_prob(stats):
    rr = stats.get("real_rates") or {}
    adm = rr.get("hr_ds_dx_admissions") or 0
    tot = rr.get("hr_ds_dx_items") or 0
    new = rr.get("hr_ds_new_dx_items") or 0
    if adm > 0 and tot > adm:
        return min(0.72, max(0.35, new / (tot - adm)))
    return P_DS_ONLY
# 没有数值的检查写成异常的概率（按检查大类给的经验值，不来自病历）：影像、内镜、病理多是为本病做的，功能检查多为常规
P_EXAM_ABNORMAL = {"影像": 0.6, "内镜": 0.6, "造影": 0.6, "病理": 0.9, "功能": 0.3}
_DUR = re.compile(r"(\d+)\s*(?:个)?(小时|天|日|周|月|年)")
_DUR_YEARS = {"小时": 1 / 8760, "天": 1 / 365, "日": 1 / 365, "周": 1 / 52, "月": 1 / 12, "年": 1}


def _has_any(text, words):
    t = text or ""
    return any(w in t for w in words)


def duration_ok(text, age):
    """病程时长不能比年龄还长：章层的时长表成人儿童混在一起，0.1.4 给儿童配过成人的时长（b7_AGE-HIST）。"""
    m = _DUR.search(text or "")
    return not m or int(m.group(1)) * _DUR_YEARS[m.group(2)] <= max(age, 1 / 365)


_INFANT_WORDS = ("哭闹", "拒奶", "吐奶", "吃奶", "奶量", "呛奶", "溢奶", "囟门")
_CHILD_DEV_WORDS = ("生长发育", "语言发育", "言语发育", "语言落后", "运动落后", "遗尿")
_FEMALE_SYM_WORDS = ("停经", "胎动", "见红", "破水", "宫缩", "阴道流血", "阴道出血", "阴道流液", "白带", "痛经", "月经")
_PREG_SYM_WORDS = ("停经", "胎动", "见红", "破水", "宫缩")
_ADULT_ONLY_SYM_WORDS = ("胸骨后压榨", "绝经", "前列腺")


def sym_ok(sym, p):
    """0.21.2 / 0.3.2 / 0.3.4：混龄/混性别/跨孕产层与早孕胎动耦合校验。"""
    s = sym or ""
    if p["age"] >= 6 and _has_any(s, _INFANT_WORDS):
        return False
    if p["age"] >= CHILD_AGE and _has_any(s, _CHILD_DEV_WORDS):
        return False
    if p["age"] < 10 and _has_any(s, _ADULT_ONLY_SYM_WORDS):
        return False
    if p["sex"] == "男" and _has_any(s, _FEMALE_SYM_WORDS):
        return False
    if (p["age"] < 12 or p["age"] >= 55) and _has_any(s, _PREG_SYM_WORDS):
        return False
    if p.get("pregnant") and p.get("gest_weeks") is not None:
        gw = p["gest_weeks"]
        if gw < 18 and "胎动" in s:
            return False
        if gw < 20 and _has_any(s, ("见红", "宫缩")):
            return False
    return True


PREG_NAME = re.compile(r"妊娠|孕\d|宫内孕|分娩|临产")
PREG_AGE = (15, 49)
# 章层和全局层把成人和儿童混在一起：这些病不给 18 岁以下 / 成人不给儿童病
ADULT_DX = re.compile(
    r"高血压|2型糖尿病|冠心病|冠状动脉|心肌梗死|脑梗|脑出血|脑卒中|脑血栓|蛛网膜下腔出血|"
    r"帕金森|阿尔茨海默|慢性阻塞|慢阻肺|肺源性心脏病|肺心病|房颤|心房颤动|痛风|前列腺|前列腺增生|"
    r"子宫肌瘤|绝经|动脉硬化|高脂血症|骨质疏松|老年"
)
_CHILD_DX_WORDS = (
    "发育障碍", "发育迟缓", "发育落后", "孤独症", "自闭症", "多动障碍", "小儿", "婴幼儿", "新生儿",
    "黏膜皮肤淋巴结综合征", "皮肤黏膜淋巴结综合征", "川崎病", "手足口病", "毛细支气管炎", "疱疹性咽峡炎",
    "幼年", "少儿", "早产儿", "足月儿", "母乳性黄疸", "佝偻病", "热性惊厥", "高热惊厥", "咽结合膜热", "百日咳", "遗尿症", "抽动障碍", "脐炎",
)
NON_EVOLVING_SYMS = {"停经", "妊娠", "待产"}
_LESION_WORDS = ("肿物", "肿块", "结节", "占位", "息肉", "囊肿", "肌瘤", "腺瘤", "白内障", "疝")
_OBSTETRIC_WORDS = ("待产", "临产", "顺产", "分娩", "活产", "剖宫产", "保胎", "先兆早产")
_ABORTIFACIENT_WORDS = ("米非司酮", "米索前列醇", "依沙吖啶", "利凡诺")
# 0.1.6：术式大类的适应证关键词（公开临床常识，不来自病历）。章层、全局层的术式只给关键词对得上的诊断；
# 不在表里的大类（气管切开）没有固定的适应证写法，仍只在主诊断名、code3 两层取
FAMILY_DX = {
    "胆囊切除": r"胆囊|胆石|胆结石", "胆道取石": r"胆总管|胆管.{0,4}结石|胆道结石", "阑尾切除": r"阑尾",
    "腹股沟疝修补": r"腹股沟.{0,4}疝", "脐疝修补": r"脐疝", "切口疝修补": r"切口疝",
    "剖宫产": r"待产|临产|足月|分娩|瘢痕子宫|胎位|臀位|头盆不称|胎儿窘迫|巨大儿|前置胎盘|胎膜早破|子痫|孕(?:2[89]|3\d|4[0-2])",
    "髋关节置换": r"股骨颈骨折|股骨头.{0,4}坏死|髋关节", "股骨头置换": r"股骨颈骨折|股骨头", "膝关节置换": r"膝关节|膝.{0,2}骨.{0,2}关节",
    "骨折内固定": r"骨折", "冠脉介入": r"冠心病|冠状动脉|心肌梗死|心绞痛|冠脉", "冠脉造影": r"冠心病|冠状动脉|心肌梗死|心绞痛|冠脉",
    "甲状腺切除": r"甲状腺.{0,4}(?:结节|肿|瘤|癌)|结节性甲状腺", "子宫肌瘤剔除": r"子宫.{0,4}肌瘤|子宫平滑肌瘤",
    "子宫切除": r"子宫.{0,4}(?:肌瘤|腺肌|内膜癌|脱垂)|宫颈癌|子宫恶性", "结肠癌根治": r"结肠.{0,4}(?:癌|恶性)", "直肠癌根治": r"直肠.{0,4}(?:癌|恶性)",
    "胃癌根治": r"胃.{0,4}(?:癌|恶性)", "胃大部切除": r"胃.{0,4}溃疡|十二指肠.{0,4}溃疡|胃.{0,2}穿孔",
    "泌尿系碎石": r"(?:输尿管|肾|尿路|膀胱|泌尿系).{0,4}结石", "前列腺电切": r"前列腺增生|前列腺肥大", "白内障": r"白内障",
    "椎间盘": r"椎间盘|椎管狭窄", "扁桃体切除": r"扁桃体", "大隐静脉": r"大隐静脉|下肢静脉曲张", "痔": r"痔",
    "肠镜息肉": r"(?:结肠|直肠|结直肠|大肠|肠).{0,4}息肉", "清宫": r"不全流产|稽留流产|难免流产|过期流产|胚胎停|葡萄胎|宫腔.{0,2}残留",
    "卵巢囊肿": r"卵巢.{0,4}(?:囊肿|肿物|畸胎瘤|囊腺瘤|肿瘤)", "乳腺癌根治": r"乳腺.{0,4}(?:癌|恶性)|乳房.{0,4}恶性",
    "乳腺肿物": r"乳腺.{0,4}(?:肿物|肿块|纤维腺瘤|结节)|乳房.{0,4}肿物", "开颅血肿": r"脑出血|颅内.{0,2}血肿|硬膜[下外].{0,2}血肿|颅内出血",
    "肺叶切除": r"肺.{0,4}(?:癌|恶性|结节|肿物|占位)",
}
FAMILY_DX_RE = {f: re.compile(rx) for f, rx in FAMILY_DX.items()}
# 骨折内固定的名字跟着诊断写（"左侧桡骨远端骨折切开复位内固定术"），大类常用名"{S}骨折切开复位内固定术"没有部位
FAMILY_FROM_DX = {"骨折内固定": "切开复位内固定术"}
# 儿童腹股沟疝做疝囊高位结扎，不做无张力修补
FAMILY_CHILD_NAME = {"腹股沟疝修补": "腹腔镜下{S}腹股沟斜疝疝囊高位结扎术"}
# 成对器官（评测台侧别读数器官表里成对的那些，公开解剖知识）。侧别加在诊断里第一个器官词前面；单字器官写"右肺""左肾"，其余写"左侧"。
# "肾脏病""肾功能不全"说的不是一侧的病变，不算
LAT_ORGAN = re.compile(r"腹股沟|髋关节|髋|膝关节|膝|股骨|胫腓骨|胫骨|腓骨|肱骨|尺桡骨|尺骨|桡骨|锁骨|踝|肘|腕|肩|乳腺|乳房|甲状腺|下肢|大隐静脉|输尿管|卵巢|肺|眼|肾(?!脏病|功能|病|衰|小球|炎|性)")
_BARE_SIDE = {"肺", "眼", "肾", "膝", "髋", "肩", "肘", "腕", "踝"}


class StatsVersionError(ValueError):
    pass


def require_stats(stats):
    """0.2 的表结构变了（用药拆表、分层加 code3 和主诊断名），旧表直接报错，不静默降级。"""
    if stats.get("version") != STATS_VERSION:
        raise StatsVersionError("stats_version")


def seed_of(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _pick(rng, items):
    """items：{键: 计数}。"""
    keys = sorted(items)
    if not keys:
        return None
    return rng.choices(keys, weights=[items[k] for k in keys])[0]


def _table_where(stats, name, strata, keep):
    """同 _table，但先按 keep 过滤键；过滤后为空就继续往粗的层退。返回 items。"""
    t = stats.get(name) or {}
    for s in strata:
        items = {x: n for x, n in ((t.get(s) or {}).get("items") or {}).items() if keep(x)}
        if items:
            return items
    return {}


def strata_of(name, code):
    """从细到粗：主诊断名、code3、节（0.1.8）、章、全局。"""
    blk = icd.block_of(code) if len(code) == 3 else None
    return [f"n:{name}"] + ([code] if len(code) == 3 else []) + ([blk] if blk else []) + ([code[:1]] if code else []) + ["*"]


# 0.2.0（老板 11-11 批准，路线 B RB1 扩充）：强全麻词仍排最前（腹腔镜疝修补保持全麻，宫腔镜除外归静脉麻醉）；内瘘归局麻；
# 部位词排到"切除、成形、摘除"这类泛动作词前面（血栓性外痔切除不再被"切除"带成全麻）；未命中不再默认全麻（返回 ""）
ANESTHESIA_RULES = (
    (re.compile(r"(?<!宫)腔镜|开颅|开胸|剖腹|根治|置换|移植|搭桥"), "全身麻醉"),
    (re.compile(r"内瘘|造影|支架|介入|消融|穿刺|置管|活检|胃镜|肠镜|内镜|支气管镜|起搏器|球囊|射频|PCI|白内障|晶状体|人工晶体|清创|缝合|拔除|注射|引流"), "局部麻醉"),
    (re.compile(r"清宫|刮宫|人工流产|宫腔镜"), "静脉麻醉"),
    (re.compile(r"剖宫产|疝|下肢|大隐静脉|肛|痔|经尿道|膀胱|前列腺|髋|膝|股骨|胫|腓|踝|足|输尿管"), "腰硬联合麻醉"),
    (re.compile(r"切除|成形|重建|摘除|剥除|剔除|切开|松解|吻合|修补|甲状腺|乳腺|骨折|内固定"), "全身麻醉"),
)


def anesthesia_for(proc_name):
    for rx, a in ANESTHESIA_RULES:
        if rx.search(proc_name or ""):
            return a
    return ""


# 0.1.11（第十次 M1）：检查名在术式名里（公开的同义写法另列），这项检查就是这台手术本身
EXAM_IN_PROC = {"冠脉造影": ("冠脉造影", "冠状动脉造影"), "肠镜": ("肠镜",), "胃镜": ("胃镜",), "支气管镜": ("支气管镜",), "膀胱镜": ("膀胱镜",),
                "喉镜": ("喉镜",), "超声内镜": ("超声内镜",), "ERCP": ("ERCP", "逆行胰胆管"), "脑血管造影": ("脑血管造影", "DSA"),
                "骨髓穿刺": ("骨髓穿刺",), "腰椎穿刺": ("腰椎穿刺",)}


def proc_has_exam(proc_name, key):
    return any(w in (proc_name or "") for w in EXAM_IN_PROC.get(key, ()))


# 0.1.11（第十次 M2）：取了组织的手术才开术后病理。显式切除/活检/取材类开病理；碎石/取石/内固定/修补/造影/剖宫产等不单独取组织，不开术后病理。
_TISSUE_WORDS = ("切除", "摘除", "活检", "清宫", "刮宫", "剔除", "电切", "剥除", "取材", "息肉", "外剥内扎", "根治", "截肢")
_NO_TISSUE_PROC_WORDS = ("碎石", "取石", "内固定", "复位", "修补", "成形", "松解", "剖宫产", "置换", "引流", "造口", "环扎", "置管", "搭桥", "介入", "造影", "消融")


def takes_tissue(a):
    nm = (a or {}).get("name") or ""
    if _has_any(nm, _TISSUE_WORDS):
        return True
    if _has_any(nm, _NO_TISSUE_PROC_WORDS):
        return False
    return anesthesia_for(nm) not in ("", "局部麻醉", "静脉麻醉")


# 0.1.8：这些诊断不是一种"病"（化疗、复查、术后），检查不写"提示＋诊断名"
_NOT_FINDING_WORDS = ("治疗", "术后", "随访", "复查", "化疗", "放疗", "康复", "维持", "检查", "观察", "待产", "分娩", "妊娠")
DIAG_LIFT = 2.0
DIAG_LIFT_SITE = 1.2
DIAG_ADD_MIN = 1.0


def finding_name(dx_name):
    n = (dx_name or "").replace("？", "").replace("?", "").strip()
    return None if not n or _has_any(n, _NOT_FINDING_WORDS) else n


# 0.1.11（第十次 M8）：检查结论、病理写三位码的类目名（公开 ICD 类目表），去掉"伴有……""不伴有……"这类限定尾巴；
# 诊断清单照旧用真实写法。0.1.10 写出了"符合大脑动脉的闭塞或狭窄引起的脑梗死表现""符合痔不伴有并发症"
_QUAL_TAIL = re.compile(r"(?:不伴有?|伴有?|未特指|不可归类在他处).*$")


def finding_label(dx_name, code3):
    base = finding_name(dx_name)
    if base is None:
        return None
    cat = icd.category(code3) if code3 and len(code3) == 3 else None
    if cat and len(cat) < len(base) and not _has_any(cat, _NOT_FINDING_WORDS):
        base = cat
    return _QUAL_TAIL.sub("", base) or base


# 0.1.9：化疗、放疗、复查这类住院（主要诊断是一次治疗，不是一种病）。主诉不写症状，写"确诊××N月，为行第N次化疗"
TREAT_DX = re.compile(r"化疗|化学治疗|放疗|放射治疗|靶向治疗|免疫治疗|生物治疗|内分泌治疗|复查|随访|内固定.{0,2}取出|取出内固定")
TREAT_CODE3 = {"Z51", "Z08", "Z09", "Z47", "Z48"}
_TREAT_TAIL = re.compile(r"(?:术后)?(?:的)?(?:维持性|辅助|新辅助|姑息性?)?(?:化学治疗|化疗|放射治疗|放疗|靶向治疗|免疫治疗|生物治疗|内分泌治疗|复查|随访|治疗)$")
_CANCER_WORDS = ("癌", "恶性", "肉瘤", "淋巴瘤", "白血病", "骨髓瘤", "母细胞瘤")


def treatment_kind(name, code3):
    """化疗/放疗/复查/取内固定，不是这类住院返回 None。"""
    n = name or ""
    if _has_any(n, ("放疗", "放射治疗")):
        return "放疗"
    if _has_any(n, ("化疗", "化学治疗", "靶向治疗", "免疫治疗", "生物治疗", "内分泌治疗")):
        return "化疗"
    if "取出" in n and "内固定" in n or code3 == "Z47":
        return "取内固定"
    if _has_any(n, ("复查", "随访")) or code3 in ("Z08", "Z09", "Z48"):
        return "复查"
    if code3 == "Z51":
        return "化疗"
    return None


_PRIOR_DX_TAIL = re.compile(r"(?:术后|治疗后|放化疗后|化疗后|放疗后)+(?:状态)?$")


def _clean_under(s):
    n = _TREAT_TAIL.sub("", (s or "").replace("？", "").replace("?", "")).strip("，,、 ")
    n = _PRIOR_DX_TAIL.sub("", n).strip("，,、 ")
    return n if len(n) >= 2 and not TREAT_DX.search(n) else None


def underlying_dx(name, comorb):
    """治疗性住院的原发病：合并症里第一个肿瘤名（去掉末尾术后/治疗后等入院前修饰）；没有就把诊断名去掉"化疗""复查"这类尾巴；都没有返回 None。"""
    for c in comorb:
        if _has_any(c, _CANCER_WORDS):
            return _clean_under(c)
    return _clean_under(name)


def treatment_chief(kind, under, rng):
    """返回 (主诉文字, 主诉关键词, 病程)。关键词进必写事实的 key（"化疗"）。"""
    months = rng.randint(2, 11)
    dur = f"{months}个月"
    if kind == "化疗":
        n = rng.randint(2, 6)
        return f"确诊{under or '恶性肿瘤'}{dur}，为行第{n}次化疗", "化疗", dur
    if kind == "放疗":
        return f"确诊{under or '恶性肿瘤'}{dur}，为行放疗", "放疗", dur
    if kind == "取内固定":
        return f"{under or '骨折'}术后{dur}，要求取出内固定", "内固定", dur
    return f"{under or '原发病'}既往治疗{dur}，此次返院随访", "随访", dur


def fine_strata(strata):
    """节层以内（主诊断名、code3、节）。0.1.9：主诉、伴随症状、诊断相关检查都只在这几层取，不退到章层。"""
    return [s for s in strata if s.startswith("n:") or s.startswith("blk:") or (len(s) == 3 and s != "*")]


# 0.1.9：常规化验和心电图不算"诊断相关"（0.1.8 写出了"尿常规提示冠状动脉粥样硬化性心脏病"）
def diag_eligible(key):
    return KIND_OF.get(key, "其他") != "检验" and key not in ("心电图", "病理")


# 结果按检查类型写（公开写法）。{f} 是诊断名。0.2.0（第 3 条）：功能检查（脑电图、肺功能……）异常结果不带诊断名，写"可见异常改变"
FINDING_TEXT = {"影像": "符合{f}表现", "内镜": "符合{f}表现", "造影": "符合{f}表现", "功能": "可见异常改变", "穿刺": "符合{f}", "取材": "符合{f}"}


def exam_lifts(stats, strata, organs=None):
    """节层以内最细的非空层里，各检查的出现率是全局层的几倍。只看能算"诊断相关"的检查：不是常规化验和心电图，
    检查的部位和诊断的器官对得上（organs，审查员第八轮补充；None 表示不查部位）。"""
    t = stats.get("exams") or {}
    fine = next((t[s] for s in fine_strata(strata) if (t.get(s) or {}).get("items")), None)
    glob = t.get("*") or {}
    out = {}
    if not fine or not glob.get("items"):
        return out
    for item, n in fine["items"].items():
        key, _, region = item.partition("|")
        if diag_eligible(key) and glob["items"].get(item) and (organs is None or site_ok(key, region, organs)):
            out[item] = (n / max(1, fine["n"])) / (glob["items"][item] / max(1, glob["n"]))
    return out


def mark_exam_findings(stats, strata, exams, fname, organs=None):
    """没有数值的检查：节层以内出现率至少是全局层 DIAG_LIFT 倍、又不是常规化验和心电图、部位和诊断的器官对得上的算诊断相关，
    一定异常，按检查类型写"符合××表现"；培养诊断相关的写"培养出致病菌"，其余"未培养出致病菌"；其余一律"未见明显异常"
    （部位对不上的伴随检查也是）。病理、活检写"符合××"。fname 是写进结果的诊断名（化疗这类住院是原发病），None 表示不写诊断。"""
    lifts = exam_lifts(stats, strata, organs)
    thr = DIAG_LIFT if organs is None else DIAG_LIFT_SITE
    for e in exams:
        if e["items"]:
            continue
        if e["key"] in ("病理", "活检"):
            e["finding"] = f"符合{fname}" if fname else "未见恶性证据"
            e["abnormal"] = bool(fname)
            continue
        if "abnormal" not in e:
            continue
        item = f"{e['key']}|{e['region'] or ''}"
        # 0.1.11：补进来的那项（guaranteed，比值 ≥ DIAG_ADD_MIN、部位对得上）一定算诊断相关
        diag = bool(fname) and (lifts.get(item, 0.0) >= thr or bool(e.get("guaranteed")))
        e["abnormal"] = diag
        e["diagnostic"] = diag
        if e["kind"] == "培养":
            e["finding"] = "培养出致病菌" if diag else "未培养出致病菌"
        elif diag:
            e["finding"] = FINDING_TEXT.get(e["kind"], "符合{f}表现").format(f=fname)
        else:
            e.pop("finding", None)
    if fname and organs and not any(e.get("diagnostic") for e in exams):
        cands = [e for e in exams if not e["items"] and "abnormal" in e and diag_eligible(e["key"])
                 and e["kind"] in ("造影", "内镜", "影像") and site_ok(e["key"], e["region"], organs)]
        if cands:
            _kind_p = {"造影": 0, "内镜": 1, "影像": 2}
            _key_p = {"冠脉造影": 0, "脑血管造影": 1, "CT": 2, "磁共振": 3, "MRCP": 4, "ERCP": 5, "胃镜": 6, "肠镜": 7, "超声": 8, "心脏超声": 9, "X线": 10}
            best_e = min(cands, key=lambda e: (_kind_p.get(e["kind"], 9), _key_p.get(e["key"], 99), e.get("id") or ""))
            best_e["abnormal"] = best_e["diagnostic"] = True
            best_e["finding"] = FINDING_TEXT.get(best_e["kind"], "符合{f}表现").format(f=fname)


def _inv_group(key):
    kind = KIND_OF.get(key, "其他")
    if kind not in INVASIVE:
        return None
    return "组织" if kind in ("取材", "病理") else kind


def ensure_diagnostic(stats, strata, exams, fname, organs=None):
    """0.1.9：一项诊断相关的检查都没抽到时，补出现率比值最高的那项（比值要够 DIAG_LIFT，部位要对得上；0.2.0.1：不补已有同大类的有创检查）。
    返回要补的 (检查, 部位)，没有返回 None。"""
    if not fname:
        return None
    lifts = exam_lifts(stats, strata, organs)
    thr, add_min = (DIAG_LIFT, DIAG_LIFT) if organs is None else (DIAG_LIFT_SITE, DIAG_ADD_MIN)
    have = {f"{e['key']}|{e['region'] or ''}" for e in exams}
    if any(lifts.get(x, 0.0) >= thr for x in have):
        return None
    used_inv = {("组织" if ek in ("取材", "病理") else ek)
                for e in exams
                for ek in (e.get("kind") or KIND_OF.get(e.get("key"), "其他"),)
                if ek in INVASIVE and e.get("name") != "术后病理"}
    best = max(((v, x) for x, v in lifts.items()
                if v >= add_min and (x in have or _inv_group(x.partition("|")[0]) not in used_inv)),
               default=None)
    return tuple(best[1].split("|", 1)) if best else None


def _table(stats, name, strata):
    """第一层 items 非空的表。0.1 用 t.get(ch) or t.get("*")，章层存在但 items 被压空时是真值，挡住了回退。"""
    t = stats.get(name) or {}
    for s in strata:
        if (t.get(s) or {}).get("items"):
            return t[s]
    return {"n": 0, "items": {}}


def _sub(stats, name, key):
    return ((stats.get(name) or {}).get(key) or {}).get("items") or {}


_NEONATAL_DX_WORDS = ("新生儿", "早产儿", "足月儿", "母乳性黄疸", "脐炎")
_DEPT_TUMOR_RULES = (
    (("血液",), ("白血病", "淋巴瘤", "骨髓瘤", "骨髓")),
    (("妇",), ("宫颈", "子宫", "卵巢", "输卵管", "外阴", "阴道", "绒毛膜", "滋养", "盆腔")),
    (("乳腺", "甲乳"), ("乳腺", "乳房", "甲状腺")),
    (("甲状腺",), ("甲状腺",)),
    (("胃肠", "肛肠", "结直肠", "消化"), ("胃", "结肠", "直肠", "食管", "肠", "肛", "消化")),
    (("肝胆",), ("肝", "胆", "胰", "壶腹")),
    (("胸外", "呼吸"), ("肺", "支气管", "食管", "纵隔", "胸膜")),
    (("泌尿",), ("肾", "膀胱", "前列腺", "输尿管", "尿路上皮", "睾丸")),
    (("头颈", "耳鼻", "口腔", "颌面"), ("鼻", "咽", "喉", "舌", "口腔", "颌", "腮腺", "甲状腺", "颈")),
)


def _is_neonatal_dx(name, code=""):
    return (code or "").startswith("P") or _has_any(name, _NEONATAL_DX_WORDS)


def _is_specific_cancer(term):
    """是否为指明具体部位或谱系的原发肿瘤诊断（排除泛称"恶性肿瘤化疗"及"继发恶性肿瘤"）。"""
    if not _has_any(term, _CANCER_WORDS) or "继发" in (term or ""):
        return False
    base = _clean_under(term) or ""
    return bool(base and base not in ("恶性肿瘤", "手术后恶性肿瘤", "术后恶性肿瘤", "原发恶性肿瘤", "癌症"))


def _dept_tumor_ok(dept, term):
    if not _is_specific_cancer(term):
        return True
    d = dept or ""
    for d_kws, t_kws in _DEPT_TUMOR_RULES:
        if _has_any(d, d_kws):
            return _has_any(term, t_kws)
    if _has_any(term, ("白血病", "骨髓瘤")) and not _has_any(d, ("肿瘤", "放疗", "内科", "儿", "全科", "急诊", "重症", "ICU")):
        return False
    return True


def dx_ok(name, sex, age):
    """主诊断、合并症、鉴别诊断都过性别和年龄：slot_filter 加上成人病不给儿童、儿童发育障碍不给成人、新生儿病不给非新生儿。"""
    if age < 18 and ADULT_DX.search(name):
        return False
    if age >= 18 and _has_any(name, _CHILD_DX_WORDS):
        return False
    if age >= 1 / 12 and _has_any(name, ("新生儿", "早产儿", "足月儿")):
        return False
    if age < 35 and _has_any(name, ("高龄妊娠", "高龄初产", "高龄经产", "高龄产妇")):
        return False
    return not slot_filter({"dx_name": name, "dx_code3": "", "age": age, "sex": sex})


def dept_ok(dept, sex, age, pregnant, code=""):
    d = dept or ""
    if sex == "男" and _has_any(d, ("妇", "产")):
        return False
    if sex == "男" and "乳腺" in d and ("甲" not in d or age < 30):
        return False
    if sex == "女" and "产" in d and "妇" not in d and not pregnant:
        return False
    if "男科" in d and sex != "男":
        return False
    if age >= 18 and _has_any(d, ("儿", "新生")):
        return False
    if age >= 1 / 12 and "新生" in d:
        return False
    if re.match(r"^N(?:2[0-3]|4\d|5[01])", code or "") and "风湿" in d:
        return False
    return not (age < 14 and "老年" in d)


def side_of(text):
    """诊断里的侧别：双、左、右，没有返回 None。"""
    s = {c for c in ("左", "右", "双") if c in (text or "")}
    if "双" in s or {"左", "右"} <= s:
        return "双"
    return next(iter(s), None)


def family_ok(fam, dx_name):
    """这个大类和诊断对得上吗：关键词命中；剖宫产不给写明 28 周以前的，清宫不给写明 13 周以后的。"""
    rx = FAMILY_DX_RE.get(fam)
    if rx is None or not rx.search(dx_name or ""):
        return False
    w = icd.gest_weeks(dx_name)
    return not (w is not None and ((fam == "剖宫产" and w < 28) or (fam == "清宫" and w > 13)))


def same_organ(a, b):
    """两个名字里有没有同一个成对器官（按器官词首字比，"乳腺""乳房"算同一个）。"""
    return bool({x[0] for x in LAT_ORGAN.findall(a or "")} & {x[0] for x in LAT_ORGAN.findall(b or "")})


def lateralize_dx(name, side, fam=None):
    """诊断没写侧别、术式分侧时，把术式的侧别加进诊断（监督员第五轮：分侧术式要配分侧诊断）。加在第一个成对器官词前面；
    没有器官词的白内障加在开头（"右眼…白内障"）；都没有返回原名。"""
    m = LAT_ORGAN.search(name or "")
    if m:
        return name[:m.start()] + side + ("" if m.group(0) in _BARE_SIDE else "侧") + name[m.start():]
    if fam == "白内障":
        return side + "眼" + name
    return name


def realize_proc(key, dx_name, rng):
    """手术名占位符换成侧别：诊断有侧别就跟诊断，没有就随机左右。返回 (手术名, 侧别)。"""
    if "{S" not in key:
        return key, "无"
    side = side_of(dx_name) or rng.choice(("左", "右"))
    return key.replace("{S侧}", side + "侧").replace("{S}", side + ("侧" if side == "双" else "")), side


def qdraw(rng, q, lo_q=0.05, hi_q=0.95):
    """分位数之间线性插值抽一个值。q 是 p5、p25、p50、p75、p95。"""
    u = rng.uniform(lo_q, hi_q)
    pts = [0.05, 0.25, 0.5, 0.75, 0.95]
    for i in range(4):
        if u <= pts[i + 1]:
            r = (u - pts[i]) / (pts[i + 1] - pts[i])
            return q[i] + (q[i + 1] - q[i]) * r
    return q[-1]


def fictional_date(jid):
    h = seed_of(jid, "date")
    return (dt.date(2025, 1, 1) + dt.timedelta(days=h % 300)).isoformat()


def age_text(age):
    if age < 1 / 12:
        return f"{max(1, int(round(age * 365)))}天"
    if age < 1:
        return f"{max(1, int(round(age * 12)))}个月"
    return f"{int(age)}岁"


def _band_age(rng, band, pregnant=False, neonatal=False):
    """年龄段里抽一个年龄。妊娠只在 15–49 岁抽，新生儿诊断只在 0–1 岁段按 3–27 天抽，和年龄段没有交集时返回 None。"""
    if neonatal:
        return round(rng.randint(3, 27) / 365, 4) if band == "0-1" else None
    for name, lo, hi in AGE_BANDS:
        if name == band:
            if pregnant:
                a, b = max(lo, PREG_AGE[0]), min(hi - 1, PREG_AGE[1])
                return rng.randint(a, b) if a <= b else None
            if hi - lo == 1 and lo == 0:
                return round(rng.randint(1, 11) / 12, 2)
            return rng.randint(lo, min(hi - 1, 98))
    return None if pregnant else 50


def dx_name_for(stats, rng, code, ch):
    """主诊断名，从细到粗：code3 下凑够 k 次的诊断名 → 凑够 k 次的诊断主干（侧别随机）→ 公开词典的标准名 → 章层的诊断名 → 章层的主干。
    返回 (诊断名, 来源)；来源是 chapter、chapter_stem 时调用方把 code 退成章，后面的表按章取，不和别的 code3 错配。"""
    if code:
        name = _pick(rng, _sub(stats, "dx_names", code))
        if name:
            return name, "name"
        stem = _pick(rng, _sub(stats, "dx_stems", code))
        if stem:
            return icd.realize(stem, rng), "stem"
        name = icd.category(code)
        if name:
            return name, "icd"
    name = _pick(rng, _sub(stats, "dx_names_ch", ch))
    if name:
        return name, "chapter"
    # 0.1.5：没有一个 code3 凑够 k 次的章（如 O 章），0.1.4 到这里整章跳过；现在再看章层主干
    stem = _pick(rng, _sub(stats, "dx_stems", ch))
    return (icd.realize(stem, rng), "chapter_stem") if stem else (None, None)


_DELIVERY_OUTCOME_WORDS = ("活产", "死产", "死胎", "顺产", "难产", "分娩", "产钳", "胎头吸引", "臀位牵引", "产后", "产褥")


def _is_delivery_dx(text):
    t = text or ""
    return _has_any(t, _DELIVERY_OUTCOME_WORDS) or ("会阴" in t and "裂伤" in t)


def gest_for(name, rng):
    """孕周：诊断名写了就用名字里的（0.1.5 随机 8–40 周，和"孕39周待产"对不上），没写按名字定范围，再没有随机 8–40 周。"""
    w = icd.gest_weeks(name)
    if w is not None:
        return w
    lo, hi = icd.week_range(name) or (8, 40)
    return rng.randint(lo, hi)


def draw_patient(stats, rng, jid, slot=None):
    if slot:
        sex, age, dept, name, code = slot["sex"], slot["age"], slot["dept"], slot["dx_name"], slot.get("dx_code3") or ""
        source = "slot"
    else:
        sex = age = dept = name = code = source = None
        for _ in range(30):
            cell = _pick(rng, stats["joint"]["items"])
            if cell is None:
                break
            sex, band, ch = cell.split("|")
            code = _pick(rng, _table(stats, "code3", [ch])["items"])
            name, source = dx_name_for(stats, rng, code, ch)
            if name is None:
                continue
            if source in ("chapter", "chapter_stem"):
                code = ch
            age = _band_age(rng, band, code.startswith("O") or bool(PREG_NAME.search(name)), neonatal=_is_neonatal_dx(name, code))
            if age is not None and not slot_filter({"dx_name": name, "dx_code3": code, "age": age, "sex": sex}) and dx_ok(name, sex, age):
                break
            name = None
        if name is None:
            return None
    pregnant = sex == "女" and (code.startswith("O") or bool(PREG_NAME.search(name)))
    strata = strata_of(name, code)
    if not slot:
        dept = _pick(rng, _table_where(stats, "dept", strata, lambda d: dept_ok(d, sex, age, pregnant, code))) or ("儿科" if age < 14 else "内科")
    comorb, raw_comorb = [], []
    p_ds_only = _ds_only_prob(stats)
    used_grades = {s for s in (_dx_grade_stem(name),) if s}

    def can_comorb(term):
        if len(raw_comorb) >= MAX_COMORB or term in name or name in term or not dx_ok(term, sex, age):
            return False
        if _is_delivery_dx(term) and not _is_delivery_dx(name):
            return False
        if "继发" in term and _has_any(term, _CANCER_WORDS):
            if not any(_is_specific_cancer(x) for x in [name] + [r for r, _ in raw_comorb]):
                return False
        elif _is_specific_cancer(term) and not _dept_tumor_ok(dept, term):
            return False
        if any(term in r_term or r_term in term for r_term, _ in raw_comorb):
            return False
        gs = _dx_grade_stem(term)
        return not (gs and gs in used_grades)

    def add_comorb(term):
        is_chron = any(past_matches(pt, term) for pt in PAST_TERMS)
        raw_comorb.append((term, is_chron))
        gs = _dx_grade_stem(term)
        if gs:
            used_grades.add(gs)

    fine_strata_c = [s for s in strata if s != "*"]
    t = _table(stats, "comorbid", fine_strata_c or strata)
    for term, n in sorted(t["items"].items()):
        if not can_comorb(term):
            continue
        if rng.random() < n / max(1, t["n"]):
            add_comorb(term)
    # 0.21 / 0.30：合并症补齐优先锁定细层（n:name/code3/blk），存在细层表时不退回粗章层跨龄跨系统借病
    strict_fine_c = fine_strata(strata)
    topup_strata = strict_fine_c if any((((stats.get("comorbid") or {}).get(s) or {}).get("items")) for s in strict_fine_c) else fine_strata_c
    rr = stats.get("real_rates") or {}
    if rr.get("ds_dx_admissions") and rr.get("ds_dx_items"):
        avg_sec = max(0.0, rr["ds_dx_items"] / rr["ds_dx_admissions"] - 1.0)
        target_c = min(4, max(2, int(round(avg_sec * 0.6))))
        if sum(1 for _, ic in raw_comorb if not ic) < 2 and len(raw_comorb) < target_c:
            for s in topup_strata:
                cand = {term: n for term, n in (((stats.get("comorbid") or {}).get(s) or {}).get("items") or {}).items()
                        if can_comorb(term) and not any(past_matches(pt, term) for pt in PAST_TERMS)}
                while sum(1 for _, ic in raw_comorb if not ic) < 2 and len(raw_comorb) < target_c and cand:
                    term = _pick(rng, cand)
                    del cand[term]
                    if can_comorb(term):
                        add_comorb(term)
                if sum(1 for _, ic in raw_comorb if not ic) >= 2 or len(raw_comorb) >= target_c:
                    break
        if len(raw_comorb) < target_c:
            for s in topup_strata:
                cand = {term: n for term, n in (((stats.get("comorbid") or {}).get(s) or {}).get("items") or {}).items() if can_comorb(term)}
                while len(raw_comorb) < target_c and cand:
                    term = _pick(rng, cand)
                    del cand[term]
                    if can_comorb(term):
                        add_comorb(term)
                if len(raw_comorb) >= target_c:
                    break
    if treatment_kind(name, code) in ("化疗", "放疗") and not any(_is_specific_cancer(x) for x in [name] + [r for r, _ in raw_comorb]) and len(raw_comorb) < MAX_COMORB:
        for s in fine_strata_c:
            cand = {term: n for term, n in (((stats.get("comorbid") or {}).get(s) or {}).get("items") or {}).items()
                    if can_comorb(term) and _is_specific_cancer(term)}
            if cand:
                z_rng = random.Random(seed_of(jid, "z51_prim_v034"))
                pick_t = _pick(z_rng, cand)
                if pick_t:
                    add_comorb(pick_t)
                    break
    if rr.get("b3_admissions") and rr.get("b3_permille_sum"):
        real_b3 = rr["b3_permille_sum"] / (rr["b3_admissions"] * 1000.0)
        target_share = min(0.32, max(0.28, real_b3 * 1.18))
        non_chron_n = sum(1 for _, ic in raw_comorb if not ic)
        p_item = min(0.85, max(0.35, target_share * (1 + len(raw_comorb)) / non_chron_n)) if non_chron_n else 0.0
    else:
        p_item = p_ds_only
    need_adm_cancer = treatment_kind(name, code) in ("化疗", "放疗") and not _is_specific_cancer(name)
    for term, is_chron in raw_comorb:
        # 0.21（B_B3 提分）：属既往慢病词表的合并症默认入院前即有，切断 ds_only 同步进 past_history 的自抵消
        adm_draw = True if is_chron else (rng.random() >= p_item)
        if need_adm_cancer and _is_specific_cancer(term):
            adm_draw = True
            need_adm_cancer = False
        if not is_chron and ("术后" in term or "产后" in term) and "术后" not in name:
            adm_draw = False
        comorb.append({"name": term, "at_admission": adm_draw})
    past = []
    g = (stats.get("past") or {}).get(next((n for n, lo, hi in AGE_BANDS if lo <= age < hi), None)) or {}
    for term, n in sorted((g.get("items") or {}).items()):
        if rng.random() < n / max(1, g["n"]):
            past.append(term)
    # 0.21：只同步入院时已列诊断进既往史，且不把 ds_only 合并症误带入既往史（防止 HR 提前暴露使 B3 失分）
    adm_listed = "；".join([name] + [c["name"] for c in comorb if c["at_admission"]])
    ds_only_text = "；".join(c["name"] for c in comorb if not c["at_admission"])
    past = [term for term in past if past_matches(term, adm_listed) or not past_matches(term, ds_only_text)]
    # 诊断清单里有的慢性病（含公开全称，如冠状动脉粥样硬化性心脏病 ↔ 冠心病），既往史里也要有，否则入院记录写"否认"、出院诊断又列出来
    listed = "；".join([name] + [c["name"] for c in comorb])
    for term in PAST_TERMS:
        if term not in past and past_matches(term, listed):
            past.append(term)
    return {"patient_id": jid, "sex": sex, "age": age, "age_text": age_text(age), "weight_kg": round(weight_p50(age), 1) if age < 18 else rng.randint(50, 78),
            "pregnant": pregnant, "gest_weeks": gest_for(name, rng) if pregnant else None, "dept": dept, "primary_dx": {"name": name, "code3": code, "source": source},
            "comorbidities": comorb, "past_history": past, "allergies": [], "source": "paired" if slot else "free"}


def med_allowed(p, name, dose_text, freq, route):
    """把这个用药元组写成一句话，让 g3-rules-0.3 判。命中任何规则（硬错误、儿科成人量、性别、警示级）都不要。
    诊断文本带上合并症和既往史：闸门按整份住院的文字判，规划时只看主诊断会漏掉合并症引起的禁忌。"""
    sentence = f"予{name}{dose_text}{(' ' + freq) if freq else ''}{route or ''}治疗。"
    age_hi = p["age"] + 1 if p["age"] == int(p["age"]) else p["age"]
    dx = [p["primary_dx"]["name"]] + [c["name"] for c in p.get("comorbidities") or []] + list(p.get("past_history") or [])
    rec = Record(rid="plan", age=p["age"], age_hi=age_hi, sex={"男": "M", "女": "F"}[p["sex"]], dx_text="；".join(dx) + ("；妊娠状态" if p["pregnant"] else ""),
                 segs=[("0:plan", sentence, "course")])
    res = evaluate(rec)
    return not res["hits"] and not res["n1"]


_QPTS = (0.05, 0.25, 0.5, 0.75, 0.95)
P_LAB_ABN_DEFAULT = 0.25   # 统计表里没有这一项的异常比例时用（0.1.8 的固定值）
LAB_RHO = 0.5              # 同一次住院各项化验共享的病情轻重占几成（公开的建模选择，不来自病历）
# 统计表的分位数没有低于（高于）参考范围的尾部时，异常往哪边偏（公开常识）
ABN_SIDE_DEFAULT = {"HGB": "low", "ALB": "low", "NA": "low", "CA": "low", "K": "low"}
_ND = NormalDist()


def qinterp(q, u):
    """分位数之间线性插值：u 在 0.05–0.95 之间。"""
    u = min(max(u, _QPTS[0]), _QPTS[-1])
    for i in range(4):
        if u <= _QPTS[i + 1]:
            r = (u - _QPTS[i]) / (_QPTS[i + 1] - _QPTS[i])
            return q[i] + (q[i + 1] - q[i]) * r
    return q[-1]


def qrank(q, v):
    """v 在分位数里的位置（0.05–0.95，超出夹到两端）。"""
    if v <= q[0]:
        return _QPTS[0]
    for i in range(4):
        if v <= q[i + 1]:
            span = q[i + 1] - q[i]
            return _QPTS[i] + (_QPTS[i + 1] - _QPTS[i]) * ((v - q[i]) / span if span else 0.5)
    return _QPTS[-1]


def _lab_q(stats, key, layers=()):
    """分位数。0.1.11（第十次 M5）：先按 layers 取分层的（stats.labs_by，每层 ≥ LAB_STRATUM_MIN 个读数才有）：成人是节 → 章，
    14 岁以下是本章儿童 → 全院儿童；都没有才用全院的（少见化验）。0.1.10 只有全院一张，尿路梗阻抽到了别的病尾部的总胆红素 108.7。"""
    by = stats.get("labs_by") or {}
    for s in layers:
        lab = (by.get(s) or {}).get(key)
        if lab:
            return lab["q"]
    lab = (stats.get("labs") or {}).get(key)
    lo, hi = REF[key]
    return lab["q"] if lab else [lo, lo + (hi - lo) * 0.3, (lo + hi) / 2, lo + (hi - lo) * 0.7, hi * 1.2 if hi else hi]


def lab_rate(stats, strata, key):
    """这项化验在本诊断层的异常比例。从细到粗取第一层"正常""异常"两格都凑够 k 的；只有一格凑够时，另一格不到 k，
    比例的上界（或下界）记下来，用更粗一层的比例夹进这个界里。都没有用 P_LAB_ABN_DEFAULT。"""
    t = stats.get("lab_state") or {}
    k = stats.get("k", 11)
    lo_b, hi_b = 0.0, 1.0
    for s in strata:
        items = (t.get(s) or {}).get("items") or {}
        a, n = items.get(f"{key}|abn"), items.get(f"{key}|norm")
        if a and n:
            return min(max(a / (a + n), lo_b), hi_b)
        if n:
            hi_b = min(hi_b, (k - 1) / (n + k - 1))
        elif a:
            lo_b = max(lo_b, a / (a + k - 1))
    return min(max(P_LAB_ABN_DEFAULT, lo_b), hi_b)


def _fmt_lab(key, v):
    v = min(max(v, PLAUSIBLE[key][0]), PLAUSIBLE[key][1])
    v = round(v, DECIMALS.get(key, 2))
    if DECIMALS.get(key, 2) == 0:
        v = int(v)
    lo, hi = REF[key]
    flag = "偏高" if v > hi else ("偏低" if v < lo else "正常")
    return {"item": LAB_NAME[key], "key": key, "value": str(v), "unit": LAB_UNIT[key], "flag": flag}


_CBC_EXT = {
    "WBC": ("high", 12.0, "wbc_gt12", "wbc_admissions"),
    "NEUT": ("high", 85.0, "neut_gt85", "neut_admissions"),
    "HGB": ("low", 90.0, "hgb_lt90", "hgb_admissions"),
    "PLT": ("low", 100.0, "plt_lt100", "plt_admissions"),
}


def _cbc_ext_cond(stats, key, side):
    """0.21：由真实首次血常规极值率反推异常时跨越极值线的条件概率，消除全住院复查混合分位数引起的极值偏高。"""
    spec = _CBC_EXT.get(key)
    if not spec or side != spec[0]:
        return None
    ext_k, adm_k = spec[2], spec[3]
    rr = stats.get("real_rates") or {}
    adm = rr.get(adm_k) or 0
    glob = ((stats.get("lab_state") or {}).get("*") or {}).get("items") or {}
    ga, gn = glob.get(f"{key}|abn", 0), glob.get(f"{key}|norm", 0)
    if adm <= 0 or ga <= 0 or (ga + gn) <= 0:
        return None
    q_g = _lab_q(stats, key, ())
    lo, hi = REF[key]
    fl, fh = qrank(q_g, lo), qrank(q_g, hi)
    lm, hm = max(0.0, fl - _QPTS[0]), max(0.0, _QPTS[-1] - fh)
    if lm + hm <= 0:
        p_side_g = 1.0 if side == ABN_SIDE_DEFAULT.get(key, "high") else 0.2
    else:
        p_side_g = (lm / (lm + hm)) if side == "low" else (hm / (lm + hm))
    p_real_ext = rr.get(ext_k, 0) / adm
    p_glob_abn = ga / (ga + gn)
    cond = p_real_ext / max(0.05, p_glob_abn * max(0.15, p_side_g))
    return min(0.95, max(0.05, cond))


def _lab_value(stats, rng, key, abnormal, sev=None, rate=None, layers=()):
    """抽一项化验值。正常的在参考范围以内抽（按真实分布落在范围内的那一段）；异常的在真实分布的尾部抽，哪边的尾部按两边尾部的大小定，
    越往外越重：位置由病情轻重 sev 和随机数各占一半，这一层异常少见（rate 小）的只取轻度。0.1.8 正常的也可能抽到范围外，
    异常的固定取全院 p75–p95。layers：分位数从哪几层取（0.1.11，见 _lab_q）。"""
    q = _lab_q(stats, key, layers)
    lo, hi = REF[key]
    f_lo, f_hi = qrank(q, lo), qrank(q, hi)
    sev = 0.5 if sev is None else sev
    if not abnormal:
        if f_hi - f_lo >= 0.05:
            v = qinterp(q, rng.uniform(f_lo, f_hi))
        else:
            v = rng.uniform(lo, hi)
        v = min(max(v, lo), hi)
        return _fmt_lab(key, v)
    low_mass, high_mass = max(0.0, f_lo - _QPTS[0]), max(0.0, _QPTS[-1] - f_hi)
    if low_mass + high_mass <= 0:
        side = ABN_SIDE_DEFAULT.get(key, "high")
    else:
        p_low = low_mass / (low_mass + high_mass)
        side = "low" if rng.random() < p_low else "high"
    step = 10 ** -DECIMALS.get(key, 2)
    cond = _cbc_ext_cond(stats, key, side)
    if cond is not None:
        ext_thr = _CBC_EXT[key][1]
        is_ext = rng.random() < cond
        u = min(1.0, max(0.0, 0.3 * sev + 0.7 * rng.random()))
        if side == "high":
            if is_ext:
                if q[-1] > ext_thr:
                    a = max(f_hi, qrank(q, ext_thr))
                    v = max(ext_thr + step, qinterp(q, a + (_QPTS[-1] - a) * u))
                else:
                    v = rng.uniform(ext_thr + step, ext_thr * 1.12)
            else:
                top = min(ext_thr, q[-1]) if q[-1] > hi + step else ext_thr
                v = rng.uniform(hi + step, max(hi + step, top))
        else:
            if is_ext:
                if q[0] < ext_thr:
                    b = min(f_lo, qrank(q, ext_thr))
                    v = min(ext_thr - step, qinterp(q, b - (b - _QPTS[0]) * u))
                else:
                    v = rng.uniform(ext_thr * 0.88, ext_thr - step)
            else:
                bot = max(ext_thr, q[0]) if q[0] < lo - step else ext_thr
                v = rng.uniform(min(lo - step, bot), lo - step)
        return _fmt_lab(key, v)
    depth = (sev + rng.random()) / 2 * min(1.0, 0.4 + (rate if rate is not None else P_LAB_ABN_DEFAULT))
    if side == "high":
        v = qinterp(q, f_hi + (_QPTS[-1] - f_hi) * depth) if high_mass > 0 else hi * (1.05 + 0.5 * depth)
        v = max(v, hi * 1.02 if hi else hi + 0.01)
    else:
        v = qinterp(q, f_lo - (f_lo - _QPTS[0]) * depth) if low_mass > 0 else lo * (0.95 - 0.3 * depth)
        v = min(v, lo * 0.98)
    out = _fmt_lab(key, v)
    if out["flag"] == "正常":
        # 四舍五入后落回范围内时往外推一位
        out = _fmt_lab(key, (hi + step) if side == "high" else (lo - step))
    return out


CHILD_AGE = 14          # 0.1.11：14 岁以下的化验、用药从儿童层取（同 child_range 的年龄线）
# 0.1.11（第十次 M5）：这两项异常要有糖尿病（公开诊断标准：糖化血红蛋白、空腹血糖是糖尿病的诊断指标）。0.1.10 两份"否认糖尿病"的病人
# 糖化血红蛋白 9.4%、9.3%
DIABETES_MARKERS = {"HBA1C", "GLU"}
_RENAL_FAIL_WORDS = ("慢性肾脏病3期", "慢性肾脏病4期", "慢性肾脏病5期", "慢性肾脏病三期", "慢性肾脏病四期", "慢性肾脏病五期", "肾衰竭", "肾功能不全", "尿毒症", "急性肾损伤")
_SEVERE_RENAL_WORDS = ("慢性肾脏病4期", "慢性肾脏病5期", "慢性肾脏病四期", "慢性肾脏病五期", "终末期肾", "尿毒症")


def child_layers(strata):
    """14 岁以下的病人：本章儿童层 → 全院儿童层（stats 0.10 起有，kid:章、kid:*）。"""
    ch = next((s for s in strata if len(s) == 1 and s != "*"), None)
    return ([f"kid:{ch}"] if ch else []) + ["kid:*"]


def lab_ctx(rng, stats, strata, p=None):
    """一次住院的病情轻重（标准正态 z 和对应的 0–1 位置），各项化验共用。
    0.1.11：p 给了时，分位数成人按节 → 章取，14 岁以下按儿童层取，异常比例也按儿童层取；记下有没有糖尿病。"""
    z = rng.gauss(0, 1)
    ctx = {"z": z, "sev": _ND.cdf(z), "stats": stats, "strata": strata, "rates": {}, "q_layers": (), "diabetes": True}
    if p is not None:
        dxt = "；".join([p["primary_dx"]["name"]] + [c["name"] for c in p.get("comorbidities") or []] + list(p.get("past_history") or []))
        ctx["diabetes"] = "糖尿病" in dxt
        ctx["renal_fail"] = _has_any(dxt, _RENAL_FAIL_WORDS)
        ctx["renal_severe"] = _has_any(dxt, _SEVERE_RENAL_WORDS)
        ctx["anemia"] = "贫血" in dxt
        ctx["is_child"] = p["age"] < CHILD_AGE
        if p["age"] < CHILD_AGE:
            kid = child_layers(strata)
            ctx["q_layers"], ctx["rate_strata"] = kid, kid
        else:
            ctx["q_layers"] = [s for s in strata if s.startswith("blk:")] + [s for s in strata if len(s) == 1 and s != "*"]
    return ctx


def draw_lab(rng, key, ctx):
    """按诊断层的异常比例定这一项异常不异常：z = √ρ·病情 + √(1−ρ)·本项噪声，超过 Φ⁻¹(1−比例) 算异常。单项的异常比例不变，各项一起偏。
    0.1.11：14 岁以下的异常比例按儿童层取；糖化血红蛋白、血糖没有糖尿病时一律正常（随机数照抽，不改后面的抽签）。"""
    stats, strata = ctx["stats"], ctx.get("rate_strata") or ctx["strata"]
    rate = ctx["rates"].get(key)
    if rate is None:
        rate = ctx["rates"][key] = lab_rate(stats, strata, key)
    z = math.sqrt(LAB_RHO) * ctx["z"] + math.sqrt(1 - LAB_RHO) * rng.gauss(0, 1)
    abnormal = rate > 0 and (rate >= 1 or z > _ND.inv_cdf(1 - rate))
    if key in DIABETES_MARKERS and not ctx.get("diabetes", True):
        abnormal = False
    elif key in ("CREA", "BUN") and ctx.get("renal_fail"):
        abnormal = True
    elif key == "HGB" and ctx.get("anemia"):
        abnormal = True
    out = _lab_value(stats, rng, key, abnormal, ctx["sev"], rate, ctx.get("q_layers") or ())
    if key in ("CREA", "BUN") and ctx.get("renal_fail"):
        hi = REF[key][1]
        floor_v = (180.0 if ctx.get("is_child") else 442.0) if (key == "CREA" and ctx.get("renal_severe")) else (hi * 1.25)
        if out["flag"] != "偏高" or float(out["value"]) < floor_v:
            out = _fmt_lab(key, floor_v + (abs(z) % 1.0) * (120.0 if key == "CREA" else 4.0))
    elif key == "HGB" and ctx.get("anemia") and out["flag"] != "偏低":
        lo = REF["HGB"][0]
        out = _fmt_lab("HGB", max(65.0, lo - 6.0 - (abs(z) % 1.0) * 18.0))
    return out


# 复查时往参考范围走的快慢（公开常识）：血红蛋白、血小板恢复慢，白细胞、中性粒细胞比例快
RECOVER_SPEED = {"HGB": 0.25, "PLT": 0.6}


# 0.1.12（第十一次合成病历 08：肌酸激酶 29U/L、同工酶 33U/L）：同一组化验里"部分 ≤ 总量"（公开检验常识：CK-MB 是 CK 的同工酶之一，
# LDL-C 是总胆固醇的一部分）。各项是分开抽的，会抽出部分大于总量
PART_OF = {"CKMB": "CK", "LDL": "TC"}


def enforce_parts(rng, items):
    """部分大于总量时，部分改在参考范围以内、且不超过总量重抽（只在违反时用随机数）。单位不同的不比。"""
    out = list(items)
    for part, total in PART_OF.items():
        ia = next((i for i, x in enumerate(out) if x["key"] == part), None)
        ib = next((i for i, x in enumerate(out) if x["key"] == total), None)
        if ia is None or ib is None or out[ia]["unit"] != out[ib]["unit"]:
            continue
        vb = float(out[ib]["value"])
        if float(out[ia]["value"]) <= vb:
            continue
        lo, hi = REF[part]
        top = min(hi, vb)
        new = _fmt_lab(part, rng.uniform(lo, top) if top > lo else top)
        if float(new["value"]) > vb:
            new = _fmt_lab(part, top)
        out[ia] = new
    return out


def recheck_value(rng, x, day, los):
    """复查值以入院值为起点：入院异常的往参考范围走（住院越往后走得越多，血红蛋白走得慢），入院正常的小幅波动、仍在范围内。"""
    key = x["key"]
    lo, hi = REF[key]
    v0 = float(x["value"])
    if lo <= v0 <= hi:
        v = v0 * (1 + rng.gauss(0, 0.04))
        return _fmt_lab(key, min(max(v, lo), hi))
    bound = hi if v0 > hi else lo
    frac = min(0.95, max(0.1, 0.35 + 0.5 * day / max(1, los) + rng.gauss(0, 0.1))) * RECOVER_SPEED.get(key, 1.0)
    return _fmt_lab(key, v0 + (bound - v0) * frac)


def _exam(stats, rng, i, key, region, los, day=None, ctx=None):
    kind = KIND_OF.get(key, "其他")
    edays = stats.get("exam_days") or {}
    d = dict(edays.get(f"{key}|{region}") or (edays.get(f"{key}|") if region else None) or {})
    if day is None:
        day = int(_pick(rng, d)) if d else rng.choice((0, 0, 1, 1, 2))
        day = min(day, los)
    elags = stats.get("exam_lags") or {}
    lh = next((elags[k] for k in (f"{key}|{region}", key, f"kind:{kind}") if elags.get(k)), None)
    lag = int(_pick(rng, lh)) if lh else (rng.randint(0, SCHED[kind]) if kind in SCHED else 0)
    order = max(0, day - lag)
    name = f"{REGION_NAME.get(region, region)}{key}" if region else key
    if ctx is None:
        items = [_lab_value(stats, rng, k, rng.random() < P_LAB_ABN_DEFAULT) for k in PANELS.get(key, ())]
    else:
        items = [draw_lab(rng, k, ctx) for k in PANELS.get(key, ())]
    items = enforce_parts(rng, items)
    e = {"id": f"e{i}", "key": key, "region": region or None, "name": name, "kind": kind, "order_day": order, "result_day": day, "items": items}
    if not items:
        # 0.1.5：没有数值的检查结果分正常、异常两种写；0.1.4 一律写"结果与主要诊断相符"，对话模型常改写成计划口吻，读数器不计
        e["abnormal"] = rng.random() < P_EXAM_ABNORMAL.get(kind, 0.5)
    if key == "病理" and day == 0:
        # 第 0 天就有的病理只能是入院前做的（外院病理）。0.1.4 写成"病理回报"，正好是 K4 的触发词
        e["name"], e["prior"] = "外院病理", True
    return e


# 0.1.9（审查员第八轮补充）：14 岁以下按年龄给脉搏、呼吸、血压的抽样范围（儿科教科书的正常范围，公开知识）。
# 统计表的儿童体征是 0–17 岁混在一起的，3 岁肺炎患儿抽到了 112/50mmHg。每项给 (p5, p95)，中间三个分位数线性插值
CHILD_RANGE = ((1, {"P": (110, 160), "R": (30, 45), "SBP": (70, 90), "DBP": (40, 55)}),
               (3, {"P": (95, 140), "R": (24, 34), "SBP": (80, 98), "DBP": (48, 62)}),
               (6, {"P": (85, 125), "R": (22, 28), "SBP": (85, 102), "DBP": (52, 66)}),
               (10, {"P": (75, 115), "R": (18, 24), "SBP": (90, 110), "DBP": (55, 70)}),
               (14, {"P": (65, 105), "R": (16, 22), "SBP": (95, 118), "DBP": (58, 75)}))


def child_range(age):
    return next((r for hi, r in CHILD_RANGE if age < hi), None)


def _vitals(stats, age):
    """生命体征的分位数。儿童先取统计表本年龄段的（child:1-6 这类，0.9 起有），没有的项 14 岁以下用 CHILD_RANGE，
    14–17 岁用统计表儿童合并的。数据优先，公开范围兜底（采纳审查员"先用统计量"的意见）。"""
    g = "child" if age < 18 else "adult"
    base = dict(VITAL_DEFAULT[g])
    vt = stats.get("vitals") or {}
    for k, v in (vt.get(g) or {}).items():
        base[k] = v["q"]
    if g == "child":
        band = next((n for n, lo, hi in AGE_BANDS if lo <= age < hi), None)
        own = vt.get(f"child:{band}") or {}
        cr = child_range(age)
        for k in ("P", "R", "SBP", "DBP"):
            if (own.get(k) or {}).get("q"):
                base[k] = own[k]["q"]
            elif cr:
                lo, hi = cr[k]
                base[k] = [lo + (hi - lo) * x for x in (0, 0.25, 0.5, 0.75, 1)]
    return base


_CHILD_ONLY_PROC_WORDS = ("鞘状突高位结扎", "疝囊高位结扎", "睾丸下降固定")


def proc_for(stats, strata, dx_name, age=None):
    """手术可取的写法。主诊断名、code3 两层：先具体写法，再大类。0.1.6：两层都没有时再看章层、全局层，只取适应证关键词
    和诊断对得上的（family_ok）：先具体写法（按它属于的大类判），再大类。返回 (items, 来源)，来源是 key、family、key_kw、family_kw；
    都没有返回 ({}, None)。"""
    fine = [s for s in strata if s.startswith("n:") or len(s) == 3]
    coarse = [s for s in strata if s not in fine]
    proc_ok = lambda k: not (age is not None and age >= CHILD_AGE and _has_any(k, _CHILD_ONLY_PROC_WORDS))
    items = _table_where(stats, "procedures", fine, proc_ok)
    if items:
        return items, "key"
    items = {f: n for f, n in _table(stats, "proc_family", fine)["items"].items() if f in PROC_FAMILY_NAME}
    if items:
        return items, "family"
    ok = {f for f in FAMILY_DX_RE if family_ok(f, dx_name)}
    if not ok:
        return {}, None
    items = _table_where(stats, "procedures", coarse, lambda k: proc_family(k) in ok and proc_ok(k))
    if items:
        return items, "key_kw"
    items = _table_where(stats, "proc_family", coarse, lambda f: f in ok and f in PROC_FAMILY_NAME)
    return (items, "family_kw") if items else ({}, None)


def proc_family_of(pkey, psource):
    return pkey if psource in ("family", "family_kw") else proc_family(pkey)


def realize_surgery(pkey, psource, p, rng, lateral_dx):
    """术式名和侧别。分侧术式、诊断没写侧别、两者是同一器官（或诊断就是这个大类的适应证）时，把侧别加进诊断（lateral_dx=True 才加，
    配对模式的诊断来自槽位，不改）。返回 (术式名, 侧别, 新诊断名)。"""
    name = p["primary_dx"]["name"]
    if psource == "slot":
        return pkey, side_of(pkey) or "无", name
    fam = proc_family_of(pkey, psource)
    if psource in ("family", "family_kw") and pkey in FAMILY_FROM_DX:
        head = name[:name.find("骨折") + 2] if "骨折" in name else name
        if lateral_dx and not side_of(head) and LAT_ORGAN.search(head):
            name = lateralize_dx(name, rng.choice(("左", "右")), fam)
            head = name[:name.find("骨折") + 2] if "骨折" in name else name
        pname = head + FAMILY_FROM_DX[pkey]
        return pname, side_of(pname) or "无", name
    if psource in ("family", "family_kw"):
        template = FAMILY_CHILD_NAME[pkey] if p["age"] < 14 and pkey in FAMILY_CHILD_NAME else PROC_FAMILY_NAME[pkey]
    else:
        template = pkey
    pname, side = realize_proc(template, name, rng)
    if lateral_dx and side in ("左", "右", "双") and not side_of(name) and (same_organ(pname, name) or (fam and family_ok(fam, name))):
        name = lateralize_dx(name, side, fam)
    return pname, side, name


def _bare_region_for(key, organ, age=None, dx_name=""):
    """0.3.4：按模态与诊断器官选择合规解剖部位前缀；缺具体器官时按模态安全兜底（不再留下裸 CT/超声）。"""
    if key == "超声":
        if organ == "心":
            return "心"
        if organ == "脑":
            return "头" if (age is not None and age < 1) else "颈"
        if organ == "咽喉":
            return "颈"
        if organ == "骨":
            return "四肢"
        if organ in ("腹", "泌尿", "盆", "颈", "乳腺", "血管"):
            return organ
        return "腹"
    if organ == "咽喉":
        return "头部" if _has_any(dx_name, ("鼻", "窦", "耳", "乳突", "颌")) else "颈"
    if organ in BARE_REGION:
        return BARE_REGION[organ]
    return _MODALITY_DEFAULT_REGION.get(key)


def dedupe_bare(exams, organs, age=None, dx_name=""):
    """0.1.9 / 0.3.4：没写部位的 CT、超声、磁共振、X线，按检查模态+诊断器官补部位；
    补完和已有的重复、或原同名检查已有带部位的就去重（避免出现胸部CT和胸部CT并存或四肢超声/胸部CT错配）。"""
    organ = next((o for o in ORGAN_ORDER if o in organs and (o in BARE_REGION or o == "血管")), None)
    have = {(e["key"], e["region"] if e["region"] in REGION_ORGAN else BENCH_REGION.get(e["region"], e["region"])) for e in exams if e["region"]}
    out = []
    for e in exams:
        if e["key"] in REGIONAL_KEYS and not e["region"]:
            if organ is None and any(k == e["key"] for k, _ in have):
                continue
            reg_txt = _bare_region_for(e["key"], organ, age=age, dx_name=dx_name)
            reg_code = BENCH_REGION.get(reg_txt, reg_txt) if reg_txt else None
            if reg_code and (e["key"], reg_code) not in have:
                disp = REGION_NAME.get(reg_code, reg_code)
                e["region"], e["name"] = reg_code, f"{disp}{e['key']}"
                have.add((e["key"], reg_code))
            else:
                continue
        out.append(e)
    return out


def exam_target(stats, rng, strata):
    """这次住院要有几种检查：按真实每次住院的检查种类数分位数抽（code3 → 章 → 全局）。表里没有返回 0（不补）。"""
    t = stats.get("exam_kinds") or {}
    for s in strata[1:]:
        if (t.get(s) or {}).get("q"):
            return int(round(qdraw(rng, t[s]["q"])))
    return 0


MAX_MEDS = 10
CORE_MIN_SHARE = 0.6   # 主治的一类至少用到住院的几成（公开的建模选择）
_DOSE_MG_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(mg|g|毫克|克)\s*$", re.I)
_PED_MG_PER_KG = {
    "pip_tazo": 112.5, "vancomycin": 20.0, "meropenem": 40.0, "imipenem": 25.0,
    "cefoperazone_sulbactam": 80.0, "ceftriaxone": 80.0, "cefuroxime": 50.0,
    "ceftazidime": 50.0, "cefotaxime": 50.0, "cefepime": 50.0, "amox_clav": 45.0,
    "levofloxacin": 10.0, "azithromycin": 12.0, "metronidazole": 15.0,
    "methylprednisolone": 2.0, "prednisone": 2.0, "prednisolone": 2.0,
    "dexamethasone": 0.5, "hydrocortisone": 5.0,
    "omeprazole": 1.5, "esomeprazole": 1.5, "pantoprazole": 1.5,
    "ibuprofen": 10.0, "acetaminophen": 15.0,
    "ambroxol": 1.5, "furosemide": 2.0, "ondansetron": 0.2,
}
_PED_EXEMPT_WORDS = ("免疫球蛋白", "丙种球蛋白", "白蛋白", "甘露醇", "葡萄糖", "氯化钠")


def _child_dose_ok(p, did, alias, dose):
    """14 岁以下患儿拦截成人级单次克级高剂量，防 g3 漏记抗生素超量。"""
    if p["age"] >= CHILD_AGE or not dose:
        return True
    m = _DOSE_MG_RE.match(dose)
    if not m:
        return True
    val, unit = float(m.group(1)), m.group(2).lower()
    mg = val * 1000.0 if unit in ("g", "克") else val
    w = float(p.get("weight_kg") or weight_p50(p["age"]))
    if did in _PED_MG_PER_KG:
        return mg <= _PED_MG_PER_KG[did] * w * 1.05
    if p["age"] < 6 and mg >= 1000.0 and not _has_any(alias, _PED_EXEMPT_WORDS):
        return mg <= 60.0 * w
    return True


def _med_entry(stats, rng, p, did, los, n_meds, cls, core):
    """一种药：别名、剂量、频次、途径各按统计表抽，过 g3 规则及儿童剂量防呆；不过返回 None。"""
    alias = _pick(rng, _sub(stats, "med_alias", did))
    if not alias:
        return None
    dose = _pick(rng, _sub(stats, "med_dose", did)) or ""
    freq = _pick(rng, _sub(stats, "med_freq", did)) or ""
    route = _pick(rng, _sub(stats, "med_route", did)) or ""
    if dose and not _child_dose_ok(p, did, alias, dose):
        alt = {d: n for d, n in _sub(stats, "med_dose", did).items()
               if _child_dose_ok(p, did, alias, d) and med_allowed(p, alias, d, freq, route)}
        if alt:
            ped_rng = random.Random(seed_of(p.get("patient_id") or "", did, n_meds, "ped_dose"))
            dose = _pick(ped_rng, alt) or ""
        else:
            dose = ""
    if not med_allowed(p, alias, dose, freq, route):
        return None
    chronic = is_chronic(did)
    if chronic:
        # 长期用药：入院就在用，住院期间不停，出院带药（静脉用的除外）
        start, stop = 0, los
        home = route not in ("静滴", "静脉滴注", "静推", "静注", "静脉注射", "泵入")
    else:
        start = rng.choice((0, 0, 0, 1)) if los >= 1 else 0
        lo = max(start, int(round(CORE_MIN_SHARE * los))) if core else start
        stop = rng.randint(min(lo, los), los)
        # 0.1.12：途径为空的只有有口服剂型的才带（drug_class.oral_ok）。随机数照 0.1.11 的条件抽，不改后面的抽签
        home = stop == los and route in ("口服", "") and rng.random() < 0.5 and oral_ok(did, route)
    return {"id": f"m{n_meds + 1}", "drug_id": did, "name": alias, "dose": dose, "freq": freq, "route": route, "start_day": min(start, los), "stop_day": stop,
            "discharge": home, "class": cls, "chronic": chronic, "core": core}


CORE_MIN_P = 0.2   # 主治大类至少要有两成住院用（太少见的大类比值不稳）


def core_class(stats, t_cls):
    """主治的药物大类：本诊断层用得够多（≥ CORE_MIN_P）的大类里，出现率比全院高得最多的那一类（Lift = P(类|诊断) / P(类)）；
    都不够多时取频率最高的。0.1.9 采纳审查员的意见：按频率取，化疗住院会取到止吐药、地塞米松，挤掉化疗药。"""
    items, n = t_cls["items"], max(1, t_cls["n"])
    g = (stats.get("med_class") or {}).get("*") or {}
    gi, gn = g.get("items") or {}, max(1, g.get("n", 1))

    def lift(c):
        return (items[c] / n) / (gi[c] / gn) if gi.get(c) else float("inf")

    ok = [c for c in items if items[c] / n >= CORE_MIN_P]
    if not ok:
        return max(items, key=lambda c: (items[c], c))
    return max(ok, key=lambda c: (lift(c), items[c], c))


def plan_meds(stats, rng, p, strata, los):
    """0.1.9：先按频率抽药物大类（节层……章层，不退到全局），主诊断层频率最高的一类一定要；每类抽一种药，按真实比例再抽第二种。
    大类里的药优先取本诊断层用药表里有的，没有再按大类里全院的分布。统计表没有大类表时照 0.1.8 逐药抽。"""
    fine = [s for s in strata if s != "*"]
    cls_s = next((s for s in fine if ((stats.get("med_class") or {}).get(s) or {}).get("items")), None)
    drug_s = next((s for s in fine if ((stats.get("meds") or {}).get(s) or {}).get("items")), None)
    cls_layer, drug_layer = _layer_name(cls_s) or "none", _layer_name(drug_s) or "none"
    t_cls = _table(stats, "med_class", fine)
    t_drug = _table(stats, "meds", fine)
    meds, seen = [], set()
    # 0.1.11：长期用药要有适应证（第十次 M4，主治的一类除外，见 drug_class.indicated）；14 岁以下只开儿童层用够 k 次的药
    # （本章儿童层 → 全院儿童层，不退到成人层；第十次 4 岁言语发育障碍开了奥氮平）。统计表没有儿童层（0.10 以前）时不过滤
    dxt = "；".join([p["primary_dx"]["name"]] + [c["name"] for c in p.get("comorbidities") or []] + list(p.get("past_history") or []))
    code = p["primary_dx"].get("code3") or ""
    late_preg = bool(p.get("pregnant") and ((p.get("gest_weeks") or 0) >= 20 or _has_any(dxt, _OBSTETRIC_WORDS)))
    kid_ok = None
    if p["age"] < CHILD_AGE:
        kid_ok = next((set(_sub(stats, "meds", s)) for s in child_layers(strata) if _sub(stats, "meds", s)), None)

    def usable(did, core):
        if late_preg and (did in ("mifepristone", "misoprostol", "ethacridine") or any(_has_any(a, _ABORTIFACIENT_WORDS) for a in _sub(stats, "med_alias", did))):
            return False
        if not antitumor_ok(did, dxt):
            return False
        return (kid_ok is None or did in kid_ok) and (core or indicated(did, dxt, code))

    if not t_cls["items"]:
        for did, n in sorted(t_drug["items"].items()):
            if rng.random() >= n / max(1, t_drug["n"]) or len(meds) >= MAX_MEDS or did in seen or not usable(did, False):
                continue
            m = _med_entry(stats, rng, p, did, los, len(meds), drug_class(did), False)
            if m:
                m.update(cls_layer="none", drug_pool=f"local_{drug_layer}", core_reason="")
                seen.add(did)
                meds.append(m)
        return meds
    items = t_cls["items"]
    order = sorted(items, key=lambda c: (-items[c], c))
    top = core_class(stats, t_cls)
    core_how = "lift" if any(items[c] / max(1, t_cls["n"]) >= CORE_MIN_P for c in items) else "freq_fallback"
    picked = [top] + [c for c in order if c != top and rng.random() < items[c] / max(1, t_cls["n"])]
    for c in picked:
        # 先取本诊断层用药表里属于这一类的药，都不过用药规则（儿童用了成人量……）再取这一类在全院的药
        local = {d: n for d, n in t_drug["items"].items() if drug_class(d) == c}
        pools = [(f"local_{drug_layer}", local), ("class_global", {d: n for d, n in _sub(stats, "med_class_drug", c).items() if d not in local})]
        mt = (stats.get("med_class_multi") or {}).get(c) or {}
        want = 2 if rng.random() < (mt.get("items") or {}).get("2+", 0) / max(1, mt.get("n", 1)) else 1
        got, subs = 0, set()
        for pool_tag, within in pools:
            while got < want and within and len(meds) < MAX_MEDS:
                did = _pick(rng, within)
                del within[did]
                # 0.1.11：第二种药不和第一种同亚类（第十次 M9：头孢他啶加头孢曲松）
                if did in seen or not usable(did, c == top) or (subclass(did) and subclass(did) in subs):
                    continue
                m = _med_entry(stats, rng, p, did, los, len(meds), c, c == top)
                if m:
                    m.update(cls_layer=cls_layer, drug_pool=pool_tag, core_reason=core_how if c == top else "")
                    seen.add(did)
                    meds.append(m)
                    subs.add(subclass(did))
                    got += 1
            if got:
                break
    return meds


def discharge_vitals(rng, v, age, base):
    """0.1.9：出院当天的生命体征回到正常范围。成人按公开的正常范围；儿童按统计表儿童体征的 p25–p75。已在范围内的不动。"""
    if age >= 18:
        rng_ = {"T": (36.0, 37.2, 36.3, 36.9), "P": (60, 100, 68, 90), "R": (12, 20, 16, 20), "SBP": (95, 139, 108, 135), "DBP": (60, 89, 64, 85)}
    else:
        rng_ = {k: (q[1], q[3], q[1], q[3]) for k, q in base.items() if k in ("T", "P", "R", "SBP", "DBP")}
    out = dict(v)
    for k, (lo, hi, a, b) in rng_.items():
        if not lo <= out[k] <= hi:
            x = rng.uniform(a, b)
            out[k] = round(x, 1) if k == "T" else int(round(x))
    out["DBP"] = min(out["DBP"], out["SBP"] - 20)
    return out


def symptom_keep_prob(stats, strata, term, in_chief, day, los):
    """0.21：按 S2 symptom_course 分层推症状单日留存概率（表空时返回 None，回退旧 0.75）。"""
    sc = stats.get("symptom_course") or {}
    if not sc:
        return None
    db = prog_day_bucket(day)
    span = 2.0 if db == "1-2" else (3.0 if db == "3-5" else max(2.0, float(los - 5)))
    k_min = stats.get("k", 11)
    for prefix in (f"sym:{term}|{db}", f"role:{'chief' if in_chief else 'accomp'}|{db}"):
        for s in strata:
            items = (sc.get(s) or {}).get("items") or {}
            kp, gn = items.get(f"{prefix}|keep"), items.get(f"{prefix}|gone")
            if kp and gn:
                p_b = kp / (kp + gn)
            elif kp:
                p_b = max(0.85, kp / (kp + k_min - 1))
            elif gn:
                p_b = min(0.15, (k_min - 1) / (gn + k_min - 1))
            else:
                continue
            return min(0.98, max(0.05, p_b ** (1.0 / span)))
    return 0.88 if in_chief else 0.75


def plan_journey(stats, rng, jid, slot=None):
    """返回 Journey；抽不出合格病人时返回 None。"""
    p = draw_patient(stats, rng, jid, slot)
    if p is None:
        return None
    code = p["primary_dx"]["code3"]
    name = p["primary_dx"]["name"]
    strata = strata_of(name, code)
    tkind_pre = treatment_kind(name, code)
    kb_data = stats.get("_kb")
    kb_ent = kb.match_kb(kb_data, code, name, tkind=tkind_pre, age=p["age"]) if kb_data else None
    kb_rng = random.Random(seed_of(jid, "kb_v03")) if kb_ent else None
    if slot:
        los, surgical, sd = int(slot["los"]), bool(slot.get("surgical")), slot.get("surgery_day")
    else:
        los = int(_pick(rng, _table(stats, "los", strata)["items"]) or 7)
        s = _table(stats, "surgery", strata)["items"]
        surgical = rng.random() < s.get("yes", 0) / max(1, s.get("yes", 0) + s.get("no", 0))
        if kb_ent and kb_ent.get("surgical") is False:
            has_c3_surg = bool((((stats.get("surgery") or {}).get(code) or {}).get("items") or {}).get("yes"))
            if not has_c3_surg:
                surgical = False
        sd = None
    los = max(1, los)
    procedures, do_day = [], None
    # 术式先取主诊断名和 code3 两层；取不到再看章层、全局层里适应证关键词对得上的（0.1.6，见 proc_for）
    if surgical and slot and slot.get("procedure"):
        pkey, psource = slot["procedure"], "slot"
    else:
        pkey, psource = proc_for(stats, strata, name, age=p["age"]) if surgical else (None, None)
        pkey = _pick(rng, pkey) if pkey else None
        if surgical and pkey is None and kb_ent and kb_ent.get("surgical") and kb_ent.get("procedure"):
            pkey, psource = kb_ent["procedure"], "pathway"
    if surgical and pkey is None:
        # 配对模式也不做：0.1.3 在这里把诊断名拼上"手术治疗"当术式
        surgical = False
    if surgical:
        if sd is None or not 0 <= int(sd) <= los:
            hist = {d: n for d, n in stats["surgery_day"]["items"].items() if int(d) <= los} if stats.get("surgery_day") else {}
            sd = int(_pick(rng, hist)) if hist else rng.randint(min(1, los), max(min(1, los), los - 1))
        do_day = int(sd)
        pname, side, new_name = realize_surgery(pkey, psource, p, rng, lateral_dx=not slot)
        if new_name != name:
            # stats_name：抽表用的原名（分层键），规划诊断按它算层级
            p["primary_dx"].update(name=new_name, side_added=True, stats_name=name)
            name = new_name
        if proc_family_of(pkey, psource) == "剖宫产" and icd.gest_weeks(name) is None and (p.get("gest_weeks") or 0) < 37:
            # 名字里没写孕周的剖宫产，孕周按足月给，入院记录的停经周数和手术对得上
            p["gest_weeks"] = rng.randint(37, 41)
        procedures.append({"id": "a1", "kind": "手术", "name": pname, "side": side, "decide_day": max(0, do_day - rng.randint(0, 1)), "do_day": do_day,
                           "anesthesia": anesthesia_for(pname), "source": psource})
        rng.random()  # 0.1.7 在这里随机抽麻醉；留一次抽签，后面的规划不跟着变
    # 0.1.9：治疗性住院（化疗、复查……）的原发病；检查结果、主诉都用它
    tkind = treatment_kind(name, code)
    under = underlying_dx(name, [c["name"] for c in p["comorbidities"]]) if tkind else None
    if tkind:
        p["primary_dx"]["treatment"] = tkind
        if under:
            p["primary_dx"]["underlying"] = under
    lctx = lab_ctx(rng, stats, strata, p)
    organs = dx_organs(under or name, code)
    p["primary_dx"]["organs"] = sorted(organs)
    fine_set = set(fine_strata(strata))
    used_invasive = set()

    glob_ex = (stats.get("exams") or {}).get("*") or {}

    def invasive_ok(item, layer):
        """审查员第八轮补充：有创检查只从节层以内取，同类（穿刺、组织、造影、内镜）一次住院只要一项。
        退到节层以外（章层）取的只要通用检查：这一层出现率比全院高不到 COARSE_LIFT 倍的，或部位和诊断器官对得上的
        （采纳审查员的意见：章层的专科检查是别的病的，0.1.8 痔从循环系统章抽到了心脏超声）。全局层的比值是 1，都算通用。"""
        key, _, region = item.partition("|")
        kind = KIND_OF.get(key, "其他")
        coarse = layer not in fine_set
        if coarse and layer != "*":
            t_l = (stats.get("exams") or {}).get(layer) or {}
            n_l, n_g = (t_l.get("items") or {}).get(item, 0), (glob_ex.get("items") or {}).get(item, 0)
            # 全局层是章层的超集，真实统计表里章层凑够 k 的全局层一定有；全局层没有（只出现在手写的自检表里）就不挡
            lift = (n_l / max(1, t_l.get("n", 1))) / (n_g / max(1, glob_ex.get("n", 1))) if n_g else 0.0
            if lift >= COARSE_LIFT and not site_ok(key, region, organs):
                return False
        if kind not in INVASIVE:
            return True
        if key == "病理" and tkind:
            return False
        grp = "组织" if kind in ("取材", "病理") else kind
        if coarse or grp in used_invasive:
            return False
        used_invasive.add(grp)
        return True

    exams, i = [], 1
    t = _table(stats, "exams", strata)
    t_layer = next((s for s in strata if ((stats.get("exams") or {}).get(s) or {}).get("items")), None)
    for item, n in sorted(t["items"].items()):
        if rng.random() >= n / max(1, t["n"]) or not invasive_ok(item, t_layer):
            continue
        key, _, region = item.partition("|")
        exams.append(_exam(stats, rng, i, key, region, los, ctx=lctx))
        i += 1
    if not any(e["key"] == "血常规" for e in exams):
        exams.append(_exam(stats, rng, i, "血常规", "", los, day=0, ctx=lctx))
        i += 1
    # 逐项独立抽的期望种类数比真实少（凑不够 k 次的少见检查都被压掉了）：不够抽到的种类数就按权重不放回补齐。
    # 0.1.5：本层候选用完了再往粗层取（0.1.4 只在本层补，候选不够就停）。0.1.9：有创检查不拿来凑数
    target = exam_target(stats, rng, strata)
    have = {f"{e['key']}|{e['region'] or ''}" for e in exams}
    for s in strata:
        rest = {x: n for x, n in (((stats.get("exams") or {}).get(s) or {}).get("items") or {}).items() if x not in have}
        while len(have) < target and rest:
            item = _pick(rng, rest)
            del rest[item]
            if not invasive_ok(item, s):
                continue
            key, _, region = item.partition("|")
            exams.append(_exam(stats, rng, i, key, region, los, ctx=lctx))
            exams[-1]["filled"] = True
            have.add(item)
            i += 1
    # 0.1.9：诊断相关的检查一项都没有时补一项（出现率比值最高、部位对得上的）；0.2.0.1：治疗性住院写"既往治疗改变"，避免"治疗后"触发 DS 入院情况 K4
    fname = (f"{under}既往治疗改变" if under else None) if tkind else finding_label(name, code)
    add = ensure_diagnostic(stats, strata, exams, fname, organs)
    if add:
        old = next((e for e in exams if (e["key"], e["region"] or "") == (add[0], add[1])), None)
        if old is not None:
            # 0.1.11：比值在 DIAG_ADD_MIN 和 DIAG_LIFT_SITE 之间、已经抽到的那项，直接算诊断相关，不再补一项同名的
            old["guaranteed"] = True
        else:
            exams.append(_exam(stats, rng, i, add[0], add[1], los, ctx=lctx))
            exams[-1]["guaranteed"] = True
            i += 1
    if kb_ent and kb_ent.get("exams_required"):
        # 0.30：融入路线 B 必查清单中缺失的核心非有创项目（最多 3 项，独立 kb_rng）
        added_kb = 0
        have_keys = {e["key"] for e in exams}
        fine0 = (fine_strata(strata) or ["*"])[0]
        for kex in kb_ent["exams_required"]:
            if added_kb >= 3:
                break
            kk, kr = kex["key"], kex.get("region") or ""
            if kk in have_keys or KIND_OF.get(kk, "其他") in INVASIVE:
                continue
            if not invasive_ok(f"{kk}|{kr}", fine0):
                continue
            exams.append(_exam(stats, kb_rng, i, kk, kr, los, ctx=lctx))
            exams[-1]["kb_required"] = True
            have_keys.add(kk)
            added_kb += 1
            i += 1
    exams = dedupe_bare(exams, organs, age=p["age"], dx_name=under or name)
    if fname and organs and not tkind and not any(
        not e["items"] and diag_eligible(e["key"]) and e["kind"] in ("造影", "内镜", "影像") and site_ok(e["key"], e["region"], organs)
        for e in exams
    ):
        dk, dr = (
            ("心脏超声", "") if "心" in organs else
            ("CT", "头") if "脑" in organs else
            ("CT", "胸") if "肺" in organs else
            ("磁共振", "脊柱") if "脊柱" in organs else
            next((("超声" if o in ("腹", "泌尿", "盆", "颈", "乳腺") else "CT", BARE_REGION[o])
                  for o in ORGAN_ORDER if o in organs and o in BARE_REGION), (None, None))
        )
        if dk and (dk, dr or None) not in {(e["key"], e["region"]) for e in exams}:
            diag_rng = random.Random(seed_of(jid, "diag_img_v032"))
            exams.append(_exam(stats, diag_rng, i, dk, dr, los, ctx=lctx))
            i += 1
    # 0.1.11（第十次 M1）：手术表里有的操作，检查表里同名的那项并进手术：开单在决定手术那天，结果在手术之后才有
    # （0.1.10 入院记录辅助检查写"冠脉造影：未见明显异常"，当天 18 时才做冠脉造影术）。手术当天出院的，结果在出院记录；
    # 入院当天做、当天出院的，检查删掉，只留手术
    for a in procedures:
        keep = []
        for e in exams:
            if not proc_has_exam(a["name"], e["key"]):
                keep.append(e)
                continue
            if los == 0:
                continue
            e["order_day"], e["result_day"], e["with_proc"] = a["decide_day"], min(los, a["do_day"] + 1), a["id"]
            e.pop("prior", None)
            keep.append(e)
        exams = keep
    base = next(e for e in exams if e["key"] == "血常规")
    first = max(base["result_day"] + 1, 1)
    # 首次血常规在出院当天回报时没有复查的日子（0.1 在这里 randint 空区间报错）
    if los >= 3 and first <= los and rng.random() < 0.5:
        exams.append(_exam(stats, rng, i, "血常规", "", los, day=rng.randint(first, los), ctx=lctx))
        rc = exams[-1]
        rc["recheck"] = True
        # 0.1.9：复查以入院值为起点，往正常方向走（0.1.8 复查和入院各抽各的，血红蛋白 118 → 69 照样"符合出院标准"）
        by = {x["key"]: x for x in base["items"]}
        rc["items"] = [recheck_value(rng, by[x["key"]], rc["result_day"], los) if x["key"] in by else x for x in rc["items"]]
        i += 1
    if surgical and rng.random() < 0.6 and takes_tissue(procedures[0]):
        exams.append({"id": f"e{i}", "key": "病理", "region": None, "name": "术后病理", "kind": "病理", "order_day": do_day, "result_day": min(los, do_day + rng.randint(2, 5)), "items": []})
        i += 1
    mark_exam_findings(stats, strata, exams, under if tkind else finding_label(name, code), organs)
    if surgical and do_day is not None:
        for e in exams:
            if (e.get("diagnostic") and e["kind"] == "影像" and not e.get("with_proc")
                    and not any(proc_has_exam(a["name"], e["key"]) for a in procedures)
                    and e["result_day"] > do_day):
                e["result_day"] = do_day
                e["order_day"] = min(e["order_day"], do_day)
    # 0.1.11（第十次合成病历 11）：阳性的培养至少在出院前一天回报（0.1.10 出院当天才回报"培养出致病菌"，照常出院）
    for e in exams:
        if e["kind"] == "培养" and e.get("abnormal") and los >= 1 and e["result_day"] >= los:
            e["result_day"] = los - 1
            e["order_day"] = min(e["order_day"], e["result_day"])
    if tkind and under:
        # 治疗性住院的影像、内镜写"符合××既往治疗改变"（0.2.0：功能检查仍只写"可见异常改变"，不带诊断名）
        for e in exams:
            if e.get("diagnostic") and e["kind"] not in ("培养", "病理", "取材", "功能"):
                e["finding"] = FINDING_TEXT.get(e["kind"], "符合{f}表现").format(f=fname).replace("表现", "")
    # 0.1.12：只记不改。没有诊断相关检查的原因（exam_diag_none 第十、十一次都是 248/500），首次血常规各项的分位数层、异常比例层
    nr = diag_none_reason(exams, tkind, fname, organs)
    exam_diag = {"none_reason": nr, "cbc": cbc_sources(stats, strata, lctx, base),
                 "no_cand_detail": no_candidate_detail(stats, strata, organs) if nr == "no_candidate" else None}
    meds = plan_meds(stats, rng, p, strata, los)
    dx_events, did = [], 1
    # 0.1.12（第十一次合成病历 13："恶性肿瘤维持性化学治疗？"）：化疗、放疗、复查这类住院是来做已定的治疗，主诊断不走疑似。
    # 随机数照抽，不改后面的抽签
    if los >= 2 and rng.random() < P_DX_CHANGE and not tkind:
        dx_events.append({"id": f"d{did}", "name": name, "day": 0, "state": "疑似"})
        confirm = min(los, max([e["result_day"] for e in exams] + [1]))
        dx_events.append({"id": f"d{did + 1}", "name": name, "day": confirm, "state": "确诊"})
        did += 2
    else:
        dx_events.append({"id": f"d{did}", "name": name, "day": 0, "state": "确诊"})
        did += 1
    differentials = []
    ch = code[:1]
    taken = [name] + [c["name"] for c in p["comorbidities"]]
    # 0.1.11（第十次 M3）：鉴别诊断只取和主诊断同一器官系统的（器官从 ICD-10 结构推，同检查部位核对）。0.1.10 心梗要和脑梗死鉴别：
    # 循环系统章里心和脑混在一起
    named = {n: c for c, v in (stats.get("dx_names") or {}).items() if c[:1] == ch and c != code for n in v["items"]}
    pool = set(named) | set(_sub(stats, "dx_names_ch", ch))
    names = sorted(n for n in pool if dx_ok(n, p["sex"], p["age"]) and not any(n in x or x in n for x in taken)
                   and dx_organs(n, named.get(n, "")) & organs)
    if names and rng.random() < P_DIFF:
        for n in rng.sample(names, min(2, len(names))):
            differentials.append(n)
            dx_events.append({"id": f"d{did}", "name": n, "day": 0, "state": "疑似"})
            did += 1
            if rng.random() < P_DIFF_CLOSE and los >= 1:
                dx_events.append({"id": f"d{did}", "name": n, "day": rng.randint(1, los), "state": "排除"})
                did += 1
    # 0.1.9：伴随症状、主诉症状表都只到节层（0.1.8 退到章层，痔抽到"胸闷""头晕"）
    fs = fine_strata(strata)
    t_sym = _table(stats, "symptoms", fs)
    syms = [(s, n) for s, n in sorted(t_sym["items"].items()) if sym_ok(s, p)]
    drawn = [s for s, n in syms if rng.random() < n / max(1, t_sym["n"])]
    cc_layer = next((s for s in fs if any(sym_ok(x, p) for x in (((stats.get("chief_symptoms") or {}).get(s) or {}).get("items") or {}))), None)
    chief, extra = [], {}
    if tkind:
        # 化疗、放疗、复查：主诉写这次来做什么，不写症状
        ctext, ckey, dur = treatment_chief(tkind, under, rng)
        extra = {"text": ctext, "key": ckey, "kind": tkind}
        source = "treatment"
    else:
        # 0.1.8：主诉从主诉症状表按出现频率抽 1–2 个；伴随症状按频率排在后面，一共不超过 5 个
        cc_pool = {x: n for x, n in stats["chief_symptoms"][cc_layer]["items"].items() if sym_ok(x, p)} if cc_layer else {}
        for _ in range(1 if rng.random() < 0.5 else 2):
            x = _pick(rng, cc_pool)
            if x is None:
                break
            chief.append(x)
            del cc_pool[x]
        source = _layer_name(cc_layer) if chief else None
        if not chief and syms:
            chief, source = [max(syms, key=lambda x: (x[1], x[0]))[0]], "symptoms"
        if not chief:
            # 0.1.9.1（第九次 S1-A：level_chief_none 49/500 写"不适"）：章的器官系统和诊断的器官系统一致时（呼吸章的病、消化章的病），
            # 可以用章层的主诉表；不一致（循环系统章里的痔、脑梗死）照旧不用。器官都从 ICD-10 的分类结构推，不按病名写规则
            ch = (code or "")[:1]
            ch_tab = {x: n for x, n in ((((stats.get("chief_symptoms") or {}).get(ch) or {}).get("items") or {}).items()) if sym_ok(x, p)}
            if ch_tab and CHAPTER_ORGAN.get(ch) and CHAPTER_ORGAN[ch] in organs:
                chief, source = [_pick(rng, ch_tab)], "chapter_organ"
            else:
                # 0.1.11（第 11 条）：记下章层回退被哪一条挡住，plan_diag 按 code3 计数，拿到计数再改
                extra["none_reason"] = ("no_chapter" if not ch else "chapter_no_organ" if not CHAPTER_ORGAN.get(ch)
                                        else "organ_mismatch" if CHAPTER_ORGAN[ch] not in organs else "chapter_table_empty")
                # 0.1.12（第十一次"不适"49 份：N13 章层表空 36、I84 章层器官对不上 13）：再退一层"器官合并层"，按诊断所在的器官系统
                # 把各章的住院合起来建主诉表（stats 0.11 的 org:器官，k 照压），器官照旧从 ICD-10 结构推。数据驱动，不按病名写症状
                for o in sorted(organs, key=lambda x: ORGAN_ORDER.index(x) if x in ORGAN_ORDER else len(ORGAN_ORDER)):
                    o_tab = {x: n for x, n in ((((stats.get("chief_symptoms") or {}).get(f"org:{o}") or {}).get("items") or {}).items()) if sym_ok(x, p)}
                    if o_tab:
                        chief, source = [_pick(rng, o_tab)], "organ"
                        extra["organ"] = o
                        break
                else:
                    extra["organ_reason"] = "no_organ" if not organs else "organ_table_empty"
        if kb_ent and source in ("organ", "none") and kb_ent.get("chief_symptoms"):
            kb_cands = [x for x in kb_ent["chief_symptoms"] if sym_ok(x, p)]
            if kb_cands:
                pick_n = min(len(kb_cands), 2 if kb_rng.random() < 0.45 else 1)
                chief = kb_rng.sample(kb_cands, pick_n)
                source = "pathway"
        if not chief:
            # 不按病名补症状（不写病种规则），照 0.1.8 写"不适"，plan_diag 计 level_chief_none 看有多少
            chief, source = ["不适"], "none"
        dur = None
    if tkind:
        cc_cat = "scheduled"
    elif p.get("pregnant") or any(x in NON_EVOLVING_SYMS or x in ("见红", "破水", "胎动", "胎动减少") for x in chief):
        cc_cat = "obstetric"
    elif _has_any("".join(chief), _LESION_WORDS):
        cc_cat = "lesion"
    else:
        cc_cat = "acute"
    extra["category"] = cc_cat
    freq = t_sym["items"]
    rest = sorted((s for s in drawn if s not in chief), key=lambda s: (-freq.get(s, 0), s))
    chosen = (chief + rest)[:5] if not tkind else rest[:3]
    # 0.21.3：主诉+伴随症状不足 3 项时，用独立子随机数按频次权重不放回补齐至随机 2-4 项（不动主 rng 序列）
    if not tkind and len(chosen) < 3 and len(syms) >= 4:
        pool_sym = {s: n for s, n in syms if s not in chosen}
        if pool_sym:
            sym_rng = random.Random(seed_of(jid, "sym_topup"))
            want_n = sym_rng.randint(2, 4)
            while len(chosen) < want_n and pool_sym:
                pick_s = _pick(sym_rng, pool_sym)
                del pool_sym[pick_s]
                chosen.append(pick_s)
    if dur is None:
        dur = _pick(rng, _table_where(stats, "durations", strata, lambda x: duration_ok(x, p["age"]))) or f"{rng.randint(1, 9)}天"

    def _ob_acute_dur(d_txt, words):
        m_w = _DUR.search(d_txt or "")
        if m_w and (m_w.group(2) in ("年", "月") or (m_w.group(2) == "周" and int(m_w.group(1)) >= 2)):
            ob_rng = random.Random(seed_of(jid, "ob_other_dur"))
            return f"{ob_rng.randint(2, 12)}小时" if _has_any("".join(words), ("阴道流液", "破水", "见红", "腹痛")) else f"{ob_rng.randint(1, 3)}天"
        return d_txt

    if not tkind and p.get("pregnant") and p.get("gest_weeks"):
        if "停经" in chief:
            if chief == ["停经"]:
                dur = f"{p['gest_weeks']}周"
            else:
                others = [s for s in chief if s != "停经"]
                extra["text"] = f"停经{p['gest_weeks']}周，{'、'.join(others)}{_ob_acute_dur(dur, others)}"
        elif _has_any("".join(chief), ("阴道流液", "破水", "见红", "胎动减少")):
            dur = _ob_acute_dur(dur, chief)
    v0 = _vitals(stats, p["age"])
    cur = {k: qdraw(rng, q) for k, q in v0.items()}
    # 夹的范围：成人照 0.1.8；14 岁以下按年龄范围放宽一点（0.1.9）
    cr = child_range(p["age"])
    clamp = {"SBP": (85, 185), "DBP": (48, 110), "P": (48, 150), "R": (12, 40)}
    if cr:
        clamp = {k: (lo - 8, hi + 15) for k, (lo, hi) in cr.items()}
    # 0.30：非演进主诉词（停经/妊娠/待产）不进入每日演进症状列表，杜绝“停经已消失”；实体肿物/疝术前不消退、术后消退
    daily, today = [], [s for s in chosen if s not in NON_EVOLVING_SYMS]
    has_sc = bool(stats.get("symptom_course"))
    for day in range(los + 1):
        # 症状逐日减少：有 S2 symptom_course 表时按分层留存概率演变，表空回退旧 0.75 逻辑
        if day:
            if not has_sc:
                base_next = today[:1] + [s for s in today[1:] if rng.random() < 0.75]
            else:
                nxt = []
                for idx_s, s in enumerate(today):
                    if day == 1 and idx_s == 0 and s in chief:
                        nxt.append(s)
                    elif rng.random() < symptom_keep_prob(stats, strata, s, s in chief, day, los):
                        nxt.append(s)
                base_next = nxt
            if cc_cat == "lesion":
                resected = surgical and do_day is not None and day > do_day
                keep_lesions = [s for s in today if _has_any(s, _LESION_WORDS) and not resected]
                today = keep_lesions + [s for s in base_next if s not in keep_lesions and not (resected and _has_any(s, _LESION_WORDS))]
            else:
                today = base_next
        if day:
            for k, sd_ in (("T", 0.2), ("P", 4), ("R", 1), ("SBP", 5), ("DBP", 3)):
                cur[k] = cur[k] + rng.gauss(0, sd_)
        t_ = round(min(max(cur["T"], 36.0), 38.8), 1)
        s_, d_ = (int(min(max(cur[k], clamp[k][0]), clamp[k][1])) for k in ("SBP", "DBP"))
        d_ = min(d_, s_ - 20)
        daily.append({"day": day, "T": t_, "P": int(min(max(cur["P"], clamp["P"][0]), clamp["P"][1])), "R": int(min(max(cur["R"], clamp["R"][0]), clamp["R"][1])),
                      "SBP": s_, "DBP": d_,
                      "trend": "同前" if day == 0 else rng.choice(("好转", "好转", "平稳", "平稳", "减轻")), "symptoms": today})
    if los >= 1:
        # 0.1.9：出院当天回到正常范围
        daily[-1].update(discharge_vitals(rng, {k: daily[-1][k] for k in ("T", "P", "R", "SBP", "DBP")}, p["age"], v0))
    j = {"journey_id": jid, "contract": CONTRACT_VERSION, "patient": p, "admit_date": fictional_date(jid), "los": los, "surgical": surgical, "surgery_day": do_day,
         "chief": dict({"symptoms": chief, "duration": dur, "source": source}, **extra), "exams": exams, "meds": meds, "procedures": procedures,
         "dx_events": dx_events, "differentials": differentials, "daily": daily, "exam_target": target, "exam_diag": exam_diag}
    if kb_ent:
        j["kb_facts"] = {
            "icd10": kb_ent["icd10"],
            "pathway": kb_ent["pathway"],
            "signs": list(kb_ent.get("signs") or ()),
            "postop_watch": list(kb_ent.get("postop_watch") or ()),
            "discharge_criteria": list(kb_ent.get("discharge_criteria") or ()),
        }
    return j


def diag_none_reason(exams, tkind, fname, organs):
    """一项诊断相关检查都没有的原因（有就返回 None）。顺序就是 mark_exam_findings、ensure_diagnostic 挡住它的顺序：
    treatment 化疗、放疗、复查这类住院；no_label 结论里没有可写的诊断名；labs_only 只有带数值的化验和病理（这两样按设计不算诊断相关）；
    no_organ 诊断查不到器官，比值线用 2.0；no_candidate 有影像、内镜这类检查，但出现率比值都不够、也没有能补的。"""
    if any(e.get("diagnostic") for e in exams):
        return None
    if tkind:
        return "treatment"
    if not fname:
        return "no_label"
    if not [e for e in exams if not e["items"] and e["key"] not in ("病理", "活检") and "abnormal" in e]:
        return "labs_only"
    if not organs:
        return "no_organ"
    return "no_candidate"


def no_candidate_detail(stats, strata, organs):
    """0.2.0（第 6 条，只计数）：no_candidate 进一步拆细——节层以内检查表为空、没有器官对得上的候选、还是比值不到 DIAG_ADD_MIN。"""
    t = stats.get("exams") or {}
    fine = next((t[s] for s in fine_strata(strata) if (t.get(s) or {}).get("items")), None)
    if not fine:
        return "fine_table_empty"
    lifts = exam_lifts(stats, strata, organs)
    if not lifts:
        return "no_eligible_site"
    return "lift_below_add_min" if max(lifts.values()) < DIAG_ADD_MIN else "other"


def _layer_kind(s):
    if s is None:
        return "ref"
    if s.startswith("kid:"):
        return "kid_all" if s == "kid:*" else "kid_chapter"
    return _layer_name(s) or "global"


def lab_rate_layer(stats, strata, key):
    """lab_rate 的比例取自哪一层：两格都凑够 k 的第一层；没有返回 bounded（用上下界夹）或 default。"""
    t = stats.get("lab_state") or {}
    bounded = False
    for s in strata:
        items = (t.get(s) or {}).get("items") or {}
        a, n = items.get(f"{key}|abn"), items.get(f"{key}|norm")
        if a and n:
            return _layer_kind(s)
        bounded = bounded or bool(a or n)
    return "bounded" if bounded else "default"


def cbc_sources(stats, strata, ctx, base):
    """首次血常规每项：分位数取自哪层、异常比例取自哪层、抽出来偏高／偏低／正常（只记，给 plan_diag 拆血小板、血红蛋白偏多的原因）。"""
    by = stats.get("labs_by") or {}
    out = {}
    for x in base["items"]:
        key = x["key"]
        qs = next((s for s in (ctx.get("q_layers") or ()) if (by.get(s) or {}).get(key)), None)
        q = _layer_kind(qs) if qs else ("global" if (stats.get("labs") or {}).get(key) else "ref")
        out[key] = {"q": q, "rate": lab_rate_layer(stats, ctx.get("rate_strata") or ctx["strata"], key), "flag": x["flag"]}
    return out


def _layer_name(s):
    if s is None:
        return None
    if s.startswith("n:"):
        return "name"
    if s.startswith("blk:"):
        return "block"
    return "code3" if len(s) == 3 else ("global" if s == "*" else "chapter")


def schedule(j, stats, rng):
    """DocSpec 列表。入院记录、首程在第 0 天；上级查房按 pb_subtypes.by_los 分档多篇分布（表空回退第 1–2 天 1 篇）；手术前后有术前小结和术后首程；其余日常病程按真实的每日篇数抽。"""
    los = j["los"]
    sd = j.get("surgery_day")
    out = [("hospital_record", "入院记录", "HR", 0, 0), ("progress_note", "首次病程记录", "FP", 0, 1)]
    q = (stats.get("notes_per_day") or {}).get("q") or [0.3, 0.5, 0.8, 1.0, 1.2]
    rate = max(0.25, qdraw(rng, q))
    b_row = ((stats.get("pb_subtypes") or {}).get("by_los") or {}).get(los_bucket(los))
    sup_weights = (
        [(SUP_SUBTYPE_CN[sub], v["notes"]) for sub, v in sorted((b_row.get("sub") or {}).items())
         if sub in SUP_SUBTYPE_CN and v.get("notes", 0) > 0]
        if b_row and b_row.get("admissions", 0) > 0 else []
    )
    if not sup_weights:
        sup_day = min(los, rng.choice((1, 1, 2)))
        if los >= 1:
            out.append(("progress_note", rng.choices([s for s, _ in SUP_TYPES], weights=[w for _, w in SUP_TYPES])[0], "SUP", sup_day, 2))
    elif los >= 1:
        mu = sum(w for _, w in sup_weights) / b_row["admissions"]
        n_sup = int(mu) + (1 if rng.random() < (mu - int(mu)) else 0)
        max_sup = 1 if los <= 1 else min(6, los - 1)
        n_sup = max(1, min(n_sup, max_sup))
        d0 = min(los - 1 if los >= 2 else los, rng.choice((1, 1, 2)))
        sup_days = [d0]
        if n_sup > 1:
            op_day = sd if j["surgical"] else None
            pref = [d for d in range(1, los) if d != d0 and d != op_day]
            pool = pref if len(pref) >= n_sup - 1 else [d for d in range(1, los) if d != d0]
            k_rem = n_sup - 1
            for idx_s in range(k_rem):
                lo_i = int(round(idx_s * len(pool) / k_rem))
                hi_i = min(len(pool), max(lo_i + 1, int(round((idx_s + 1) * len(pool) / k_rem))))
                sup_days.append(rng.choice(pool[lo_i:hi_i]))
            sup_days.sort()
        cns, cws = [s for s, _ in sup_weights], [w for _, w in sup_weights]
        for sup_day in sup_days:
            out.append(("progress_note", rng.choices(cns, weights=cws)[0], "SUP", sup_day, 2))
    if j["surgical"]:
        if sd >= 1:
            out.append(("progress_note", "术前小结", "PRE", max(1, sd - 1) if sd > 1 else sd, 3))
        out.append(("progress_note", "术后首次病程记录", "POST", sd, 4))
    for day in range(1, los):
        if rng.random() < min(1.0, rate):
            out.append(("progress_note", "日常病程记录", "DAILY", day, 5))
    out.append(("discharge_summary", "出院记录", "DS", los, 9))
    out.sort(key=lambda x: (x[3], x[4]))
    specs = [{"k": k, "doc_type": t, "doc_type_cn": cn, "role": r, "day": d, "author": "主治医师" if r == "SUP" else "住院医师"}
             for k, (t, cn, r, d, _) in enumerate(out)]
    times = doc_times({"slot_id": j["journey_id"], "skeleton": specs})
    for s in specs:
        s["time"] = times[s["k"]]
    return specs


def check_plan(j, specs):
    return validate_journey(j) + validate_docspecs(specs, j["los"])


def journey_to_draft(j):
    """换成 g2_kit 的底稿格式，oracle2 按它量真未来泄露。"""
    p = j["patient"]
    obs = [{"id": e["id"], "name": e["name"], "kind": e["kind"] if e["kind"] in ("影像", "检验", "内镜", "病理", "功能") else "其他", "order_day": e["order_day"], "result_day": e["result_day"],
            "result": exam_result_text(e)} for e in j["exams"]]
    jud = [{"day": d["day"], "diagnosis": d["name"], "state": d["state"], "basis": []} for d in j["dx_events"]]
    act = [{"id": a["id"], "kind": a["kind"], "name": a["name"], "side": a["side"], "plan_day": a["decide_day"], "do_day": a["do_day"]} for a in j["procedures"]]
    return {"final_diagnoses": final_diagnoses(j), "presentation": {"chief_complaint": chief_text(j), "history": "", "past_history": "、".join(p["past_history"]), "exam": ""},
            "observations": obs, "judgments": jud, "actions": act}


def exam_result_text(e):
    if e["items"]:
        return "，".join(f"{x['item']}{x['value']}{x['unit']}" for x in e["items"])
    if e.get("finding"):
        # 0.1.9：按检查类型写（mark_exam_findings 给的）：影像"符合××表现"、培养"未培养出致病菌"、病理"符合××"
        return e["finding"]
    if e["key"] == "病理":
        return "未见恶性证据"
    if "abnormal" in e:
        return "可见异常改变" if e["abnormal"] else "未见明显异常"
    return "未见明显异常"


def chief_text(j):
    c = j["chief"]
    return c.get("text") or ("、".join(c["symptoms"]) + c["duration"])


def chief_key(j):
    """必写事实里主诉的关键词：第一个主诉症状；化疗这类住院是"化疗"。"""
    c = j["chief"]
    return c.get("key") or (c["symptoms"][0] if c["symptoms"] else None)


def final_diagnoses(j):
    return [j["patient"]["primary_dx"]["name"]] + [c["name"] for c in j["patient"]["comorbidities"]]

