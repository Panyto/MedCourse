"""L2 统计底座 v0.2（S1）：现有规则包在 fold 0 上抽实体、用药、化验数值，分层计数，只留 k≥11 的格子。

python -B -m medcourse.stats --path 导出文件 --dx-map dx_map.csv --copyops S0_copyops.csv --out product/derived/fold0/stats_v0.json

产物 stats_v0.json 含真实表面写法（诊断名、科室名），算真实派生材料，不出 ys。stats_summary.csv 只有表大小和压掉的格子数，可以进包。
S2 的抽取产物出来以后，这张表换成 Events 算的，字段不变。

0.2（S1S2 第一轮结果）：fold 0 只有约 1,600 次住院，0.1 按原串计诊断、按五元组计用药、手术名带侧别，k=11 下几乎全被压掉。
现在诊断和合并症按 norm_dx 归一后计；用药拆成药、别名、剂量、频次、途径五张表分别计；手术名去掉麻醉前缀，侧别换成占位符。
分层从 全局/章 扩到 全局/章/code3/主诊断名，规划器从最细的非空层往粗的退。

0.3（S1S2 第二次结果：模板写手 10.8% 丢弃，全是 N1 的血红蛋白）：血红蛋白的读数排除 MCH、MCHC（"平均红细胞血红蛋白浓度"约 330 g/L），
和 g3-rules-0.3 的口径一致；所有化验读数先过生理范围再算分位数，剔掉的条数记进 labs_dropped。另记手术住院数和读到术式的住院数。

0.4（S1S2 第三次结果和监督员第三轮审查：规划 0 例手术，外科、产科章一例没有）：dx_map 只做整串精确匹配，fold 0 有 42% 的住院
没有 code3，抽不成主诊断。没有 code3 的住院用公开词典按主干补（去"伴…"、去侧别），补之前先在已有 code3 的住院上自检，
一致率够才启用（dx_map_meta）。另加：诊断名按主干计数（dx_stems），给规划器兜底；术式去掉"腹腔镜下"一类写法差异，
再按大类计数（proc_family）；每次住院的检查种类数分位数（exam_kinds，和评测台 B_exam 同口径）；血红蛋白新旧两种读法
各匹配多少条（labs_meta），直接核对第二次丢弃的根因。

0.5（S1S2 第四次结果和监督员第四轮审查）：术式大类表在 0.4 里漏登记分母（没调 adm），整张表被 k 压空，大类兜底没生效；
现在补上。诊断主干同时按章计数：O 章这类没有一个 code3 凑够 k 次的章，0.4 仍整章抽不到。另记真实病程里写了生命体征的比例
（note_vitals），0.4 每篇病程都由程序写全四项体征，不像真实病程。

0.7（S1S2 第六次结果）：表的内容不变，只多一项 blocked_chapters：联合分布里有、但 code3、诊断名、主干都没有一项凑够 k 次的章
（第六次的 O 章，58 次住院），记住院数、code3 种数和最多一种的次数、诊断名和主干同样两个数，看离 k 还差多少。只有计数。

0.8（第七次合成病历审阅、监督员第七轮）：
- 症状：0.7 把主诉、现病史里出现的症状词全算成"有"，没排除否定。真实现病史几乎每份都写"无发热、恶心、呕吐"，这几个词成了
  所有诊断最常见的症状，合成病历 32/47 的主诉以"发热"开头。现在按分句判否定（分句里在症状词之前出现"无、否认、未见、未诉、
  不伴、没有"就不算；"无明显诱因"不算否定），另立一张主诉症状表（只数主诉段），规划器的主诉从这张表抽。
- 科室：0.7 只截 2–12 个字，把紧跟着的"年龄"也截进去了（"心血管内科南院病区年龄"），长的被截断。现在截到 24 个字，
  再在表头字段（年龄、性别、姓名、床号……）处切开；院区字样（南院、北院……）去掉。
- 途径：药名后 30 字里找途径，跨过标点吸了别的药的"雾化"。现在只看到第一个标点（含顿号）为止。
- 分层多一层 ICD-10 的节（icd.block_of，公开分类），在 code3 和章之间。
- 只出计数的诊断：首次病程以外的全部病程按小类（主任医师查房、日常查房……）各几篇（pb_subtypes），
  看真实每次住院 6.3 篇、合成 3.0 篇差在哪类。0.8 的说明写成"首程之后、出院记录之前"，实际没按 progress_between 截窗（伙计第八次指出），
  0.9 改了说明，口径不变。

0.9（第八次合成病历审阅，老板批准的 v0.1.9 十条）：
- 症状词表补一份公开的专科主诉词（CHIEF_EXTRA：肛门肿物、排尿不畅、阴道流血……）。评测台的词表没有"肛门肿物""脱出"，
  痔的主诉数不到症状，主诉症状表在名称、code3、节三层被 k 压空，规划退到章层，痔抽到了"胸闷""头晕"。
- 化验异常的比例按诊断层计（lab_state：每次住院每项第一次读数在参考范围内外各几次）。0.8 每项固定 25% 异常，入院血常规 8 份里 7 份有极端值。
- 用药按公开药物大类计（med_class，drug_class.py）；大类里各药几次（med_class_drug）、一次住院用两种以上同类药的比例（med_class_multi）。
- 病程里写"精神、睡眠、饮食""二便"的比例（note_general，同 out_diag 套话口径）。规划按这个比例给一般情况的事实，不靠禁止套话。
- 病程小类按住院天数分档计（pb_subtypes.by_los），给下一版排查房和会诊。
- 真实的几项比例（real_rates，只有计数）：入院血常规有极端值、血红蛋白波动 ≥30、总胆红素 ≥100、没有用药、成人出院生命体征超出范围，和规划同口径直接比。

0.6（S1S2 第五次结果和监督员第五轮审查）：本文件的计数代码没改；诊断主干的键（icd.stem_key）把产科的孕周、孕产次换成占位符，
补 code3 用的主干候选（icd.stems）多一个去掉这些写法的。键变了，表要重算，版本跟着升。
"""
import argparse
import csv
import importlib.util
import json
import os
import re

from . import icd
from .drug_class import drug_class
from .rules import setup
from .safe import SafeError, guarded, log

K = 11
# 0.10（v0.1.11）：化验分位数按节、章、儿童层另记（labs_by）；用药、用药大类、化验异常比例多儿童层（kid:章、kid:*）；
# 主诉词补全（部位＋疼痛／不适、词＋修饰词）；主诉覆盖计数（chief_cov）
# 0.11（v0.1.12）：主诉表、主诉覆盖多"器官合并层"（org:器官，按主诊断的器官系统把各章的住院合起来）；描述用的主诉计数两组：
# 公开肛肠、泌尿主诉词各命中几次（chief_probe），去侧别、去部位以后的主诉词表（chief_lat、chief_base）。k 照压
# 0.20（v0.2.0）：支持多折（fold 2–7）和 S2 甲换表（检查非检验类、主诉症状、既往史、合并症），并列核对计数（s2_switch_meta）；
# 主诉词表补"脱出"；规则路径既往史否定窗口扩到整句（past_negated）并补慢性病同义全称（PAST_SYNONYMS）；real_rates 补首次血常规单项极值和出院新增诊断计数
# 0.21（v0.2.1）：新增 S2 病程同期检查日（exam_days 非检验类）、计划→做完间隔（exam_lags）、病程逐日症状留存/消退（symptom_course）及并列核对计数
STATS_VERSION = "mc-stats-0.21"
LAB_STRATUM_MIN = 30    # 分层化验分位数：这一层这一项至少 30 个读数，并且来自至少 k 次住院
CHILD_AGE = 14          # 儿童层的年龄线，同 planner.CHILD_AGE
# 0.10（v0.1.11 第 13 条）：评测台症状词表里的泛词"疼痛"不带部位，"右上腹疼痛""肛周疼痛"都只记成"疼痛"，规划写出"主诉：疼痛1天"；
# "语言发育落后"只记成"语言发育"。紧挨着的部位词（评测台部位词表＋这里补的公开解剖词）连着记，"不适"只在前面有部位词时记；
# 词后紧跟固定修饰词的连着记。都是通用写法，不按病种
SITE_EXTRA = ("肛周", "肛门", "腰部", "腰", "背部", "背", "咽部", "咽喉", "咽", "耳", "眼", "牙", "会阴", "腹股沟", "臀部", "骶尾部", "足", "手", "腿", "全身")
SITE_TERMS = ("疼痛",)
QUAL_TAIL = ("迟缓", "落后", "延迟", "障碍", "困难", "不清")
# 0.9：公开的专科主诉词（不来自病历），补在评测台症状词表后面。不收比已有词更短的（会把已有词拆开计数）
# 0.11：只计数、不进词表的公开主诉词（肛肠：痔的公开诊治指南列的症状；泌尿：泌尿系梗阻、结石的常见主诉），给 plan_diag 看"不适"的住院
# 主诉里实际写的是什么，计数够了再决定补不补进 CHIEF_EXTRA
CHIEF_PROBE = ("便血", "大便带血", "便后出血", "肛门出血", "肛门肿物", "肛周肿物", "肿物脱出", "脱出", "肛门疼痛", "肛周疼痛", "肛门瘙痒", "肛周瘙痒",
               "肛门坠胀", "肛门不适", "腰痛", "腰部疼痛", "腰酸", "腰部不适", "血尿", "尿频", "尿急", "尿痛", "排尿困难", "排尿不畅", "尿线变细", "少尿",
               "无尿", "肾绞痛", "腹痛", "发热")
_LATERAL = re.compile(r"^[左右双]侧?")
CHIEF_EXTRA = ("肛门肿物", "肛周肿物", "肿物脱出", "脱出", "肛门坠胀", "排便困难", "尿线变细", "排尿不畅", "阴道流血", "阴道出血", "阴道流液", "停经",
               "胎动减少", "下腹坠胀", "视物不清", "眼痛", "眼胀", "颈部肿物", "乳房肿物", "乳腺肿物", "腹股沟肿物", "肿胀", "活动受限", "外伤", "跌倒")
# 文献常用参考范围（公开知识，不来自真实病历）。0.9 从 planner 挪来：统计表按它判每项化验在不在范围内
REF = {"WBC": (3.5, 9.5), "NEUT": (40, 75), "HGB": (115, 175), "PLT": (125, 350), "K": (3.5, 5.3), "NA": (137, 147), "CL": (99, 110), "CA": (2.1, 2.7), "GLU": (3.9, 6.1),
       "CREA": (45, 110), "BUN": (2.8, 7.2), "UA": (150, 420), "ALT": (7, 40), "AST": (13, 35), "TBIL": (3.4, 20.5), "ALB": (40, 55), "CRP": (0, 8), "PCT": (0, 0.05),
       "PT": (10, 14), "APTT": (25, 37), "INR": (0.8, 1.2), "FIB": (2, 4), "DD": (0, 0.5), "CK": (40, 200), "CKMB": (0, 25), "LDH": (120, 250), "TNI": (0, 0.04), "BNP": (0, 125),
       "HBA1C": (4, 6), "TC": (2.8, 5.2), "TG": (0.5, 1.7), "LDL": (0, 3.4), "AMY": (30, 110)}
# 真实和规划同口径的几项极端值线（第八次合成病历审阅用的线）
CBC_EXTREME = {"WBC": (None, 12), "NEUT": (None, 85), "HGB": (90, None), "PLT": (100, None)}
DS_VITALS_OFF = {"SBP": (95, 160), "P": (None, 100), "R": (None, 22)}
LOS_BUCKETS = (("1-3", 0, 3), ("4-7", 4, 7), ("8-14", 8, 14), ("15+", 15, 10 ** 6))


def cbc_extreme(vals):
    """vals：{项目: 数值}。有一项越过 CBC_EXTREME 的线就算极端。"""
    for k, (lo, hi) in CBC_EXTREME.items():
        v = vals.get(k)
        if v is not None and ((lo is not None and v < lo) or (hi is not None and v > hi)):
            return True
    return False


def ds_vitals_off(sbp, p, r):
    return (sbp is not None and not DS_VITALS_OFF["SBP"][0] <= sbp <= DS_VITALS_OFF["SBP"][1]) or (p is not None and p > 100) or (r is not None and r > 22)


def los_bucket(los):
    return next((n for n, lo, hi in LOS_BUCKETS if lo <= los <= hi), None)


def prog_day_bucket(day):
    """病程天分档：早期(1-2)、中期(3-5)、后期(6+)，供病程症状演变与对照统计共用。"""
    return "1-2" if day <= 2 else ("3-5" if day <= 5 else "6+")
# 公开词典补 code3 的启用线：补出来的 code3 和原来一致的比例、章一致的比例。已有 code3 的真实住院里能用这种查法的
# 不少于 DX_MAP_MIN_N 次时按真实住院判，不够时按公开词典自身判（主干查法在词典上三位码一致约 85%、章一致约 96%）
DX_MAP_MIN, DX_MAP_MIN_CH, DX_MAP_MIN_N = 0.80, 0.95, 30
SEX_CN = {"M": "男", "F": "女"}
AGE_BANDS = (("0-1", 0, 1), ("1-6", 1, 6), ("6-18", 6, 18), ("18-40", 18, 40), ("40-65", 40, 65), ("65+", 65, 200))
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
_DEPT = re.compile(r"科\s*室\s*[：:]\s*([^\s|｜，,；;:：]{2,24})")
# 0.8：科室名后面紧跟的表头字段（没有空白隔开时会被一起截进来）和院区字样
_DEPT_LABEL = re.compile(r"(?:年龄|性别|姓名|床号|床位|住院号|病案号|病历号|门诊号|入院|出院|日期|时间|ID)")
_DEPT_CAMPUS = re.compile(r"(?:南|北|东|西|中|新|老)院(?:区)?|本部|分院")
# 0.8：症状否定按分句判。"无明显诱因"不算否定（"无明显诱因出现腹痛"）
_SYM_NEG = re.compile(r"否认|无(?!明显诱因|诱因)|未见|未诉|未再|未出现|不伴|没有")
_SYM_RESOLVED = re.compile(r"\s*(?:已|均已|基本)?(?:消失|消退)")
_SYM_CLAUSE_END = re.compile(r"[，,。；;：:\n]")
# 0.8：途径只看到药名后第一个标点为止（0.7 取 30 字，跨过逗号、顿号吸到别的药的途径）
_ROUTE_STOP = re.compile(r"[，,。；;、\n]")
_CC_DUR = re.compile(r"(\d{1,3})\s*(小时|天|日|周|月|年)")
_PROC = re.compile(r"(?:手术方式|手术名称|拟施手术名称和方式|拟施手术名称|拟施手术|拟行手术名称|拟行手术|术式)\s*[:：]\s*([^\n，,。；;]{2,30})")
_ROUTE = re.compile(r"静滴|静脉滴注|静推|静注|静脉注射|口服|皮下注射|肌注|肌肉注射|雾化吸入|雾化|泵入|吸入|含服|鼻饲")
_FREQ = re.compile(r"(?<![A-Za-z])(qd|bid|tid|qid|qn|q\d+h|prn|st)(?![A-Za-z])|每日\s*[一二三四1234]\s*次|每\s*\d+\s*小时\s*[一1]?\s*次?", re.I)
_DOSE_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|μg|ug|毫克|克|微克|IU|U|万U|万单位|ml|mL)")

LAB_UNIT = {"WBC": "×10^9/L", "NEUT": "%", "HGB": "g/L", "PLT": "×10^9/L", "K": "mmol/L", "NA": "mmol/L", "CL": "mmol/L", "CA": "mmol/L", "GLU": "mmol/L",
            "CREA": "μmol/L", "BUN": "mmol/L", "UA": "μmol/L", "ALT": "U/L", "AST": "U/L", "TBIL": "μmol/L", "ALB": "g/L", "CRP": "mg/L", "PCT": "ng/mL",
            "PT": "s", "APTT": "s", "INR": "", "FIB": "g/L", "DD": "mg/L", "CK": "U/L", "CKMB": "U/L", "LDH": "U/L", "TNI": "ng/mL", "BNP": "pg/mL",
            "HBA1C": "%", "TC": "mmol/L", "TG": "mmol/L", "LDL": "mmol/L", "AMY": "U/L"}
LAB_NAME = {"WBC": "白细胞", "NEUT": "中性粒细胞比例", "HGB": "血红蛋白", "PLT": "血小板", "K": "血钾", "NA": "血钠", "CL": "血氯", "CA": "血钙", "GLU": "血糖",
            "CREA": "肌酐", "BUN": "尿素氮", "UA": "尿酸", "ALT": "谷丙转氨酶", "AST": "谷草转氨酶", "TBIL": "总胆红素", "ALB": "白蛋白", "CRP": "C反应蛋白",
            "PCT": "降钙素原", "PT": "凝血酶原时间", "APTT": "活化部分凝血活酶时间", "INR": "INR", "FIB": "纤维蛋白原", "DD": "D-二聚体", "CK": "肌酸激酶",
            "CKMB": "肌酸激酶同工酶", "LDH": "乳酸脱氢酶", "TNI": "肌钙蛋白I", "BNP": "BNP", "HBA1C": "糖化血红蛋白", "TC": "总胆固醇", "TG": "甘油三酯",
            "LDL": "低密度脂蛋白", "AMY": "血淀粉酶"}
_LAB_ALIAS = {"WBC": "白细胞(?:计数)?|WBC", "NEUT": "中性粒细胞(?:比例|百分比|%)?|NEUT%?|N%", "HGB": "(?<!平均)(?<!平均红细胞)(?<!红细胞平均)血红蛋白|HGB|Hb", "PLT": "血小板(?:计数)?|PLT", "K": "血钾|血清钾|钾离子|K\\+?",
              "NA": "血钠|血清钠|钠离子|Na\\+?", "CL": "血氯|氯离子|Cl-?", "CA": "血钙|钙离子|Ca2?\\+?", "GLU": "血糖|葡萄糖|GLU", "CREA": "肌酐|Cr|CREA", "BUN": "尿素氮|尿素|BUN",
              "UA": "尿酸|UA", "ALT": "谷丙转氨酶|丙氨酸氨基转移酶|ALT", "AST": "谷草转氨酶|天门冬氨酸氨基转移酶|AST", "TBIL": "总胆红素|TBIL", "ALB": "白蛋白|ALB",
              "CRP": "C反应蛋白|CRP", "PCT": "降钙素原|PCT", "PT": "凝血酶原时间|PT", "APTT": "活化部分凝血活酶时间|APTT", "INR": "INR", "FIB": "纤维蛋白原|FIB",
              "DD": "D-?二聚体", "CK": "肌酸激酶|CK", "CKMB": "肌酸激酶同工酶|CK-?MB", "LDH": "乳酸脱氢酶|LDH", "TNI": "肌钙蛋白I?|cTnI|TnI", "BNP": "NT-proBNP|BNP",
              "HBA1C": "糖化血红蛋白|HbA1c", "TC": "总胆固醇|TC", "TG": "甘油三酯|TG", "LDL": "低密度脂蛋白(?:胆固醇)?|LDL(?:-C)?", "AMY": "血?淀粉酶|AMY"}


def _unit_pat(u):
    if u.startswith("×"):
        return r"\s*(?:[×xX*]\s*)?10\^?9\s*[/／]\s*L"
    return r"\s*" + re.escape(u) if u else ""


# 生理上可能的范围，比 N1（ruler_g3/numbers.py）不更窄：范围外的读数多半是别的项目被读成了这一项，不进分位数
PLAUSIBLE = {"WBC": (0.1, 500), "NEUT": (1, 100), "HGB": (15, 260), "PLT": (1, 2000), "K": (1.0, 12), "NA": (90, 200), "CL": (50, 160), "CA": (0.5, 5),
             "GLU": (0.5, 150), "CREA": (5, 5000), "BUN": (0.1, 150), "UA": (10, 2000), "ALT": (1, 10000), "AST": (1, 10000), "TBIL": (0.5, 1000),
             "ALB": (5, 80), "CRP": (0, 1000), "PCT": (0, 1000), "PT": (5, 200), "APTT": (10, 300), "INR": (0.5, 20), "FIB": (0.1, 20), "DD": (0, 200),
             "CK": (1, 100000), "CKMB": (0, 5000), "LDH": (10, 20000), "TNI": (0, 1000), "BNP": (0, 100000), "HBA1C": (2, 25), "TC": (0.5, 30),
             "TG": (0.1, 50), "LDL": (0.1, 20), "AMY": (1, 20000)}
LAB_RE = {k: re.compile(rf"(?<![A-Za-z])(?:{a})[^\d，,。；;\n]{{0,5}}?(\d+(?:\.\d+)?){_unit_pat(LAB_UNIT[k])}", re.I) for k, a in _LAB_ALIAS.items()}
# 0.2 的血红蛋白读法（没排除 MCH、MCHC）。只用来计数，核对第二次丢弃的根因，不进分位数
_HGB_OLD = re.compile(rf"(?<![A-Za-z])(?:血红蛋白|HGB|Hb)[^\d，,。；;\n]{{0,5}}?(\d+(?:\.\d+)?){_unit_pat('g/L')}", re.I)
PAST_TERMS = ("高血压", "糖尿病", "冠心病", "脑梗死", "慢性肾脏病", "慢性阻塞性肺疾病", "房颤", "高脂血症", "乙肝", "痛风")
# 0.20：慢性病公开全称/简称对照（"冠状动脉粥样硬化性心脏病" ↔ "冠心病"），供规划器同步既往史、编译器排除否认清单
PAST_SYNONYMS = {
    "冠心病": ("冠心病", "冠状动脉粥样硬化性心脏病", "冠状动脉性心脏病", "缺血性心脏病", "心肌梗死", "心绞痛"),
    "高血压": ("高血压",),
    "糖尿病": ("糖尿病",),
    "脑梗死": ("脑梗死", "脑梗塞"),
    "慢性肾脏病": ("慢性肾脏病", "慢性肾功能不全", "慢性肾衰竭", "尿毒症"),
    "慢性阻塞性肺疾病": ("慢性阻塞性肺疾病", "慢阻肺"),
    "房颤": ("房颤", "心房颤动"),
    "高脂血症": ("高脂血症", "高脂蛋白血症"),
    "乙肝": ("乙肝", "乙型肝炎", "慢性乙型病毒性肝炎"),
    "痛风": ("痛风",),
}
_NEG_BEFORE = re.compile(r"(?:否认|无|未见|没有)[^，,。；;\n]{0,10}$")
_PAST_NEG = re.compile(r"否认|无(?!明显诱因|诱因)|未见|没有")
_PAST_POS = re.compile(r"(?:^|[，,、\s])(?:有|患|确诊|发现(?!无))")


def past_matches(term, text):
    return any(w in (text or "") for w in PAST_SYNONYMS.get(term, (term,)))


def past_negated(text, pos):
    """0.20（第 4 条规则回退）：同一句（到上一个句号、分号、换行为止）里前面有否定词，且最后一个否定词之后没有"有、患、确诊"这类转折。"""
    t = text or ""
    start = max(t.rfind(ch, 0, pos) for ch in "。；;\n") + 1
    negs = list(_PAST_NEG.finditer(t, start, pos))
    if not negs:
        return False
    return not _PAST_POS.search(t[negs[-1].end():pos])
_PROC_PRE = re.compile(r"^(?:在|于)?(?:全身麻醉|全麻|局部麻醉|局麻|腰硬联合麻醉|腰麻|硬膜外麻醉|椎管内麻醉|神经阻滞麻醉|静脉麻醉)下?|^行")
_SIDE_TOK = re.compile(r"(左|右|双)(侧)?")
_SCOPE = re.compile(r"^经(?=腹腔镜|胸腔镜|宫腔镜|关节镜|输尿管镜|膀胱镜|尿道)|(?<=镜)(?:辅助)?下")
# 术式大类：同一大类的不同写法合起来计数，凑够 k 次但单个写法凑不够时，规划器用右边的常用名（公开知识，不来自病历）。
# 按顺序取第一个匹配的；常用名里的 {S} 由规划器按诊断定侧别
# 大类要窄：常用名不能把良性手术写成癌根治、把肌瘤剔除写成子宫切除
PROC_FAMILY = (
    ("胆囊切除", r"胆囊切除", "腹腔镜胆囊切除术"), ("胆道取石", r"胆总管.{0,6}取石|胆道.{0,4}取石|ERCP", "内镜逆行胰胆管造影取石术"),
    ("阑尾切除", r"阑尾切除", "腹腔镜阑尾切除术"),
    ("腹股沟疝修补", r"腹股沟.{0,6}疝", "{S}腹股沟疝无张力修补术"), ("脐疝修补", r"脐疝", "脐疝修补术"), ("切口疝修补", r"切口疝", "切口疝修补术"),
    ("剖宫产", r"剖宫产", "子宫下段剖宫产术"), ("髋关节置换", r"髋关节置换", "{S}全髋关节置换术"), ("股骨头置换", r"股骨头置换", "{S}人工股骨头置换术"),
    ("膝关节置换", r"膝关节置换", "{S}全膝关节置换术"), ("骨折内固定", r"复位.{0,6}内固定", "{S}骨折切开复位内固定术"),
    ("冠脉介入", r"冠状动脉.{0,8}(?:支架|球囊|介入)|PCI", "经皮冠状动脉支架植入术"), ("冠脉造影", r"冠状动脉造影", "冠状动脉造影术"),
    ("甲状腺切除", r"甲状腺.{0,6}切除", "{S}甲状腺腺叶切除术"), ("子宫肌瘤剔除", r"肌瘤.{0,2}(?:剔除|切除|剥除)", "腹腔镜子宫肌瘤剔除术"),
    ("子宫切除", r"子宫(?:全|次全)?切除", "腹腔镜全子宫切除术"),
    ("结肠癌根治", r"结肠癌.{0,4}(?:根治|切除)", "腹腔镜结肠癌根治术"), ("直肠癌根治", r"直肠癌.{0,4}(?:根治|切除)", "腹腔镜直肠癌根治术"),
    ("胃癌根治", r"胃癌.{0,4}根治|全胃切除", "腹腔镜胃癌根治术"), ("胃大部切除", r"胃大部.{0,2}切除", "胃大部切除术"),
    ("泌尿系碎石", r"(?:输尿管|肾|尿路|膀胱).{0,8}(?:碎石|取石)", "输尿管镜钬激光碎石术"),
    ("前列腺电切", r"前列腺.{0,6}(?:电切|切除|剜除)", "经尿道前列腺电切术"), ("白内障", r"白内障|人工晶体", "{S}白内障超声乳化摘除联合人工晶体植入术"),
    ("椎间盘", r"椎间盘.{0,6}(?:摘除|切除)|髓核", "腰椎间盘髓核摘除术"), ("扁桃体切除", r"扁桃体.{0,4}切除", "扁桃体切除术"),
    ("大隐静脉", r"大隐静脉", "{S}大隐静脉高位结扎剥脱术"), ("痔", r"痔", "混合痔外剥内扎术"),
    ("肠镜息肉", r"(?:结肠|肠镜|直肠).{0,8}息肉|EMR|ESD", "内镜下结肠息肉切除术"), ("清宫", r"清宫|刮宫", "清宫术"),
    ("卵巢囊肿", r"卵巢.{0,6}(?:囊肿|剥除)", "腹腔镜卵巢囊肿剥除术"), ("乳腺癌根治", r"乳腺癌.{0,4}根治|改良根治", "{S}乳腺癌改良根治术"),
    ("乳腺肿物", r"乳腺.{0,6}(?:肿物|肿块|纤维腺瘤|区段)", "{S}乳腺肿物切除术"), ("气管切开", r"气管切开", "气管切开术"),
    ("开颅血肿", r"开颅.{0,8}血肿|血肿清除", "开颅血肿清除术"), ("肺叶切除", r"肺.{0,4}(?:叶|段|楔形).{0,2}切除", "胸腔镜{S}肺叶切除术"),
)
_FAMILY_RE = [(f, re.compile(rx, re.I)) for f, rx, _ in PROC_FAMILY]
PROC_FAMILY_NAME = {f: name for f, _, name in PROC_FAMILY}


def proc_family(key):
    """归一后的手术名属于哪个大类，认不出返回 None。"""
    for f, rx in _FAMILY_RE:
        if rx.search(key or ""):
            return f
    return None


def proc_key(raw):
    """手术名归一：去括号、空白、麻醉前缀和"行"；"腹腔镜下""经腹腔镜"一类写法合成"腹腔镜"；第一个侧别换成占位符 {S} 或 {S侧}，侧别由规划器按诊断定。"""
    s = re.sub(r"\s", "", re.sub(r"[（(][^）)]{0,20}[）)]", "", raw))
    for _ in range(2):
        s = _PROC_PRE.sub("", s)
    s = _SCOPE.sub("", s.strip("，,。；;：:、"))
    m = _SIDE_TOK.search(s)
    if m:
        s = s[:m.start()] + ("{S侧}" if m.group(2) else "{S}") + s[m.end():]
    return s if len(s.replace("{S侧}", "").replace("{S}", "")) >= 3 else None


def band_of(age):
    if age is None:
        return None
    for name, lo, hi in AGE_BANDS:
        if lo <= age < hi:
            return name
    return None


def _terms():
    """症状词表直接按路径读，不走 import：g45_core 里带的 ruler_g3 不能进这个进程。"""
    path = os.path.join(os.environ["BENCH_G45"], "ruler_g4", "terms.py")
    spec = importlib.util.spec_from_file_location("mc_terms", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 0.9：补公开的专科主诉词
    return list(mod.SYMPTOMS) + [x for x in CHIEF_EXTRA if x not in mod.SYMPTOMS]


def clean_dept(raw):
    """科室名：在表头字段处切开，去掉院区字样。太短返回 None。"""
    s = _DEPT_LABEL.split(raw or "", 1)[0]
    s = _DEPT_CAMPUS.sub("", s).strip("（(　 ")
    return s if len(s) >= 2 else None


def site_regex(region_words):
    """部位词（评测台部位词表的写法＋SITE_EXTRA），前面可带侧别和上下，结尾对齐。"""
    words = sorted(set(region_words) | set(SITE_EXTRA), key=len, reverse=True)
    alt = "|".join(re.escape(w) for w in words)
    return re.compile(r"(?:[左右双]侧?)?[上下中全]?(?:" + alt + r")部?$"), re.compile(r"(?:[左右双]侧?)?[上下中全]?(?:" + alt + r")部?不适")


def _iter_symptom_hits(text, sym_re, sites=None):
    """扫描文本中的症状命中，依次产出 (归一化词, 分句前有否定, 词后紧跟消失/消退)。"""
    text = text or ""
    hits = [(m.start(), m.end(), m.group(0)) for m in sym_re.finditer(text)]
    if sites:
        hits += [(m.start(), m.end(), m.group(0)) for m in sites[1].finditer(text)]
        hits.sort()
    for s0, e0, term in hits:
        start = 0
        for e in _SYM_CLAUSE_END.finditer(text, 0, s0):
            start = e.end()
        neg_before = bool(_SYM_NEG.search(text[start:s0]))
        end_c = _SYM_CLAUSE_END.search(text, e0)
        post = text[e0:end_c.start() if end_c else e0 + 6]
        resolved_after = bool(_SYM_RESOLVED.match(post))
        if sites:
            if term in SITE_TERMS:
                m = sites[0].search(text[max(start, s0 - 8):s0])
                term = (m.group(0) + term) if m else term
            q = next((x for x in QUAL_TAIL if text.startswith(x, e0)), None)
            if q and not term.endswith(q):
                term += q
        yield term, neg_before, resolved_after


def positive_symptoms(text, sym_re, sites=None):
    """分句里没有被否定的症状词。否定词出现在同一分句、症状词之前才算否定（"无发热、寒战"两个都算否定）。
    sites（0.10，site_regex 的返回值）给了时补全：泛词"疼痛"带上紧挨着的部位词，"部位＋不适"也记，词后紧跟修饰词的连着记。"""
    return [term for term, neg_before, _ in _iter_symptom_hits(text, sym_re, sites) if not neg_before]


def negated_symptoms(text, sym_re, sites=None):
    """同一分句前被否定，或词后紧跟"已消失/消退"的症状词。"""
    return [term for term, neg_before, resolved_after in _iter_symptom_hits(text, sym_re, sites) if neg_before or resolved_after]


def route_tail(text, end):
    tail = text[end:end + 30]
    m = _ROUTE_STOP.search(tail)
    return tail[:m.start()] if m else tail


def _q(xs):
    xs = sorted(xs)
    return [xs[int(q * (len(xs) - 1))] for q in QS]


class Tally:
    """分层计数：cnt[层][键]，den[层] 是这一层的住院数。全局层是 "*"。"""

    def __init__(self):
        self.cnt, self.den = {}, {}

    def adm(self, strata):
        for s in strata:
            self.den[s] = self.den.get(s, 0) + 1

    def add(self, strata, key, n=1):
        for s in strata:
            d = self.cnt.setdefault(s, {})
            d[key] = d.get(key, 0) + n

    def table(self, k, stats):
        """压掉 <k 的格子，也压掉分母 <k 的层。压掉多少记进 stats["suppressed"]。"""
        out, gone = {}, 0
        for s, d in self.cnt.items():
            if self.den.get(s, 0) < k:
                gone += len(d)
                continue
            keep = {x: n for x, n in d.items() if n >= k}
            gone += len(d) - len(keep)
            out[s] = {"n": self.den[s], "items": keep}
        stats["suppressed"] = stats.get("suppressed", 0) + gone
        return out


def _hist_table(hist, k, stats):
    """平直直方图表（键→分箱→次数）按 k≥11 压格，把被压掉的格子数计入 stats["suppressed"]。"""
    out, gone = {}, 0
    for x, h in hist.items():
        if sum(h.values()) < k:
            gone += len(h)
            continue
        keep = {b: n for b, n in h.items() if n >= k}
        gone += len(h) - len(keep)
        if keep:
            out[x] = keep
    stats["suppressed"] = stats.get("suppressed", 0) + gone
    return out


def _record_sym_course(tally, sc_st, obs, base_syms, chief_syms):
    """把一次住院在病程各天分档里的症状(留存/消退)记进分层 Tally，每格至多加 1 次（保持住院级口径）。"""
    role_emitted = set()
    for (term, db), (pos_d, neg_d) in sorted(obs.items()):
        state = "keep" if pos_d >= neg_d else "gone"
        tally.add(sc_st, f"sym:{term}|{db}|{state}")
        if term in base_syms:
            rtag = "chief" if term in chief_syms else "accomp"
            rk = (rtag, db, state)
            if rk not in role_emitted:
                role_emitted.add(rk)
                tally.add(sc_st, f"role:{rtag}|{db}|{state}")


def _primary_name(ds, dxm):
    items = [it for it in dxm.list_items(ds) if it.section == "出院诊断"] if ds is not None else []
    return items[0].norm if items else None


def dx_map_check(pnames, codes):
    """公开词典补 code3 之前先自检，只出计数。整串、主干两种查法分别看：在已有 code3 的住院上查出来的 code3、章和原来一致的比例，
    两条线都过才启用；能查的不到 DX_MAP_MIN_N 次时，整串查法默认启用，主干查法改按公开词典自身判。"""
    m = {x: 0 for x in ("present", "missing", "exact_found", "exact_agree", "exact_chapter_agree", "stem_found", "stem_agree", "stem_chapter_agree")}
    for rid, pname in pnames.items():
        c3 = codes.get(rid)
        if not c3:
            m["missing"] += 1
            continue
        m["present"] += 1
        for how, got in (("exact", icd.code3_exact(pname) if pname else None), ("stem", icd.code3_stem(pname) if pname else None)):
            if got:
                m[f"{how}_found"] += 1
                m[f"{how}_agree"] += got == c3
                m[f"{how}_chapter_agree"] += got[:1] == c3[:1]

    def ok(found, agree, ch):
        return found > 0 and agree >= DX_MAP_MIN * found and ch >= DX_MAP_MIN_CH * found

    # 整串查法就是 D0 的做法（加上常用写法对照），样本不够时默认启用
    m["use_exact"] = int(ok(m["exact_found"], m["exact_agree"], m["exact_chapter_agree"])) if m["exact_found"] >= DX_MAP_MIN_N else 1
    if m["stem_found"] >= DX_MAP_MIN_N:
        m["stem_basis_dictionary"] = 0
        m["use_stem"] = int(ok(m["stem_found"], m["stem_agree"], m["stem_chapter_agree"]))
    else:
        d = icd.self_check()
        m.update({"stem_basis_dictionary": 1, "dict_stem_found": d["found"], "dict_stem_agree": d["agree"], "dict_stem_chapter_agree": d["chapter_agree"]})
        m["use_stem"] = int(ok(d["found"], d["agree"], d["chapter_agree"]))
    return m


def blocked_chapters(code3, names_ch, stems, k):
    """有住院、但 code3、诊断名、主干三层都没有一项凑够 k 次的章（规划器抽不出主诊断名）。只有计数。"""
    out = {}
    for ch, n in sorted(code3.den.items()):
        if ch == "*" or n < k:
            continue
        cnts = {"code3": code3.cnt.get(ch) or {}, "name": names_ch.cnt.get(ch) or {}, "stem": stems.cnt.get(ch) or {}}
        if any(v >= k for d in cnts.values() for v in d.values()):
            continue
        row = {"admissions": n}
        for t, d in cnts.items():
            row[f"{t}_kinds"] = len(d)
            row[f"{t}_max"] = max(d.values()) if d else 0
        out[ch] = row
    return out


def _switch_compare(rt, st_, k):
    """0.20（M2 方案第 1 节）：一张换了的表，规则版和 S2 版并列核对计数（只出整数计数，不出词名）。"""
    def eval_t(t):
        out, gone = {}, 0
        for s, d in t.cnt.items():
            if t.den.get(s, 0) < k:
                gone += len(d)
                continue
            keep = {x: n for x, n in d.items() if n >= k}
            gone += len(d) - len(keep)
            out[s] = keep
        return out, gone

    def top10(d):
        return [x for x, _ in sorted(d.items(), key=lambda z: (-z[1], z[0]))[:10]]

    def ov(a, b):
        return len(set(a) & set(b)) / min(len(a), len(b)) if a and b else None

    rd, rg = eval_t(rt)
    sd, sg = eval_t(st_)
    gov = ov(top10(rd.get("*") or {}), top10(sd.get("*") or {}))
    sov = sorted(x for s in set(rd) | set(sd) if s != "*" and not s.startswith("n:")
                 for x in (ov(top10(rd.get(s) or {}), top10(sd.get(s) or {})),) if x is not None)
    p50 = sov[len(sov) // 2] if sov else None
    return {
        "rule_strata_nonempty": sum(1 for v in rd.values() if v),
        "s2_strata_nonempty": sum(1 for v in sd.values() if v),
        "s2_emptied_strata": sum(1 for s, v in rd.items() if v and not sd.get(s)),
        "rule_cells": sum(len(v) for v in rd.values()),
        "s2_cells": sum(len(v) for v in sd.values()),
        "rule_suppressed": rg,
        "s2_suppressed": sg,
        "top10_global_permille": round(1000 * gov) if gov is not None else -1,
        "top10_strata_n": len(sov),
        "top10_strata_p50_permille": round(1000 * p50) if p50 is not None else -1,
    }


def build_stats(lines, codes, k=K, copy_table=None, events_by=None):
    setup(("BENCH_G3", "BENCH_G2", "BENCH_G45"))
    from ruler_g2 import build_real
    from ruler_g2 import diagnoses as dxm
    from ruler_g2.events import admission_mentions
    from ruler_g3.adapters import from_admission
    from ruler_g3.engine import mentions
    from ruler_g3.numbers import _BP, _RATE_CN, _RATE_EN, _T
    from ruler_g2.events import _REGION
    symptoms = sorted(_terms(), key=len, reverse=True)
    sym_re = re.compile("|".join(re.escape(s) for s in symptoms))
    sites = site_regex(_REGION)
    joint, code3, names, names_ch, dept, los, surg, sday, procs, exams, meds, syms, durs, past, comorb, subtypes = (Tally() for _ in range(16))
    stems, proc_fam, chief_syms = Tally(), Tally(), Tally()
    # 0.20：events_by 给了时走 S2 甲换四张表，同时用平行 Tally 留规则版做并列核对计数
    # 0.21：再增病程同期检查日、计划→做完间隔、病程逐日症状演变的 S2 建表与规则并列计数
    use_s2 = events_by is not None
    r_exams_nl, s2_exams_nl, r_chief, r_past, r_comorb = (Tally() for _ in range(5))
    r_exam_days, s2_exam_days, r_exam_lags, s2_exam_lags, r_sym_course, sym_course = (Tally() for _ in range(6))
    s2_day_hist, r_nonlab_day_hist, lag_hist = {}, {}, {}
    s2_extra_counts = {
        "s2_course_exam_stays": 0, "s2_exam_lag_stays": 0,
        "s2_exam_lag_pairs": 0, "rule_exam_lag_pairs": 0, "s2_prog_sym_stays": 0,
    }
    vocab = None
    if use_s2:
        from .s2_table_compare import Vocab, s2_reading
        vocab = Vocab()
    # 0.10：主诉覆盖（有没有入院记录、主诉节、匹配到的主诉词），给 plan_diag 查"不适"的原因；化验分层读数
    chief_cov, labs_by, labs_by_adm = Tally(), {}, {}
    # 0.11
    chief_probe, chief_lat, chief_base = Tally(), Tally(), Tally()
    from .planner import KIND_OF, dx_organs
    # 0.9
    lab_state, med_class, med_class_drug, med_class_multi = Tally(), Tally(), Tally(), Tally()
    from .out_diag import BOILER
    general_rx = {c: re.compile(rx) for c, rx in BOILER if c in ("精神睡眠饮食", "二便")}
    note_general = {c: 0 for c in general_rx}
    pb_los = {}
    real = {"cbc_admissions": 0, "cbc_any_extreme": 0, "wbc_admissions": 0, "wbc_gt12": 0, "neut_admissions": 0, "neut_gt85": 0,
            "hgb_admissions": 0, "hgb_lt90": 0, "plt_admissions": 0, "plt_lt100": 0,
            "hgb_2plus": 0, "hgb_swing_ge30": 0, "tbil_admissions": 0, "tbil_ge100": 0,
            "admissions": 0, "zero_meds": 0, "ds_vitals_adult_read": 0, "ds_vitals_adult_off": 0,
            "ds_dx_admissions": 0, "ds_dx_items": 0, "hr_ds_dx_admissions": 0, "hr_ds_dx_items": 0,
            "hr_ds_new_dx_items": 0, "hr_ds_new_share_permille_sum": 0, "b3_admissions": 0, "b3_permille_sum": 0}
    pb_sub, pb_total = {}, []
    med_parts = {"alias": Tally(), "dose": Tally(), "freq": Tally(), "route": Tally()}
    labs, vitals, sec_chars, day_hist, labs_dropped, exam_kinds = {}, {}, {}, {}, {}, {}
    labs_meta = {"HGB_new_matches": 0, "HGB_old_matches": 0, "HGB_old_only": 0, "HGB_old_only_out_of_range": 0}
    notes_per_day, mentions_per_adm, dose_written, note_vitals, n_total = [], [], [0, 0], [0, 0], 0
    surgery_meta = {"surgical_admissions": 0, "proc_key_found": 0, "proc_family_found": 0}
    # 先过一遍只取主诊断名（顺带缓存 build_real 和出院诊断清单，给 S2 诊断词表用），自检公开词典补 code3 的一致率，够线才在下面补
    lines = list(lines)
    adms, ds_dx_by, pnames, all_ds_norms = {}, {}, {}, set()
    for o in lines:
        rid = str(o["admission_id"])
        adm0 = build_real({"admission_id": rid, "docs": o["docs"]})
        adms[rid] = adm0
        its0 = [it for it in dxm.list_items(adm0.ds) if it.section == "出院诊断"] if adm0.ds is not None else []
        ds_dx_by[rid] = its0
        pnames[rid] = its0[0].norm if its0 else None
        all_ds_norms.update(it.norm for it in its0)
    if use_s2:
        vocab.set_dx_vocab(all_ds_norms)
    dx_meta = dx_map_check(pnames, codes)
    dx_meta.update({"added_exact": 0, "added_stem": 0, "still_missing": 0})
    for obj in lines:
        rid = str(obj["admission_id"])
        adm = adms[rid]
        rec = from_admission({"admission_id": rid, "docs": obj["docs"]}, rid)
        sr = s2_reading(vocab, events_by.get(rid, ()), adm)[0]["jia"] if use_s2 else None
        n_total += 1
        hr, ds = adm.hr, adm.ds
        items = ds_dx_by[rid]
        pname = items[0].norm if items else None
        c3 = codes.get(rid) or ""
        if not c3:
            got = (dx_meta["use_exact"] and pname and icd.code3_exact(pname)) or None
            if got:
                dx_meta["added_exact"] += 1
            else:
                got = (dx_meta["use_stem"] and pname and icd.code3_stem(pname)) or None
                dx_meta["added_stem"] += bool(got)
                dx_meta["still_missing"] += not got
            c3 = got or ""
        ch = c3[:1] or None
        # 分层：全局、章、节（0.8）、code3、主诊断名。规划器从最细的非空层往粗的退
        blk = icd.block_of(c3)
        st = ("*",) + ((ch,) if ch else ()) + ((blk,) if blk else ()) + ((c3,) if len(c3) == 3 else ()) + ((f"n:{pname}",) if pname else ())
        est = ("*",) + ((ch,) if ch else ())
        sc_st = ("*",) + ((ch,) if ch else ()) + ((blk,) if blk else ()) + ((c3,) if len(c3) == 3 else ())
        for t in (dept, los, surg, procs, proc_fam, exams, meds, syms, durs, comorb, subtypes, chief_syms, lab_state, med_class, chief_cov):
            t.adm(st)
        if use_s2:
            for t in (r_exams_nl, s2_exams_nl, r_comorb):
                t.adm(st)
            for t in (r_exam_days, s2_exam_days, r_exam_lags, s2_exam_lags):
                t.adm(est)
            for t in (r_sym_course, sym_course):
                t.adm(sc_st)
        # 0.10：14 岁以下另记儿童层（全院儿童、本章儿童），只进用药、用药大类、化验异常比例和化验分位数
        kst = ((("kid:*",) + ((f"kid:{ch}",) if ch else ())) if rec.age is not None and rec.age < CHILD_AGE else ())
        for t in (meds, lab_state, med_class):
            t.adm(kst)
        lst = ((blk,) if blk else ()) + ((ch,) if ch else ()) + kst
        # 0.11：主诉表的器官合并层（和规划器同一个 dx_organs）；描述用的三组主诉计数用同样的层
        ost = tuple(f"org:{o}" for o in sorted(dx_organs(pname or "", c3)))
        cst = st + ost
        for t in (chief_syms, chief_cov):
            t.adm(ost)
        if use_s2:
            r_chief.adm(cst)
        for t in (chief_probe, chief_lat, chief_base):
            t.adm(cst)
        real["admissions"] += 1
        code3.adm(("*",) + ((ch,) if ch else ()))
        b = band_of(rec.age)
        sex = SEX_CN.get(rec.sex)
        joint.adm(("*",))
        pst = ("*",) + ((b,) if b else ())
        past.adm(pst)
        if use_s2:
            r_past.adm(pst)
        if ch and b and sex:
            joint.add(("*",), f"{sex}|{b}|{ch}")
        if ch:
            code3.add((ch,), c3)
        hr_dx_items = []
        hr_seen_syms, rule_cc = set(), set()
        if hr is not None:
            hr_dx_items = [it for it in dxm.list_items(hr) if it.section == "入院诊断"]
            m = _DEPT.search(hr.text[:400])
            dname = clean_dept(m.group(1)) if m else None
            if dname:
                dept.add(st, dname)
            rule_past_set = set()
            sec = hr.section("既往史")
            if sec:
                t = hr.text[sec[0]:sec[1]]
                for name in PAST_TERMS:
                    for f in re.finditer(name, t):
                        if not past_negated(t, f.start()):
                            rule_past_set.add(name)
                            break
            if use_s2:
                for name in rule_past_set:
                    r_past.add(pst, name)
            else:
                for name in rule_past_set:
                    past.add(pst, name)
            cc = hr.section("主诉")
            hpi = hr.section("现病史")
            seen = set()
            for sec in (cc, hpi):
                if sec:
                    seen |= set(positive_symptoms(hr.text[sec[0]:sec[1]], sym_re, sites))
            hr_seen_syms = set(seen)
            for s in seen:
                syms.add(st, s)
            chief_cov.add(cst, "hr")
            if cc:
                chief_cov.add(cst, "cc")
                cc_text = hr.text[cc[0]:cc[1]]
                rule_cc = set(positive_symptoms(cc_text, sym_re, sites))
                for w in CHIEF_PROBE:
                    if w in cc_text:
                        chief_probe.add(cst, w)
                d = _CC_DUR.search(cc_text)
                if d:
                    durs.add(st, f"{d.group(1)}{d.group(2)}")
            if use_s2:
                for s in rule_cc:
                    r_chief.add(cst, s)
            got_cc = sr["chief"] if use_s2 else rule_cc
            if got_cc:
                chief_cov.add(cst, "term")
            for s in got_cc:
                chief_syms.add(cst, s)
            for s in {_LATERAL.sub("", x) for x in got_cc}:
                chief_lat.add(cst, s)
            for s in {next((b_ for b_ in SITE_TERMS + ("不适",) if x.endswith(b_) and len(x) > len(b_)), x) for x in got_cc}:
                chief_base.add(cst, s)
            sec = hr.section("体格检查")
            if sec:
                t = hr.text[sec[0]:sec[1]][:200]
                vb = "child" if rec.age is not None and rec.age < 18 else "adult"
                # 0.9：儿童另按年龄段记一份（child:0-1、child:1-6、child:6-18），规划按年龄段取（审查员第八轮补充：3 岁抽到 112/50mmHg）
                groups = [vitals.setdefault(vb, {"T": [], "P": [], "R": [], "SBP": [], "DBP": []})]
                if vb == "child" and b:
                    groups.append(vitals.setdefault(f"child:{b}", {"T": [], "P": [], "R": [], "SBP": [], "DBP": []}))
                x = _T.search(t)
                for v in groups:
                    if x:
                        v["T"].append(float(x.group(1)))
                for rx in (_RATE_CN, _RATE_EN):
                    for x in rx.finditer(t):
                        key = {"脉搏": "P", "心率": "P", "呼吸": "R", "P": "P", "HR": "P", "R": "R"}[x.group(1)]
                        for v in groups:
                            v[key].append(float(x.group(2)))
                x = _BP.search(t)
                if x:
                    for v in groups:
                        v["SBP"].append(float(x.group(1)))
                        v["DBP"].append(float(x.group(2)))
        if use_s2:
            for name in sr["past"]:
                past.add(pst, name)
        if ds is not None:
            if items:
                real["ds_dx_admissions"] += 1
                real["ds_dx_items"] += len(items)
                if hr_dx_items:
                    real["hr_ds_dx_admissions"] += 1
                    real["hr_ds_dx_items"] += len(items)
                    new_n = sum(1 for it in items if not any(dxm.dx_match(it.norm, h.norm) for h in hr_dx_items))
                    real["hr_ds_new_dx_items"] += new_n
                    real["hr_ds_new_share_permille_sum"] += round(1000 * new_n / len(items))
            if pname and len(c3) == 3:
                names.adm((c3,))
                names.add((c3,), pname)
            if pname and ch:
                names_ch.adm((ch,))
                names_ch.add((ch,), pname)
            # 诊断主干按 code3 和章两层计；章层给没有一个 code3 凑够 k 次的章兜底（0.5）
            sk = icd.stem_key(pname) if pname else None
            for s in ((c3,) if pname and len(c3) == 3 else ()) + ((ch,) if pname and ch else ()):
                stems.adm((s,))
                if sk:
                    stems.add((s,), sk)
            rule_comorb = sorted({it.norm for it in items[1:]} - {pname})
            if use_s2:
                for norm in rule_comorb:
                    r_comorb.add(st, norm)
                for norm in sorted(sr["dx"] - ({pname} if pname else set())):
                    comorb.add(st, norm)
            else:
                for norm in rule_comorb:
                    comorb.add(st, norm)
            los.add(st, str(ds.day))
            # 0.9：成人出院生命体征超出范围的比例（出院情况一节，只计数）
            sec = ds.section("出院情况")
            if sec and rec.age is not None and rec.age >= 18:
                t = ds.text[sec[0]:sec[1]]
                bp = _BP.search(t)
                rate = {"P": None, "R": None}
                for rx in (_RATE_CN, _RATE_EN):
                    for x in rx.finditer(t):
                        key = {"脉搏": "P", "心率": "P", "呼吸": "R", "P": "P", "HR": "P", "R": "R"}[x.group(1)]
                        rate[key] = rate[key] if rate[key] is not None else float(x.group(2))
                if bp or rate["P"] is not None:
                    real["ds_vitals_adult_read"] += 1
                    real["ds_vitals_adult_off"] += ds_vitals_off(float(bp.group(1)) if bp else None, rate["P"], rate["R"])
            nd = max(1, ds.day)
            n_notes = sum(1 for d in adm.docs if d.doc_type == "progress_note" and d.subtype in ("日常查房", "其他病程"))
            notes_per_day.append(n_notes / nd)
        post = adm.post
        surg.add(st, "yes" if post is not None else "no")
        if post is not None:
            sday.adm(("*",))
            sday.add(("*",), str(post.day))
            surgery_meta["surgical_admissions"] += 1
            for d in [post] + [x for x in adm.docs if x is not post]:
                m = _PROC.search(d.text)
                key = proc_key(m.group(1)) if m else None
                if key:
                    procs.add(st, key)
                    surgery_meta["proc_key_found"] += 1
                    fam = proc_family(key)
                    if fam:
                        proc_fam.add(st, fam)
                        surgery_meta["proc_family_found"] += 1
                    break
        # 0.8：首次病程以外的全部病程按小类各几篇（只出计数；不按 progress_between 截窗，0.9 改了说明）
        prog = [d for d in adm.docs if d.doc_type == "progress_note" and d.subtype != "首次病程"]
        if ds is not None and prog:
            per = {}
            for d in prog:
                per[d.subtype] = per.get(d.subtype, 0) + 1
            for sub_, n in per.items():
                pb_sub.setdefault(sub_, []).append(n)
            pb_total.append(len(prog))
            # 0.9：按住院天数分档
            bk = pb_los.setdefault(los_bucket(ds.day), {"admissions": 0, "notes": 0, "sub": {}})
            bk["admissions"] += 1
            bk["notes"] += len(prog)
            for sub_, n in per.items():
                x = bk["sub"].setdefault(sub_, [0, 0])
                x[0] += 1
                x[1] += n
        seen = set()
        first_lab, hgb_all, tbil_all, adm_labs = {}, [], [], {}
        for d in adm.docs:
            subtypes.add(st, f"docs|{d.subtype}")
            for s in d.sections:
                if s.name in ("抬头", "签名"):
                    continue
                sec_chars.setdefault(f"{d.subtype}|{s.name}", []).append(s.end - s.start)
            if d.doc_type == "progress_note" and d.subtype in ("日常查房", "主任医师查房", "主治医师查房", "上级医师查房", "其他病程"):
                sec_chars.setdefault(f"{d.subtype}|正文", []).append(len(d.text))
                # 这篇病程写没写生命体征：体温、心率或呼吸、血压三样里有两样
                hits = bool(_T.search(d.text)) + bool(_BP.search(d.text)) + any(rx.search(d.text) for rx in (_RATE_CN, _RATE_EN))
                note_vitals[0] += hits >= 2
                note_vitals[1] += 1
                # 0.9：一般情况（精神睡眠饮食、二便），同 out_diag 套话口径：去掉空白和标点后匹配
                flat = re.sub(r"[\s，,。；;：:、！？!?（）()]", "", d.text)
                for c, rx in general_rx.items():
                    note_general[c] += bool(rx.search(flat))
            seen.add(d.subtype)
            for key, rx in LAB_RE.items():
                lo, hi = PLAUSIBLE[key]
                for x in rx.finditer(d.text):
                    v = float(x.group(1))
                    if lo <= v <= hi:
                        labs.setdefault(key, []).append(v)
                        adm_labs.setdefault(key, []).append(v)
                        first_lab.setdefault(key, v)
                        if key == "HGB":
                            hgb_all.append(v)
                        elif key == "TBIL":
                            tbil_all.append(v)
                    else:
                        labs_dropped[key] = labs_dropped.get(key, 0) + 1
            starts = {x.start() for x in LAB_RE["HGB"].finditer(d.text)}
            labs_meta["HGB_new_matches"] += len(starts)
            for x in _HGB_OLD.finditer(d.text):
                labs_meta["HGB_old_matches"] += 1
                if x.start() not in starts:
                    labs_meta["HGB_old_only"] += 1
                    labs_meta["HGB_old_only_out_of_range"] += not PLAUSIBLE["HGB"][0] <= float(x.group(1)) <= PLAUSIBLE["HGB"][1]
        for s in seen:
            subtypes.add(st, f"has|{s}")
        # 0.9：每项化验第一次读数在参考范围内外（规划按诊断层的异常比例抽）；真实的极端值比例只计数
        for key, v in first_lab.items():
            lo, hi = REF[key]
            lab_state.add(st + kst, f"{key}|{'abn' if (v < lo or v > hi) else 'norm'}")
        # 0.10：化验读数按节、章、儿童层另记
        for s in lst:
            for key, vs in adm_labs.items():
                labs_by.setdefault(s, {}).setdefault(key, []).extend(vs)
                labs_by_adm.setdefault(s, {})[key] = labs_by_adm.setdefault(s, {}).get(key, 0) + 1
        if "WBC" in first_lab or "HGB" in first_lab:
            real["cbc_admissions"] += 1
            real["cbc_any_extreme"] += cbc_extreme(first_lab)
        if "WBC" in first_lab:
            real["wbc_admissions"] += 1
            real["wbc_gt12"] += first_lab["WBC"] > 12
        if "NEUT" in first_lab:
            real["neut_admissions"] += 1
            real["neut_gt85"] += first_lab["NEUT"] > 85
        if "HGB" in first_lab:
            real["hgb_admissions"] += 1
            real["hgb_lt90"] += first_lab["HGB"] < 90
        if "PLT" in first_lab:
            real["plt_admissions"] += 1
            real["plt_lt100"] += first_lab["PLT"] < 100
        if len(hgb_all) >= 2:
            real["hgb_2plus"] += 1
            real["hgb_swing_ge30"] += max(hgb_all) - min(hgb_all) >= 30
        if tbil_all:
            real["tbil_admissions"] += 1
            real["tbil_ge100"] += max(tbil_all) >= 100
        ments, first = admission_mentions(adm), {}
        first_p_course, first_r_course = {}, {}
        ds_idx = ds.idx if ds is not None else None
        from ruler_g2.readings import Reading, b_group
        rb3 = Reading("B3")
        b_group(adm, ments, Reading("B1"), Reading("B2"), rb3, Reading("B4"), Reading("B4b"), Reading("B5"))
        if rb3.value is not None:
            real["b3_admissions"] += 1
            real["b3_permille_sum"] += round(1000 * rb3.value)
        for d in adm.docs:
            for x in ments[d.idx]:
                if x.tone == "R":
                    first.setdefault((x.key, x.region or ""), d.day)
                    if use_s2 and d.idx != ds_idx and d.day is not None:
                        first_r_course.setdefault((x.key, x.region or ""), int(d.day))
                elif use_s2 and x.tone == "P" and d.idx != ds_idx and d.day is not None:
                    first_p_course.setdefault((x.key, x.region or ""), int(d.day))
        # 每次住院结果口吻的检查种类数，和评测台 B_exam（ruler_g2.c2parts 的 exam_kinds）同口径
        for s in ("*",) + ((ch,) if ch else ()) + ((c3,) if len(c3) == 3 else ()):
            exam_kinds.setdefault(s, []).append(len(first))
        for (key, region), day in first.items():
            item_k = f"{key}|{region}"
            bin_ = str(min(max(0, int(day)), 5))
            if use_s2:
                if vocab.is_lab(item_k):
                    day_hist.setdefault(item_k, {})[bin_] = day_hist.get(item_k, {}).get(bin_, 0) + 1
                    exams.add(st, item_k)
                else:
                    r_nonlab_day_hist.setdefault(item_k, {})[bin_] = r_nonlab_day_hist.get(item_k, {}).get(bin_, 0) + 1
                    r_exams_nl.add(st, item_k)
            else:
                day_hist.setdefault(item_k, {})[bin_] = day_hist.get(item_k, {}).get(bin_, 0) + 1
                exams.add(st, item_k)
        if use_s2:
            for item_k in sr["exams"]:
                exams.add(st, item_k)
                s2_exams_nl.add(st, item_k)
            # 0.21：病程同期检查日、planned→done 间隔、病程逐日症状演变（规则与 S2 并列计数）
            for (key, region), d_r in first_r_course.items():
                item_k = f"{key}|{region}"
                if not vocab.is_lab(item_k):
                    r_exam_days.add(est, f"{item_k}@{str(min(max(0, d_r), 5))}")
            r_lag_seen = set()
            for (key, region), d_p in first_p_course.items():
                item_k = f"{key}|{region}"
                if vocab.is_lab(item_k):
                    continue
                d_r = first_r_course.get((key, region))
                if d_r is not None and d_r >= d_p:
                    s2_extra_counts["rule_exam_lag_pairs"] += 1
                    lag_b = str(min(int(d_r - d_p), 5))
                    kind = KIND_OF.get(key, "其他")
                    for tk in (f"{item_k}@{lag_b}", f"kind:{kind}@{lag_b}"):
                        r_lag_seen.add(tk)
            for tk in r_lag_seen:
                r_exam_lags.add(est, tk)
            if sr["course_exam_done_day"]:
                s2_extra_counts["s2_course_exam_stays"] += 1
            for item_k, d_done in sr["course_exam_done_day"].items():
                bin_ = str(min(max(0, int(d_done)), 5))
                s2_day_hist.setdefault(item_k, {})[bin_] = s2_day_hist.get(item_k, {}).get(bin_, 0) + 1
                s2_exam_days.add(est, f"{item_k}@{bin_}")
            adm_lag_cells, adm_s2_tcells, has_lag = set(), set(), False
            for item_k, d_plan in sr["course_exam_planned_day"].items():
                d_done = sr["course_exam_done_day"].get(item_k)
                if d_done is not None and d_done >= d_plan:
                    has_lag = True
                    s2_extra_counts["s2_exam_lag_pairs"] += 1
                    lag_b = str(min(int(d_done - d_plan), 5))
                    key_only = item_k.split("|", 1)[0]
                    kind = KIND_OF.get(key_only, "其他")
                    for tgt in (item_k, key_only, f"kind:{kind}"):
                        adm_lag_cells.add((tgt, lag_b))
                    for tk in (f"{item_k}@{lag_b}", f"kind:{kind}@{lag_b}"):
                        adm_s2_tcells.add(tk)
            if has_lag:
                s2_extra_counts["s2_exam_lag_stays"] += 1
            for tgt, lag_b in adm_lag_cells:
                lag_hist.setdefault(tgt, {})[lag_b] = lag_hist.get(tgt, {}).get(lag_b, 0) + 1
            for tk in adm_s2_tcells:
                s2_exam_lags.add(est, tk)
            r_sym_obs = {}
            for d in adm.docs:
                if d.doc_type == "progress_note" and d.day is not None and d.day >= 1:
                    db = prog_day_bucket(d.day)
                    for term, neg_before, resolved_after in _iter_symptom_hits(d.text, sym_re, sites):
                        cur = r_sym_obs.setdefault((term, db), [-1, -1])
                        if neg_before or resolved_after:
                            cur[1] = max(cur[1], int(d.day))
                        else:
                            cur[0] = max(cur[0], int(d.day))
            if r_sym_obs:
                _record_sym_course(r_sym_course, sc_st, r_sym_obs, hr_seen_syms, rule_cc)
            if sr["prog_sym_days"]:
                s2_extra_counts["s2_prog_sym_stays"] += 1
                s2_sym_obs = {}
                s2_base = sr["syms"] | sr.get("fp_day0_syms", set())
                for term, dsets in sr["prog_sym_days"].items():
                    for db in ("1-2", "3-5", "6+"):
                        pos_d = max((dy for dy in dsets["done"] if prog_day_bucket(dy) == db), default=-1)
                        neg_d = max((dy for dy in dsets["negated"] if prog_day_bucket(dy) == db), default=-1)
                        if pos_d >= 0 or neg_d >= 0:
                            s2_sym_obs[(term, db)] = (pos_d, neg_d)
                _record_sym_course(sym_course, sc_st, s2_sym_obs, s2_base, sr["chief"])
        # 用药：药按住院计一次；别名、剂量、频次、途径各取这次住院里该药第一个写出来的值，分别计
        drugs, count = {}, 0
        for x in mentions(rec):
            if x["negated"] or x["not_current"] or x["topical"]:
                continue
            count += 1
            dose_written[1] += 1
            dose_written[0] += bool(x["dose"])
            got = drugs.setdefault(x["drug_id"], {"alias": x["alias"], "dose": None, "freq": None, "route": None})
            text = next((t for name, t, _ in rec.segs if name == x["seg"]), "")
            tail = route_tail(text, x["end"])
            route, freq = _ROUTE.search(tail), _FREQ.search(tail)
            if got["dose"] is None and x["dose"] and x["dose"].get("single_mg") is not None:
                got["dose"] = x["dose"]["text"]
            if got["freq"] is None and freq:
                got["freq"] = freq.group(0)
            if got["route"] is None and route:
                got["route"] = route.group(0)
        by_cls = {}
        for did, got in drugs.items():
            meds.add(st + kst, did)
            by_cls.setdefault(drug_class(did), []).append(did)
            for part, t in med_parts.items():
                t.adm((did,))
                v = got[part] if got[part] is not None else ("" if part in ("freq", "route") else None)
                if v is not None:
                    t.add((did,), v)
        # 0.9：按公开药物大类计：这次住院用了哪些大类；大类里各药几次；同类药用了两种以上的住院
        for cls, ds_ in by_cls.items():
            med_class.add(st + kst, cls)
            med_class_drug.adm((cls,))
            med_class_multi.adm((cls,))
            for did in ds_:
                med_class_drug.add((cls,), did)
            if len(ds_) >= 2:
                med_class_multi.add((cls,), "2+")
        real["zero_meds"] += not drugs
        mentions_per_adm.append(count)
    stats = {"version": STATS_VERSION, "k": k, "n_admissions": n_total}
    stats["joint"] = joint.table(k, stats).get("*", {"n": 0, "items": {}})
    stats["code3"] = code3.table(k, stats)
    stats["dx_names"] = names.table(k, stats)
    stats["dx_names_ch"] = names_ch.table(k, stats)
    stats["blocked_chapters"] = blocked_chapters(code3, names_ch, stems, k)
    stats["dx_stems"] = stems.table(k, stats)
    stats["dx_map_meta"] = dx_meta
    stats["dept"] = dept.table(k, stats)
    stats["los"] = los.table(k, stats)
    stats["surgery"] = surg.table(k, stats)
    stats["surgery_day"] = sday.table(k, stats).get("*", {"n": 0, "items": {}})
    stats["procedures"] = procs.table(k, stats)
    stats["proc_family"] = proc_fam.table(k, stats)
    stats["exams"] = exams.table(k, stats)
    stats["exam_kinds"] = {s: {"q": _q(v), "n": len(v)} for s, v in exam_kinds.items() if len(v) >= k}
    if use_s2:
        day_hist.update(s2_day_hist if s2_day_hist else r_nonlab_day_hist)
    stats["exam_days"] = _hist_table(day_hist, k, stats)
    stats["exam_lags"] = _hist_table(lag_hist, k, stats) if use_s2 else {}
    stats["meds"] = meds.table(k, stats)
    for part, t in med_parts.items():
        stats[f"med_{part}"] = t.table(k, stats)
    stats["symptoms"] = syms.table(k, stats)
    stats["chief_symptoms"] = chief_syms.table(k, stats)
    stats["chief_cov"] = chief_cov.table(k, stats)
    stats["chief_probe"] = chief_probe.table(k, stats)
    stats["chief_lat"] = chief_lat.table(k, stats)
    stats["chief_base"] = chief_base.table(k, stats)
    stats["durations"] = durs.table(k, stats)
    stats["past"] = past.table(k, stats)
    stats["comorbid"] = comorb.table(k, stats)
    stats["subtypes"] = subtypes.table(k, stats)
    stats["labs"] = {x: {"q": _q(v), "n": len(v), "unit": LAB_UNIT[x]} for x, v in labs.items() if len(v) >= k}
    by = {s: {x: {"q": _q(v), "n": len(v)} for x, v in d.items() if len(v) >= LAB_STRATUM_MIN and labs_by_adm[s][x] >= k} for s, d in labs_by.items()}
    stats["labs_by"] = {s: d for s, d in by.items() if d}
    stats["labs_dropped"] = dict(sorted(labs_dropped.items()))
    stats["labs_meta"] = labs_meta
    stats["surgery_meta"] = surgery_meta
    stats["vitals"] = {b: {x: {"q": _q(v), "n": len(v)} for x, v in g.items() if len(v) >= k} for b, g in vitals.items()}
    stats["section_chars"] = {x: {"q": _q(v), "n": len(v)} for x, v in sec_chars.items() if len(v) >= k}
    stats["notes_per_day"] = {"q": _q(notes_per_day), "n": len(notes_per_day)} if len(notes_per_day) >= k else None
    stats["mentions_per_adm"] = {"q": _q(mentions_per_adm), "n": len(mentions_per_adm)} if len(mentions_per_adm) >= k else None
    stats["p_dose_written"] = dose_written[0] / dose_written[1] if dose_written[1] >= k else None
    stats["note_vitals"] = {"rate": note_vitals[0] / note_vitals[1], "n": note_vitals[1]} if note_vitals[1] >= k else None
    stats["copy"] = copy_table or {}
    stats["pb_subtypes"] = {"admissions": len(pb_total), "notes_q": _q(pb_total) if len(pb_total) >= k else None,
                            "by_subtype": {x: {"admissions": len(v), "notes": sum(v), "per_adm_q": _q(v)} for x, v in sorted(pb_sub.items()) if len(v) >= k},
                            "by_los": {b: {"admissions": v["admissions"], "notes": v["notes"],
                                           "sub": {s: {"admissions": a, "notes": n} for s, (a, n) in sorted(v["sub"].items()) if a >= k}}
                                       for b, v in sorted(pb_los.items()) if b and v["admissions"] >= k}}
    # 0.9
    stats["lab_state"] = lab_state.table(k, stats)
    stats["med_class"] = med_class.table(k, stats)
    stats["med_class_drug"] = med_class_drug.table(k, stats)
    stats["med_class_multi"] = med_class_multi.table(k, stats)
    stats["note_general"] = {"n": note_vitals[1], **{c: note_general[c] / note_vitals[1] for c in note_general}} if note_vitals[1] >= k else None
    stats["real_rates"] = real
    stats["symptom_course"] = sym_course.table(k, stats) if use_s2 else {}
    if use_s2:
        sw = {}
        for tname, rt, st_ in (("exams_nonlab", r_exams_nl, s2_exams_nl), ("chief_symptoms", r_chief, chief_syms),
                               ("past", r_past, past), ("comorbid", r_comorb, comorb),
                               ("exam_days", r_exam_days, s2_exam_days), ("exam_lags", r_exam_lags, s2_exam_lags),
                               ("symptom_course", r_sym_course, sym_course)):
            for mk, mv in _switch_compare(rt, st_, k).items():
                sw[f"{tname}_{mk}"] = mv
        sw.update(s2_extra_counts)
        stats["s2_switch_meta"] = sw
    return stats


def summary_rows(stats):
    """stats_summary.csv 的行：每张表的层数和格子数，只有计数。0.1 只数嵌套的 items，平表（joint、surgery_day）和分位数表都记成 0。"""
    rows = []
    for t, v in stats.items():
        if not isinstance(v, dict) or t == "copy":
            continue
        if t == "note_vitals":
            rows.append({"table": "note_vitals:rate_per_mille", "strata": 1, "cells": round(1000 * v["rate"])})
            continue
        if t == "note_general":
            rows += [{"table": f"note_general:{x}_per_mille", "strata": 1, "cells": round(1000 * r)} for x, r in v.items() if x != "n"]
            continue
        if t == "pb_subtypes":
            rows.append({"table": "pb_subtypes", "strata": len(v.get("by_subtype") or {}), "cells": len(v.get("by_los") or {})})
            continue
        if t in ("labs_dropped", "labs_meta", "surgery_meta", "dx_map_meta", "real_rates", "s2_switch_meta"):
            rows += [{"table": f"{t}:{x}", "strata": 1, "cells": n} for x, n in v.items()]
            continue
        if "items" in v:
            strata, cells = 1, len(v["items"])
        elif "q" in v:
            strata, cells = 1, 1
        elif all(isinstance(x, dict) and "items" in x for x in v.values()) and v:
            strata, cells = len(v), sum(len(x["items"]) for x in v.values())
        else:
            strata, cells = len(v), sum(len(x) if isinstance(x, dict) and "q" not in x else 1 for x in v.values())
        rows.append({"table": t, "strata": strata, "cells": cells})
    rows.append({"table": "copy", "strata": len(stats.get("copy") or {}), "cells": sum(len(x) for x in (stats.get("copy") or {}).values())})
    rows.append({"table": "suppressed_cells_lt_k", "strata": 1, "cells": stats.get("suppressed", 0)})
    return rows


def read_copy_table(path):
    """S0_copyops.csv 的真实行：细类和天数间隔的复用占比（exact+numeric_changed+near）。这张表已在审查包里。"""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["row"] != "R" or r["group"] not in ("全部", "文书细类", "天数间隔"):
                continue
            share = float(r["exact_share"]) + float(r["numeric_changed_share"]) + float(r["near_share"])
            out.setdefault(r["group"], {})[r["value"]] = {"reuse": share, "exact": float(r["exact_share"]), "pairs": int(r["pairs"])}
    return out


def _split_csv_arg(val):
    return [p.strip() for p in (val or "").split(",") if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="住院 JSONL 路径，多折用逗号分隔")
    ap.add_argument("--dx-map", required=True, help="dx_map CSV 路径，多折用逗号分隔")
    ap.add_argument("--copyops", required=True)
    ap.add_argument("--events", default="", help="可选：S2 events_norm.jsonl 路径，多折用逗号分隔")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    os.umask(0o077)
    codes = {}
    for mp in _split_csv_arg(a.dx_map):
        with open(mp, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("code3"):
                    codes[str(r["admission_id"])] = r["code3"]
    lines, seen_ids = [], set()
    for p in _split_csv_arg(a.path):
        with open(p, encoding="utf-8") as f:
            for x in f:
                if x.strip():
                    o = json.loads(x)
                    rid = str(o["admission_id"])
                    if rid in seen_ids:
                        raise SafeError("duplicate_admission_id", count=len(seen_ids))
                    seen_ids.add(rid)
                    lines.append(o)
    events_by = None
    ev_paths = _split_csv_arg(a.events)
    if ev_paths:
        from .s2_table_compare import read_events
        events_by = {}
        for ep in ev_paths:
            for rid, evs in read_events(ep, seen_ids).items():
                events_by.setdefault(rid, []).extend(evs)
    stats = build_stats(lines, codes, a.k, read_copy_table(a.copyops), events_by=events_by)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)
    rows = summary_rows(stats)
    with open(os.path.join(os.path.dirname(a.out) or ".", "stats_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "strata", "cells"])
        w.writeheader()
        w.writerows(rows)
    log("stats_done", admissions=stats["n_admissions"], suppressed=stats.get("suppressed", 0), s2=int(events_by is not None))


if __name__ == "__main__":
    guarded(main)
