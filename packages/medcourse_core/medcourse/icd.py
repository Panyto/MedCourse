"""公开 ICD 词典（《疾病分类与代码国家临床版 2.0》，生成方法见 tools/build_icd_tables.py）。不读病历。

诊断名查三位码（整串、主干两种；词典没有的临床常用写法查 SYNONYM）；三位码取一个标准名；诊断名取主干（去掉"伴…"，
侧别换成占位符），给统计表按主干计数。前缀、后缀模糊匹配在词典自检上三位码一致只有 41%–72%，不用。

0.1.6（监督员第五轮）：产科诊断名里的孕周、孕产次、胎方位每个病人都不一样（"孕39+2周G1P0LOA待产"），按原串、按主干都凑不够 k，
O 章一例抽不到。主干键把孕周换成 {W}、孕产次换成 {G}{P}、胎方位去掉；查词典用的主干候选多一个去掉这些写法的；
规划时 realize 按名字填孕周（待产、临产 37–41 周），gest_weeks 从名字里读，入院记录的停经周数和诊断名一致。
"""
import csv
import os
import re

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_PUNCT = re.compile(r"[\W_]+", flags=re.UNICODE)
_WITH = re.compile(r"伴有|伴|合并|并发")
_SIDE = re.compile(r"(左|右|双)(侧)?")
_LEAD = re.compile(r"^(?:左侧|右侧|双侧|左|右|双)")
_OB = re.compile(r"孕|妊娠|产|胎")
_GP = re.compile(r"[Gg]\s*(\d{1,2})\s*[Pp]\s*(\d{1,2})|孕\s*(\d{1,2})\s*产\s*(\d{1,2})")
_GEST = re.compile(r"(孕|妊娠)\s*(\d{1,2})(?:\s*[+＋]\s*\d)?\s*周(?:\s*[+＋]?\s*\d\s*天)?")
_FETAL_POS = re.compile(r"[LR][OS][ATP]")
_WEEKS = re.compile(r"(?:孕|妊娠)(\d{1,2})(?:\+\d)?周")
_cache = {}
# 临床常用写法但词典里没有的（公开医学知识，不来自病历）。只在词典整串查不到时用
SYNONYM = {"社区获得性肺炎": "J18", "重症社区获得性肺炎": "J18", "高血压病": "I10", "冠心病": "I25", "慢性阻塞性肺疾病": "J44", "慢性阻塞性肺疾病急性加重": "J44",
           "慢阻肺": "J44", "急性脑梗死": "I63", "脑梗塞": "I63", "慢性心功能不全": "I50", "房颤": "I48", "早产": "O60", "腰椎间盘突出症": "M51", "肺癌": "C34",
           "慢性肾脏病": "N18", "尿路感染": "N39", "支气管扩张": "J47", "急性肾损伤": "N17", "类风湿关节炎": "M06", "脑出血": "I61", "乳腺癌": "C50", "胃癌": "C16",
           "结肠癌": "C18", "直肠癌": "C20", "肝癌": "C22", "子宫肌瘤": "D25", "良性前列腺增生": "N40", "腹股沟疝": "K40"}
# 病名打头、后面是分级分期的写法，三位码由打头的病名定
SYNONYM_PREFIX = ((re.compile(r"^高血压病?[123一二三]级"), "I10"), (re.compile(r"^冠状动脉粥样硬化性心脏病"), "I25"),
                  (re.compile(r"^慢性阻塞性肺疾病"), "J44"), (re.compile(r"^2型糖尿病"), "E11"))


def _table(fname, k, v):
    if fname not in _cache:
        with open(os.path.join(_DIR, fname), encoding="utf-8", newline="") as f:
            _cache[fname] = {r[k]: r[v] for r in csv.DictReader(f)}
    return _cache[fname]


def key(name):
    """同 D0 的 normalize_name：去标点和空白，转小写。"""
    return _PUNCT.sub("", name or "").lower()


def _lookup(k):
    got = _table("icd_names.csv", "name", "code3").get(k) or SYNONYM.get(k)
    return got or next((c for rx, c in SYNONYM_PREFIX if rx.match(k)), None)


def code3_exact(name):
    return _lookup(key(name))


def obstetric_key(name, keep=True):
    """产科写法归一：孕产次换成 G{G}P{P} 或 孕{G}产{P}，孕周换成 孕{W}周，胎方位去掉。keep=False 时这些写法整个去掉（查词典用）。
    不像产科诊断的名字原样返回。"""
    s = name or ""
    if not _OB.search(s):
        return s
    s = _GP.sub(lambda m: ("G{G}P{P}" if m.group(1) else "孕{G}产{P}") if keep else "", s)
    s = _GEST.sub(lambda m: m.group(1) + "{W}周" if keep else "", s)
    s = _FETAL_POS.sub("", s)
    return s if keep else re.sub(r"^宫内", "", s)


def stems(name):
    """主干候选：去掉"伴…"以后的部分、去掉开头的侧别、两样都去；产科名再加一个去掉孕周孕产次的（0.1.6）。不含原串。"""
    k = key(name)
    head = _WITH.split(k, 1)[0]
    ob = key(obstetric_key(name, keep=False))
    out = []
    for s in (head, _LEAD.sub("", k), _LEAD.sub("", head), ob, _WITH.split(ob, 1)[0]):
        if len(s) >= 2 and s != k and s not in out:
            out.append(s)
    return out


def code3_stem(name):
    for s in stems(name):
        got = _lookup(s)
        if got:
            return got
    return None


def self_check():
    """主干查法在公开词典自身上的一致率：能取主干的名字当作整串查不到，只用主干查，三位码、章和原来一致的各多少。"""
    names = _table("icd_names.csv", "name", "code3")
    out = {"found": 0, "agree": 0, "chapter_agree": 0}
    for n, c3 in names.items():
        s = next((names[x] for x in stems(n) if x in names), None)
        if s:
            out["found"] += 1
            out["agree"] += s == c3
            out["chapter_agree"] += s[:1] == c3[:1]
    return out


def category(code3):
    return _table("icd_category.csv", "code3", "name").get(code3)


def stem_key(name):
    """统计表按主干计数用的键：去掉"伴…"以后的部分；第一个侧别换成 {S} 或 {S侧}（同 stats.proc_key）；产科的孕周、孕产次换成占位符（0.1.6）。
    太短返回 None。"""
    head = _WITH.split(obstetric_key(name), 1)[0]
    m = _SIDE.search(head)
    if m:
        head = head[:m.start()] + ("{S侧}" if m.group(2) else "{S}") + head[m.end():]
    return head if len(head.replace("{S侧}", "").replace("{S}", "")) >= 2 else None


def realize_side(text, rng):
    """占位符随机换成左或右。"""
    if "{S" not in text:
        return text
    side = rng.choice(("左", "右"))
    return text.replace("{S侧}", side + "侧").replace("{S}", side)


def week_range(name):
    """按诊断名定孕周范围（公开产科常识）：流产、胚胎停育、早孕 6–13 周，早产 28–36 周，待产、临产、足月、分娩 37–41 周，其余 None。"""
    if re.search(r"流产|胚胎|葡萄胎|早孕|宫外孕|异位妊娠", name or ""):
        return 6, 13
    if "早产" in (name or ""):
        return 28, 36
    if re.search(r"待产|临产|足月|分娩", name or ""):
        return 37, 41
    return None


def realize_obstetric(text, rng):
    """孕周、孕产次占位符换成数：孕周按 week_range（没有就 28–40 周），零头 0–6 天；孕次 1–4，产次小于孕次。"""
    if "{W}" not in text and "{G}" not in text:
        return text
    lo, hi = week_range(text) or (28, 40)
    w, d = rng.randint(lo, hi), rng.randint(0, 6)
    g = rng.choice((1, 1, 1, 2, 2, 3, 4))
    return text.replace("{W}", f"{w}+{d}" if d else str(w)).replace("{G}", str(g)).replace("{P}", str(rng.randint(0, g - 1)))


def realize(text, rng):
    return realize_obstetric(realize_side(text, rng), rng)


def gest_weeks(name):
    """诊断名里写的孕周（整周），没写返回 None。"""
    m = _WEEKS.search(name or "")
    return int(m.group(1)) if m else None


# 0.1.8（第七次合成病历审阅）：ICD-10 的"节"（block），公开分类，WHO ICD-10 2019 版的节界；国家临床版 2.0 的三位码沿用同样的节。
# 统计表在 code3 和章之间多一层：诊断名、code3 都没有凑够 k 次的格子时先退到节，再退到章。章层太粗：K 章把口腔病和胰腺炎放在一起，
# I 章把脑梗死和冠心病放在一起，第七次的科室有 199/500 退到章层，排出了"急性胰腺炎住口腔颌面外科"。
BLOCKS = (
    "A00-A09 A15-A19 A20-A28 A30-A49 A50-A64 A65-A69 A70-A74 A75-A79 A80-A89 A90-A99 B00-B09 B15-B19 B20-B24 B25-B34 B35-B49 "
    "B50-B64 B65-B83 B85-B89 B90-B94 B95-B98 B99-B99 "
    "C00-C14 C15-C26 C30-C39 C40-C41 C43-C44 C45-C49 C50-C50 C51-C58 C60-C63 C64-C68 C69-C72 C73-C75 C76-C80 C81-C96 C97-C97 "
    "D00-D09 D10-D36 D37-D48 D50-D53 D55-D59 D60-D64 D65-D69 D70-D77 D80-D89 "
    "E00-E07 E10-E14 E15-E16 E20-E35 E40-E46 E50-E64 E65-E68 E70-E90 "
    "F00-F09 F10-F19 F20-F29 F30-F39 F40-F48 F50-F59 F60-F69 F70-F79 F80-F89 F90-F98 F99-F99 "
    "G00-G09 G10-G14 G20-G26 G30-G32 G35-G37 G40-G47 G50-G59 G60-G64 G70-G73 G80-G83 G90-G99 "
    "H00-H06 H10-H13 H15-H22 H25-H28 H30-H36 H40-H42 H43-H45 H46-H48 H49-H52 H53-H54 H55-H59 H60-H62 H65-H75 H80-H83 H90-H95 "
    "I00-I02 I05-I09 I10-I15 I20-I25 I26-I28 I30-I52 I60-I69 I70-I79 I80-I89 I95-I99 "
    "J00-J06 J09-J18 J20-J22 J30-J39 J40-J47 J60-J70 J80-J84 J85-J86 J90-J94 J95-J99 "
    "K00-K14 K20-K31 K35-K38 K40-K46 K50-K52 K55-K64 K65-K67 K70-K77 K80-K87 K90-K93 "
    "L00-L08 L10-L14 L20-L30 L40-L45 L50-L54 L55-L59 L60-L75 L80-L99 "
    "M00-M03 M05-M14 M15-M19 M20-M25 M30-M36 M40-M43 M45-M49 M50-M54 M60-M63 M65-M68 M70-M79 M80-M85 M86-M90 M91-M94 M95-M99 "
    "N00-N08 N10-N16 N17-N19 N20-N23 N25-N29 N30-N39 N40-N51 N60-N64 N70-N77 N80-N98 N99-N99 "
    "O00-O08 O10-O16 O20-O29 O30-O48 O60-O75 O80-O84 O85-O92 O94-O99 "
    "P00-P04 P05-P08 P10-P15 P20-P29 P35-P39 P50-P61 P70-P74 P75-P78 P80-P83 P90-P96 "
    "Q00-Q07 Q10-Q18 Q20-Q28 Q30-Q34 Q35-Q37 Q38-Q45 Q50-Q56 Q60-Q64 Q65-Q79 Q80-Q89 Q90-Q99 "
    "R00-R09 R10-R19 R20-R23 R25-R29 R30-R39 R40-R46 R47-R49 R50-R69 R70-R79 R80-R82 R83-R89 R90-R94 R95-R99 "
    "S00-S09 S10-S19 S20-S29 S30-S39 S40-S49 S50-S59 S60-S69 S70-S79 S80-S89 S90-S99 "
    "T00-T07 T08-T14 T15-T19 T20-T32 T33-T35 T36-T50 T51-T65 T66-T78 T79-T79 T80-T88 T90-T98 "
    "V01-X59 X60-X84 X85-Y09 Y10-Y34 Y35-Y36 Y40-Y84 Y85-Y89 Y90-Y98 "
    "Z00-Z13 Z20-Z29 Z30-Z39 Z40-Z54 Z55-Z65 Z70-Z76 Z80-Z99 U00-U49 U82-U85"
).split()
_BLOCK_RANGES = tuple(tuple(b.split("-")) for b in BLOCKS)


def block_of(code3):
    """三位码所在的节，返回 "blk:I20-I25"；不是三位码或不在任何一节返回 None。"""
    c = (code3 or "").upper()
    if len(c) != 3 or not c[0].isalpha() or not c[1:].isdigit():
        return None
    for lo, hi in _BLOCK_RANGES:
        if lo <= c <= hi:
            return f"blk:{lo}-{hi}"
    return None
