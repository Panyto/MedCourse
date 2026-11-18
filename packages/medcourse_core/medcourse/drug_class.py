"""药物大类（0.1.9）：g3-rules-0.3 药名词表里 280 种药按公开药理分类归大类，标出长期用药。公开知识，不来自病历。
0.1.11：长期用药的适应证（INDICATION、DRUG_INDICATION、CLASS_CHAPTER，公开说明书）；同类第二种药的亚类（SUBCLASS，INN 词干）。

用途：
- 统计表按大类计数（stats.med_class）。化疗方案、降压方案的药名分散，按药名计数凑不够 k 次，按大类就够了。
- 规划先按频率抽大类，再在大类里抽具体药，保证主诊断层频率最高的那一类在里面（第八次合成病历：4 次化疗住院一个化疗药都没有）。
- 长期用药（降压、降糖、抗血小板、调脂……）住院期间不停，出院带药；不写"疗程已足"（第八次 6/20 份把波立维、美托洛尔写成疗程已足）。
- 停药理由按大类写。
"""
import re

# 大类 → 药（drug_id，同 ruler_g3/data/drug_lexicon.csv）
CLASSES = {
    "抗菌药": "levofloxacin moxifloxacin ciprofloxacin ofloxacin norfloxacin gatifloxacin gemifloxacin sitafloxacin nemonoxacin pazufloxacin "
              "lomefloxacin fleroxacin pefloxacin antofloxacin prulifloxacin tetracycline doxycycline minocycline oxytetracycline chlortetracycline "
              "tigecycline amoxicillin amox_clav cefuroxime ceftriaxone ceftazidime meropenem pip_tazo azithromycin metronidazole cefixime cefaclor "
              "cefdinir cefprozil cefazolin cefoperazone_sulbactam cefoperazone cefotaxime cefepime cefoxitin cefathiamidine ceftizoxime cefmetazole "
              "ceph_unspecified penicillin_g ampicillin piperacillin amp_sulbactam imipenem ertapenem biapenem vancomycin linezolid clindamycin "
              "erythromycin clarithromycin roxithromycin gentamicin amikacin etimicin ornidazole tinidazole smz_tmp nitrofurantoin fosfomycin",
    "抗真菌药": "fluconazole voriconazole",
    "抗病毒药": "acyclovir ganciclovir oseltamivir ribavirin entecavir tenofovir",
    "抗结核药": "isoniazid rifampicin ethambutol pyrazinamide",
    "解热镇痛抗炎药": "nimesulide acetaminophen ibuprofen diclofenac celecoxib loxoprofen flurbiprofen parecoxib",
    "阿片类镇痛药": "codeine dihydrocodeine tramadol tramadol_apap morphine oxycodone fentanyl pethidine",
    "RAS抑制剂": "captopril enalapril benazepril perindopril ramipril fosinopril lisinopril imidapril cilazapril quinapril losartan valsartan "
                "irbesartan telmisartan candesartan olmesartan allisartan azilsartan eprosartan sacubitril_valsartan",
    "钙通道阻滞剂": "amlodipine levamlodipine nifedipine_er nifedipine felodipine nitrendipine nimodipine diltiazem verapamil",
    "β受体阻滞剂": "metoprolol bisoprolol carvedilol propranolol labetalol",
    "利尿剂": "furosemide torasemide hydrochlorothiazide spironolactone indapamide",
    "α受体阻滞剂": "terazosin doxazosin tamsulosin",
    "调脂药": "atorvastatin rosuvastatin simvastatin pravastatin pitavastatin ezetimibe",
    "抗血小板药": "aspirin clopidogrel ticagrelor",
    "抗凝药": "warfarin rivaroxaban dabigatran heparin lmwh",
    "抗心绞痛药": "isosorbide_mononitrate isosorbide_dinitrate nitroglycerin trimetazidine",
    "强心和抗心律失常药": "digoxin amiodarone propafenone",
    "血管活性药": "dobutamine dopamine norepinephrine epinephrine",
    "降糖药": "metformin sitagliptin glimepiride gliclazide glipizide glibenclamide repaglinide acarbose voglibose pioglitazone insulin "
              "dapagliflozin empagliflozin liraglutide",
    "抑酸药": "omeprazole esomeprazole pantoprazole lansoprazole rabeprazole ilaprazole famotidine ranitidine",
    "胃肠动力和止吐药": "mosapride domperidone metoclopramide ondansetron",
    "消化系统其他用药": "sucralfate ursodeoxycholic_acid lactulose montmorillonite",
    "祛痰镇咳药": "ambroxol acetylcysteine carbocisteine bromhexine dextromethorphan",
    "平喘药": "theophylline salbutamol budesonide ipratropium tiotropium montelukast",
    "糖皮质激素": "prednisone prednisolone methylprednisolone dexamethasone hydrocortisone",
    "抗肿瘤药": "cyclophosphamide capecitabine oxaliplatin cisplatin carboplatin paclitaxel docetaxel gemcitabine fluorouracil",
    "内分泌治疗药": "tamoxifen letrozole bicalutamide",
    "抗风湿药": "methotrexate hydroxychloroquine leflunomide",
    "神经系统用药": "pregabalin gabapentin carbamazepine valproate levetiracetam oxcarbazepine phenytoin donepezil",
    "镇静催眠药": "diazepam alprazolam estazolam zolpidem",
    "精神科用药": "olanzapine quetiapine risperidone haloperidol sertraline escitalopram paroxetine",
    "脑保护和脱水药": "mannitol citicoline edaravone butylphthalide",
    "甲状腺用药": "levothyroxine methimazole propylthiouracil",
    "骨代谢用药": "calcium_carbonate calcitriol alendronate",
    "电解质和酸碱用药": "potassium_chloride sodium_bicarbonate magnesium_sulfate",
    "抗痛风药": "allopurinol febuxostat benzbromarone colchicine",
    "抗贫血和维生素": "iron folic_acid mecobalamin",
    "子宫收缩药": "oxytocin carbetocin carboprost dinoprostone mifepristone",
    "性激素": "progesterone dydrogesterone estradiol",
    "前列腺用药": "finasteride dutasteride",
    "止血药": "tranexamic_acid ethamsylate",
    "溶栓药": "urokinase alteplase",
    "围术期用药": "atropine neostigmine lidocaine propofol midazolam",
    "皮肤科用药": "isotretinoin",
}
DRUG_CLASS = {d: c for c, ds in CLASSES.items() for d in ds.split()}

# 长期用药：住院期间不停，出院带药，不写"疗程已足"。按药定（同是抗凝药，华法林长期、低分子肝素不是）
CHRONIC = set((
    "RAS抑制剂 钙通道阻滞剂 β受体阻滞剂 α受体阻滞剂 调脂药 抗血小板药 降糖药 内分泌治疗药 抗风湿药 神经系统用药 精神科用药 甲状腺用药 "
    "骨代谢用药 前列腺用药 抗结核药").split())
CHRONIC_DRUGS = {"warfarin", "rivaroxaban", "dabigatran", "isosorbide_mononitrate", "trimetazidine", "digoxin", "amiodarone", "spironolactone",
                 "indapamide", "entecavir", "tenofovir", "tiotropium", "montelukast", "allopurinol", "febuxostat", "benzbromarone", "iron", "folic_acid",
                 "isotretinoin"}
NOT_CHRONIC_DRUGS = {"haloperidol", "nimodipine"}

# 0.1.12（第十一次合成病历：顺铂、氟尿嘧啶、头孢曲松写成出院带药）：国内没有口服剂型的药（公开说明书、药典的剂型；注射、静脉、
# 吸入麻醉、阴道用）。统计表里途径多为空（第十一次 A256 出院带药 329 条里 299 条途径为空），途径为空时按这里判能不能出院带走。
# 只列词表里有的药；有口服剂型的（氨苄西林、环磷酰胺、甲泼尼龙、咪达唑仑……）不列
NO_ORAL = set((
    "ceftriaxone ceftazidime cefazolin cefoperazone_sulbactam cefoperazone cefotaxime cefepime cefoxitin cefathiamidine ceftizoxime cefmetazole "
    "meropenem imipenem ertapenem biapenem pip_tazo piperacillin penicillin_g amp_sulbactam vancomycin amikacin etimicin tigecycline "
    "parecoxib fentanyl pethidine heparin lmwh dobutamine dopamine norepinephrine epinephrine "
    "oxaliplatin cisplatin carboplatin paclitaxel docetaxel gemcitabine fluorouracil mannitol edaravone urokinase alteplase "
    "lidocaine propofol neostigmine oxytocin carbetocin carboprost dinoprostone").split())


def oral_ok(did, route):
    """出院能带走：途径写明口服；途径为空时药有口服剂型（不在 NO_ORAL 里）。0.1.11 途径为空一律当口服。"""
    return route == "口服" or (route == "" and did not in NO_ORAL)

# 停药理由（病程写法的公开常识）；不在表里的写"症状改善，予停用"
STOP_REASON = {"抗菌药": "抗感染疗程已足，予停用", "抗真菌药": "抗真菌疗程已足，予停用", "抗病毒药": "疗程已足，予停用",
               "解热镇痛抗炎药": "疼痛缓解，予停用", "阿片类镇痛药": "疼痛缓解，予停用", "抗肿瘤药": "本周期化疗已完成",
               "糖皮质激素": "症状控制，予停用", "利尿剂": "水肿减轻，予停用", "电解质和酸碱用药": "复查电解质正常，予停用",
               "血管活性药": "血压平稳，予停用", "围术期用药": "术后予停用", "子宫收缩药": "产后宫缩良好，予停用", "止血药": "无活动性出血，予停用"}
DEFAULT_STOP = "症状改善，予停用"


# 0.1.11（第十次 M4）：长期用药的适应证（公开药品说明书的适应证，按大类写；个别药另写）。病人的主诊断、合并症、既往史里一个都对不上，
# 就不开这类长期药（主治的一类除外：它是按本诊断层的 Lift 选出来的，数据本身就是适应证）。0.1.10 心衰病人"否认糖尿病"却开了二甲双胍、阿卡波糖
INDICATION = {
    "RAS抑制剂": r"高血压|心力衰竭|心衰|心功能不全|冠心病|冠状动脉|心肌梗死|心肌病|糖尿病肾病|肾病|蛋白尿|慢性肾",
    "钙通道阻滞剂": r"高血压|心绞痛|冠心病|冠状动脉|蛛网膜下腔出血|雷诺",
    "β受体阻滞剂": r"高血压|冠心病|冠状动脉|心绞痛|心肌梗死|心力衰竭|心衰|心功能不全|心律失常|心房颤动|房颤|心动过速|早搏|甲状腺功能亢进|甲亢|心肌病",
    "α受体阻滞剂": r"前列腺|高血压|输尿管.{0,4}结石|尿潴留|排尿困难",
    "调脂药": r"高脂血症|血脂|胆固醇|冠心病|冠状动脉|心肌梗死|心绞痛|脑梗|动脉粥样硬化|动脉硬化|糖尿病|斑块|支架",
    "抗血小板药": r"冠心病|冠状动脉|心肌梗死|心绞痛|脑梗|缺血性|动脉粥样硬化|动脉硬化|支架|短暂性脑缺血|动脉闭塞|动脉狭窄",
    "降糖药": r"糖尿病|高血糖",
    "内分泌治疗药": r"乳腺.{0,4}(?:癌|恶性)|乳房.{0,4}恶性|前列腺.{0,4}(?:癌|恶性)|子宫内膜.{0,4}(?:癌|恶性)",
    "抗风湿药": r"类风湿|关节炎|红斑狼疮|狼疮|银屑病|干燥综合征|皮肌炎|血管炎|强直性脊柱炎|风湿",
    "神经系统用药": r"癫痫|抽搐|惊厥|神经痛|带状疱疹|神经病变|痴呆|阿尔茨海默|认知|帕金森|周围神经|三叉神经",
    "精神科用药": r"精神|抑郁|焦虑|分裂|双相|躁狂|谵妄|心境|强迫|失眠|睡眠障碍",
    "甲状腺用药": r"甲状腺|甲亢|甲减|桥本",
    "骨代谢用药": r"骨质疏松|骨折|骨量|低钙|佝偻病|维生素D|肾衰|肾功能不全|慢性肾|甲状旁腺|骨软化",
    "前列腺用药": r"前列腺",
    "抗结核药": r"结核",
}
# 公开 ICD-10 的章：精神和行为障碍（F）、神经系统疾病（G）、肌肉骨骼和结缔组织（M）的主诊断，本章的长期用药算有适应证
CLASS_CHAPTER = {"精神科用药": "F", "神经系统用药": "G", "抗风湿药": "M"}
DRUG_INDICATION = {
    "warfarin": r"房颤|心房颤动|血栓|栓塞|瓣膜|深静脉", "rivaroxaban": r"房颤|心房颤动|血栓|栓塞|深静脉", "dabigatran": r"房颤|心房颤动|血栓|栓塞|深静脉",
    "isosorbide_mononitrate": r"冠心病|冠状动脉|心绞痛|心肌梗死|心肌缺血|心力衰竭|心衰", "trimetazidine": r"冠心病|冠状动脉|心绞痛|心肌梗死|心肌缺血",
    "digoxin": r"心力衰竭|心衰|心功能不全|房颤|心房颤动", "amiodarone": r"心律失常|房颤|心房颤动|心动过速|早搏|心力衰竭|心衰",
    "spironolactone": r"心力衰竭|心衰|心功能不全|高血压|肝硬化|腹水|醛固酮|水肿|低钾", "indapamide": r"高血压",
    "entecavir": r"乙型|乙肝|肝炎|肝硬化", "tenofovir": r"乙型|乙肝|肝炎|肝硬化|艾滋|HIV",
    "tiotropium": r"慢性阻塞|慢阻肺|哮喘|支气管", "montelukast": r"哮喘|支气管|喘息|过敏性鼻炎",
    "allopurinol": r"痛风|高尿酸", "febuxostat": r"痛风|高尿酸", "benzbromarone": r"痛风|高尿酸",
    "iron": r"贫血|妊娠|孕|缺铁|肾衰|慢性肾|出血", "folic_acid": r"贫血|妊娠|孕|叶酸|肾衰|慢性肾", "isotretinoin": r"痤疮",
}
_IND_RE = {k: re.compile(v) for k, v in {**INDICATION, **DRUG_INDICATION}.items()}


def indicated(did, dx_text, code3=""):
    """长期用药在这个病人身上有没有适应证：不是长期用药、或没写适应证的，算有。"""
    if not is_chronic(did):
        return True
    rx = _IND_RE.get(did) or _IND_RE.get(drug_class(did))
    if rx is None:
        return True
    if rx.search(dx_text or ""):
        return True
    ch = CLASS_CHAPTER.get(drug_class(did))
    return bool(ch and (code3 or "")[:1] == ch)


# 0.1.11（第十次 M9）：同一大类的第二种药不和第一种同属一个亚类。亚类按国际非专利药名的词干分（公开命名规则）：
# 规则包药名词表的类别标记（ceph 头孢类、fq 喹诺酮类、tetra 四环素类、ras 普利和沙坦）都在里面，另补青霉素类、碳青霉烯类、
# 大环内酯类、氨基糖苷类、硝基咪唑类、他汀、拉唑、地平、洛尔。0.1.10 写出了头孢他啶加头孢曲松
# 0.2.0.2（第十五次 S1-B 停线：O 章缩宫素＋卡贝缩宫素同大类并开触发 med_conflict 3/498）：按国际非专利药名与药理词干补齐
# 催产素类（-tocin）、前列腺素类（-prost-）、硝酸酯类、肝素类、泼尼松类、可待因类、曲马多类、肾上腺素类，全词表 280 种药同大类子串配对均不可并开
SUBCLASS = (("ceph", r"^cef|^ceph"), ("fq", r"oxacin$"), ("tetra", r"cycline$"), ("carbapenem", r"penem$"),
            ("penicillin", r"cillin$|^penicillin|^amox|^amp_|^pip_"), ("macrolide", r"thromycin$|^erythromycin$"), ("aminoglycoside", r"micin$|^amikacin$"),
            ("nitroimidazole", r"nidazole$"), ("pril", r"pril$"), ("sartan", r"sartan"), ("statin", r"statin$"), ("prazole", r"prazole$"),
            ("dipine", r"dipine"), ("olol", r"olol$"),
            ("tocin", r"tocin$"), ("prost", r"prost"), ("nitrate", r"^isosorbide_|^nitroglycerin$"),
            ("heparin", r"^heparin$|^lmwh$"), ("prednis", r"^prednis"), ("codeine", r"codeine$"),
            ("tramadol", r"^tramadol"), ("epinephrine", r"epinephrine$"))
_SUB_RE = tuple((n, re.compile(rx)) for n, rx in SUBCLASS)


def subclass(did):
    return next((n for n, rx in _SUB_RE if rx.search(did or "")), None)


def drug_class(did):
    return DRUG_CLASS.get(did, "其他")


def is_chronic(did):
    if did in NOT_CHRONIC_DRUGS:
        return False
    return did in CHRONIC_DRUGS or drug_class(did) in CHRONIC


def stop_reason(cls):
    return STOP_REASON.get(cls, DEFAULT_STOP)
