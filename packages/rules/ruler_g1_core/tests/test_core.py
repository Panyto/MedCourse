"""总经理编写的对抗性测试。

每条用例写明：文书类型、正文、要看的对、是否应可比、是否应报冲突。
大部分用例是"硬负例"：真实病历里常见、看起来像冲突但不是冲突的写法。
这些用例由总经理维护；伙计不得为了让测试通过而修改用例，只能报告失败。
"""
import unittest

from ruler_g1 import evaluate_document, is_first_progress, render_summary7, split_sections

H_ADM = "入院记录\n姓名：[患者] | 性别：{sex} | 年龄：{age}\n科室：内科 | 床号：[床号]\n"
H_DIS = "出院记录\n姓名：[患者]\n性别：\n{sex}\n年龄：\n{age}\n入院时间：2025-03-01 10:00\n出院时间：2025-03-{dd}\n住院天数：{los}天\n"
H_ADM_D = "入院记录\n姓名：[患者] | 性别：{sex} | 年龄：{age}\n入院时间：2025-03-10 10:00\n"
H_FPN = "首次病程记录\n2025-03-01 15:00\n姓名：[患者] 年龄：{age} 科室：内科\n"


def adm(body, sex="男", age="45岁"):
    return H_ADM.format(sex=sex, age=age) + body


def admd(body, sex="男", age="45岁"):
    return H_ADM_D.format(sex=sex, age=age) + body


def dis(body, sex="女", age="60岁", dd="08", los="7"):
    return H_DIS.format(sex=sex, age=age, dd=dd, los=los) + body


def fpn(body, age="50岁"):
    return H_FPN.format(age=age) + body


def s7(sex="Male", age="60", dx="肺炎", cc="咳嗽3天", hpi="患者3天前受凉后咳嗽。", hc="予抗感染治疗。", di="门诊随诊。"):
    return render_summary7({"Gender": sex, "Age": age, "Primary Diagnosis": dx, "Chief Complaint": cc,
                            "History of Present Illness": hpi, "Hospital Course": hc,
                            "Discharge Instructions": di})


# (名称, 文书类型, 正文, 对, 应可比, 应冲突)
CASES = [
    # ---- G1-1 性别 ----
    ("性别一致", "hospital_record", adm("主诉：胸痛3天。\n现病史：患者，男，45岁，3天前出现胸痛。"), "G1-1", True, False),
    ("性别矛盾", "hospital_record", adm("主诉：胸痛3天。\n现病史：患者，女，45岁，3天前出现胸痛。"), "G1-1", True, True),
    ("占位符主语", "hospital_record", adm("现病史：[患者]，女，45岁，3天前出现胸痛。"), "G1-1", True, True),
    ("女性患者", "hospital_record", adm("现病史：女性患者，3天前出现胸痛。"), "G1-1", True, True),
    ("硬负例：育有1子1女", "hospital_record", adm("现病史：3天前出现胸痛。\n婚育史：适龄结婚，育有1子1女。"), "G1-1", False, False),
    ("硬负例：家族史里的男", "hospital_record", adm("现病史：3天前出现胸痛。\n家族史：其父，男，70岁，患高血压。", sex="女"), "G1-1", False, False),
    ("硬负例：女儿陪同", "hospital_record", adm("现病史：3天前出现胸痛，由其女儿陪同就诊。"), "G1-1", False, False),
    ("出院记录入院情况", "discharge_summary", dis("入院情况：患者，女，60岁，因“咳嗽2周”入院。"), "G1-1", True, False),
    ("出院记录性别矛盾", "discharge_summary", dis("入院情况：患者，男，60岁，因“咳嗽2周”入院。"), "G1-1", True, True),
    # ---- G1-2 性别特异内容 ----
    ("男性有初潮", "hospital_record", adm("月经史：初潮13岁，经期5天。"), "G1-2", True, True),
    ("男性配偶孕产", "hospital_record", adm("婚育史：适龄结婚，配偶孕2产1，体健。"), "G1-2", True, False),
    ("男性月经史模板", "hospital_record", adm("月经史：/\n家族史：无特殊。"), "G1-2", True, False),
    ("女性有前列腺", "hospital_record", adm("专科情况：前列腺无肿大。", sex="女"), "G1-2", True, True),
    ("女性丈夫前列腺", "hospital_record", adm("婚育史：丈夫前列腺增生。", sex="女"), "G1-2", True, False),
    ("家族史不算", "hospital_record", adm("家族史：其母子宫肌瘤，其姐孕3产2。"), "G1-2", True, False),
    # v0.3：儿科出生史里的孕产次、分娩方式说的是母亲
    ("v0.3：男童出生史G1P1顺产", "hospital_record", adm("个人史：系G1P1，孕39周顺产，出生体重3.2kg。", age="2岁"), "G1-2", True, False),
    ("v0.3：男婴剖宫产娩出", "hospital_record", adm("个人史：第1胎第1产，足月剖宫产娩出。", age="20天"), "G1-2", True, False),
    ("v0.3：成年男性出生史", "hospital_record", adm("个人史：足月顺产，生长发育同同龄人。", age="35岁"), "G1-2", True, False),
    ("v0.3：男童有初潮仍冲突", "hospital_record", adm("月经史：初潮12岁，经期5天。", age="10岁"), "G1-2", True, True),
    ("v0.3：成年男性孕产次仍冲突", "hospital_record", adm("婚育史：已婚，孕2产1。", age="40岁"), "G1-2", True, True),
    ("v0.3：女性查体有前列腺（查体段落已能识别）", "hospital_record", adm("家族史：无特殊。\n体格检查T：36.5℃，直肠指检前列腺无肿大。", sex="女"), "G1-2", True, True),
    ("v0.3：家族史后的查体不再被吞", "hospital_record", adm("家族史：无特殊。\n体格检查T：36.5℃，外生殖器发育正常，经期查体未做。"), "G1-2", True, True),
    # ---- G1-3 年龄 ----
    ("年龄一致", "hospital_record", adm("现病史：患者，男，45岁。"), "G1-3", True, False),
    ("年龄矛盾", "hospital_record", adm("现病史：患者，男，55岁。"), "G1-3", True, True),
    ("年龄差1岁只算轻差", "hospital_record", adm("现病史：患者，男，46岁。"), "G1-3", True, False),
    ("硬负例：初潮13岁", "hospital_record", adm("现病史：3天前胸痛。\n月经史：初潮13岁。", sex="女"), "G1-3", False, False),
    ("硬负例：13岁时", "hospital_record", adm("现病史：患者13岁时曾患肺炎。"), "G1-3", False, False),
    ("硬负例：家属年龄", "hospital_record", adm("现病史：3天前胸痛。\n家族史：父亲，男，75岁，体健。"), "G1-3", False, False),
    ("高龄区间一致", "hospital_record", adm("现病史：患者，男，≥90岁。", age="≥90岁"), "G1-3", True, False),
    ("高龄区间矛盾", "hospital_record", adm("现病史：患者，男，75岁。", age="≥90岁"), "G1-3", True, True),
    ("婴儿月龄", "hospital_record", adm("现病史：患儿，男，3个月。", age="3个月"), "G1-3", True, False),
    ("首次病程病例特点", "first_progress", fpn("一、病例特点\n患者年龄、性别，病程：男，50岁，病程3天。"), "G1-3", True, False),
    ("首次病程年龄矛盾", "first_progress", fpn("一、病例特点\n患者年龄、性别，病程：男，60岁，病程3天。"), "G1-3", True, True),
    # ---- G1-4 起病时长 ----
    ("时长一致", "hospital_record", adm("主诉：胸痛3天。\n现病史：患者于3天前无诱因出现胸痛。"), "G1-4", True, False),
    ("时长矛盾", "hospital_record", adm("主诉：胸痛3天。\n现病史：患者于3月前无诱因出现胸痛。"), "G1-4", True, True),
    ("年余对年前", "hospital_record", adm("主诉：咳嗽2年余。\n现病史：2年前出现咳嗽。"), "G1-4", True, False),
    ("周对天", "hospital_record", adm("主诉：发热1周。\n现病史：7天前出现发热。"), "G1-4", True, False),
    ("半月对两周", "hospital_record", adm("主诉：腹痛半月。\n现病史：2周前出现腹痛。"), "G1-4", True, False),
    ("中文数字", "hospital_record", adm("主诉：腹痛两天。\n现病史：2天前出现腹痛。"), "G1-4", True, False),
    ("十余年", "hospital_record", adm("主诉：反复咳嗽十余年，加重3天。\n现病史：十余年前开始反复咳嗽，3天前加重。"), "G1-4", True, False),
    ("小时一致", "hospital_record", adm("主诉：胸痛2小时。\n现病史：2小时前突发胸痛。"), "G1-4", True, False),
    ("小时对天矛盾", "hospital_record", adm("主诉：胸痛2小时。\n现病史：2天前突发胸痛。"), "G1-4", True, True),
    ("硬负例：后面的既往事件", "hospital_record", adm("主诉：胸痛3天。\n现病史：患者3天前出现胸痛。既往高血压10年，5年前行阑尾切除术。"), "G1-4", True, False),
    ("硬负例：现病史用日期", "hospital_record", adm("主诉：胸痛3天。\n现病史：患者于2025年2月26日出现胸痛。"), "G1-4", False, False),
    ("硬负例：主诉里的年份", "hospital_record", adm("主诉：2020年体检发现肺结节。\n现病史：5年前体检发现肺结节。"), "G1-4", False, False),
    ("出院记录一致", "discharge_summary", dis("入院情况：患者因“咳嗽2周”入院。患者2周前受凉后咳嗽。"), "G1-4", True, False),
    ("出院记录矛盾", "discharge_summary", dis("入院情况：患者因“咳嗽2周”入院。患者2月前受凉后咳嗽。"), "G1-4", True, True),
    # ---- G1-5 侧别 ----
    ("侧别一致", "hospital_record", adm("主诉：左膝疼痛1月。\n入院诊断：左膝骨关节炎"), "G1-5", True, False),
    ("侧别矛盾", "hospital_record", adm("主诉：左膝疼痛1月。\n入院诊断：右膝骨关节炎"), "G1-5", True, True),
    ("左侧写法", "hospital_record", adm("主诉：左侧膝关节疼痛1月。\n入院诊断：右膝骨关节炎"), "G1-5", True, True),
    ("硬负例：既往史里另一侧", "hospital_record", adm("主诉：右膝疼痛1月。\n既往史：左膝置换术后。\n入院诊断：右膝骨关节炎"), "G1-5", True, False),
    # v0.3 起"双"只在被比的两段里排除：主诉↔专科情况 这一对因"双膝"不比，主诉↔入院诊断 仍可比且一致
    ("硬负例：专科写双膝", "hospital_record", adm("主诉：左膝疼痛1月。\n专科情况：双膝无畸形，左膝压痛。\n入院诊断：左膝骨关节炎"), "G1-5", True, False),
    ("v0.3：查体双肺不连带排除肺", "hospital_record", adm("主诉：体检发现右肺结节1月。\n体格检查T：36.5℃，双肺呼吸音清。\n入院诊断：左肺结节"), "G1-5", True, True),
    ("v0.3：诊断里写双侧则该对不比", "hospital_record", adm("主诉：右下肢肿痛3天。\n入院诊断：双下肢深静脉血栓"), "G1-5", False, False),
    ("硬负例：诊断里切除术后", "hospital_record", adm("主诉：右肾结石1年。\n入院诊断：右肾结石；左肾切除术后"), "G1-5", True, False),
    ("硬负例：不成对结构", "hospital_record", adm("主诉：左侧肢体无力3天。\n入院诊断：脑梗死"), "G1-5", False, False),
    ("出院记录侧别矛盾", "discharge_summary", dis("入院诊断：左侧股骨颈骨折\n出院诊断：右侧股骨颈骨折"), "G1-5", True, True),
    # ---- v0.4 诊断口径（v0.5 起即 G1-4） ----
    # v0.5 起 G1-4 即原 G1-4b 的证据口径，下面四条改为测 G1-4
    ("v0.5：首句是既往病，G1-4 不再冲突", "hospital_record", adm("主诉：胸痛3天。\n现病史：患者10年前诊断高血压，3天前出现胸痛。"), "G1-4", True, False),
    ("v0.5：证据口径真矛盾", "hospital_record", adm("主诉：腹痛3天。\n现病史：患者3月前出现腹痛，1月前加重。"), "G1-4", True, True),
    ("v0.5：现病史只有日期、无入院日期不可比", "hospital_record", adm("主诉：腹痛3天。\n现病史：患者于2025年2月26日出现腹痛。"), "G1-4", False, False),
    # ---- v0.4 summary7（LLM-CARe 论文格式） ----
    ("v0.4：s7 一致", "summary7", s7(hpi="患者，男，60岁，3天前受凉后咳嗽。"), "G1-1", True, False),
    ("v0.4：s7 性别矛盾", "summary7", s7(sex="Female", hpi="患者，男，60岁，3天前受凉后咳嗽。"), "G1-1", True, True),
    ("v0.4：s7 年龄矛盾", "summary7", s7(age="47", hpi="患者，男，72岁，3天前受凉后咳嗽。"), "G1-3", True, True),
    ("v0.4：s7 时长矛盾", "summary7", s7(hpi="患者3月前受凉后咳嗽。"), "G1-4", True, True),
    ("v0.4：s7 侧别矛盾", "summary7", s7(dx="右膝骨关节炎", cc="左膝疼痛1年", hpi="患者1年前出现左膝疼痛。"), "G1-5", True, True),
    ("v0.4：s7 男性有孕产", "summary7", s7(hc="患者孕2产1，予对症治疗。"), "G1-2", True, True),
    ("v0.4：s7 空泛病历不可比", "summary7", s7(cc="咳嗽", hpi="患者咳嗽，予对症治疗。"), "G1-4", False, False),
    # ---- G1-6 住院天数 ----
    ("住院天数一致", "discharge_summary", dis("出院诊断：肺炎"), "G1-6", True, False),
    ("住院天数矛盾", "discharge_summary", dis("出院诊断：肺炎", los="8"), "G1-6", True, True),
    # ---- v0.5：r1 标注发现的误报（全部虚构） ----
    ("v0.5：30余年在单位前", "hospital_record", adm("主诉：反复咳嗽30余年，加重3天。\n现病史：患者30余年前开始反复咳嗽，2周前受凉后加重，3天前出现发热。"), "G1-4", True, False),
    ("v0.5：7+年", "hospital_record", adm("主诉：腰痛7+年。\n现病史：患者7+年前无诱因出现腰痛。"), "G1-4", True, False),
    ("v0.5：3个多月", "hospital_record", adm("主诉：腹胀3个多月。\n现病史：患者3个多月前出现腹胀。"), "G1-4", True, False),
    ("v0.5：1年半", "hospital_record", adm("主诉：咳嗽1年半。\n现病史：患者1年半前出现咳嗽。"), "G1-4", True, False),
    ("v0.5：两段式主诉配病史时长", "hospital_record", adm("主诉：发现血压升高10年，头晕3天。\n现病史：患者有高血压病史10年，3天前出现头晕。"), "G1-4", True, False),
    ("v0.5：现病史里的远期手术不制造冲突", "hospital_record", adm("主诉：腰痛4年。\n现病史：患者30年前行甲状腺手术，4年前出现腰痛。"), "G1-4", True, False),
    ("v0.5：主诉从确诊算起，症状更早不算冲突", "hospital_record", adm("主诉：确诊肺癌7月。\n现病史：患者1年前出现咳嗽，后确诊肺癌。"), "G1-4", True, False),
    ("v0.5：主诉不从确诊算起，症状更早仍冲突", "hospital_record", adm("主诉：咳嗽7月。\n现病史：患者1年前出现咳嗽。"), "G1-4", True, True),
    ("v0.5：反复治疗的肿瘤病历（现病史只有治疗事件，不可比）", "hospital_record", adm("主诉：肺癌术后2年，末次化疗后20天。\n现病史：患者2年前行肺癌根治术，术后规律化疗。"), "G1-4", False, False),
    ("v0.5：今晨起病", "hospital_record", adm("主诉：胸痛5小时。\n现病史：患者今晨无明显诱因突发胸痛。"), "G1-4", True, False),
    ("v0.5：今日来院不算起病", "hospital_record", adm("主诉：胸痛3天。\n现病史：患者今日来我院就诊。"), "G1-4", False, False),
    ("v0.5：入院前3天", "hospital_record", adm("主诉：发热3天。\n现病史：患者入院前3天受凉后发热。"), "G1-4", True, False),
    ("v0.5：日期起病一致", "hospital_record", admd("主诉：胸痛3天。\n现病史：患者于2025年3月7日无诱因出现胸痛。"), "G1-4", True, False),
    ("v0.5：日期起病矛盾", "hospital_record", admd("主诉：胸痛3天。\n现病史：患者于2024年12月无诱因出现胸痛。"), "G1-4", True, True),
    ("v0.5：月份精度的日期起病", "hospital_record", admd("主诉：咳嗽3月。\n现病史：患者2024年12月起无明显诱因出现咳嗽。"), "G1-4", True, False),
    ("v0.5：住院日期不算起病", "hospital_record", admd("主诉：胸痛3天。\n现病史：患者2024年12月曾在外院住院治疗。"), "G1-4", False, False),
    ("v0.5：孕周不是病程", "hospital_record", adm("主诉：停经36周，下腹痛2小时。\n现病史：患者2小时前出现下腹阵痛。", sex="女", age="28岁"), "G1-4", True, False),
    ("v0.5：孕38+2周", "hospital_record", adm("主诉：孕38+2周，阴道流液1小时。\n现病史：患者1小时前出现阴道流液。", sex="女", age="28岁"), "G1-4", True, False),
    ("v0.5：加重时间不能圆掉起病矛盾", "hospital_record", adm("主诉：发热伴咳嗽2天，加重伴气促1天。\n现病史：患者7天前无明显诱因出现发热，伴咳嗽。近1天来症状急剧加重，出现气促。"), "G1-4", True, True),
    ("v0.5：N天前症状加重不算起病", "hospital_record", adm("主诉：反复胸痛5天，加重1天。\n现病史：患者1个月前无明显诱因出现胸痛，休息后缓解。3天前症状加重。"), "G1-4", True, True),
    ("v0.5：两段式主诉配起病和加重", "hospital_record", adm("主诉：胸闷1个月，加重3天。\n现病史：患者1个月前出现胸闷，3天前症状加重。"), "G1-4", True, False),
    ("v0.5：主诉写的是加重时长", "hospital_record", adm("主诉：腹痛加重3天。\n现病史：患者1年前出现腹痛，3天前腹痛加重。"), "G1-4", True, False),
    ("v0.5：只有加重时间不可比", "hospital_record", adm("主诉：腹痛3天。\n现病史：患者3天前腹痛加重。"), "G1-4", False, False),
    ("v0.5：出现并逐渐加重仍是起病", "hospital_record", adm("主诉：头晕2年。\n现病史：患者2年前出现头晕并逐渐加重。"), "G1-4", True, False),
    ("v0.5：再次出现算新一次发作", "hospital_record", adm("主诉：反复咳嗽10年，再发3天。\n现病史：患者10年前开始反复咳嗽，3天前受凉后再次出现咳嗽加重。"), "G1-4", True, False),
    ("v0.5：反复发作不是再发", "hospital_record", adm("主诉：胸痛反复发作2天，加重1天。\n现病史：患者3个月前开始无明显诱因出现胸痛。近1周来症状逐渐加重。"), "G1-4", True, True),
    ("v0.5：确诊口径不靠病史时长放宽", "hospital_record", adm("主诉：确诊肺癌7月。\n现病史：患者3月前体检发现肺部结节，后确诊肺癌。既往高血压病史10年。"), "G1-4", True, True),
    ("v0.5：只有远期手术不可比", "hospital_record", adm("主诉：化疗后出现乏力、恶心，持续约10天。\n现病史：患者5年前因胃癌行手术切除治疗，近期因肿瘤复发于我院化疗。"), "G1-4", False, False),
    ("v0.5：远期手术加真起病仍可比", "hospital_record", adm("主诉：腰痛4年。\n现病史：患者30年前行甲状腺手术，1年前出现腰痛。"), "G1-4", True, True),
    ("v0.5：多段主诉，现病史只写这次发作", "hospital_record", adm("主诉：发现血压升高10年，头晕3天。\n现病史：患者3天前无明显诱因出现头晕，伴恶心。"), "G1-4", True, False),
    ("v0.5：多段主诉，现病史写了同一病但时长不符", "hospital_record", adm("主诉：发现血压升高10年，头晕3天。\n现病史：患者5年前体检发现血压升高，3天前出现头晕。"), "G1-4", True, True),
    ("v0.5：多段主诉，同一症状时长不符", "hospital_record", adm("主诉：胸闷3个月，头晕1周。\n现病史：患者1个月前出现胸闷，1周前出现头晕。"), "G1-4", True, True),
    ("v0.5：多段主诉都对得上", "hospital_record", adm("主诉：确诊2型糖尿病10余年，透析3年余，乏力2天。\n现病史：患者10余年前确诊2型糖尿病，3年前开始规律透析，2天前出现乏力。"), "G1-4", True, False),
    ("v0.5：加重段比病程还长", "hospital_record", adm("主诉：多饮、多尿2个月，加重伴视力模糊1年。\n现病史：患者2个月前出现多饮、多尿，1周前出现视力模糊。"), "G1-4", True, True),
    ("v0.5：加重段现病史没写具体时间", "hospital_record", adm("主诉：胸痛1个月，加重3天。\n现病史：患者1个月前出现胸痛，近来加重。"), "G1-4", True, False),
    ("v0.5：主诉里的入院时间不是病程", "hospital_record", adm("主诉：多饮多尿，于1周前入院治疗。\n现病史：患者近3个月出现多饮、多尿。"), "G1-4", False, False),
    ("v0.5：近N月起病矛盾", "hospital_record", adm("主诉：尿量减少3个月。\n现病史：患者近一个月内出现尿量减少。"), "G1-4", True, True),
    ("v0.5：阴性所见的侧别不比", "hospital_record", adm("主诉：发现左侧卵巢囊肿1月。\n专科情况：右侧卵巢未见异常，左附件区可及包块。\n入院诊断：左侧卵巢囊肿", sex="女", age="30岁"), "G1-5", True, False),
    ("v0.5：阳性所见仍比", "hospital_record", adm("主诉：发现左侧卵巢囊肿1月。\n专科情况：右侧卵巢可及包块。\n入院诊断：左侧卵巢囊肿", sex="女", age="30岁"), "G1-5", True, True),
    ("v0.5：锁骨中线是定位线", "hospital_record", adm("主诉：右锁骨骨折后疼痛2天。\n专科情况：心尖搏动位于左锁骨中线第5肋间，局部压痛。\n入院诊断：右锁骨骨折"), "G1-5", True, False),
    ("v0.5：现病史第一句的侧别不再被当成既往", "hospital_record", adm("主诉：膝痛1月。\n现病史：左膝疼痛1月，活动后加重。\n入院诊断：右膝骨关节炎"), "G1-5", True, True),
    ("v0.5：椎动脉两侧病灶不比", "hospital_record", adm("主诉：头晕3天。\n现病史：外院头颅MRA示右侧椎动脉狭窄。\n入院诊断：左侧椎动脉闭塞"), "G1-5", False, False),
    ("v0.5：当天入出院记1天", "discharge_summary", dis("出院诊断：肺炎", dd="01", los="1"), "G1-6", True, False),
    ("v0.5：当天入出院记2天仍冲突", "discharge_summary", dis("出院诊断：肺炎", dd="01", los="2"), "G1-6", True, True),
]


class TestCases(unittest.TestCase):
    def test_cases(self):
        failures = []
        for name, dtype, text, pair, want_check, want_conf in CASES:
            r = evaluate_document(text, "SYN", 0, dtype)
            got_check = r["checkable"][pair] > 0
            got_conf = any(c.pair_id == pair for c in r["conflicts"])
            if got_check != want_check or got_conf != want_conf:
                failures.append(f"{name}: 可比 {got_check}/{want_check}，冲突 {got_conf}/{want_conf}")
        self.assertEqual(failures, [], "\n" + "\n".join(failures))

    def test_mild_age(self):
        r = evaluate_document(adm("现病史：患者，男，46岁。"))
        self.assertEqual(r["mild"].get("G1-3"), 1)


class TestSections(unittest.TestCase):
    def test_spaced_title_alone_on_line(self):
        names = [s.name for s in split_sections(adm("现病史：胸痛。\n体 格 检 查\nT 36.5℃\n专科情况：无。"))]
        self.assertIn("体格检查", names)
        self.assertIn("专科情况", names)

    def test_title_word_in_running_text(self):
        names = [s.name for s in split_sections(adm("现病史：患者主诉胸痛，入院诊断待定。"))]
        self.assertEqual(names.count("入院诊断"), 0)

    def test_vital_signs_after_title(self):
        # v0.3：入院记录常写"体格检查T：36.5℃"，标题后面直接接体温
        for body in ("家族史：无特殊。\n体格检查T：36.5℃ P：80次/分\n专科情况：无。",
                     "家族史：无特殊。\n体格检查 T:36.5℃ P:80次/分\n专科情况：无。",
                     "家族史：无特殊。\n体格检查体温：36.5℃\n专科情况：无。"):
            secs = split_sections(adm(body))
            names = [s.name for s in secs]
            self.assertIn("体格检查", names, body)
            fam = [s for s in secs if s.name == "家族史"][0]
            self.assertNotIn("36.5", fam.text, body)

    def test_vital_lookahead_needs_number(self):
        # "体格检查欠合作"之类不是标题
        names = [s.name for s in split_sections(adm("现病史：患者体格检查Tom配合差。"))]
        self.assertNotIn("体格检查", names)

    def test_revision_and_signature_end_admit_diagnosis(self):
        secs = split_sections(adm("入院诊断：左膝骨关节炎\n修正诊断：右膝骨关节炎\n住院医师签名：[医师]"))
        diag = [s for s in secs if s.name == "入院诊断"][0]
        self.assertNotIn("右膝", diag.text)
        self.assertIn("修正诊断", [s.name for s in secs])
        self.assertIn("签名", [s.name for s in secs])

    def test_summary7_adapter(self):
        text = render_summary7({"性别": "女性", "年龄": "47", "主要诊断": "肺炎", "主诉": "咳嗽\n3天",
                                "现病史": "患者\n主诉：咳嗽", "住院经过": "好转", "出院医嘱": "随诊"})
        self.assertIn("性别：女\n", text)
        self.assertIn("年龄：47岁\n", text)
        names = [s.name for s in split_sections(text)]
        # 正文里的换行被抹平，模型自己写的"主诉："不会多切出一个段落
        self.assertEqual(names.count("主诉"), 1)

    def test_first_progress(self):
        self.assertTrue(is_first_progress("首次病程记录\n2025-03-01"))
        self.assertFalse(is_first_progress("术后首次病程记录\n2025-03-05"))


if __name__ == "__main__":
    unittest.main()
