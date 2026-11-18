# -*- coding: utf-8 -*-
"""路线 B 原型：从公开临床路径建病种知识表 kb_facts。只处理公开文本，不读任何病历和派生材料。

sections    把路径正文切成（一）…（十二）小节和逐日表单
schema      kb_facts 的字段和校验
provenance  每条事实的引文核对：引文在原文里、取值在引文里，才算有出处
chunks/retrieve  把路径和指导原则切成段落、BM25 检索（只切不抽）
agentic     路线 B 正式的事实获取：8B 两阶段自主提问检索 + 引文核对（build_kb_agentic 批量跑）
extract/build_kb  正则抽取，只出 kb_v0 当消融靶子（compare_kb 对比）
rule_compare  规则层单挑：手写规则 vs 8B（RB1）
"""
VERSION = "route-b-kit-0.5"
