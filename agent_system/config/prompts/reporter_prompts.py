"""
Reporter 层 Prompt 模板
"""

# 1. 大纲生成 Prompt
OUTLINE_GEN_PROMPT = """你是一个专业的报告规划师。基于用户的查询和收集到的证据，请构建一份结构清晰的报告大纲。

**用户查询**: {query}

**任务背景**:
{background}

**已收集证据摘要**:
{evidence_summary}

请生成 JSON 格式的大纲，包含以下字段：
- title: 报告标题
- abstract: 全文摘要
- sections: 章节列表（每个章节包含 id, title, description）

要求：
1. 逻辑流畅，从背景到分析再到结论。
2. 章节数量控制在 3-6 个。
3. 确保覆盖用户查询的核心痛点。

输出示例：
{{
    "title": "...",
    "abstract": "...",
    "sections": [
        {{ "section_id": "sec_1", "title": "背景", "description": "..." }}
    ]
}}
"""

# 2. 章节写作 Prompt (传统模式)
SECTION_WRITE_PROMPT = """你是一个严谨的学术/商业报告撰写人。请基于提供的证据，撰写报告的指定章节。

**章节标题**: {section_title}
**章节说明**: {section_description}

**可用证据**:
{evidence_context}

**撰写要求**:
1. 内容必须严格基于提供的证据，严禁编造。
2. 引用证据时，请在句尾标注 [ID]。例如：...票价为210元[evidence_1]。
3. 语言风格专业、客观。
4. 如果证据不足以覆盖某些点，请如实说明。

请直接输出章节的正文内容（Markdown格式），不要包含标题。
"""