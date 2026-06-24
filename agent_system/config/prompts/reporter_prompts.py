"""
Reporter 层 Prompt 模板
"""

# 1. 大纲生成 Prompt
OUTLINE_GEN_PROMPT = """你是一个专业的报告规划师。基于用户的查询和收集到的证据，请构建一份结构清晰的报告大纲。

**用户查询**: {query}

**任务背景**:
{background}

**输出模式要求**:
{plan_mode_instruction}

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

**输出模式要求**:
{plan_mode_instruction}

**撰写要求**:
1. 内容必须严格基于提供的证据，严禁编造。
2. 引用证据时，请在句尾标注 [ID]。例如：...票价为210元[evidence_1]。
3. 语言风格专业、客观。
4. 如果证据不足以覆盖某些点，请如实说明。

请直接输出章节的正文内容（Markdown格式），不要包含标题。
"""

# 2b. 批量章节写作 Prompt，用于 full 模式减少 LLM 调用次数
BATCH_SECTION_WRITE_PROMPT = """你是一个严谨的文旅报告撰写人。请基于提供的证据，一次性撰写所有章节。

**报告标题**: {title}
**报告摘要**: {abstract}

**章节大纲 JSON**:
{sections_json}

**可用证据**:
{evidence_context}

**输出模式要求**:
{plan_mode_instruction}

**撰写要求**:
1. 内容必须严格基于提供的证据，严禁编造。
2. 每个章节只写对应章节内容，不要重复章节标题。
3. 引用证据时，可在句尾标注 [evidence_1]、[evidence_2]。
4. 如果证据不足以覆盖某些点，请如实说明。

请只输出 JSON，格式如下：
{{
  "sections": [
    {{
      "section_id": "sec_1",
      "title": "章节标题",
      "content": "Markdown 正文"
    }}
  ]
}}
"""

# 3. 简短汇总 Prompt，用于默认 Agent 快速响应模式
CONCISE_REPORT_PROMPT = """你是一个文旅问答助手。请基于多智能体已经收集到的证据，直接回答用户问题。

**用户查询**:
{query}

**可用证据**:
{evidence_context}

**输出模式要求**:
{plan_mode_instruction}

**要求**:
1. 严格依据证据回答，不要编造。
2. 用简洁的 Markdown 输出，优先覆盖用户明确要求的信息点。
3. 涉及交通、票价、时间、优惠、注意事项时保留关键数字和限制条件。
4. 如果证据不足，请明确说明缺口。

请直接输出最终回答，不要输出大纲或过程说明。
"""

# 4. 结构化行程输出 Prompt
ITINERARY_JSON_PROMPT = """你是一个严谨的文旅行程结构化助手。请基于用户问题、自然语言答案和多智能体证据，生成可被前端或地图工具消费的结构化行程 JSON。

**用户查询**:
{query}

**自然语言答案**:
{answer}

**可用证据**:
{evidence_context}

**要求**:
1. 只输出 JSON，不要输出 Markdown、解释或代码块。
2. 严格依据证据和自然语言答案，不要编造不存在的票价、开放时间或交通耗时。
3. 如果证据不足，请在 warnings 或 assumptions 中说明。
4. days 至少包含 1 天；每个 day 的 slots 尽量按时间顺序排列。
5. source_refs 使用 evidence_1、evidence_2 这类证据编号。

JSON 格式：
{{
  "days": [
    {{
      "date_label": "第 1 天",
      "slots": [
        {{
          "start_time": "09:00",
          "end_time": "11:00",
          "title": "景点或活动名称",
          "location": "地点",
          "activity": "活动说明",
          "transport_to_next": "前往下一站的交通说明",
          "estimated_cost": "费用估计或证据不足说明",
          "ticket_info": "票务信息或证据不足说明",
          "source_refs": ["evidence_1"],
          "notes": "注意事项"
        }}
      ]
    }}
  ],
  "total_budget": "总预算估计或证据不足说明",
  "assumptions": ["必要假设"],
  "warnings": ["信息缺口或风险提示"]
}}
"""
