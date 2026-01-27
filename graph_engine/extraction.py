import json
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from .schema import GraphResult, Entity, Relationship

# 配置 LLM
llm = OllamaLLM(model="qwen2.5:14b", base_url="http://172.28.16.1:11434")

# 修复要点：
# 1. JSON 示例中的所有 { 和 } 都变成了 {{ 和 }}
# 2. 只有底部的 {text} 保持单层大括号，作为变量输入
GRAPH_EXTRACTION_PROMPT = """
你是一个旅游领域的知识图谱构建专家。你的任务是从给定的【旅游攻略/景区问答】文本中提取结构化知识。

### 核心目标：
请重点提取以下几类关键信息，并将其转化为实体和关系：

1. **票价与政策**（重点）：
   - 提取具体的票种（如“夜游两江四湖成人票”）。
   - 提取价格关系。例如：(夜游两江四湖成人票)-[COSTS]->(210元)。
   - 提取适用条件。例如：(儿童票)-[APPLIES_TO]->(身高1.2米-1.5米儿童)。

2. **交通与路线**：
   - 提取公交、码头、站点。
   - 提取到达路径。例如：(桂林站)-[ROUTE_TO {{ "description": "乘坐100路" }}]->(阳桥站)。
   - 提取步行信息。例如：(阳桥站)-[ROUTE_TO {{ "description": "步行400米" }}]->(日月湾码头)。

3. **景点与设施**：
   - 提取景点及其包含的子景点（如“两江四湖”包含“木龙湖”）。
   - 提取设施位置（如“渔人码头”拥有“母婴室”）。

### JSON 输出示例（注意格式）：
{{
  "entities": [
    {{
      "name": "夜游两江四湖成人票",
      "type": "票务/价格",
      "description": "适用于成人，包含夜游两江四湖行程"
    }},
    {{
      "name": "210元",
      "type": "票务/价格",
      "description": "价格金额"
    }}
  ],
  "relationships": [
    {{
      "source": "夜游两江四湖成人票",
      "target": "210元",
      "relation_type": "COSTS",
      "description": "门市价"
    }}
  ]
}}

### 文本内容：
{text}

### 输出要求：
1. 必须返回严格的 JSON 格式，包含 "entities" 和 "relationships"。
2. Entity.type 必须是以下之一：[景点, 地点, 交通工具, 票务/价格, 规则/政策, 活动/演艺, 设施, 组织/公司]。
3. Relationship.relation_type 尽量使用：[LOCATED_IN, NEAR, COSTS, CONTAINS, ROUTE_TO, REQUIRES, APPLIES_TO, HAS_FACILITY]。
4. 对于复杂的交通指引，请将动作（如“乘坐100路”）放入关系的 description 中。
5. 对于票价规则，请将具体金额或条件放入实体的 description 或关系的 description 中。

请直接输出 JSON，不要包含 Markdown 标记。
"""


def clean_json_text(text: str) -> str:
    """清洗 LLM 可能输出的 markdown 符号"""
    text = text.strip()
    # 移除 ```json 和 ``` 包裹
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_graph_from_text(text_chunk: str) -> GraphResult:
    """
    调用 LLM 提取实体和关系，返回结构化对象
    """
    try:
        # 使用 from_template 解析 Prompt，此时 {{}} 会被转义为 {}，{text} 会被识别为变量
        prompt = PromptTemplate.from_template(GRAPH_EXTRACTION_PROMPT)
        chain = prompt | llm

        response = chain.invoke({"text": text_chunk})
        clean_res = clean_json_text(response)

        # 解析 JSON
        data = json.loads(clean_res)

        # 转换为 Pydantic 对象
        entities = []
        relationships = []

        for e in data.get("entities", []):
            # 容错处理：确保所有字段都是字符串
            e["description"] = str(e.get("description", ""))
            entities.append(Entity(**e))

        for r in data.get("relationships", []):
            # 容错处理
            r["description"] = str(r.get("description", ""))
            relationships.append(Relationship(**r))

        return GraphResult(entities=entities, relationships=relationships)

    except json.JSONDecodeError:
        print("[Extraction Error] JSON 解析失败，跳过该块。")
        return GraphResult(entities=[], relationships=[])
    except Exception as e:
        print(f"[Extraction Error] 提取过程出错: {e}")
        return GraphResult(entities=[], relationships=[])