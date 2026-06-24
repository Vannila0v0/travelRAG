# Travel Plan Clarifier Skill

```json skill-runtime
{
  "name": "travel_plan_clarifier",
  "route_scope": ["auto", "global", "agent"],
  "trigger_terms": [
    "旅游",
    "旅行",
    "游玩",
    "行程",
    "路线",
    "攻略",
    "怎么玩",
    "怎么安排",
    "一日游",
    "两日游",
    "三日游",
    "多日游",
    "一天",
    "两天",
    "三天",
    "几天"
  ],
  "planning_verbs": ["规划", "安排", "计划", "设计", "推荐", "做一个", "帮我"],
  "broad_destinations": ["桂林", "阳朔", "龙胜", "市区"],
  "detail_mode_terms": [
    "详细路线",
    "精确到小时",
    "按小时",
    "时间安排",
    "上午",
    "下午",
    "晚上",
    "交通",
    "票价",
    "门票",
    "预算",
    "从",
    "出发",
    "路线安排"
  ],
  "recommendation_mode_terms": [
    "推荐一些",
    "推荐几个",
    "好玩的地方",
    "有哪些好玩",
    "景点推荐",
    "只推荐",
    "不用路线",
    "不用安排",
    "值得去"
  ],
  "clarification_type": "travel_plan_mode",
  "clarification_answer_lines": [
    "你是想要哪一种结果？",
    "",
    "1. 详细路线安排：按天或按上午/下午/晚上安排，包含景点顺序、交通、票价、预算和注意事项。",
    "2. 景点/项目推荐：先给你推荐值得玩的地方和适合人群，不强行排成完整路线。",
    "",
    "你可以直接回复“详细路线安排”或“景点推荐”。"
  ],
  "options": [
    {
      "id": "detailed_itinerary",
      "label": "详细路线安排",
      "description": "按时间顺序组织路线，补充交通、票价、预算和注意事项。",
      "next_request": {
        "plan_mode": "detailed_itinerary",
        "route": "agent"
      }
    },
    {
      "id": "place_recommendations",
      "label": "景点/项目推荐",
      "description": "先推荐值得玩的景点或项目，适合还没确定行程细节的用户。",
      "next_request": {
        "plan_mode": "place_recommendations",
        "route": "agent"
      }
    }
  ]
}
```

## 目标

当用户提出较宽泛的整段旅游规划需求时，先澄清用户想要的输出类型，避免系统直接进入复杂 Agent 流程后生成不符合预期的长报告。

## 触发场景

用户表达了“去某地玩一段时间”或“帮我安排旅行”的宽泛需求，但没有明确说明想要：

- 详细路线安排
- 景点或项目推荐

典型输入：

```text
我想去桂林玩三天，帮我安排一下。
帮我做一个阳朔旅游计划。
龙胜怎么玩比较好？
```

## 不触发场景

如果用户已经明确选择了输出模式，则不再追问，直接进入原有查询或 Agent 流程。

详细路线类输入：

```text
帮我规划一天桂林市区详细路线，包含交通和票价。
从龙胜县城汽车总站出发，帮我安排龙胜温泉一日游。
给我一个精确到上午、下午、晚上的路线安排。
```

景点推荐类输入：

```text
推荐几个桂林好玩的地方，不用路线。
阳朔有哪些值得去的景点？
先给我推荐一些适合亲子游的项目。
```

## 澄清问题

命中后返回：

```text
你是想要哪一种结果？

1. 详细路线安排：按天或按上午/下午/晚上安排，包含景点顺序、交通、票价、预算和注意事项。
2. 景点/项目推荐：先给你推荐值得玩的地方和适合人群，不强行排成完整路线。

你可以直接回复“详细路线安排”或“景点推荐”。
```

## 输出 metadata

```json
{
  "skill": "travel_plan_clarifier",
  "skill_spec": "agent_system/skills/travel_plan_clarifier.md",
  "clarification_required": true,
  "clarification_type": "travel_plan_mode",
  "options": [
    {
      "id": "detailed_itinerary",
      "label": "详细路线安排",
      "description": "按时间顺序组织路线，补充交通、票价、预算和注意事项。",
      "next_request": {
        "plan_mode": "detailed_itinerary",
        "route": "agent"
      }
    },
    {
      "id": "place_recommendations",
      "label": "景点/项目推荐",
      "description": "先推荐值得玩的景点或项目，适合还没确定行程细节的用户。",
      "next_request": {
        "plan_mode": "place_recommendations",
        "route": "agent"
      }
    }
  ]
}
```

## 工程实现

运行时代码在：

```text
agent_system/skills/travel_plan_clarifier.py
```

服务入口在：

```text
server/routers/query.py
```

该 skill 在 `/query` 和 `/agent/query` 的前置阶段执行。命中澄清时不会调用依赖检查、QueryEngine、Neo4j、FAISS 或 LLM，只返回澄清问题并写入 trace。
