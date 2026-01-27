import json
import logging
from typing import List, Dict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalSearch:
    def __init__(self, llm, neo4j_handler, token_limit: int = 4000):
        self.llm = llm
        self.neo4j = neo4j_handler
        self.token_limit = token_limit

    def _get_all_community_summaries(self) -> List[Dict]:
        """
        从 Neo4j 获取所有社区的 ID、标题和摘要
        """
        cypher = "MATCH (c:Community) RETURN c.id as id, c.title as title, c.summary as summary"
        summaries = []
        with self.neo4j.driver.session() as session:
            result = session.run(cypher)
            for record in result:
                summaries.append({
                    "id": record["id"],
                    "title": record["title"],
                    "summary": record["summary"]
                })
        return summaries

    def _map_communities(self, query: str, communities: List[Dict]) -> List[Dict]:
        """
        [Map 阶段] 让 LLM 给社区打分，筛选出相关的社区。
        为了节省 Token，这一步只把“标题”和“摘要”发给 LLM。
        """
        # 构建一个精简的列表供 LLM 评估
        community_text_list = []
        for c in communities:
            # 格式: [ID: 0] 标题: 票务体系 \n 摘要: ...
            text = f"[ID: {c['id']}] 标题: {c['title']}\n摘要: {c['summary'][:100]}..."  # 摘要只取前100字预览
            community_text_list.append(text)

        context_str = "\n\n".join(community_text_list)

        prompt = f"""
        你是一个智能筛选器。用户提出了一个问题，请评估以下【社区摘要列表】，找出哪些社区包含回答该问题所需的信息。

        用户问题: {query}

        社区列表:
        {context_str}

        要求:
        1. 请筛选出相关度高（评分 > 5）的社区。
        2. 返回一个 JSON 列表，包含社区 ID 和评分。

        输出示例:
        [
            {{"id": 0, "score": 10}},
            {{"id": 3, "score": 8}}
        ]

        请直接返回 JSON 列表，不要包含其他文字。
        """

        try:
            response = self.llm.invoke(prompt).strip()
            # 清洗
            response = response.replace("```json", "").replace("```", "").strip()
            if response.startswith("{") and "items" in response:  # 容错处理
                import json
                return json.loads(response)["items"]

            import json
            scores = json.loads(response)

            # 按分数降序排列
            scores.sort(key=lambda x: x.get("score", 0), reverse=True)
            return scores

        except Exception as e:
            logger.warning(f"Map 阶段筛选失败: {e}，将使用所有社区（降级策略）")
            # 如果 LLM 挂了，就返回所有 ID，每个 1 分
            return [{"id": c["id"], "score": 1} for c in communities]

    def search(self, query: str):
        logger.info(f"🔎 [Global Search] 开始处理: {query}")

        # 1. 获取所有社区摘要
        all_communities = self._get_all_community_summaries()

        if not all_communities:
            return "数据库中没有社区摘要，请先运行 Step 5 (summarize.py)。"

        logger.info(f"   -> 获取到 {len(all_communities)} 个社区摘要")

        # 2. Map: 筛选相关社区
        # 如果社区很少（比如少于 5 个），直接全用，不筛选了
        if len(all_communities) > 5:
            logger.info("   -> 正在执行 Map 筛选...")
            scored_communities = self._map_communities(query, all_communities)
            # 选出 Top K 或者 分数 > 5 的
            valid_ids = [item['id'] for item in scored_communities if item.get('score', 0) >= 5]

            # 至少保留 Top 3，防止过滤太狠
            if not valid_ids:
                valid_ids = [item['id'] for item in scored_communities[:3]]

            target_communities = [c for c in all_communities if c['id'] in valid_ids]
        else:
            target_communities = all_communities

        logger.info(f"   -> 筛选出 {len(target_communities)} 个高相关社区")

        # 3. Reduce: 构建最终 Prompt
        # 拼接完整的摘要内容
        context_parts = []
        for c in target_communities:
            part = f"### 社区: {c['title']}\n{c['summary']}"
            context_parts.append(part)

        final_context = "\n\n".join(context_parts)

        prompt = f"""
        你是一个资深的旅游分析师。请基于以下【社区报告】（Community Reports），对用户的问题进行全面、宏观的回答。

        这些报告是对原始数据的高层级总结，请综合不同社区的信息，生成一篇连贯的回答。
        如果不清楚，请回答“根据现有报告无法总结”。

        【社区报告集】:
        {final_context}

        【用户问题】:
        {query}

        请生成回答（支持 Markdown 格式）：
        """

        return self.llm.invoke(prompt)