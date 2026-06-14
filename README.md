# Local RAG / GraphRAG 文旅问答原型

这是一个面向文旅资料的 RAG / GraphRAG / Multi-Agent 原型项目。当前阶段已将生成模型入口从本地 Ollama 收敛为 DeepSeek OpenAI-compatible API，Embedding 与 Reranker 仍默认使用本地模型。

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，至少配置：

```text
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

如需使用本地图谱功能，还需要配置并启动 Neo4j：

```text
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

## 启动 Neo4j

项目提供 Docker Compose 配置，可以自动启动 Neo4j 并初始化基础约束和索引：

```powershell
docker compose up -d neo4j neo4j-init
```

启动后可访问：

- Neo4j Browser: http://localhost:7474
- Bolt: `bolt://localhost:7687`

默认账号密码来自 `.env` 或 Compose 默认值：

```text
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

如果应用也运行在 Docker Compose 网络内，应用侧的 `NEO4J_URI` 应配置为：

```text
NEO4J_URI=bolt://neo4j:7687
```

如果应用在宿主机运行，保持：

```text
NEO4J_URI=bolt://localhost:7687
```

## 当前 LLM 入口

- `agent_system/integration/llm_factory.py`
  - `get_llm_model()`：返回 DeepSeek `ChatOpenAI` 实例，适合 Agent、LangGraph、工具调用。
  - `get_stream_llm_model()`：返回流式 DeepSeek `ChatOpenAI` 实例。
  - `get_text_llm()`：兼容旧脚本，`invoke(prompt)` 直接返回字符串。
- `llm.py`
  - 保留旧接口 `get_llm()`，内部转到 `get_text_llm()`。

## 注意

当前仓库内旧 `venv` 指向不存在的 Conda 环境，不建议继续使用。请按上面的步骤重建 `.venv`。

DeepSeek API 为 OpenAI-compatible API，默认 base URL 为 `https://api.deepseek.com`。模型名通过 `DEEPSEEK_MODEL` 配置，后续可按官方文档更新。

## 标准构图命令

Neo4j 启动并配置好 `.env` 后，使用统一入口构建知识图谱：

```powershell
python .\manage.py build-graph --source .\data
```

常用参数：

```powershell
# 只查看会处理哪些文件，不调用 LLM，不写 Neo4j
python .\manage.py build-graph --source .\data --dry-run

# 先清空已有 Entity / Community，再重新构图
python .\manage.py build-graph --source .\data --clear

# 构图后执行 DQA、社区发现和社区摘要
python .\manage.py build-graph --source .\data --community --summary

# 默认不处理 xlsx；确实需要处理表格时显式开启
python .\manage.py build-graph --source .\data --include-xlsx

# 小批量验证
python .\manage.py build-graph --source .\data --limit 3
```

默认流程是：文档解析 -> chunk 切分 -> 实体/关系抽取 -> 写入 Neo4j -> DQA 清洗。加上 `--community` 后会执行社区发现；加上 `--summary` 后会生成社区摘要，作为后续 Global Search 的图谱索引基础。

## 标准向量索引命令

构建普通 RAG 使用的 FAISS 向量索引：

```powershell
python .\manage.py build-index --source .\data
```

常用参数：

```powershell
# 只查看会处理哪些文件
python .\manage.py build-index --source .\data --dry-run

# 小批量验证
python .\manage.py build-index --source .\data --limit 3

# 指定索引输出目录
python .\manage.py build-index --source .\data --index-dir .\.cache\faiss_index

# 默认不处理 xlsx；确实需要处理表格时显式开启
python .\manage.py build-index --source .\data --include-xlsx
```

索引中的每个 chunk 都会保留 `doc_id`、`chunk_id`、`source_path`、`chunk_index`、`page`、`section` 等 metadata，后续 Query Engine 可以用这些字段返回引用来源。

## 统一查询命令

图谱和向量索引构建完成后，可以通过统一 Query Engine 查询：

```powershell
python .\manage.py query "两江四湖成人票多少钱？"
```

默认 `--route auto` 会根据问题自动选择检索路径，也可以显式指定：

```powershell
# 普通向量 RAG
python .\manage.py query "龙胜温泉有什么特色？" --route vector

# Local GraphRAG：实体相关的票价、位置、交通、政策等问题
python .\manage.py query "两江四湖成人票多少钱？" --route local

# Global GraphRAG：宏观总结、路线规划、多景点主题问题
python .\manage.py query "帮我总结桂林市区一日游可以怎么玩" --route global

# Hybrid：向量检索 + 局部图谱检索
python .\manage.py query "两江四湖成人票多少钱？" --route hybrid

# 多智能体复杂任务入口
python .\manage.py query "帮我规划一天桂林市区游玩路线，包含交通和票价" --route agent
```

查看结构化结果和引用来源：

```powershell
python .\manage.py query "两江四湖成人票多少钱？" --json
python .\manage.py query "两江四湖成人票多少钱？" --show-sources 5 --show-source-text
```

## FastAPI 服务

启动在线查询服务：

```powershell
python .\manage.py serve --host 127.0.0.1 --port 8000
```

启动后可访问：

```text
GET  http://127.0.0.1:8000/health
GET  http://127.0.0.1:8000/graph/stats
POST http://127.0.0.1:8000/query
POST http://127.0.0.1:8000/agent/query
```

普通查询示例：

```powershell
curl -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"两江四湖成人票多少钱？\",\"route\":\"auto\",\"max_sources\":5}"
```

多智能体查询示例：

```powershell
curl -X POST http://127.0.0.1:8000/agent/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"帮我规划一天桂林市区游玩路线，包含交通和票价\"}"
```

当前服务层只负责在线查询和状态检查；`build-graph` / `build-index` 仍保留为 CLI 离线任务，避免长任务、并发写库和 token 成本失控。
