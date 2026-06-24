import argparse
import json
import os
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / ".cache" / "faiss_index"
DEFAULT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".txt", ".md"}


def parse_extension_list(value: str) -> set[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    return extensions


def collect_source_files(source: Path, extensions: set[str], limit: int | None = None) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in extensions else []

    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    files = [
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    files.sort(key=lambda item: str(item).lower())
    return files[:limit] if limit else files


def clear_graph() -> None:
    from core.config import neo4j_password, neo4j_url, neo4j_user
    from core.neo4j_handler import Neo4jHandler

    handler = Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)
    try:
        with handler.driver.session() as session:
            session.run("MATCH (c:Community) DETACH DELETE c")
            session.run("MATCH (e:Entity) DETACH DELETE e")
            session.run("MATCH (c:Chunk) DETACH DELETE c")
            session.run("MATCH (d:Document) DETACH DELETE d")
    finally:
        handler.close()


def ensure_graph_schema() -> None:
    from core.config import neo4j_password, neo4j_url, neo4j_user
    from core.neo4j_handler import Neo4jHandler

    statements = [
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
        "CREATE INDEX community_level_index IF NOT EXISTS FOR (c:Community) ON (c.level)",
        "CREATE INDEX community_title_index IF NOT EXISTS FOR (c:Community) ON (c.title)",
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE INDEX chunk_doc_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
    ]

    handler = Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)
    try:
        with handler.driver.session() as session:
            for statement in statements:
                session.run(statement)
    finally:
        handler.close()


def run_dqa() -> None:
    from core.config import neo4j_password, neo4j_url, neo4j_user
    from core.neo4j_handler import Neo4jHandler

    handler = Neo4jHandler(neo4j_url, neo4j_user, neo4j_password)
    try:
        handler.perform_dqa()
    finally:
        handler.close()


def build_graph_command(args: argparse.Namespace) -> int:
    extensions = set(DEFAULT_EXTENSIONS)
    if args.extensions:
        extensions = parse_extension_list(args.extensions)
    if args.include_xlsx:
        extensions.add(".xlsx")

    source = Path(args.source).resolve()
    files = collect_source_files(source, extensions, args.limit)

    print(f"Source: {source}")
    print(f"Extensions: {', '.join(sorted(extensions))}")
    print(f"Matched files: {len(files)}")

    if args.dry_run:
        for file_path in files:
            print(f" - {file_path}")
        return 0

    if not files:
        print("No supported files found.")
        return 1

    from graph_engine.builder import build_knowledge_graph

    ensure_graph_schema()

    if args.clear:
        print("Clearing existing Entity, Community, Document, and Chunk data...")
        clear_graph()

    failures: list[tuple[Path, str]] = []
    total_chunks = 0
    total_processed = 0
    total_entities = 0
    total_relationships = 0

    for index, file_path in enumerate(files, start=1):
        print(f"\n=== [{index}/{len(files)}] Build graph from {file_path} ===")
        try:
            stats = build_knowledge_graph(str(file_path), min_chunk_chars=args.min_chunk_chars)
        except Exception as exc:
            failures.append((file_path, str(exc)))
            print(f"[FAILED] {file_path}: {exc}")
            if args.fail_fast:
                break
            continue

        total_chunks += stats.chunks
        total_processed += stats.processed_chunks
        total_entities += stats.entities
        total_relationships += stats.relationships

    if not args.skip_dqa:
        print("\n=== Run DQA ===")
        run_dqa()

    if args.community or args.summary:
        print("\n=== Run community detection ===")
        from graph_engine.run_community import main as run_community

        run_community()

    if args.summary:
        print("\n=== Run community summarization ===")
        from graph_engine.summarize import generate_community_summaries

        generate_community_summaries()

    print("\n=== Build graph summary ===")
    print(f"Files attempted: {len(files)}")
    print(f"Files failed: {len(failures)}")
    print(f"Chunks: {total_chunks}")
    print(f"Processed chunks: {total_processed}")
    print(f"Extracted entities: {total_entities}")
    print(f"Extracted relationships: {total_relationships}")

    if failures:
        print("\nFailures:")
        for file_path, message in failures:
            print(f" - {file_path}: {message}")
        return 1

    return 0


def build_index_command(args: argparse.Namespace) -> int:
    extensions = set(DEFAULT_EXTENSIONS)
    if args.extensions:
        extensions = parse_extension_list(args.extensions)
    if args.include_xlsx:
        extensions.add(".xlsx")

    source = Path(args.source).resolve()
    files = collect_source_files(source, extensions, args.limit)

    print(f"Source: {source}")
    print(f"Extensions: {', '.join(sorted(extensions))}")
    print(f"Matched files: {len(files)}")
    print(f"Index dir: {Path(args.index_dir).resolve()}")

    if args.dry_run:
        for file_path in files:
            print(f" - {file_path}")
        return 0

    if not files:
        print("No supported files found.")
        return 1

    from indexing.build_index import build_faiss_index

    stats, failures = build_faiss_index(
        files=files,
        index_dir=Path(args.index_dir),
        min_chunk_chars=args.min_chunk_chars,
    )

    print("\n=== Build index summary ===")
    print(f"Files attempted: {stats.files_attempted}")
    print(f"Files failed: {stats.files_failed}")
    print(f"Chunks indexed: {stats.chunks}")
    print(f"Index dir: {stats.index_dir}")

    if failures:
        print("\nFailures:")
        for file_path, message in failures:
            print(f" - {file_path}: {message}")
        return 1

    return 0


def check_command(args: argparse.Namespace) -> int:
    from server.deps import (
        check_embedding_config,
        check_faiss_index,
        check_llm_config,
        check_neo4j,
    )

    checks: dict[str, dict[str, str | None]] = {}

    try:
        check_neo4j()
        checks["neo4j"] = {"status": "ok", "detail": None}
    except Exception as exc:
        checks["neo4j"] = {"status": "failed", "detail": str(exc)}

    checks["faiss_index"] = (
        {"status": "ok", "detail": None}
        if check_faiss_index(Path(args.index_dir))
        else {"status": "failed", "detail": "FAISS index files not found"}
    )

    llm_ok, llm_detail = check_llm_config()
    checks["llm_config"] = {
        "status": "ok" if llm_ok else "failed",
        "detail": llm_detail,
    }

    embedding_ok, embedding_detail = check_embedding_config()
    checks["embedding_config"] = {
        "status": "ok" if embedding_ok else "failed",
        "detail": embedding_detail,
    }

    status_value = "ok" if all(item["status"] == "ok" for item in checks.values()) else "degraded"
    payload = {"status": status_value, "checks": checks}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Status: {status_value}")
        for name, item in checks.items():
            line = f"- {name}: {item['status']}"
            if item["detail"]:
                line += f" ({item['detail']})"
            print(line)

    return 0 if status_value == "ok" else 1


def query_command(args: argparse.Namespace) -> int:
    from query_engine import QueryEngine

    engine = QueryEngine(index_dir=args.index_dir)
    try:
        result = engine.ask(
            args.question,
            route=args.route,
            plan_mode=args.plan_mode,
            response_format=args.response_format,
        )
    finally:
        engine.close()

    if args.json:
        payload = {
            "route": result.route,
            "answer": result.answer,
            "sources": [
                {
                    "doc_id": source.doc_id,
                    "chunk_id": source.chunk_id,
                    "source_path": source.source_path,
                    "file_name": source.file_name,
                    "chunk_index": source.chunk_index,
                    "page": source.page,
                    "section": source.section,
                    "score": source.score,
                    "text": source.text,
                }
                for source in result.sources
            ],
            "metadata": result.metadata,
            "structured_output": result.structured_output,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Route: {result.route}\n")
    print(result.answer)
    if result.structured_output:
        print("\nStructured output:")
        print(json.dumps(result.structured_output, ensure_ascii=False, indent=2))
    if result.sources:
        print("\nSources:")
        for index, source in enumerate(result.sources[: args.show_sources], start=1):
            label = source.file_name or source.source_path or source.doc_id or "unknown"
            chunk = f" [{source.chunk_id}]" if source.chunk_id else ""
            print(f"{index}. {label}{chunk}")
            if args.show_source_text and source.text:
                preview = source.text.replace("\n", " ").strip()
                print(f"   {preview[:240]}")
    return 0


def _mask_secret(value: str | None) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _dump_model(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _web_tool_source_summary(source: dict) -> dict:
    return {
        "title": source.get("title") or source.get("file_name"),
        "url": source.get("url") or source.get("source_path"),
        "source_type": source.get("source_type"),
        "score": source.get("score"),
        "text_preview": str(source.get("text") or "").replace("\n", " ")[:240],
    }


def _smoke_record_summary(record) -> dict:
    payload = _dump_model(record)
    metadata = payload.get("metadata") or {}
    return {
        "task_id": payload.get("task_id"),
        "route": payload.get("route"),
        "task_type": (payload.get("inputs") or {}).get("task_type"),
        "description": (payload.get("inputs") or {}).get("description"),
        "error": metadata.get("error"),
        "latency_seconds": metadata.get("latency_seconds"),
        "tool_metadata": payload.get("tool_metadata") or {},
        "source_count": len(payload.get("sources") or []),
        "sources": [
            _web_tool_source_summary(source)
            for source in (payload.get("sources") or [])
            if isinstance(source, dict)
        ],
        "output_preview": str(payload.get("output") or "").replace("\n", " ")[:600],
    }


def smoke_web_tools_command(args: argparse.Namespace) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass

    provider = os.getenv("WEB_SEARCH_PROVIDER", "").strip()
    api_key = os.getenv("WEB_SEARCH_API_KEY", "").strip() or os.getenv("EXA_API_KEY", "").strip()
    config_summary = {
        "WEB_SEARCH_PROVIDER": provider or "<missing>",
        "WEB_SEARCH_API_KEY": _mask_secret(api_key),
        "WEB_SEARCH_EXA_MCP_ENDPOINT": os.getenv("WEB_SEARCH_EXA_MCP_ENDPOINT", ""),
        "WEB_SEARCH_EXA_TOOL_NAME": os.getenv("WEB_SEARCH_EXA_TOOL_NAME", "web_search_exa"),
        "WEB_SEARCH_MAX_RESULTS": os.getenv("WEB_SEARCH_MAX_RESULTS", "5"),
        "WEB_FETCH_MAX_BYTES": os.getenv("WEB_FETCH_MAX_BYTES", "1048576"),
        "WEB_FETCH_MAX_CHARS": os.getenv("WEB_FETCH_MAX_CHARS", "12000"),
    }

    if args.require_exa and provider.lower() not in {"exa", "exa_mcp"}:
        print("WEB_SEARCH_PROVIDER is not exa_mcp/exa. Use WEB_SEARCH_PROVIDER=exa_mcp or remove --require-exa.")
        print(json.dumps(config_summary, ensure_ascii=False, indent=2))
        return 2
    os.environ.setdefault("AGENT_MAX_WORKERS", "1")

    from agent_system.orchestrator import MultiAgentOrchestrator

    state = MultiAgentOrchestrator().run(
        args.query,
        report_mode=args.report_mode,
        plan_mode=args.plan_mode,
    )

    tasks = []
    if state.plan and state.plan.task_graph:
        tasks = [_dump_model(node) for node in state.plan.task_graph.nodes]
    records = [_smoke_record_summary(record) for record in state.execution_records]
    payload = {
        "query": args.query,
        "config": config_summary,
        "task_graph": {
            "execution_mode": state.plan.task_graph.execution_mode if state.plan else None,
            "nodes": tasks,
        },
        "execution_records": records,
        "agent_trace": state.agent_trace,
        "sources": [
            _web_tool_source_summary(source)
            for source in state.sources
            if isinstance(source, dict)
        ],
        "final_answer": state.final_report,
    }

    has_error = any(record.get("error") for record in records)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=== Web Tools Smoke ===")
        print(f"Query: {args.query}")
        print("\nConfig:")
        for name, value in config_summary.items():
            print(f"- {name}: {value}")

        print("\nTask Graph:")
        for index, task in enumerate(tasks, start=1):
            params = task.get("parameters") or {}
            suffix = f" params={json.dumps(params, ensure_ascii=False)}" if params else ""
            print(f"{index}. {task.get('task_id')} [{task.get('task_type')}] {task.get('description')}{suffix}")

        print("\nExecution Records:")
        for index, record in enumerate(records, start=1):
            status = "ERROR" if record.get("error") else "OK"
            latency = record.get("latency_seconds")
            latency_text = f"{latency:.3f}s" if isinstance(latency, (int, float)) else "n/a"
            print(f"{index}. [{status}] {record.get('task_id')} route={record.get('route')} latency={latency_text}")
            if record.get("error"):
                print(f"   error: {record.get('error')}")
            if record.get("tool_metadata"):
                print(f"   metadata: {json.dumps(record['tool_metadata'], ensure_ascii=False)}")
            for source_index, source in enumerate(record.get("sources") or [], start=1):
                print(f"   source {source_index}: {source.get('title')} {source.get('url')}")
                if source.get("text_preview"):
                    print(f"      {source.get('text_preview')}")
            if args.show_outputs and record.get("output_preview"):
                print(f"   output: {record.get('output_preview')}")

        print("\nFinal Answer:")
        print(state.final_report or "<empty>")

    return 1 if has_error else 0


def eval_command(args: argparse.Namespace) -> int:
    from evaluation.run_eval import evaluate

    payload = evaluate(args)
    summary = payload["summary"]
    print("\n=== Evaluation Summary ===")
    print(f"evaluated_samples: {summary['evaluated_samples']}")
    print(f"route_accuracy: {summary['route_accuracy']:.2%}")
    print(f"avg_answer_keyword_recall: {summary['avg_answer_keyword_recall']:.2%}")
    print(f"avg_source_recall: {summary['avg_source_recall']:.2%}")
    print(f"has_source_rate: {summary['has_source_rate']:.2%}")
    print(f"avg_latency_seconds: {summary['avg_latency_seconds']:.2f}")
    print(f"reports: {Path(args.report_dir).resolve()}")
    return 0


def serve_command(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG project management commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_graph = subparsers.add_parser(
        "build-graph",
        help="Parse documents, extract GraphRAG triples, and write them to Neo4j",
    )
    build_graph.add_argument("--source", default=str(DEFAULT_SOURCE), help="Document file or directory")
    build_graph.add_argument("--extensions", help="Comma-separated extensions, e.g. .md,.docx,.pdf")
    build_graph.add_argument("--include-xlsx", action="store_true", help="Include .xlsx files in default scan")
    build_graph.add_argument("--limit", type=int, help="Process at most N files")
    build_graph.add_argument("--min-chunk-chars", type=int, default=50, help="Skip chunks shorter than this")
    build_graph.add_argument("--skip-dqa", action="store_true", help="Skip post-build graph cleanup")
    build_graph.add_argument("--community", action="store_true", help="Run community detection after build")
    build_graph.add_argument("--summary", action="store_true", help="Generate community summaries after detection")
    build_graph.add_argument("--clear", action="store_true", help="Delete existing Entity and Community data first")
    build_graph.add_argument("--dry-run", action="store_true", help="Only list matched files")
    build_graph.add_argument("--fail-fast", action="store_true", help="Stop on first failed file")
    build_graph.set_defaults(func=build_graph_command)

    build_index = subparsers.add_parser(
        "build-index",
        help="Parse documents, build embeddings, and persist a FAISS vector index",
    )
    build_index.add_argument("--source", default=str(DEFAULT_SOURCE), help="Document file or directory")
    build_index.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="FAISS index output directory")
    build_index.add_argument("--extensions", help="Comma-separated extensions, e.g. .md,.docx,.pdf")
    build_index.add_argument("--include-xlsx", action="store_true", help="Include .xlsx files in default scan")
    build_index.add_argument("--limit", type=int, help="Process at most N files")
    build_index.add_argument("--min-chunk-chars", type=int, default=20, help="Skip chunks shorter than this")
    build_index.add_argument("--dry-run", action="store_true", help="Only list matched files")
    build_index.set_defaults(func=build_index_command)

    check = subparsers.add_parser(
        "check",
        help="Check local runtime dependencies without loading heavy models",
    )
    check.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="FAISS index directory")
    check.add_argument("--json", action="store_true", help="Print structured JSON status")
    check.set_defaults(func=check_command)

    query = subparsers.add_parser(
        "query",
        help="Ask a question through the unified Query Engine",
    )
    query.add_argument("question", help="User question")
    query.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="FAISS index directory")
    query.add_argument(
        "--route",
        choices=["auto", "vector", "local", "global", "hybrid", "agent"],
        default="auto",
        help="Force a retrieval route, or use auto routing",
    )
    query.add_argument("--json", action="store_true", help="Print structured JSON result")
    query.add_argument(
        "--plan-mode",
        choices=["auto", "detailed_itinerary", "place_recommendations"],
        default="auto",
        help="Planning mode for agent travel planning queries",
    )
    query.add_argument(
        "--response-format",
        choices=["text", "itinerary"],
        default="text",
        help="Return text only or include structured itinerary output for detailed agent planning",
    )
    query.add_argument("--show-sources", type=int, default=5, help="Number of sources to print")
    query.add_argument("--show-source-text", action="store_true", help="Print source text previews")
    query.set_defaults(func=query_command)

    smoke_web_tools = subparsers.add_parser(
        "smoke-web-tools",
        help="Run a manual end-to-end smoke check for web_search/web_fetch agent tools",
    )
    smoke_web_tools.add_argument(
        "--query",
        default="桂林两江四湖今天是否开放？请查最新信息",
        help="Question to run through the agent web tools chain",
    )
    smoke_web_tools.add_argument(
        "--report-mode",
        choices=["concise", "full"],
        default="concise",
        help="Reporter mode for the smoke run",
    )
    smoke_web_tools.add_argument(
        "--plan-mode",
        choices=["auto", "detailed_itinerary", "place_recommendations"],
        default="auto",
        help="Planning mode for the smoke run",
    )
    smoke_web_tools.add_argument(
        "--require-exa",
        action="store_true",
        help="Fail before running unless WEB_SEARCH_PROVIDER is exa_mcp/exa",
    )
    smoke_web_tools.add_argument("--json", action="store_true", help="Print structured JSON output")
    smoke_web_tools.add_argument("--show-outputs", action="store_true", help="Print execution output previews")
    smoke_web_tools.set_defaults(func=smoke_web_tools_command)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run the tourism QA evaluation benchmark",
    )
    eval_parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "evaluation" / "datasets" / "tourism_qa.jsonl"),
        help="JSONL evaluation dataset",
    )
    eval_parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "evaluation" / "reports"),
        help="Report output directory",
    )
    eval_parser.add_argument("--report-name", help="Optional named report file, without extension")
    eval_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="FAISS index directory")
    eval_parser.add_argument("--limit", type=int, help="Evaluate at most N samples")
    eval_parser.add_argument("--ids", help="Comma-separated sample IDs to evaluate")
    eval_parser.add_argument("--skip-agent", action="store_true", help="Skip samples whose expected route is agent")
    eval_parser.add_argument(
        "--route",
        choices=["auto", "dataset", "vector", "local", "global", "hybrid", "agent"],
        default="auto",
        help="Route mode. auto uses router; dataset forces each sample's expected_route.",
    )
    eval_parser.add_argument("--include-source-text", action="store_true", help="Store source text in JSON report")
    eval_parser.add_argument(
        "--report-mode",
        choices=["concise", "full"],
        default="concise",
        help="Reporter mode for agent samples.",
    )
    eval_parser.add_argument(
        "--plan-mode",
        choices=["auto", "detailed_itinerary", "place_recommendations"],
        default="auto",
        help="Planning mode for agent travel planning samples.",
    )
    eval_parser.add_argument(
        "--response-format",
        choices=["text", "itinerary"],
        default="text",
        help="Response format for agent planning samples.",
    )
    eval_parser.add_argument(
        "--allow-hf-network",
        action="store_true",
        help="Allow HuggingFace network checks/downloads during evaluation.",
    )
    eval_parser.set_defaults(func=eval_command)

    serve = subparsers.add_parser(
        "serve",
        help="Start the FastAPI service",
    )
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", type=int, default=8000, help="Bind port")
    serve.add_argument("--reload", action="store_true", help="Enable uvicorn reload")
    serve.add_argument("--log-level", default="info", help="Uvicorn log level")
    serve.set_defaults(func=serve_command)

    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
