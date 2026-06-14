import argparse
import json
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


def query_command(args: argparse.Namespace) -> int:
    from query_engine import QueryEngine

    engine = QueryEngine(index_dir=args.index_dir)
    try:
        result = engine.ask(args.question, route=args.route)
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
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Route: {result.route}\n")
    print(result.answer)
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
    query.add_argument("--show-sources", type=int, default=5, help="Number of sources to print")
    query.add_argument("--show-source-text", action="store_true", help="Print source text previews")
    query.set_defaults(func=query_command)

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
