"""
Foundry Local RAG Assistant — entry point.

This file is intentionally thin. It parses the command-line arguments and
delegates all work to src/. No business logic lives here.

Usage:
    python main.py index [--data-dir DATA_DIR] [--index-dir INDEX_DIR]
    python main.py query "<question>" [--index-dir INDEX_DIR] [--k K]
"""

import argparse
import sys
from pathlib import Path

DEFAULT_DATA_DIR = "data"
DEFAULT_INDEX_DIR = "data/index"
DEFAULT_K = 5


def _non_empty_question(value: str) -> str:
    """argparse type= validator: reject a blank/whitespace-only question."""
    if not value.strip():
        raise argparse.ArgumentTypeError("question must not be empty")
    return value


def cmd_index(args: argparse.Namespace) -> None:
    from src.embeddings.foundry_embedder import FoundryEmbedder
    from src.llm.client import FoundryRuntime
    from src.pipeline.index import index_documents

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Create it and add .txt/.md files, or pass --data-dir PATH."
        )

    print("Foundry Local RAG Assistant — Indexing")
    print("=" * 50)

    with FoundryRuntime(load_chat=False) as runtime:
        embedder = FoundryEmbedder(
            client=runtime.get_embedding_client(),
            model=runtime._embed_alias,
        )
        count = index_documents(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            embedder=embedder,
        )

    print(f"\nDone. {count} chunk(s) indexed and saved to {args.index_dir}")


def cmd_query(args: argparse.Namespace) -> None:
    from src.embeddings.foundry_embedder import FoundryEmbedder
    from src.llm.client import FoundryRuntime
    from src.pipeline.query import query

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(
            f"No index found in {index_dir}\n"
            f"Run 'python main.py index' first."
        )

    print("Foundry Local RAG Assistant — Query")
    print("=" * 50)
    print(f"Question: {args.question}\n")

    with FoundryRuntime() as runtime:
        embedder = FoundryEmbedder(
            client=runtime.get_embedding_client(),
            model=runtime._embed_alias,
        )
        print("Answer:")
        query(
            question=args.question,
            index_dir=args.index_dir,
            embedder=embedder,
            runtime=runtime,
            k=args.k,
            stream=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundry-rag",
        description="Foundry Local RAG Assistant — ask questions over local "
        ".txt/.md documents, fully offline, using Microsoft Foundry Local.",
        epilog=(
            "examples:\n"
            "  python main.py index\n"
            "  python main.py index --data-dir my_docs --index-dir my_index\n"
            "  python main.py query \"What is retrieval-augmented generation?\"\n"
            "  python main.py query \"Summarize the report\" --k 8\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index sub-command
    p_index = sub.add_parser(
        "index",
        help="Build/rebuild the vector index from local documents.",
        description="Load, chunk, embed, and index every .txt/.md file "
        "under --data-dir, writing a FAISS index and SQLite chunk store "
        "to --index-dir. Only the embedding model is loaded for this "
        "command.",
        epilog="example:\n  python main.py index --data-dir data --index-dir data/index\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_index.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        metavar="PATH",
        help=f"Directory containing .txt and .md files (default: {DEFAULT_DATA_DIR})",
    )
    p_index.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        metavar="PATH",
        help=f"Where to write the FAISS index and SQLite store (default: {DEFAULT_INDEX_DIR})",
    )

    # query sub-command
    p_query = sub.add_parser(
        "query",
        help="Ask a question against a previously built index.",
        description="Retrieve the most relevant chunks for QUESTION from "
        "--index-dir and generate a grounded answer with source citations. "
        "If nothing relevant is found, the chat model is not called and a "
        "deterministic refusal is printed instead.",
        epilog='example:\n  python main.py query "What is retrieval-augmented generation?"\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_query.add_argument(
        "question", type=_non_empty_question, help="The question to answer (must not be empty)."
    )
    p_query.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
        metavar="PATH",
        help=f"Directory containing the saved index (default: {DEFAULT_INDEX_DIR})",
    )
    p_query.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        metavar="N",
        help=f"Number of chunks to retrieve (default: {DEFAULT_K})",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "index":
            cmd_index(args)
        elif args.command == "query":
            cmd_query(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
