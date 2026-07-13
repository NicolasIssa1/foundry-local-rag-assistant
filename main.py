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


def cmd_index(args: argparse.Namespace) -> None:
    from src.embeddings.foundry_embedder import FoundryEmbedder
    from src.llm.client import FoundryRuntime
    from src.pipeline.index import index_documents

    print("Foundry Local RAG Assistant — Indexing")
    print("=" * 50)

    with FoundryRuntime() as runtime:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="foundry-rag",
        description="Foundry Local RAG Assistant — ask questions over local documents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index sub-command
    p_index = sub.add_parser("index", help="Index documents from data-dir.")
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
    p_query = sub.add_parser("query", help="Ask a question against the index.")
    p_query.add_argument("question", help="The question to answer.")
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

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
