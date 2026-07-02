"""
Foundry Local RAG Assistant — entry point.

This file is intentionally thin. It parses the command and delegates
all work to the modules in src/. No business logic lives here.
"""

import sys


def main() -> None:
    print("Foundry Local RAG Assistant")
    print("Run with: python main.py index | python main.py query '<question>'")
    print()
    print("Implementation coming in M2–M5. Scaffold is ready.")


if __name__ == "__main__":
    main()
