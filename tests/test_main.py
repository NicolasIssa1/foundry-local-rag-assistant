"""
Tests for the CLI entry point — main.py.

Covers --help output, error handling (missing data dir, missing index,
empty question), and graceful Ctrl-C handling. FoundryRuntime is mocked
wherever a code path might otherwise try to construct it, so these tests
never touch the real Foundry Local SDK.
"""
from unittest.mock import MagicMock, patch

import pytest

import main


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["main.py"] + argv)
    with pytest.raises(SystemExit) as exc_info:
        main.main()
    return exc_info.value.code


# ── --help output ──────────────────────────────────────────────────────────────

def test_top_level_help_exits_zero_and_lists_subcommands(monkeypatch, capsys):
    code = _run(monkeypatch, ["--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "index" in out
    assert "query" in out
    assert "example" in out.lower()


def test_index_help_exits_zero_and_describes_data_dir(monkeypatch, capsys):
    code = _run(monkeypatch, ["index", "--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "--data-dir" in out
    assert "--index-dir" in out


def test_query_help_exits_zero_and_describes_question(monkeypatch, capsys):
    code = _run(monkeypatch, ["query", "--help"])
    out = capsys.readouterr().out
    assert code == 0
    assert "question" in out
    assert "--k" in out


# ── Empty question handling ──────────────────────────────────────────────────────

def test_empty_question_is_rejected_by_argparse(monkeypatch, capsys):
    code = _run(monkeypatch, ["query", "   "])
    err = capsys.readouterr().err
    assert code == 2
    assert "must not be empty" in err


def test_missing_question_argument_is_rejected(monkeypatch, capsys):
    code = _run(monkeypatch, ["query"])
    assert code == 2


# ── Missing data directory (index) — fails fast, no model load ──────────────────

def test_index_missing_data_dir_fails_fast_without_loading_runtime(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "index", "--data-dir", str(tmp_path / "does_not_exist")],
    )
    with patch("src.llm.client.FoundryRuntime") as mock_runtime_cls:
        with pytest.raises(SystemExit) as exc_info:
            main.main()
        mock_runtime_cls.assert_not_called()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "not found" in err.lower()


# ── Missing index (query) — fails fast, no model load ────────────────────────────

def test_query_missing_index_fails_fast_without_loading_runtime(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "query", "hello", "--index-dir", str(tmp_path / "no_index_here")],
    )
    with patch("src.llm.client.FoundryRuntime") as mock_runtime_cls:
        with pytest.raises(SystemExit) as exc_info:
            main.main()
        mock_runtime_cls.assert_not_called()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "no index found" in err.lower()
    assert "python main.py index" in err


# ── Graceful Ctrl-C handling ───────────────────────────────────────────────────────

def test_keyboard_interrupt_during_query_exits_cleanly(monkeypatch, capsys, tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv", ["main.py", "query", "hello", "--index-dir", str(index_dir)]
    )
    monkeypatch.setattr(main, "cmd_query", MagicMock(side_effect=KeyboardInterrupt))

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 130
    assert "Interrupted" in capsys.readouterr().err


def test_keyboard_interrupt_during_index_exits_cleanly(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", ["main.py", "index", "--data-dir", str(tmp_path)])
    monkeypatch.setattr(main, "cmd_index", MagicMock(side_effect=KeyboardInterrupt))

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 130
    assert "Interrupted" in capsys.readouterr().err
