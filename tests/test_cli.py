"""Tests for nl2sh.cli."""

from unittest.mock import MagicMock, patch

import pytest

from nl2sh import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(argv: list[str]) -> int:
    """Run CLI with the given argv and return the exit code."""
    return cli.main(argv)


def _mock_converter(command: str = "ls -la") -> MagicMock:
    conv = MagicMock()
    conv.convert.return_value = command
    return conv


# ---------------------------------------------------------------------------
# Argument parsing / help
# ---------------------------------------------------------------------------

class TestArgParsing:
    def test_no_args_prints_help_and_returns_zero(self, capsys):
        code = _run([])
        assert code == 0
        out = capsys.readouterr().out
        assert "nl2sh" in out

    def test_query_without_execute(self, capsys):
        conv = _mock_converter("echo hi")
        with patch("nl2sh.cli.Converter", return_value=conv):
            code = _run(["say hi"])
        assert code == 0
        assert capsys.readouterr().out.strip() == "echo hi"

    def test_unknown_api_key_exits_with_one(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        code = _run(["list files"])
        assert code == 1
        assert "nl2sh:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _single_shot
# ---------------------------------------------------------------------------

class TestSingleShot:
    def test_prints_command_only(self, capsys):
        conv = _mock_converter("ls -la")
        with patch("nl2sh.cli.Converter", return_value=conv):
            code = _run(["list all files"])
        assert code == 0
        assert capsys.readouterr().out.strip() == "ls -la"

    def test_execute_flag_runs_command(self):
        conv = _mock_converter("echo hello")
        with patch("nl2sh.cli.Converter", return_value=conv), \
             patch("nl2sh.cli._run_command", return_value=0) as mock_run:
            code = _run(["-e", "say hello"])
        assert code == 0
        mock_run.assert_called_once_with("echo hello")

    def test_converter_error_returns_one(self, capsys):
        conv = MagicMock()
        conv.convert.side_effect = ValueError("Cannot express as shell command")
        with patch("nl2sh.cli.Converter", return_value=conv):
            code = _run(["tell me a joke"])
        assert code == 1
        assert "nl2sh:" in capsys.readouterr().err

    def test_model_flag_passed_to_converter(self):
        with patch("nl2sh.cli.Converter") as mock_cls:
            mock_cls.return_value = _mock_converter()
            _run(["--model", "gpt-4o", "list files"])
        mock_cls.assert_called_once_with(model="gpt-4o")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

class TestInteractiveMode:
    def test_exit_command_ends_loop(self, capsys):
        conv = _mock_converter()
        with patch("nl2sh.cli.Converter", return_value=conv), \
             patch("builtins.input", side_effect=["exit"]):
            code = _run(["-i"])
        assert code == 0

    def test_eof_ends_loop(self, capsys):
        conv = _mock_converter()
        with patch("nl2sh.cli.Converter", return_value=conv), \
             patch("builtins.input", side_effect=EOFError):
            code = _run(["-i"])
        assert code == 0

    def test_query_in_interactive_mode(self, capsys):
        conv = _mock_converter("ls -la")
        with patch("nl2sh.cli.Converter", return_value=conv), \
             patch("builtins.input", side_effect=["list files", "exit"]):
            code = _run(["-i"])
        assert code == 0
        out = capsys.readouterr().out
        assert "ls -la" in out

    def test_empty_input_skipped(self, capsys):
        conv = _mock_converter("ls")
        with patch("nl2sh.cli.Converter", return_value=conv), \
             patch("builtins.input", side_effect=["", "exit"]):
            code = _run(["-i"])
        conv.convert.assert_not_called()

    def test_converter_error_in_interactive_continues(self, capsys):
        conv = MagicMock()
        conv.convert.side_effect = ValueError("oops")
        with patch("nl2sh.cli.Converter", return_value=conv), \
             patch("builtins.input", side_effect=["bad query", "exit"]):
            code = _run(["-i"])
        assert code == 0

    def test_execute_in_interactive_mode(self, capsys):
        conv = _mock_converter("echo hello")
        with patch("nl2sh.cli.Converter", return_value=conv), \
             patch("builtins.input", side_effect=["say hello", "exit"]), \
             patch("nl2sh.cli._run_command", return_value=0) as mock_run:
            code = _run(["-i", "-e"])
        assert code == 0
        mock_run.assert_called_once_with("echo hello")
