"""Tests for nl2sh.converter."""

from unittest.mock import MagicMock, patch

import pytest

from nl2sh.converter import Converter, _strip_code_fence


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_converter(api_key: str = "test-key") -> Converter:
    """Return a Converter whose OpenAI client is fully mocked."""
    with patch("nl2sh.converter.OpenAI"):
        conv = Converter(api_key=api_key)
    return conv


def _mock_response(text: str) -> MagicMock:
    """Build a fake openai ChatCompletion response object."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# _strip_code_fence
# ---------------------------------------------------------------------------

class TestStripCodeFence:
    def test_no_fence(self):
        assert _strip_code_fence("ls -la") == "ls -la"

    def test_bash_fence(self):
        assert _strip_code_fence("```bash\nls -la\n```") == "ls -la"

    def test_sh_fence(self):
        assert _strip_code_fence("```sh\nls -la\n```") == "ls -la"

    def test_shell_fence(self):
        assert _strip_code_fence("```shell\nls -la\n```") == "ls -la"

    def test_plain_fence(self):
        assert _strip_code_fence("```\nls -la\n```") == "ls -la"

    def test_multiline_inside_fence(self):
        result = _strip_code_fence("```bash\nfind . -name '*.py' | xargs wc -l\n```")
        assert result == "find . -name '*.py' | xargs wc -l"


# ---------------------------------------------------------------------------
# Converter initialisation
# ---------------------------------------------------------------------------

class TestConverterInit:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            Converter()

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        with patch("nl2sh.converter.OpenAI") as mock_openai:
            Converter()
        mock_openai.assert_called_once()

    def test_explicit_api_key(self):
        with patch("nl2sh.converter.OpenAI") as mock_openai:
            Converter(api_key="explicit-key")
        mock_openai.assert_called_once()

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("NL2SH_MODEL", raising=False)
        with patch("nl2sh.converter.OpenAI"):
            conv = Converter(api_key="k")
        assert conv._model == "gpt-4o-mini"

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("NL2SH_MODEL", "gpt-4o")
        with patch("nl2sh.converter.OpenAI"):
            conv = Converter(api_key="k")
        assert conv._model == "gpt-4o"

    def test_explicit_model_overrides_env(self, monkeypatch):
        monkeypatch.setenv("NL2SH_MODEL", "gpt-4o")
        with patch("nl2sh.converter.OpenAI"):
            conv = Converter(api_key="k", model="gpt-3.5-turbo")
        assert conv._model == "gpt-3.5-turbo"

    def test_base_url_passed_to_client(self):
        with patch("nl2sh.converter.OpenAI") as mock_openai:
            Converter(api_key="k", base_url="http://localhost:11434/v1")
        _, kwargs = mock_openai.call_args
        assert kwargs.get("base_url") == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# Converter.convert
# ---------------------------------------------------------------------------

class TestConverterConvert:
    def _converter_with_mock_response(self, text: str) -> Converter:
        conv = _make_converter()
        conv._client.chat.completions.create.return_value = _mock_response(text)
        return conv

    def test_returns_command(self):
        conv = self._converter_with_mock_response("ls -la")
        assert conv.convert("list all files") == "ls -la"

    def test_strips_whitespace(self):
        conv = self._converter_with_mock_response("  ls -la  ")
        assert conv.convert("list files") == "ls -la"

    def test_strips_code_fence(self):
        conv = self._converter_with_mock_response("```bash\nls -la\n```")
        assert conv.convert("list files") == "ls -la"

    def test_error_response_raises_value_error(self):
        conv = self._converter_with_mock_response("ERROR: Cannot express as shell command")
        with pytest.raises(ValueError, match="Cannot express"):
            conv.convert("tell me a joke")

    def test_empty_description_raises(self):
        conv = _make_converter()
        with pytest.raises(ValueError, match="empty"):
            conv.convert("   ")

    def test_correct_messages_sent_to_api(self):
        conv = _make_converter()
        conv._client.chat.completions.create.return_value = _mock_response("echo hello")
        conv.convert("print hello")
        call_kwargs = conv._client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "print hello"

    def test_temperature_is_zero(self):
        conv = _make_converter()
        conv._client.chat.completions.create.return_value = _mock_response("echo hi")
        conv.convert("say hi")
        call_kwargs = conv._client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0
