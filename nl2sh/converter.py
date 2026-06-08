"""Core converter: translates natural-language descriptions into shell commands."""

from __future__ import annotations

import os
import re

from openai import OpenAI

_SYSTEM_PROMPT = """\
You are a shell-command assistant. When the user describes what they want to do \
in plain English, you respond with a single, concise shell command that \
accomplishes the task. Output ONLY the shell command – no explanations, no \
markdown code fences, no extra text. If multiple commands are needed, join them \
with semicolons or use a pipe. If the request is ambiguous or cannot be expressed \
as a shell command, output exactly: ERROR: <brief reason>.\
"""


class Converter:
    """Translates natural-language descriptions into shell commands via an LLM.

    Parameters
    ----------
    api_key:
        OpenAI API key. Falls back to the ``OPENAI_API_KEY`` environment
        variable when *None*.
    model:
        OpenAI model to use.  Defaults to the ``NL2SH_MODEL`` environment
        variable or ``gpt-4o-mini``.
    base_url:
        Optional alternative base URL for the OpenAI-compatible API endpoint
        (e.g. for local models).  Falls back to the ``OPENAI_BASE_URL``
        environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key is required. "
                "Set the OPENAI_API_KEY environment variable or pass api_key=."
            )

        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        client_kwargs: dict = {"api_key": resolved_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self._client = OpenAI(**client_kwargs)
        self._model = model or os.environ.get("NL2SH_MODEL", "gpt-4o-mini")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(self, description: str) -> str:
        """Return the shell command that corresponds to *description*.

        Parameters
        ----------
        description:
            Plain-English description of what the user wants to do.

        Returns
        -------
        str
            The shell command string.

        Raises
        ------
        ValueError
            If the LLM reports that the request cannot be expressed as a
            shell command.
        """
        description = description.strip()
        if not description:
            raise ValueError("Description must not be empty.")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
            temperature=0,
        )

        raw = (response.choices[0].message.content or "").strip()
        # Strip accidental markdown code fences the model might add anyway
        raw = _strip_code_fence(raw)

        if raw.startswith("ERROR:"):
            raise ValueError(raw[len("ERROR:"):].strip())

        return raw


def _strip_code_fence(text: str) -> str:
    """Remove optional leading/trailing markdown code fences from *text*."""
    pattern = r"^```(?:bash|shell|sh)?\s*\n?(.*?)\n?```$"
    match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text
