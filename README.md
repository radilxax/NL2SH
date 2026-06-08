# NL2SH

A project that makes LLM transfer Natural Language to Shell/bash commands.

NL2SH takes a plain-English description of what you want to do and uses an
OpenAI-compatible LLM to produce the corresponding shell command.

---

## Features

- **Single-shot mode** – pass your query as a CLI argument
- **Execute mode** (`-e`) – run the generated command immediately
- **Interactive REPL** (`-i`) – stay in a loop and keep asking
- **Any OpenAI-compatible endpoint** – works with GPT-4o, local Ollama models, etc.

---

## Installation

```bash
pip install .
```

Or install in editable / development mode:

```bash
pip install -e ".[dev]"
```

---

## Configuration

NL2SH is configured via environment variables (or a `.env` file loaded by your
shell):

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | **yes** | – | Your OpenAI API key |
| `NL2SH_MODEL` | no | `gpt-4o-mini` | Model to use |
| `OPENAI_BASE_URL` | no | OpenAI default | Override for compatible endpoints |

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

---

## Usage

### Single-shot

```bash
nl2sh "list all Python files modified in the last 7 days"
# → find . -name "*.py" -mtime -7
```

### Execute the generated command immediately

```bash
nl2sh -e "show disk usage sorted by size"
# prints: du -sh * | sort -h
# then runs it
```

### Interactive REPL

```bash
nl2sh -i
# NL2SH interactive mode. Type 'exit' or press Ctrl-D to quit.
# nl2sh> count lines in all .py files
# find . -name "*.py" | xargs wc -l
# nl2sh> exit
```

### Use a different model

```bash
nl2sh --model gpt-4o "recursively find all empty directories"
```

---

## Running Tests

```bash
pytest
```

---

## License

MIT – see [LICENSE](LICENSE).
