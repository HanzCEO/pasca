# pasca
[LocalWisdom] Understand writing system with ease

Drop sample texts (`.txt`, `.md`, `.docx`, `.pdf`) onto the window. Pasca measures the
style locally (sentence rhythm, pronouns, hedges/boosters, connectors, lexical diversity…),
then asks any OpenAI-compatible model to turn those numbers plus excerpts into a
writing-system codex, including a ready-to-use generation prompt.

## Run

```bash
uv run pasca
```

Set the base URL, model and API key in the left panel (or export `PASCA_API_KEY`).
Settings are saved to `~/.config/pasca/config.toml`; the key is only written there if
you tick "Remember API key on disk".

## Test

```bash
uv run pytest
```
