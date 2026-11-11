"""Build the analysis prompt and stream it from an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .config import Settings
from .ingest import Document

SYSTEM_PROMPT = """\
You are a stylistician and editorial strategist. You reverse-engineer an author's \
writing system from samples so that other writers, or another LLM, can reproduce it \
consistently on new topics.

You receive two inputs:
1. A METRICS profile computed by deterministic code. These numbers are accurate; do \
not recompute or contradict them. Interpret them and cite them where they support a rule.
2. EXCERPTS from the author's documents (possibly truncated).

Separate style from subject: describe how the author writes, not what the samples \
are about. Every rule must be observable in the samples. Quote short real examples \
(under 20 words) from the excerpts as evidence. Do not invent quotes.

Write the result in Markdown with exactly these sections:

# Writing System
One paragraph: who this writer sounds like and to whom.

## 1. Structure & Pacing
How pieces open, progress and close; typical section order and proportions; how \
transitions between sections work.

## 2. Sentence Rhythm
Sentence length and variation, fragments, questions, paragraph length, punctuation \
habits. Use the metrics.

## 3. Word Choice
Register, vocabulary level, jargon handling, signature words and recurring phrases.

## 4. Voice & Stance
Relationship to the reader (pronouns), confidence vs. hedging, humour, emotion, \
contractions.

## 5. Argument Moves
The repeatable pattern for making a point (e.g. claim -> example -> consequence); \
evidence types; analogy sources.

## 6. Rules
Two bulleted lists, "Always" and "Never", each 6-12 concrete, checkable rules.

## 7. Generation Prompt
A fenced code block containing a ready-to-use system prompt for another LLM that \
must write new content in this exact style. Express rules as concrete behaviours \
the model can follow (not numeric targets it cannot count), and include 2-3 short \
verbatim example passages from the excerpts.

## 8. Confidence & Gaps
What the samples were too small or too uniform to show.
"""


@dataclass
class PreparedRequest:
    messages: list[dict]
    truncated: bool
    excerpt_chars: int


def build_excerpts(docs: Sequence[Document], budget: int) -> tuple[str, bool]:
    """Share ``budget`` characters across documents; short docs give their unused share to long ones."""
    remaining = budget
    order = sorted(range(len(docs)), key=lambda i: len(docs[i].text))
    allotted: dict[int, int] = {}
    for position, i in enumerate(order):
        share = remaining // (len(docs) - position)
        allotted[i] = min(len(docs[i].text), share)
        remaining -= allotted[i]

    truncated = False
    parts = []
    for i, doc in enumerate(docs):
        text = doc.text
        if len(text) > allotted[i]:
            truncated = True
            cut = text[: allotted[i]]
            space = cut.rfind(" ")
            text = (cut[:space] if space > allotted[i] * 0.8 else cut) + "\n[... truncated ...]"
        parts.append(f'<document name="{doc.name}" words="{doc.word_count}">\n{text}\n</document>')
    return "\n\n".join(parts), truncated


def prepare(docs: Sequence[Document], metrics: dict, settings: Settings) -> PreparedRequest:
    excerpts, truncated = build_excerpts(docs, settings.max_excerpt_chars)
    user = (
        f"METRICS (corpus of {len(docs)} document(s)):\n```json\n{json.dumps(metrics, indent=2)}\n```\n\n"
        f"EXCERPTS:\n{excerpts}\n\n"
        "Extract this author's writing system."
    )
    return PreparedRequest(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        truncated=truncated,
        excerpt_chars=len(excerpts),
    )


def stream_completion(
    settings: Settings,
    messages: list[dict],
    on_delta: Callable[[str], None],
    cancel: threading.Event,
) -> None:
    """Stream the model's answer, calling ``on_delta`` per chunk. Raises on API errors."""
    from openai import OpenAI

    client = OpenAI(
        base_url=settings.base_url.strip() or None,
        api_key=settings.effective_api_key() or "not-set",
        timeout=300,
    )
    stream = client.chat.completions.create(
        model=settings.model.strip(),
        messages=messages,
        temperature=settings.temperature,
        stream=True,
    )
    try:
        for chunk in stream:
            if cancel.is_set():
                break
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                on_delta(chunk.choices[0].delta.content)
    finally:
        stream.close()
