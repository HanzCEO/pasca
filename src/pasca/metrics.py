"""Deterministic style metrics computed locally (no LLM).

LLMs cannot count reliably, so the numbers the model interprets come from here.
Rates are normalised per 1,000 words so corpora of different sizes compare.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+|\n{2,}")
_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*")
_CONTRACTION = re.compile(r"\b[A-Za-z]+['’](?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_EXPANDED = re.compile(
    r"\b(?:do not|does not|did not|is not|are not|was not|were not|cannot|can not|will not|would not|"
    r"should not|could not|have not|has not|had not|it is|that is|what is|there is|I am|you are|we are|"
    r"they are|I have|you have|we have|I will|you will|we will|it will)\b",
    re.IGNORECASE,
)

FIRST_SINGULAR = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'll", "i'd"}
FIRST_PLURAL = {"we", "us", "our", "ours", "ourselves", "we're", "we've", "we'll", "we'd", "let's"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves", "you're", "you've", "you'll", "you'd"}

HEDGES = {
    "might", "may", "could", "perhaps", "possibly", "probably", "maybe", "somewhat", "seemingly",
    "apparently", "arguably", "likely", "unlikely", "suggest", "suggests", "seem", "seems", "appear",
    "appears", "roughly", "approximately", "generally", "tend", "tends", "relatively", "fairly",
}
BOOSTERS = {
    "definitely", "clearly", "obviously", "certainly", "literally", "honestly", "completely", "totally",
    "absolutely", "always", "never", "really", "truly", "undoubtedly", "indeed", "entirely", "extremely",
    "incredibly", "massively", "actually", "surely", "huge", "insane", "crazy",
}

# Sentence-initial connectors, grouped by function.
OPENERS: dict[str, tuple[str, ...]] = {
    "additive": ("and", "also", "plus", "in addition", "besides"),
    "adversative": ("but", "yet", "however", "although", "though", "still", "instead", "on the other hand"),
    "causal": ("so", "because", "therefore", "thus", "hence", "as a result", "that's why"),
    "temporal": ("then", "next", "now", "meanwhile", "eventually", "finally", "first", "later", "after"),
    "formal_conjunctive": ("furthermore", "moreover", "consequently", "additionally", "nevertheless",
                           "nonetheless", "in conclusion", "accordingly"),
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "is", "it", "that",
    "this", "was", "as", "be", "by", "are", "from", "its", "it's", "i", "you", "he", "she", "they", "we",
    "his", "her", "their", "our", "your", "my", "so", "if", "not", "do", "have", "has", "had", "were",
}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n"))
    parts = (s.strip() for s in _SENTENCE_SPLIT.split(text))
    return [s for s in parts if s and _WORD.search(s)]


def words_of(text: str) -> list[str]:
    return [w.replace("’", "'").lower() for w in _WORD.findall(text)]


def count_syllables(word: str) -> int:
    word = word.lower().strip("'")
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", word)))


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010), averaged both directions."""
    if len(tokens) < 10:
        return 0.0

    def one_pass(seq: list[str]) -> float:
        factors, types, count = 0.0, set(), 0
        for tok in seq:
            count += 1
            types.add(tok)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold)
        return len(seq) / factors if factors else float(len(seq))

    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def _per_k(count: int, total: int) -> float:
    return round(count * 1000 / total, 2) if total else 0.0


def _pct(count: int, total: int) -> float:
    return round(count * 100 / total, 1) if total else 0.0


def _opener_class(sentence: str) -> str:
    lowered = sentence.lower().lstrip("\"'“‘(-— ")
    for name, starters in OPENERS.items():
        for s in starters:
            if lowered.startswith(s) and (len(lowered) == len(s) or not lowered[len(s)].isalnum()):
                return name
    return "none"


def recurring_phrases(tokens: list[str], n: int = 3, top: int = 15, min_count: int = 2) -> list[tuple[str, int]]:
    grams = Counter(
        " ".join(tokens[i : i + n])
        for i in range(len(tokens) - n + 1)
        if not all(t in STOPWORDS for t in tokens[i : i + n])
    )
    return [(g, c) for g, c in grams.most_common(top) if c >= min_count]


def profile(text: str) -> dict:
    """Return a JSON-serialisable style profile of ``text``."""
    sentences = split_sentences(text)
    tokens = words_of(text)
    n_words = len(tokens)
    lengths = [len(words_of(s)) for s in sentences]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    word_counts = Counter(tokens)
    contractions = len(_CONTRACTION.findall(text))
    expanded = len(_EXPANDED.findall(text))
    hedges = sum(word_counts[w] for w in HEDGES)
    boosters = sum(word_counts[w] for w in BOOSTERS)
    openers = Counter(_opener_class(s) for s in sentences)

    return {
        "size": {
            "words": n_words,
            "sentences": len(sentences),
            "paragraphs": len(paragraphs),
        },
        "rhythm": {
            "mean_sentence_length": round(statistics.fmean(lengths), 1) if lengths else 0.0,
            "median_sentence_length": statistics.median(lengths) if lengths else 0,
            "sentence_length_stdev": round(statistics.pstdev(lengths), 1) if len(lengths) > 1 else 0.0,
            "short_sentences_pct_under_6_words": _pct(sum(l < 6 for l in lengths), len(lengths)),
            "long_sentences_pct_over_25_words": _pct(sum(l > 25 for l in lengths), len(lengths)),
            "mean_sentences_per_paragraph": round(len(sentences) / len(paragraphs), 1) if paragraphs else 0.0,
            "questions_pct_of_sentences": _pct(sum(s.rstrip("\"'”’) ").endswith("?") for s in sentences), len(sentences)),
            "exclamations_pct_of_sentences": _pct(sum(s.rstrip("\"'”’) ").endswith("!") for s in sentences), len(sentences)),
        },
        "vocabulary": {
            "mtld_lexical_diversity": round(mtld(tokens), 1),
            "mean_word_length_chars": round(statistics.fmean(len(t) for t in tokens), 2) if tokens else 0.0,
            "mean_syllables_per_word": round(statistics.fmean(count_syllables(t) for t in tokens), 2) if tokens else 0.0,
            "long_words_pct_3plus_syllables": _pct(sum(count_syllables(t) >= 3 for t in tokens), n_words),
        },
        "voice": {
            "first_person_singular_per_1k": _per_k(sum(word_counts[w] for w in FIRST_SINGULAR), n_words),
            "first_person_plural_per_1k": _per_k(sum(word_counts[w] for w in FIRST_PLURAL), n_words),
            "second_person_per_1k": _per_k(sum(word_counts[w] for w in SECOND_PERSON), n_words),
            "contractions_per_1k": _per_k(contractions, n_words),
            "expanded_forms_per_1k": _per_k(expanded, n_words),
            "contraction_share_pct": _pct(contractions, contractions + expanded),
        },
        "stance": {
            "hedges_per_1k": _per_k(hedges, n_words),
            "boosters_per_1k": _per_k(boosters, n_words),
            "hedge_to_booster_ratio": round(hedges / boosters, 2) if boosters else None,
            "top_boosters": [w for w, _ in Counter({w: word_counts[w] for w in BOOSTERS if word_counts[w]}).most_common(6)],
            "top_hedges": [w for w, _ in Counter({w: word_counts[w] for w in HEDGES if word_counts[w]}).most_common(6)],
        },
        "sentence_openers_pct": {name: _pct(openers[name], len(sentences)) for name in [*OPENERS, "none"]},
        "punctuation_per_1k_words": {
            "em_dash": _per_k(text.count("—") + text.count(" -- ") + text.count(" - "), n_words),
            "parentheses": _per_k(text.count("("), n_words),
            "ellipsis": _per_k(text.count("…") + text.count("..."), n_words),
            "colon": _per_k(text.count(":"), n_words),
            "semicolon": _per_k(text.count(";"), n_words),
        },
        "recurring_phrases": [{"phrase": p, "count": c} for p, c in recurring_phrases(tokens)],
    }
