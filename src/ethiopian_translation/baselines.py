"""Transparent non-neural translation baselines."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable

_TOKEN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_LEFT_PUNCT = set(",.;:!?%)]}»”’")
_RIGHT_PUNCT = set("([{«“‘")


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode text while retaining Ethiopic words and punctuation."""
    return _TOKEN.findall(unicodedata.normalize("NFC", text))


def detokenize(tokens: Iterable[str]) -> str:
    """Join tokens with conventional spacing around punctuation."""
    output = ""
    previous = ""
    for token in tokens:
        if not output:
            output = token
        elif token in _LEFT_PUNCT or previous in _RIGHT_PUNCT:
            output += token
        else:
            output += " " + token
        previous = token
    return output


class CopyBaseline:
    """Return the input unchanged as a deliberately simple floor baseline."""

    def translate_many(self, texts: list[str], source: str, target: str) -> list[str]:
        del source, target
        return list(texts)


class IBMModel1Lexicon:
    """A small unsupervised IBM Model 1 word-translation table.

    It translates one source token to its most likely target token, falls back
    to copying unseen tokens, and preserves source word order. This is a
    transparent lexical baseline, not a fluent statistical MT decoder.
    """

    def __init__(self, iterations: int = 5) -> None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.iterations = iterations
        self.probabilities: dict[str, dict[str, float]] = {}

    def fit(
        self, pairs: list[dict[str, str]], source_key: str, target_key: str
    ) -> IBMModel1Lexicon:
        token_pairs = [(tokenize(row[source_key]), tokenize(row[target_key])) for row in pairs]
        cooccurring: dict[str, set[str]] = defaultdict(set)
        for source_tokens, target_tokens in token_pairs:
            for source_word in sorted(set(source_tokens)):
                cooccurring[source_word].update(target_tokens)
        probabilities: dict[str, dict[str, float]] = {
            source_word: {
                target_word: 1.0 / len(target_words) for target_word in sorted(target_words)
            }
            for source_word, target_words in cooccurring.items()
            if target_words
        }
        for _ in range(self.iterations):
            counts: dict[str, Counter[str]] = defaultdict(Counter)
            totals: Counter[str] = Counter()
            for source_tokens, target_tokens in token_pairs:
                unique_source = sorted(set(source_tokens))
                for target_word in target_tokens:
                    denominator = sum(
                        probabilities.get(source_word, {}).get(target_word, 0.0)
                        for source_word in unique_source
                    )
                    if denominator <= 0.0:
                        continue
                    for source_word in unique_source:
                        probability = probabilities.get(source_word, {}).get(target_word, 0.0)
                        if probability:
                            contribution = probability / denominator
                            counts[source_word][target_word] += contribution
                            totals[source_word] += contribution
            probabilities = {
                source_word: {
                    target_word: count / totals[source_word]
                    for target_word, count in target_counts.items()
                }
                for source_word, target_counts in counts.items()
                if totals[source_word] > 0
            }
        self.probabilities = probabilities
        return self

    def translate(self, text: str) -> str:
        translated: list[str] = []
        for token in tokenize(text):
            options = self.probabilities.get(token)
            if not options:
                translated.append(token)
                continue
            translated.append(max(options.items(), key=lambda item: (item[1], item[0]))[0])
        return detokenize(translated)

    def translate_many(self, texts: list[str], source: str, target: str) -> list[str]:
        del source, target
        return [self.translate(text) for text in texts]
