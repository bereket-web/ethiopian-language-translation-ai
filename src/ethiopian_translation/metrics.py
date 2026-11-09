"""Machine-translation metrics with explicit SacreBLEU implementations."""

from __future__ import annotations

from typing import Any

from sacrebleu.metrics import BLEU, CHRF


def score_translations(references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    """Return corpus BLEU and chrF++ on SacreBLEU's 0–100 scale."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal lengths")
    if not references:
        raise ValueError("at least one reference/hypothesis pair is required")
    bleu = BLEU(effective_order=True).corpus_score(hypotheses, [references])
    chrf = CHRF(word_order=2).corpus_score(hypotheses, [references])
    return {
        "bleu": round(float(bleu.score), 4),
        "chrf++": round(float(chrf.score), 4),
        "n": len(references),
        "metric_scale": "SacreBLEU score, 0–100; higher is better",
        "bleu_signature": str(bleu._score_signature)
        if getattr(bleu, "_score_signature", None)
        else None,
        "chrf_signature": str(chrf._score_signature)
        if getattr(chrf, "_score_signature", None)
        else None,
    }
