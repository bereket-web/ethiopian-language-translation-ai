"""Leakage-aware held-out evaluation for English–Amharic translation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ethiopian_translation.baselines import CopyBaseline, IBMModel1Lexicon
from ethiopian_translation.data import DATASET_LICENSE, DATASET_REVISION, DATASET_URL, load_pairs
from ethiopian_translation.metrics import score_translations
from ethiopian_translation.model import MODEL_ID, MODEL_REVISION, M2M100Translator

DIRECTIONS = (
    ("en->am", "english", "amharic", "en", "am"),
    ("am->en", "amharic", "english", "am", "en"),
)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4)


def evaluate(
    split: str = "test",
    train_path: Path = Path("data/processed/train.jsonl"),
    eval_path: Path = Path("data/processed/test.jsonl"),
    output_path: Path = Path("reports/benchmark.json"),
    prediction_dir: Path = Path("runs/evaluation"),
    adapter_path: Path | None = Path("models/m2m100-fleurs-am-en"),
    model_id: str = MODEL_ID,
    model_revision: str | None = None,
    limit: int = 0,
    seed: int = 42,
    threads: int = 8,
) -> dict[str, Any]:
    """Score all approaches on one split; do not use test scores for model selection."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    training = load_pairs(train_path)
    evaluation_rows = load_pairs(eval_path)
    if set(row["id"] for row in training).intersection(row["id"] for row in evaluation_rows):
        raise ValueError("training and evaluation sentence IDs overlap")
    if limit < 0:
        raise ValueError("limit cannot be negative")
    if 0 < limit < len(evaluation_rows):
        evaluation_rows = random.Random(seed).sample(evaluation_rows, limit)
        evaluation_rows.sort(key=lambda row: int(row["id"]))
    if not evaluation_rows:
        raise ValueError("evaluation split is empty")

    import torch

    torch.set_num_threads(max(1, threads))
    base_revision = model_revision or MODEL_REVISION
    if not base_revision:
        raise RuntimeError(f"Could not resolve immutable revision for {model_id}")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    candidates: dict[str, Any] = {
        "Copy baseline": CopyBaseline(),
        "IBM Model 1 lexical baseline": {
            "en->am": IBMModel1Lexicon(iterations=5).fit(training, "english", "amharic"),
            "am->en": IBMModel1Lexicon(iterations=5).fit(training, "amharic", "english"),
        },
    }
    base_translator = M2M100Translator(model_id=model_id, revision=base_revision, device="cpu")
    candidates["M2M100 zero-shot"] = base_translator
    adapter_exists = bool(adapter_path and (adapter_path / "adapter_config.json").exists())
    if adapter_exists:
        candidates["M2M100 FLEURS LoRA"] = M2M100Translator(
            model_id=model_id,
            revision=base_revision,
            adapter_path=adapter_path,
            device="cpu",
        )

    for name, candidate in candidates.items():
        direction_scores: dict[str, Any] = {}
        for direction, source_key, target_key, source_code, target_code in DIRECTIONS:
            sources = [row[source_key] for row in evaluation_rows]
            references = [row[target_key] for row in evaluation_rows]
            if isinstance(candidate, dict):
                translator = candidate[direction]
                hypotheses = translator.translate_many(sources, source_code, target_code)
            else:
                hypotheses = candidate.translate_many(sources, source_code, target_code)
            direction_scores[direction] = score_translations(references, hypotheses)
            prediction_file = (
                prediction_dir / f"{split}-{name.lower().replace(' ', '-')}-{direction}.jsonl"
            )
            with prediction_file.open("w", encoding="utf-8") as out:
                for row, hypothesis in zip(evaluation_rows, hypotheses, strict=True):
                    out.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "source": row[source_key],
                                "reference": row[target_key],
                                "hypothesis": hypothesis,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        results[name] = {
            "directions": direction_scores,
            "mean_bleu": _mean([score["bleu"] for score in direction_scores.values()]),
            "mean_chrf++": _mean([score["chrf++"] for score in direction_scores.values()]),
        }

    report = {
        "task": "Amharic–English bidirectional sentence translation",
        "dataset": {
            "id": "google/fleurs",
            "url": DATASET_URL,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "split": split,
            "aligned_pairs_evaluated": len(evaluation_rows),
            "seed": seed,
            "selection_note": (
                "Final test results are reported once after validation-based selection."
            )
            if split == "test"
            else "Validation split used only for model selection and development.",
        },
        "model": {
            "id": model_id,
            "revision": base_revision,
            "license": "MIT",
            "adapter_evaluated": adapter_exists,
            "decoding": {
                "num_beams": base_translator.num_beams,
                "batch_size": 16,
                "max_new_tokens": 128,
            },
        },
        "metrics": (
            "SacreBLEU corpus BLEU and chrF++ (0–100; higher is better); one reference per source."
        ),
        "results": results,
        "notes": [
            "Each aligned sentence ID is deduplicated; conflicting transcript IDs are excluded.",
            (
                "Only aggregate metrics are committed; predictions and raw transcripts stay local "
                "and are ignored by Git."
            ),
            (
                "FLEURS is small and read-speech-domain; scores do not establish conversational "
                "or domain-general quality."
            ),
            (
                "BLEU/chrF are lexical overlap metrics and do not replace human adequacy, "
                "fluency, or safety review."
            ),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved aggregate metrics to {output_path}; prediction examples remain in {prediction_dir}"
    )
    for name, score in results.items():
        print(f"{name}: mean BLEU={score['mean_bleu']:.2f}, mean chrF++={score['mean_chrf++']:.2f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-dir", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--adapter", type=Path, default=Path("models/m2m100-fleurs-am-en"))
    parser.add_argument("--model-revision")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    default_input = Path(f"data/processed/{args.split}.jsonl")
    evaluate(
        split=args.split,
        train_path=args.train,
        eval_path=args.input or default_input,
        output_path=args.output or Path(f"reports/{args.split}_benchmark.json"),
        prediction_dir=args.prediction_dir,
        adapter_path=args.adapter,
        model_revision=args.model_revision,
        limit=args.limit,
        seed=args.seed,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
