#!/usr/bin/env python3
"""Estimate paired bootstrap uncertainty for the held-out M2M100 LoRA gain."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from sacrebleu.metrics import CHRF

DIRECTIONS = ("en->am", "am->en")
BASE_PREFIX = "test-m2m100-zero-shot"
ADAPTER_PREFIX = "test-m2m100-fleurs-lora"


def _load(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"Duplicate sentence ID in local predictions: {row['id']}")
            rows[row["id"]] = row
    return rows


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    prediction_dir: Path = Path("runs/evaluation"),
    output_path: Path = Path("reports/test_bootstrap.json"),
    repetitions: int = 1000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Resample sentence IDs jointly across directions and compare chrF++ scores."""
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    data: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for direction in DIRECTIONS:
        base_path = prediction_dir / f"{BASE_PREFIX}-{direction}.jsonl"
        adapter_path = prediction_dir / f"{ADAPTER_PREFIX}-{direction}.jsonl"
        base_rows = _load(base_path)
        adapter_rows = _load(adapter_path)
        if set(base_rows) != set(adapter_rows):
            raise ValueError(f"Base and adapter sentence IDs differ in {direction}")
        ids = sorted(base_rows)
        if not ids:
            raise ValueError(f"No predictions found for {direction}")
        for sentence_id in ids:
            if base_rows[sentence_id]["reference"] != adapter_rows[sentence_id]["reference"]:
                raise ValueError(f"References differ for sentence {sentence_id} in {direction}")
        data[direction] = {
            "ids": {sentence_id: {} for sentence_id in ids},
            "base": base_rows,
            "adapter": adapter_rows,
        }

    ids = sorted(data[DIRECTIONS[0]]["base"])
    if any(sorted(data[direction]["base"]) != ids for direction in DIRECTIONS[1:]):
        raise ValueError("Sentence IDs do not align across translation directions")
    references = {
        direction: [data[direction]["base"][sentence_id]["reference"] for sentence_id in ids]
        for direction in DIRECTIONS
    }
    base_hypotheses = {
        direction: [data[direction]["base"][sentence_id]["hypothesis"] for sentence_id in ids]
        for direction in DIRECTIONS
    }
    adapter_hypotheses = {
        direction: [data[direction]["adapter"][sentence_id]["hypothesis"] for sentence_id in ids]
        for direction in DIRECTIONS
    }
    metric = CHRF(word_order=2)
    observed: dict[str, dict[str, float]] = {}
    for direction in DIRECTIONS:
        base_score = float(
            metric.corpus_score(base_hypotheses[direction], [references[direction]]).score
        )
        adapter_score = float(
            metric.corpus_score(adapter_hypotheses[direction], [references[direction]]).score
        )
        observed[direction] = {
            "base_chrf++": round(base_score, 4),
            "adapter_chrf++": round(adapter_score, 4),
            "delta": round(adapter_score - base_score, 4),
        }

    rng = random.Random(seed)
    direction_deltas: dict[str, list[float]] = {direction: [] for direction in DIRECTIONS}
    macro_deltas: list[float] = []
    n = len(ids)
    for _ in range(repetitions):
        sample = rng.choices(range(n), k=n)
        current: list[float] = []
        for direction in DIRECTIONS:
            refs = [references[direction][index] for index in sample]
            base = [base_hypotheses[direction][index] for index in sample]
            adapter = [adapter_hypotheses[direction][index] for index in sample]
            base_score = float(metric.corpus_score(base, [refs]).score)
            adapter_score = float(metric.corpus_score(adapter, [refs]).score)
            delta = adapter_score - base_score
            direction_deltas[direction].append(delta)
            current.append(delta)
        macro_deltas.append(sum(current) / len(current))

    ci: dict[str, dict[str, float]] = {}
    for name, values in [*direction_deltas.items(), ("macro_mean", macro_deltas)]:
        ci[name] = {
            "lower_95": round(_percentile(values, 0.025), 4),
            "upper_95": round(_percentile(values, 0.975), 4),
        }
    report: dict[str, Any] = {
        "metric": "SacreBLEU chrF++",
        "method": "Paired sentence-ID bootstrap; same sampled IDs used in both directions",
        "split": "test",
        "n_aligned_pairs": n,
        "repetitions": repetitions,
        "seed": seed,
        "observed": observed,
        "observed_macro_mean_delta": round(
            sum(item["delta"] for item in observed.values()) / len(observed), 4
        ),
        "delta_ci95": ci,
        "release_gate": {
            "criterion": "macro mean chrF++ gain lower 95% paired-bootstrap bound > 0",
            "passed": ci["macro_mean"]["lower_95"] > 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Paired test bootstrap n={n}, reps={repetitions}: "
        f"macro chrF++ gain {report['observed_macro_mean_delta']:+.3f}; "
        f"95% CI [{ci['macro_mean']['lower_95']:.3f}, {ci['macro_mean']['upper_95']:.3f}]"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--output", type=Path, default=Path("reports/test_bootstrap.json"))
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    paired_bootstrap(args.predictions, args.output, args.repetitions, args.seed)


if __name__ == "__main__":
    main()
