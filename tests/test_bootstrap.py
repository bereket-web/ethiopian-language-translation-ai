import json

import pytest

from scripts.bootstrap_eval import paired_bootstrap


def _write_predictions(directory, prefix, direction, rows):
    path = directory / f"{prefix}-{direction}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(sentence_id, reference, hypothesis):
    return {"id": sentence_id, "reference": reference, "hypothesis": hypothesis}


def _prepare(directory, missing_adapter_id=False):
    directory.mkdir()
    for direction, references in {
        "en->am": ["Good morning friend", "We are ready", "Thank you very much"],
        "am->en": ["ሰላም ወዳጄ", "ዝግጁ ነን", "በጣም አመሰግናለሁ"],
    }.items():
        ids = ["1", "2", "3"]
        base = [
            _row(key, reference, "unrelated words")
            for key, reference in zip(ids, references, strict=True)
        ]
        adapter_ids = ids[:-1] if missing_adapter_id else ids
        adapter = [
            _row(key, references[index], references[index]) for index, key in enumerate(adapter_ids)
        ]
        _write_predictions(directory, "test-m2m100-zero-shot", direction, base)
        _write_predictions(directory, "test-m2m100-fleurs-lora", direction, adapter)


def test_paired_bootstrap_is_deterministic_and_aggregate_only(tmp_path) -> None:
    predictions = tmp_path / "predictions"
    _prepare(predictions)
    output_a = tmp_path / "first.json"
    output_b = tmp_path / "second.json"
    first = paired_bootstrap(predictions, output_a, repetitions=100, seed=7)
    second = paired_bootstrap(predictions, output_b, repetitions=100, seed=7)
    assert first == second
    assert first["n_aligned_pairs"] == 3
    assert first["release_gate"]["passed"]
    serialized = output_a.read_text(encoding="utf-8")
    assert "Good morning friend" not in serialized
    assert "unrelated words" not in serialized


def test_paired_bootstrap_rejects_mismatched_sentence_ids(tmp_path) -> None:
    predictions = tmp_path / "predictions"
    _prepare(predictions, missing_adapter_id=True)
    with pytest.raises(ValueError, match="sentence IDs differ"):
        paired_bootstrap(predictions, tmp_path / "report.json", repetitions=100)
