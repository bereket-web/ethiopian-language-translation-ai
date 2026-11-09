"""Rights-aware FLEURS transcript preparation for English–Amharic translation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET_ID = "google/fleurs"
DATASET_URL = "https://huggingface.co/datasets/google/fleurs"
DATASET_LICENSE = "CC BY 4.0"
DATASET_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
DATASET_CITATION = (
    "Conneau et al. (2022), FLEURS: Few-shot Learning Evaluation of Universal "
    "Representations of Speech, https://arxiv.org/abs/2205.12446"
)
LANGUAGE_CONFIGS = {"amharic": "am_et", "english": "en_us"}
SPLIT_FILENAMES = {"train": "train", "validation": "dev", "test": "test"}
EXPECTED_TSV_SHA256 = {
    "train": {
        "am_et": "8ed7f208e511a08be00bca686706368ced79c876a576ac5bba03308377c6e233",
        "en_us": "3ccfc83672cc03a835143e325abb38b4163e3a21725bc1a7d1165bc309b95852",
    },
    "validation": {
        "am_et": "6e37bdc090862a810a4e28b5f2148e35ec838b642dbc01ee6ad9504848aa1569",
        "en_us": "9d57ee7e91e9d4c92edb39f6bbea668ef8dc2a3ff96eb510d5580b2ad05d17ec",
    },
    "test": {
        "am_et": "92c5a8c1837a83d719e890946872dc69d164f5e41011919e49ea86984e8eca0d",
        "en_us": "74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca",
    },
}
_ZERO_WIDTH = re.compile("[\u200b-\u200f\u202a-\u202e\ufeff]")


def clean_text(value: str) -> str:
    """Normalize Unicode and whitespace without transliterating either language."""
    value = unicodedata.normalize("NFC", value)
    value = _ZERO_WIDTH.sub("", value)
    return " ".join(value.split())


def download_tsv(language_config: str, split: str, output_dir: Path) -> tuple[Path, str]:
    """Download a small transcript-only TSV from the immutable FLEURS revision."""
    if language_config not in LANGUAGE_CONFIGS.values():
        raise ValueError(f"Unsupported FLEURS config: {language_config}")
    if split not in SPLIT_FILENAMES:
        raise ValueError(f"Unsupported split: {split}")
    filename = SPLIT_FILENAMES[split]
    path = output_dir / language_config / f"{filename}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"{DATASET_URL}/resolve/{DATASET_REVISION}/data/{language_config}/{filename}.tsv"
        request = urllib.request.Request(
            url, headers={"User-Agent": "ethiopian-translation-ai/0.1"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        if len(payload) > 20_000_000:
            raise ValueError("Unexpectedly large TSV; refusing to save it")
        path.write_bytes(payload)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = EXPECTED_TSV_SHA256[split][language_config]
    if digest != expected:
        raise ValueError(f"Checksum mismatch for {path}; expected the pinned FLEURS TSV")
    return path, digest


def read_transcripts(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Return one raw transcript per sentence ID; exclude conflicting IDs."""
    by_id: dict[str, set[str]] = defaultdict(set)
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.reader(source, delimiter="\t"):
            if len(row) < 3:
                continue
            sentence_id, transcript = row[0].strip(), clean_text(row[2])
            if not sentence_id.isdigit() or not transcript:
                continue
            rows += 1
            by_id[sentence_id].add(transcript)
    ambiguous = {sentence_id for sentence_id, texts in by_id.items() if len(texts) > 1}
    transcripts = {
        sentence_id: next(iter(texts)) for sentence_id, texts in by_id.items() if len(texts) == 1
    }
    counts = {
        "rows_with_transcripts": rows,
        "distinct_sentence_ids": len(by_id),
        "conflicting_sentence_ids_excluded": len(ambiguous),
        "usable_sentence_ids": len(transcripts),
    }
    return transcripts, counts


def _load_language_split(
    language_config: str,
    split: str,
    raw_dir: Path,
) -> tuple[dict[str, str], dict[str, int], str]:
    path, digest = download_tsv(language_config, split, raw_dir)
    transcript_map, counts = read_transcripts(path)
    return transcript_map, counts, digest


def prepare_data(
    raw_dir: Path = Path("data/raw/fleurs"),
    processed_dir: Path = Path("data/processed"),
    audit_path: Path = Path("reports/data_audit.json"),
) -> dict[str, Any]:
    """Align by the FLEURS source sentence ID, deduplicate, and audit leakage."""
    all_pairs: dict[str, list[dict[str, str]]] = {}
    split_audit: dict[str, Any] = {}
    checksums: dict[str, str] = {}
    for split in SPLIT_FILENAMES:
        amharic, am_counts, am_hash = _load_language_split("am_et", split, raw_dir)
        english, en_counts, en_hash = _load_language_split("en_us", split, raw_dir)
        ids = sorted(set(amharic).intersection(english), key=int)
        pairs = [
            {"id": sentence_id, "amharic": amharic[sentence_id], "english": english[sentence_id]}
            for sentence_id in ids
        ]
        all_pairs[split] = pairs
        split_audit[split] = {
            "amharic": am_counts,
            "english": en_counts,
            "aligned_pairs_before_cross_split_filter": len(pairs),
            "amharic_only_ids": len(set(amharic).difference(english)),
            "english_only_ids": len(set(english).difference(amharic)),
        }
        checksums[f"{split}/am_et.tsv"] = am_hash
        checksums[f"{split}/en_us.tsv"] = en_hash

    ids = {split: {row["id"] for row in rows} for split, rows in all_pairs.items()}
    id_overlaps = {
        f"{left}_and_{right}": len(ids[left].intersection(ids[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(id_overlaps.values()):
        raise ValueError(f"FLEURS sentence IDs cross split boundaries: {id_overlaps}")

    test_text = {
        clean_text(row[key]).casefold()
        for row in all_pairs["test"]
        for key in ("english", "amharic")
    }
    val_before = len(all_pairs["validation"])
    all_pairs["validation"] = [
        row
        for row in all_pairs["validation"]
        if not any(clean_text(row[key]).casefold() in test_text for key in ("english", "amharic"))
    ]
    validation_test_exact_text_overlaps_removed = val_before - len(all_pairs["validation"])

    heldout_text = {
        clean_text(row[key]).casefold()
        for split in ("validation", "test")
        for row in all_pairs[split]
        for key in ("english", "amharic")
    }
    train_before = len(all_pairs["train"])
    all_pairs["train"] = [
        row
        for row in all_pairs["train"]
        if not any(
            clean_text(row[key]).casefold() in heldout_text for key in ("english", "amharic")
        )
    ]
    train_heldout_exact_text_overlaps_removed = train_before - len(all_pairs["train"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in all_pairs.items():
        with (processed_dir / f"{split}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_audit[split]["usable_aligned_pairs"] = len(rows)

    audit: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "dataset_license": DATASET_LICENSE,
        "dataset_revision": DATASET_REVISION,
        "dataset_citation": DATASET_CITATION,
        "languages": {"amharic": "am_et", "english": "en_us"},
        "alignment_key": "FLEURS source sentence ID (first column in official TSVs)",
        "text_field": "raw transcript (third TSV column); one transcript per sentence ID",
        "source_tsv_sha256": checksums,
        "splits": split_audit,
        "cross_split_sentence_id_overlaps": id_overlaps,
        "validation_test_exact_text_overlaps_removed": validation_test_exact_text_overlaps_removed,
        "train_heldout_exact_text_overlaps_removed": train_heldout_exact_text_overlaps_removed,
        "redistribution": (
            "Source TSVs and processed text are local-only and excluded from public releases."
        ),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def load_pairs(path: Path) -> list[dict[str, str]]:
    """Load one processed split without printing or exposing examples."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run `uv run python scripts/prepare_data.py` first"
        )
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/fleurs"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    args = parser.parse_args()
    audit = prepare_data(args.raw_dir, args.processed_dir, args.audit)
    print(
        "Prepared aligned pairs: "
        + ", ".join(
            f"{split}={details['usable_aligned_pairs']}"
            for split, details in audit["splits"].items()
        )
    )
    print(f"Aggregate audit: {args.audit}")


if __name__ == "__main__":
    main()
