from pathlib import Path

from ethiopian_translation.data import clean_text, read_transcripts


def test_clean_text_normalizes_unicode_and_whitespace() -> None:
    assert clean_text("  አማርኛ\u200b\t  text  ") == "አማርኛ text"


def test_read_transcripts_excludes_conflicting_sentence_ids(tmp_path: Path) -> None:
    path = tmp_path / "sample.tsv"
    path.write_text(
        "10\ta.wav\tFirst translation\tnormalized\tphones\t10\tMALE\n"
        "10\tb.wav\tDifferent translation\tnormalized\tphones\t12\tFEMALE\n"
        "11\tc.wav\tOne aligned sentence\tnormalized\tphones\t9\tFEMALE\n"
        "bad\td.wav\tInvalid ID\tnormalized\tphones\t9\tFEMALE\n",
        encoding="utf-8",
    )
    texts, counts = read_transcripts(path)
    assert texts == {"11": "One aligned sentence"}
    assert counts["rows_with_transcripts"] == 3
    assert counts["conflicting_sentence_ids_excluded"] == 1
    assert counts["usable_sentence_ids"] == 1
