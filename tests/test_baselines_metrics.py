import pytest

from ethiopian_translation.baselines import CopyBaseline, IBMModel1Lexicon, detokenize, tokenize
from ethiopian_translation.metrics import score_translations


def test_copy_baseline_is_identity() -> None:
    assert CopyBaseline().translate_many(["Hello!", "አማርኛ"], "en", "am") == ["Hello!", "አማርኛ"]


def test_tokenizer_and_detokenizer_preserve_unicode_punctuation() -> None:
    tokens = tokenize("Hello, አማርኛ!")
    assert tokens == ["Hello", ",", "አማርኛ", "!"]
    assert detokenize(tokens) == "Hello, አማርኛ!"


def test_ibm_model_1_learns_observed_lexical_pairs() -> None:
    pairs = [
        {"source": "hello world", "target": "ሰላም ዓለም"},
        {"source": "hello friend", "target": "ሰላም ጓደኛ"},
    ]
    model = IBMModel1Lexicon(iterations=8).fit(pairs, "source", "target")
    output = model.translate("hello")
    assert output in {"ሰላም", "ዓለም", "ጓደኛ"}
    assert model.translate("unknown") == "unknown"


def test_metrics_return_bleu_and_chrf_on_0_to_100_scale() -> None:
    result = score_translations(["ሰላም ዓለም"], ["ሰላም ዓለም"])
    assert result["n"] == 1
    assert result["bleu"] == pytest.approx(100.0)
    assert result["chrf++"] == pytest.approx(100.0)


def test_metrics_reject_misaligned_lists() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        score_translations(["reference"], [])
