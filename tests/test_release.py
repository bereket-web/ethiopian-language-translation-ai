from scripts.publish_hf import build_model_card, release_gate


def make_training(adapter_saved: bool = True) -> dict:
    return {
        "adapter_saved": adapter_saved,
        "best_epoch": 1 if adapter_saved else 0,
        "base_validation_mean_chrf++": 10.0,
        "best_validation_mean_chrf++": 11.0 if adapter_saved else 10.0,
        "dataset_split_counts": {"train": 100, "validation": 20},
    }


def make_benchmark(adapter_en: float = 12.0, adapter_am: float = 12.0) -> dict:
    return {
        "dataset": {"split": "test", "aligned_pairs_evaluated": 40},
        "results": {
            "M2M100 zero-shot": {
                "mean_chrf++": 10.0,
                "directions": {
                    "en->am": {"bleu": 5.0, "chrf++": 10.0},
                    "am->en": {"bleu": 6.0, "chrf++": 10.0},
                },
            },
            "M2M100 FLEURS LoRA": {
                "mean_chrf++": (adapter_en + adapter_am) / 2,
                "directions": {
                    "en->am": {"bleu": 7.0, "chrf++": adapter_en},
                    "am->en": {"bleu": 8.0, "chrf++": adapter_am},
                },
            },
        },
    }


def make_bootstrap(lower: float = 0.5, upper: float = 2.5, n: int = 40) -> dict:
    return {
        "split": "test",
        "n_aligned_pairs": n,
        "observed_macro_mean_delta": 2.0,
        "delta_ci95": {
            "macro_mean": {"lower_95": lower, "upper_95": upper},
            "en->am": {"lower_95": -0.2, "upper_95": 1.2},
            "am->en": {"lower_95": 1.0, "upper_95": 3.0},
        },
    }


def test_release_gate_accepts_validation_and_test_gain_with_positive_ci() -> None:
    allowed, reason = release_gate(make_training(), make_benchmark(), make_bootstrap())
    assert allowed
    assert "paired-bootstrap lower 95% bound" in reason


def test_release_gate_rejects_adapter_that_did_not_win_validation() -> None:
    allowed, reason = release_gate(
        make_training(adapter_saved=False), make_benchmark(), make_bootstrap()
    )
    assert not allowed
    assert "No LoRA adapter" in reason


def test_release_gate_rejects_non_improving_test_results() -> None:
    allowed, reason = release_gate(
        make_training(), make_benchmark(adapter_en=9.0, adapter_am=10.0), make_bootstrap()
    )
    assert not allowed
    assert "does not improve" in reason


def test_release_gate_rejects_large_directional_regression() -> None:
    allowed, reason = release_gate(
        make_training(), make_benchmark(adapter_en=7.0, adapter_am=14.0), make_bootstrap()
    )
    assert not allowed
    assert "declines by more than 2" in reason


def test_release_gate_rejects_ci_that_includes_zero() -> None:
    allowed, reason = release_gate(make_training(), make_benchmark(), make_bootstrap(lower=-0.1))
    assert not allowed
    assert "does not establish a positive" in reason


def test_release_gate_rejects_mismatched_pair_counts() -> None:
    allowed, reason = release_gate(make_training(), make_benchmark(), make_bootstrap(n=39))
    assert not allowed
    assert "pair counts do not match" in reason


def test_model_card_has_metrics_and_no_unfilled_placeholders() -> None:
    card = build_model_card(make_training(), make_benchmark(), make_bootstrap())
    assert "100 aligned train pairs" in card
    assert "40 aligned pairs" in card
    assert "M2M100 LoRA adapter | 7.00 / 12.00 | 8.00 / 12.00" in card
    assert "paired sentence-ID bootstrap 95% CI: [0.50, 2.50]" in card
    assert "English→Amharic [-0.20, +1.20] (includes zero)" in card
    assert "Amharic→English [+1.00, +3.00]" in card
    assert "bereket-web" not in card
    assert "{{" not in card
