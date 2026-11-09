#!/usr/bin/env python3
"""Stage or publish the validation-selected M2M100 adapter without source text."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

OWNER = "klohdfhk"
REPO_ID = f"{OWNER}/ethiopian-language-translation-ai"
BASE_MODEL = "facebook/m2m100_418M"
BASE_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
DATASET_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
ADAPTER_DIR = Path("models/m2m100-fleurs-am-en")
VALIDATION_REPORT = Path("runs/training/validation.json")
TEST_REPORT = Path("reports/test_benchmark.json")
BOOTSTRAP_REPORT = Path("reports/test_bootstrap.json")
STAGING_DIR = Path("release/huggingface")


def release_gate(
    training: dict[str, Any], benchmark: dict[str, Any], bootstrap: dict[str, Any]
) -> tuple[bool, str]:
    """Require validation and test gains with positive paired-bootstrap evidence."""
    if not training.get("adapter_saved") or int(training.get("best_epoch", 0)) < 1:
        return False, "No LoRA adapter improved on zero-shot validation chrF++."
    validation_base = float(training.get("base_validation_mean_chrf++", 0.0))
    validation_best = float(training.get("best_validation_mean_chrf++", 0.0))
    if validation_best <= validation_base:
        return False, "The adapter did not improve validation chrF++ over zero-shot."
    if benchmark.get("dataset", {}).get("split") != "test":
        return False, "A full held-out test benchmark is required before release."
    if bootstrap.get("split") != "test":
        return False, "A paired-bootstrap report on the held-out test set is required."
    if int(bootstrap.get("n_aligned_pairs", -1)) != int(
        benchmark.get("dataset", {}).get("aligned_pairs_evaluated", -2)
    ):
        return False, "Bootstrap and benchmark pair counts do not match."
    results = benchmark.get("results", {})
    base = results.get("M2M100 zero-shot")
    adapter = results.get("M2M100 FLEURS LoRA")
    if not base or not adapter:
        return False, "The test report does not include both base and adapter results."
    gain = float(adapter.get("mean_chrf++", 0.0)) - float(base.get("mean_chrf++", 0.0))
    if gain <= 0:
        return False, f"The adapter does not improve held-out mean chrF++ (gain={gain:.3f})."
    base_directions = base.get("directions", {})
    adapter_directions = adapter.get("directions", {})
    for direction in ("en->am", "am->en"):
        if direction not in base_directions or direction not in adapter_directions:
            return False, f"Missing held-out direction: {direction}."
        delta = float(adapter_directions[direction]["chrf++"]) - float(
            base_directions[direction]["chrf++"]
        )
        if delta < -2.0:
            return False, f"Adapter declines by more than 2 chrF++ points in {direction}."
    lower_bound = float(bootstrap.get("delta_ci95", {}).get("macro_mean", {}).get("lower_95", 0.0))
    if lower_bound <= 0:
        return False, "The paired-bootstrap 95% CI does not establish a positive macro chrF++ gain."
    observed = float(bootstrap.get("observed_macro_mean_delta", 0.0))
    if abs(gain - observed) > 0.1:
        return False, "Bootstrap and benchmark aggregate gains are inconsistent."
    return True, (
        f"Adapter passes release gate: mean test chrF++ gain {gain:.3f}, "
        f"paired-bootstrap lower 95% bound {lower_bound:.3f}."
    )


def _format_score(results: dict[str, Any], name: str, direction: str) -> str:
    score = results[name]["directions"][direction]
    return f"{score['bleu']:.2f} / {score['chrf++']:.2f}"


def build_model_card(
    training: dict[str, Any], benchmark: dict[str, Any], bootstrap: dict[str, Any]
) -> str:
    """Render a model card containing only aggregate metrics and source attribution."""
    results = benchmark["results"]
    en_base = _format_score(results, "M2M100 zero-shot", "en->am")
    am_base = _format_score(results, "M2M100 zero-shot", "am->en")
    en_adapter = _format_score(results, "M2M100 FLEURS LoRA", "en->am")
    am_adapter = _format_score(results, "M2M100 FLEURS LoRA", "am->en")
    test_n = benchmark["dataset"].get("aligned_pairs_evaluated", "unknown")
    validation_n = training.get("dataset_split_counts", {}).get("validation", "unknown")
    training_n = training.get("dataset_split_counts", {}).get("train", "unknown")
    epoch = training.get("best_epoch", "unknown")
    gain = float(results["M2M100 FLEURS LoRA"]["mean_chrf++"]) - float(
        results["M2M100 zero-shot"]["mean_chrf++"]
    )
    confidence_interval = bootstrap["delta_ci95"]["macro_mean"]
    values = {
        "{{BASE_MODEL}}": BASE_MODEL,
        "{{BASE_REVISION}}": BASE_REVISION,
        "{{DATASET_REVISION}}": DATASET_REVISION,
        "{{REPO_ID}}": REPO_ID,
        "{{EN_BASE}}": en_base,
        "{{AM_BASE}}": am_base,
        "{{EN_ADAPTER}}": en_adapter,
        "{{AM_ADAPTER}}": am_adapter,
        "{{GAIN}}": f"{gain:+.2f}",
        "{{CI_LOW}}": f"{confidence_interval['lower_95']:.2f}",
        "{{CI_HIGH}}": f"{confidence_interval['upper_95']:.2f}",
        "{{CI_EN_LOW}}": f"{bootstrap['delta_ci95']['en->am']['lower_95']:+.2f}",
        "{{CI_EN_HIGH}}": f"{bootstrap['delta_ci95']['en->am']['upper_95']:+.2f}",
        "{{CI_AM_LOW}}": f"{bootstrap['delta_ci95']['am->en']['lower_95']:+.2f}",
        "{{CI_AM_HIGH}}": f"{bootstrap['delta_ci95']['am->en']['upper_95']:+.2f}",
        "{{TRAIN_N}}": str(training_n),
        "{{VALIDATION_N}}": str(validation_n),
        "{{TEST_N}}": str(test_n),
        "{{BEST_EPOCH}}": str(epoch),
    }
    template_path = Path(__file__).resolve().parents[1] / "docs" / "model_card_template.md"
    card = template_path.read_text(encoding="utf-8")
    for marker, value in values.items():
        card = card.replace(marker, value)
    return card


def stage_release(
    adapter_dir: Path = ADAPTER_DIR,
    validation_path: Path = VALIDATION_REPORT,
    test_path: Path = TEST_REPORT,
    bootstrap_path: Path = BOOTSTRAP_REPORT,
    output_dir: Path = STAGING_DIR,
) -> tuple[bool, str]:
    training = json.loads(validation_path.read_text(encoding="utf-8"))
    benchmark = json.loads(test_path.read_text(encoding="utf-8"))
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    allowed, reason = release_gate(training, benchmark, bootstrap)
    if not allowed:
        return False, reason
    required = [adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors"]
    if not all(path.is_file() for path in required):
        return False, "The selected adapter files are incomplete; refusing to stage a release."
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(adapter_dir / filename, output_dir / filename)
    (output_dir / "README.md").write_text(
        build_model_card(training, benchmark, bootstrap), encoding="utf-8"
    )
    return True, reason


def publish(repo_id: str = REPO_ID) -> str:
    """Create a new public repo and upload only the card plus LoRA adapter files."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN to a separate Project 18 write token before publishing.")
    if repo_id != REPO_ID:
        raise RuntimeError(f"Publication is restricted to the reviewed repo ID {REPO_ID}.")
    allowed, reason = stage_release()
    if not allowed:
        raise RuntimeError(f"Release gate failed: {reason}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    identity = api.whoami()
    username = str(identity.get("name", "")).lower()
    if username != OWNER.lower():
        raise RuntimeError(
            f"Token belongs to {username or 'an unknown account'}, not the approved owner."
        )
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=False)
    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish evaluated Amharic-English LoRA adapter and model card",
        allow_patterns=["README.md", "adapter_config.json", "adapter_model.safetensors"],
    )
    info = api.model_info(repo_id, files_metadata=True)
    expected = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
    uploaded = {item.rfilename for item in info.siblings}
    if not expected.issubset(uploaded):
        raise RuntimeError("Hub verification failed: one or more required model files are missing.")
    return f"https://huggingface.co/{repo_id} (revision {info.sha})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage locally, but do not create or upload a Hub repo",
    )
    args = parser.parse_args()
    allowed, reason = stage_release()
    print(reason)
    if not allowed:
        raise SystemExit(2)
    print(f"Staged only README.md and adapter files in {STAGING_DIR}")
    if args.dry_run:
        print("Dry run: no Hugging Face repository was created or changed.")
        return
    print(publish(args.repo_id))


if __name__ == "__main__":
    main()
