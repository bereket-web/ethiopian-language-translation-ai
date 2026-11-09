"""Validation-selected LoRA fine-tuning for the M2M100 translation model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from sacrebleu.metrics import CHRF
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from ethiopian_translation.data import DATASET_REVISION, load_pairs
from ethiopian_translation.model import MODEL_ID, MODEL_REVISION


def _directions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for row in rows:
        examples.extend(
            [
                {"source": "en", "target": "am", "text": row["english"], "label": row["amharic"]},
                {"source": "am", "target": "en", "text": row["amharic"], "label": row["english"]},
            ]
        )
    return examples


def _encode_examples(
    tokenizer: M2M100Tokenizer,
    examples: list[dict[str, str]],
    max_length: int,
) -> list[dict[str, list[int]]]:
    encoded: list[dict[str, list[int]]] = []
    for example in examples:
        tokenizer.src_lang = example["source"]
        source = tokenizer(example["text"], truncation=True, max_length=max_length)
        tokenizer.src_lang = example["target"]
        target = tokenizer(example["label"], truncation=True, max_length=max_length)
        encoded.append(
            {
                "input_ids": source["input_ids"],
                "attention_mask": source["attention_mask"],
                "labels": target["input_ids"],
                "source": [tokenizer.get_lang_id(example["source"])],
                "target": [tokenizer.get_lang_id(example["target"])],
            }
        )
    return encoded


def _batch(
    examples: list[dict[str, list[int]]], tokenizer: M2M100Tokenizer
) -> dict[str, torch.Tensor]:
    inputs = tokenizer.pad(
        [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in examples
        ],
        padding=True,
        return_tensors="pt",
    )
    label_tensors = [torch.tensor(item["labels"], dtype=torch.long) for item in examples]
    labels = torch.nn.utils.rnn.pad_sequence(label_tensors, batch_first=True, padding_value=-100)
    return {**inputs, "labels": labels}


def _translate_for_validation(
    model: Any,
    tokenizer: M2M100Tokenizer,
    rows: list[dict[str, str]],
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    directional_scores: dict[str, float] = {}
    directions = (("en", "am", "english", "amharic"), ("am", "en", "amharic", "english"))
    metric = CHRF(word_order=2)
    model.eval()
    for source, target, source_key, target_key in directions:
        texts = [row[source_key] for row in rows]
        references = [row[target_key] for row in rows]
        hypotheses: list[str] = []
        tokenizer.src_lang = source
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.get_lang_id(target),
                    num_beams=2,
                    max_new_tokens=max_length,
                    early_stopping=True,
                )
            hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        directional_scores[f"{source}->{target}"] = float(
            metric.corpus_score(hypotheses, [references]).score
        )
    return {
        "directions": {key: round(value, 4) for key, value in directional_scores.items()},
        "mean_chrf++": round(sum(directional_scores.values()) / len(directional_scores), 4),
        "n_aligned_validation_pairs": len(rows),
    }


def train(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    model_id: str = MODEL_ID,
    model_revision: str | None = None,
    epochs: int = 2,
    batch_size: int = 16,
    gradient_accumulation: int = 1,
    max_length: int = 128,
    learning_rate: float = 2e-4,
    threads: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit an adapter and retain the epoch with the highest mean validation chrF++."""
    if epochs < 1 or batch_size < 1 or gradient_accumulation < 1 or max_length < 8:
        raise ValueError("epochs, batch size, accumulation, and max length must be positive")
    train_rows = load_pairs(train_path)
    validation_rows = load_pairs(validation_path)
    if set(row["id"] for row in train_rows).intersection(row["id"] for row in validation_rows):
        raise ValueError("train and validation sentence IDs overlap")
    if not train_rows or not validation_rows:
        raise ValueError("both train and validation data must be non-empty")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    revision = model_revision or MODEL_REVISION
    if not revision:
        raise RuntimeError(f"Could not resolve immutable revision for {model_id}")
    tokenizer = M2M100Tokenizer.from_pretrained(model_id, revision=revision)
    base_model = M2M100ForConditionalGeneration.from_pretrained(model_id, revision=revision)
    adapter_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = get_peft_model(base_model, adapter_config)
    model.to(torch.device("cpu"))
    model.config.use_cache = False
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA did not expose trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    train_examples = _encode_examples(tokenizer, _directions(train_rows), max_length)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path("runs/training")
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    start = time.perf_counter()

    def write_validation_report() -> dict[str, Any]:
        report = {
            "base_model": model_id,
            "base_model_revision": revision,
            "dataset_revision": DATASET_REVISION,
            "dataset_license": "CC BY 4.0",
            "dataset_split_counts": {"train": len(train_rows), "validation": len(validation_rows)},
            "validation_decoding": {
                "num_beams": 2,
                "batch_size": batch_size,
                "max_new_tokens": max_length,
            },
            "train_directions": len(train_examples),
            "lora": {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "max_length": max_length,
            "seed": seed,
            "epochs_requested": epochs,
            "epochs_completed": max(0, len(history) - 1),
            "run_complete": max(0, len(history) - 1) >= epochs,
            "best_epoch": best_epoch,
            "base_validation_mean_chrf++": history[0]["mean_chrf++"],
            "best_validation_mean_chrf++": best_score,
            "adapter_saved": best_epoch > 0,
            "elapsed_seconds": round(time.perf_counter() - start, 2),
            "history": history,
            "test_split_used_for_selection": False,
        }
        temporary = run_dir / "validation.json.tmp"
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(run_dir / "validation.json")
        return report

    # Validation before updates is the zero-shot M2M100 score on the same dev set.
    initial = _translate_for_validation(model, tokenizer, validation_rows, batch_size, max_length)
    initial["epoch"] = 0
    initial["checkpoint"] = "base model before fine-tuning"
    history.append(initial)
    best_score = initial["mean_chrf++"]
    print(f"epoch=0 base validation mean chrF++={best_score:.4f}", flush=True)
    write_validation_report()

    total_steps = (len(train_examples) + batch_size - 1) // batch_size
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for batch_number, start_index in enumerate(range(0, len(order), batch_size), start=1):
            records = [
                train_examples[index] for index in order[start_index : start_index + batch_size]
            ]
            batch = _batch(records, tokenizer)
            result = model(**batch)
            loss = result.loss
            running_loss += float(loss.detach())
            (loss / gradient_accumulation).backward()
            final_batch = batch_number == total_steps
            if batch_number % gradient_accumulation == 0 or final_batch:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_number % 100 == 0 or final_batch:
                print(
                    f"epoch={epoch} batch={batch_number}/{total_steps} "
                    f"mean_loss={running_loss / batch_number:.4f}",
                    flush=True,
                )
        validation = _translate_for_validation(
            model, tokenizer, validation_rows, batch_size, max_length
        )
        validation.update(
            {
                "epoch": epoch,
                "mean_training_batch_loss": round(running_loss / total_steps, 6),
                "checkpoint": "LoRA adapter",
            }
        )
        history.append(validation)
        score = float(validation["mean_chrf++"])
        print(f"epoch={epoch} validation mean chrF++={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            (output_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "base_model": model_id,
                        "base_model_revision": revision,
                        "dataset": "google/fleurs",
                        "dataset_revision": DATASET_REVISION,
                        "dataset_license": "CC BY 4.0",
                        "best_validation": validation,
                        "seed": seed,
                        "epochs_requested": epochs,
                        "max_length": max_length,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        write_validation_report()

    report = write_validation_report()
    print(f"Training complete; best epoch={best_epoch}; adapter saved={best_epoch > 0}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/m2m100-fleurs-am-en"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        args.train,
        args.validation,
        args.output_dir,
        model_id=args.model_id,
        model_revision=args.model_revision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        threads=args.threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
