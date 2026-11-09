import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ethiopian_translation import training


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def get_lang_id(self, language):
        return 1 if language == "en" else 2

    def pad(self, examples, **kwargs):
        return {
            "input_ids": torch.tensor([example["input_ids"] for example in examples]),
            "attention_mask": torch.tensor([example["attention_mask"] for example in examples]),
        }

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer-placeholder.txt").write_text("test only\n", encoding="utf-8")


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.config = SimpleNamespace(use_cache=True)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def forward(self, **kwargs):
        return SimpleNamespace(loss=self.weight.square())

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter-placeholder.txt").write_text("test only\n", encoding="utf-8")


def test_completed_epoch_report_survives_later_interruption(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "M2M100Tokenizer", FakeTokenizer)
    monkeypatch.setattr(training, "M2M100ForConditionalGeneration", FakeModel)
    monkeypatch.setattr(training, "LoraConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(training, "get_peft_model", lambda model, config: model)

    scores = iter(
        [
            {"mean_chrf++": 10.0, "directions": {"en->am": 10.0, "am->en": 10.0}},
            {"mean_chrf++": 11.0, "directions": {"en->am": 11.0, "am->en": 11.0}},
            RuntimeError("simulated interruption during epoch 2 validation"),
        ]
    )

    def fake_validation(*args, **kwargs):
        result = next(scores)
        if isinstance(result, Exception):
            raise result
        return {**result, "n_aligned_validation_pairs": 1}

    monkeypatch.setattr(training, "_translate_for_validation", fake_validation)
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    row = {"id": "one", "english": "hello", "amharic": "ሰላም"}
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps({**row, "id": "validation-one"}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train(
            train_path,
            validation_path,
            tmp_path / "adapter",
            epochs=2,
            batch_size=2,
            gradient_accumulation=1,
            max_length=8,
            threads=1,
        )

    report = json.loads((tmp_path / "runs/training/validation.json").read_text())
    assert report["epochs_requested"] == 2
    assert report["epochs_completed"] == 1
    assert report["run_complete"] is False
    assert report["best_epoch"] == 1
    assert report["adapter_saved"] is True
    assert report["base_validation_mean_chrf++"] == 10.0
    assert report["best_validation_mean_chrf++"] == 11.0
