"""Inference wrapper around the MIT-licensed M2M100 multilingual model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_ID = "facebook/m2m100_418M"
MODEL_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
LANGUAGE_CODES = {"en": "en", "am": "am"}


class M2M100Translator:
    """Translate sentences in either direction using M2M100 and an optional adapter."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str | None = MODEL_REVISION,
        adapter_path: str | Path | None = None,
        device: str = "cpu",
        max_source_tokens: int = 128,
        num_beams: int = 2,
    ) -> None:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.max_source_tokens = max_source_tokens
        self.num_beams = num_beams
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_id, **kwargs)
        model = M2M100ForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.adapter_path = str(adapter_path) if adapter_path else None
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
        self.model = model.to(self.device)
        self.model.eval()

    def translate_many(
        self,
        texts: list[str],
        source: str,
        target: str,
        batch_size: int = 16,
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Run batched beam search using M2M100's explicit language IDs."""
        if source not in LANGUAGE_CODES or target not in LANGUAGE_CODES or source == target:
            raise ValueError("source and target must be different values from {'en', 'am'}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        output: list[str] = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            self.tokenizer.src_lang = LANGUAGE_CODES[source]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_source_tokens,
            ).to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(LANGUAGE_CODES[target]),
                    num_beams=self.num_beams,
                    max_new_tokens=max_new_tokens,
                    early_stopping=True,
                )
            output.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return output
