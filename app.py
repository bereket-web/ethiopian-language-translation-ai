"""Local English–Amharic translation API and browser UI."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


class Translator(Protocol):
    def translate_many(
        self, texts: list[str], source: str, target: str, **kwargs: object
    ) -> list[str]: ...


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    source_language: str = Field(pattern="^(en|am)$")


class TranslationResponse(BaseModel):
    source_language: str
    target_language: str
    translation: str
    model: str


app = FastAPI(
    title="English–Amharic Translation AI",
    version="0.1.0",
    description="Research demo for bidirectional English–Amharic sentence translation.",
)
_translator: Translator | None = None
_translator_lock = threading.Lock()
app.state.translator = None


def _load_translator() -> Translator:
    """Load the model lazily; tests may inject a stub translator via app.state."""
    global _translator
    if app.state.translator is not None:
        return app.state.translator
    with _translator_lock:
        if _translator is None:
            from ethiopian_translation.model import M2M100Translator

            adapter_path = os.getenv("TRANSLATION_ADAPTER_PATH")
            _translator = M2M100Translator(
                adapter_path=adapter_path
                if adapter_path and os.path.exists(adapter_path)
                else None,
                device=os.getenv("TRANSLATION_DEVICE", "cpu"),
                max_source_tokens=int(os.getenv("TRANSLATION_MAX_SOURCE_TOKENS", "128")),
            )
    return _translator


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "facebook/m2m100_418M (optional local adapter)"}


@app.post("/api/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest) -> TranslationResponse:
    cleaned = " ".join(request.text.split())
    if not cleaned:
        raise HTTPException(status_code=422, detail="text must contain a non-space character")
    source = request.source_language
    target = "am" if source == "en" else "en"
    translator = _load_translator()
    try:
        output = translator.translate_many([cleaned], source, target, batch_size=1)[0]
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return TranslationResponse(
        source_language=source,
        target_language=target,
        translation=output,
        model="facebook/m2m100_418M (optional local adapter)",
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    template = Path(__file__).parent / "templates" / "index.html"
    return template.read_text(encoding="utf-8")
