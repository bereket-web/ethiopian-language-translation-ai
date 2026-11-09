from fastapi.testclient import TestClient

from app import app


class StubTranslator:
    def translate_many(self, texts, source, target, **kwargs):
        return [f"{source}->{target}: {texts[0]}"]


def test_translation_api_uses_opposite_target_language() -> None:
    app.state.translator = StubTranslator()
    client = TestClient(app)
    response = client.post("/api/translate", json={"text": "Hello", "source_language": "en"})
    assert response.status_code == 200
    assert response.json()["target_language"] == "am"
    assert response.json()["translation"] == "en->am: Hello"
    app.state.translator = None


def test_translation_api_rejects_empty_text_and_unknown_language() -> None:
    app.state.translator = StubTranslator()
    client = TestClient(app)
    empty = client.post("/api/translate", json={"text": "   ", "source_language": "en"})
    bad_language = client.post("/api/translate", json={"text": "Hello", "source_language": "fr"})
    assert empty.status_code == 422
    assert bad_language.status_code == 422
    app.state.translator = None
