import pytest
from fastapi.testclient import TestClient

import api
from api import app
from pipeline import EvidenceTooLong


class FixedModels:
    def verify(self, question, answer, evidence):
        scores = {
            "verdict": "SUPPORTED", "combined_score": 0.89,
            "ner_score": 1.0, "embedding_score": 0.8,
            "nli_label": "entailment", "nli_probabilities": {
                "contradiction": 0.05, "entailment": 0.9, "neutral": 0.05,
            },
        }
        return {
            "verdict": "SUPPORTED",
            "claims": [{
                **scores, "id": "claim_1", "text": answer, "source_text": answer,
                "start_char": 0, "end_char": len(answer),
                "review_required": False, "extraction_note": None,
                "explanation": "At least one supplied passage supports this claim.",
                "evidence": [{**evidence[0], **scores, "retrieval_score": 0.8,
                              "rerank_score": 4.0,
                              "explanation": "This passage supports the claim."}],
                "verified_evidence_id": evidence[0]["id"],
            }],
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "get_pipeline", FixedModels)
    with TestClient(app) as test_client:
        yield test_client


def payload():
    return {
        "question": "How many moons does Mars have?",
        "answer": "Mars has two moons.",
        "evidence": [{"id": "mars", "text": "Mars has two moons.",
                      "source": "User-supplied astronomy passage"}],
    }


def test_verification_returns_scores_and_evidence(client):
    response = client.post("/api/v1/verify", json=payload())
    assert response.status_code == 200
    result = response.json()
    assert result["verdict"] == "SUPPORTED"
    claim = result["claims"][0]
    assert claim["verified_evidence_id"] == "mars"
    assert claim["text"] == "Mars has two moons."
    assert claim["source_text"] == claim["text"]
    assert claim["evidence"][0]["verdict"] == "SUPPORTED"
    assert claim["explanation"]
    assert claim["evidence"][0]["source"] == "User-supplied astronomy passage"
    assert "confidence" not in result
    assert "combined_score" not in result


@pytest.mark.parametrize("field,value", [
    ("question", "   "), ("answer", ""), ("answer", "a" * 2001),
    ("evidence", []), ("evidence", [{"id": "x", "text": "  "}]),
])
def test_invalid_requests_are_rejected(client, field, value):
    body = payload()
    body[field] = value
    assert client.post("/api/v1/verify", json=body).status_code == 422


def test_duplicate_evidence_ids_are_rejected(client):
    body = payload()
    body["evidence"] *= 2
    assert client.post("/api/v1/verify", json=body).status_code == 422


def test_model_input_limit_becomes_validation_error(client, monkeypatch):
    class OversizeInput:
        def verify(self, question, answer, evidence):
            raise EvidenceTooLong("Evidence and answer exceed the NLI token limit.")

    monkeypatch.setattr(api, "get_pipeline", OversizeInput)
    response = client.post("/api/v1/verify", json=payload())
    assert response.status_code == 422
    assert "token limit" in response.json()["detail"]


def test_health_and_openapi_do_not_require_model_downloads():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        schema = client.get("/openapi.json").json()
        assert "/api/v1/verify" in schema["paths"]


def test_interface_and_assets_are_served_without_loading_models(monkeypatch):
    def unexpected_model_load():
        pytest.fail("Opening the interface must not load models.")

    monkeypatch.setattr(api, "get_pipeline", unexpected_model_load)
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "HalluciGuard" in response.text
        for asset in ("app.js", "styles.css"):
            assert client.get(f"/static/{asset}").status_code == 200


def test_invalid_request_does_not_load_models(monkeypatch):
    def unexpected_model_load():
        pytest.fail("Invalid requests must be rejected before loading models.")

    monkeypatch.setattr(api, "get_pipeline", unexpected_model_load)
    with TestClient(app) as client:
        assert client.post("/api/v1/verify", json={}).status_code == 422


def test_internal_model_errors_are_not_reported_as_bad_user_input(monkeypatch):
    class BrokenModel:
        def verify(self, question, answer, evidence):
            raise ValueError("Internal model configuration error")

    monkeypatch.setattr(api, "get_pipeline", BrokenModel)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/verify", json=payload())
    assert response.status_code == 500
    assert "configuration" not in response.text


def test_evidence_urls_pass_through_and_must_be_web_links(client):
    body = payload()
    body["evidence"][0]["url"] = "https://immi.homeaffairs.gov.au/visas"
    response = client.post("/api/v1/verify", json=body)
    assert response.status_code == 200
    assert response.json()["claims"][0]["evidence"][0]["url"] == "https://immi.homeaffairs.gov.au/visas"
    body["evidence"][0]["url"] = "javascript:alert(1)"
    assert client.post("/api/v1/verify", json=body).status_code == 422
