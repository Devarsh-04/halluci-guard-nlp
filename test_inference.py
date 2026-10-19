import os

import pytest
from fastapi.testclient import TestClient

from api import app


pytestmark = pytest.mark.skipif(
    os.getenv("HALLUCIGUARD_RUN_MODEL_TESTS") != "1",
    reason="Set HALLUCIGUARD_RUN_MODEL_TESTS=1 to load and test real local models.",
)


@pytest.mark.parametrize("answer,expected", [
    ("Mars has two moons.", "SUPPORTED"),
    ("Mars has no moons.", "CONTRADICTED"),
    ("Mars has underground cities.", "INSUFFICIENT_EVIDENCE"),
])
def test_real_model_verification(answer, expected):
    with TestClient(app) as client:
        response = client.post("/api/v1/verify", json={
            "question": "How many moons does Mars have?",
            "answer": answer,
            "evidence": [
                {"id": "mars", "text": "Mars has two moons, Phobos and Deimos.",
                 "source": "Synthetic test fixture"},
                {"id": "jupiter", "text": "Jupiter is the largest planet.",
                 "source": "Synthetic distractor"},
            ],
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["claims"][0]["verified_evidence_id"] == "mars"
    assert result["verdict"] == expected, result


def test_distinct_claims_retrieve_their_own_evidence():
    answer = "Mars has two moons. Jupiter is the largest planet."
    with TestClient(app) as client:
        response = client.post("/api/v1/verify", json={
            "question": "Describe Mars and Jupiter.",
            "answer": answer,
            "evidence": [
                {"id": "mars", "text": "Mars has two moons, Phobos and Deimos."},
                {"id": "jupiter", "text": "Jupiter is the largest planet."},
            ],
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["verdict"] == "SUPPORTED", result
    assert [c["verified_evidence_id"] for c in result["claims"]] == [
        "mars", "jupiter",
    ]
    for claim in result["claims"]:
        assert answer[claim["start_char"]:claim["end_char"]] == claim["text"]


@pytest.mark.parametrize("second_claim,expected", [
    ("Mars has no moons.", "CONTRADICTED"),
    ("Mars has underground cities.", "INSUFFICIENT_EVIDENCE"),
])
def test_unsupported_second_claim_changes_overall_verdict(second_claim, expected):
    with TestClient(app) as client:
        response = client.post("/api/v1/verify", json={
            "question": "Describe Jupiter and Mars.",
            "answer": f"Jupiter is the largest planet. {second_claim}",
            "evidence": [
                {"id": "mars", "text": "Mars has two moons, Phobos and Deimos."},
                {"id": "jupiter", "text": "Jupiter is the largest planet."},
            ],
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["claims"][0]["verdict"] == "SUPPORTED", result
    assert result["claims"][1]["verdict"] == expected, result
    assert result["verdict"] == expected, result


def test_punctuation_only_answer_is_rejected_by_the_real_pipeline():
    with TestClient(app) as client:
        response = client.post("/api/v1/verify", json={
            "question": "Describe Mars.", "answer": "... !!!",
            "evidence": [{"id": "mars", "text": "Mars has two moons."}],
        })
    assert response.status_code == 422
    assert "verifiable" in response.json()["detail"]


def test_compound_claims_find_separate_support_and_contradiction():
    answer = "Mars has two moons and Jupiter is the smallest planet."
    with TestClient(app) as client:
        response = client.post("/api/v1/verify", json={
            "question": "Describe Mars and Jupiter.", "answer": answer,
            "evidence": [
                {"id": "mars", "text": "Mars has two moons, Phobos and Deimos."},
                {"id": "jupiter", "text": "Jupiter is the largest planet."},
            ],
        })
    assert response.status_code == 200, response.text
    result = response.json()
    claims = result["claims"]
    assert [claim["verdict"] for claim in claims] == [
        "SUPPORTED", "CONTRADICTED",
    ], result
    assert result["verdict"] == "CONTRADICTED"
    assert [claim["verified_evidence_id"] for claim in claims] == [
        "mars", "jupiter",
    ], result
    assert [
        {item["id"]: item["verdict"] for item in claim["evidence"]}
        for claim in claims
    ] == [
        {"mars": "SUPPORTED", "jupiter": "INSUFFICIENT_EVIDENCE"},
        {"mars": "INSUFFICIENT_EVIDENCE", "jupiter": "CONTRADICTED"},
    ], result
    assert all(claim["explanation"] for claim in claims)
    for claim in claims:
        assert answer[claim["start_char"]:claim["end_char"]] == claim["source_text"]


@pytest.mark.parametrize("reverse", [False, True])
def test_real_models_disclose_conflicting_sources(reverse):
    evidence = [
        {"id": "supports", "text": "Mars has two moons."},
        {"id": "contradicts", "text": "Mars has no moons."},
        {"id": "unrelated", "text": "Jupiter is the largest planet."},
    ]
    with TestClient(app) as client:
        response = client.post("/api/v1/verify", json={
            "question": "How many moons does Mars have?",
            "answer": "Mars has two moons.",
            "evidence": evidence[::-1] if reverse else evidence,
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["verdict"] == "CONFLICTING_EVIDENCE", result
    claim = result["claims"][0]
    verdicts = {item["id"]: item["verdict"] for item in claim["evidence"]}
    assert verdicts == {
        "supports": "SUPPORTED", "contradicts": "CONTRADICTED",
        "unrelated": "INSUFFICIENT_EVIDENCE",
    }, result
    assert claim["explanation"]


@pytest.mark.parametrize("answer,count,review", [
    ("🪐 Mars is red and has two moons.", 2, False),
    ("Mars may be red and have two moons.", 1, True),
])
def test_shared_subject_and_ambiguous_scope_through_the_api(answer, count, review):
    with TestClient(app) as client:
        response = client.post("/api/v1/verify", json={
            "question": "Describe Mars.", "answer": answer,
            "evidence": [
                {"id": "color", "text": "Mars is red."},
                {"id": "moons", "text": "Mars has two moons."},
            ],
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["claims"]) == count
    assert all(claim["review_required"] == review for claim in result["claims"])
    for claim in result["claims"]:
        assert answer[claim["start_char"]:claim["end_char"]] == claim["source_text"]
    if count == 2:
        assert result["claims"][1]["text"] == "Mars has two moons."
        assert result["verdict"] == "SUPPORTED", result
