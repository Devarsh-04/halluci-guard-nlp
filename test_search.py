import pytest
from fastapi.testclient import TestClient

import api
from search import verify_with_search


@pytest.fixture(scope="module")
def nlp():
    import spacy

    return spacy.load("en_core_web_sm")


class StubIndex:
    def __init__(self, size):
        self.size = size
        self.queries = []

    def search(self, query, start, count):
        self.queries.append((query, start, count))
        return [{"id": f"p{i}", "text": f"passage {i}", "source": "stub"}
                for i in range(start, min(start + count, self.size))]


class StubPipeline:
    """Returns a verdict decided by how many passages it has been given."""

    def __init__(self, nlp, by_passage_count):
        self.nlp = nlp
        self.by_passage_count = by_passage_count

    def verify(self, question, claim, evidence):
        verdict, probability = self.by_passage_count(len(evidence))
        label = {"SUPPORTED": "entailment", "CONTRADICTED": "contradiction"}.get(verdict)
        probabilities = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
        if label:
            probabilities[label] = probability
        return {"claims": [{
            "verdict": verdict, "nli_probabilities": probabilities,
            "evidence": [{"verdict": verdict, "nli_probabilities": probabilities}],
        }]}


def run(nlp, by_passage_count, index_size=100, **kwargs):
    index = StubIndex(index_size)
    pipeline = StubPipeline(nlp, by_passage_count)
    return verify_with_search(pipeline, index, "q", "Mars has two moons.", **kwargs), index


def test_confident_verdict_stops_after_one_round(nlp):
    result, index = run(nlp, lambda n: ("SUPPORTED", 0.97))
    trace = result["claims"][0]["search"]
    assert (trace["rounds"], trace["stopped"]) == (1, "confident verdict")
    assert len(index.queries) == 1


def test_insufficient_evidence_retries_deeper_until_budget(nlp):
    result, index = run(nlp, lambda n: ("INSUFFICIENT_EVIDENCE", 0.0), max_retries=2)
    claim = result["claims"][0]
    assert claim["search"] == {
        "rounds": 3, "passages_searched": 15, "stopped": "retry budget used",
    }
    assert [start for _, start, _ in index.queries] == [0, 5, 10]
    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_retry_finds_evidence_in_later_page(nlp):
    result, _ = run(
        nlp, lambda n: ("SUPPORTED", 0.95) if n >= 10 else ("INSUFFICIENT_EVIDENCE", 0.0),
        max_retries=2,
    )
    claim = result["claims"][0]
    assert claim["verdict"] == "SUPPORTED"
    assert claim["search"]["rounds"] == 2


def test_low_confidence_verdict_keeps_searching(nlp):
    result, _ = run(nlp, lambda n: ("CONTRADICTED", 0.7), max_retries=2)
    assert result["claims"][0]["search"]["rounds"] == 3


def test_conflict_stops_search(nlp):
    result, _ = run(nlp, lambda n: ("CONFLICTING_EVIDENCE", 0.0))
    assert result["claims"][0]["search"]["stopped"] == "conflicting evidence found"


def test_default_does_not_retry(nlp):
    result, index = run(nlp, lambda n: ("INSUFFICIENT_EVIDENCE", 0.0))
    assert result["claims"][0]["search"]["rounds"] == 1
    assert len(index.queries) == 1


def test_zero_retries_searches_once(nlp):
    result, index = run(nlp, lambda n: ("INSUFFICIENT_EVIDENCE", 0.0), max_retries=0)
    assert result["claims"][0]["search"]["rounds"] == 1
    assert len(index.queries) == 1


def test_small_index_stops_when_exhausted(nlp):
    result, _ = run(nlp, lambda n: ("INSUFFICIENT_EVIDENCE", 0.0), index_size=7, max_retries=2)
    assert result["claims"][0]["search"] == {
        "rounds": 2, "passages_searched": 7, "stopped": "index exhausted",
    }


def test_search_endpoint_is_off_without_a_corpus(monkeypatch):
    monkeypatch.delenv("HALLUCIGUARD_CORPUS", raising=False)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/verify-search", json={"question": "q", "answer": "Mars has two moons."}
        )
    assert response.status_code == 503
    assert "HALLUCIGUARD_CORPUS" in response.json()["detail"]


def test_search_endpoint_returns_trace(monkeypatch):
    from test_api import FixedModels

    def fake_search(pipeline, index, question, answer, max_retries):
        result = FixedModels().verify(question, answer, [{"id": "p1", "text": "t", "source": "s"}])
        result["claims"][0]["search"] = {
            "rounds": 2, "passages_searched": 10, "stopped": "confident verdict",
        }
        return result

    monkeypatch.setattr(api, "get_index", lambda: None)
    monkeypatch.setattr(api, "get_pipeline", lambda: None)
    monkeypatch.setattr(api, "verify_with_search", fake_search)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/verify-search", json={"question": "q", "answer": "Mars has two moons."}
        )
    assert response.status_code == 200
    assert response.json()["claims"][0]["search"]["rounds"] == 2
