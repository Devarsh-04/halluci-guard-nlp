import pytest
from fastapi.testclient import TestClient

import api
from search import PAGE_SIZE, PassageIndex, verify_with_search


@pytest.fixture(scope="module")
def nlp():
    import spacy

    return spacy.load("en_core_web_sm")


class StubIndex:
    def __init__(self, size):
        self.size = size
        self.queries = []

    def topics_in(self, text):
        return set()

    def search(self, query, start, count, topics=()):
        self.queries.append((query, start, count))
        return [{"id": f"p{i}", "text": f"passage {i}", "source": "stub"}
                for i in range(start, min(start + count, self.size))]


class StubPipeline:
    """Returns a verdict decided by how many passages it has been given."""

    def __init__(self, nlp, by_passage_count):
        self.nlp = nlp
        self.by_passage_count = by_passage_count

    def verify(self, question, claim, evidence, retrieved=False):
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
        "rounds": 3, "passages_searched": 3 * PAGE_SIZE, "stopped": "retry budget used",
    }
    assert [start for _, start, _ in index.queries] == [0, PAGE_SIZE, 2 * PAGE_SIZE]
    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_retry_finds_evidence_in_later_page(nlp):
    result, _ = run(
        nlp, lambda n: ("SUPPORTED", 0.95) if n > PAGE_SIZE else ("INSUFFICIENT_EVIDENCE", 0.0),
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
        "rounds": 1, "passages_searched": 7, "stopped": "index exhausted",
    }


def test_search_uses_the_visa_knowledge_base_by_default(monkeypatch):
    built = []
    monkeypatch.delenv("HALLUCIGUARD_CORPUS", raising=False)
    monkeypatch.setattr(api, "_index", None)
    monkeypatch.setattr(api, "get_pipeline", lambda: type("P", (), {"encoder": None})())
    monkeypatch.setattr(api.PassageIndex, "build",
                        lambda corpus, encoder, cache=None: built.append(corpus) or "index")
    assert api.get_index() == "index"
    assert built == [api.DEFAULT_CORPUS]
    assert all(path.exists() for path in api.DEFAULT_CORPUS)


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


class WordEncoder:
    """Deterministic bag-of-words stand-in for the sentence encoder."""

    VOCAB = ["mars", "moons", "jupiter", "planet", "water", "ocean"]

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        import numpy as np

        self.calls += 1
        single = isinstance(texts, str)
        rows = np.array([[float(word in text.lower()) for word in self.VOCAB]
                         for text in ([texts] if single else texts)]) + 1e-6
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        return rows[0] if single else rows


@pytest.fixture
def corpus(tmp_path):
    import json

    docs = [
        {"doc_id": 1, "title": "Mars", "abstract": [
            "Mars has two moons.", "They are Phobos and Deimos.", "Mars is red."]},
        {"doc_id": 2, "title": "Jupiter", "abstract": ["Jupiter is the largest planet."]},
        {"doc_id": 3, "title": "Long", "abstract": ["water " * 300, "water " * 300]},
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(doc) for doc in docs))
    return path


def test_index_builds_two_sentence_windows(corpus):
    index = PassageIndex.build(corpus, WordEncoder())
    assert [p["id"] for p in index.passages] == ["1:0", "1:1", "2:0"]
    assert index.passages[0]["text"] == "Mars has two moons. They are Phobos and Deimos."
    assert index.passages[2]["source"] == "Jupiter (2, part 1)"


def test_index_search_ranks_and_pages(corpus):
    index = PassageIndex.build(corpus, WordEncoder())
    assert index.search("Which planet is largest, Jupiter?", 0, 1)[0]["id"] == "2:0"
    first, second = index.search("Mars moons", 0, 1), index.search("Mars moons", 1, 1)
    assert first[0]["id"] != second[0]["id"]
    assert index.search("Mars moons", 5, 3) == []


def test_index_cache_skips_encoding(corpus, tmp_path):
    cache = tmp_path / "passage_index"
    first = WordEncoder()
    built = PassageIndex.build(corpus, first, cache=cache)
    second = WordEncoder()
    loaded = PassageIndex.build(corpus, second, cache=cache)
    assert first.calls == 1 and second.calls == 0
    assert loaded.passages == built.passages


def test_server_builds_the_index_at_startup(monkeypatch):
    built = []
    monkeypatch.setenv("HALLUCIGUARD_CORPUS", "corpus.jsonl")
    monkeypatch.setattr(api, "get_index", lambda: built.append(True))
    with TestClient(api.app):
        assert built == [True]


def test_official_passages_carry_publisher_url_and_date(tmp_path):
    import json

    path = tmp_path / "official.jsonl"
    path.write_text(json.dumps({
        "doc_id": "ha-1", "title": "Student visa (subclass 500): Stay",
        "abstract": ["Up to 6 years and in line with your enrolment."],
        "publisher": "Department of Home Affairs", "url": "https://example.gov.au/500",
        "retrieved": "2026-10-22",
    }) + "\n")
    index = PassageIndex.build([path], WordEncoder())
    assert index.passages[0]["source"] == (
        "Department of Home Affairs: Student visa (subclass 500): Stay (ha-1, part 1), "
        "https://example.gov.au/500, retrieved 2026-10-22"
    )


def test_search_is_limited_to_the_visa_the_question_names(tmp_path):
    import json

    path = tmp_path / "kb.jsonl"
    path.write_text("".join(json.dumps(doc) + "\n" for doc in [
        {"doc_id": "a", "title": "Student", "abstract": ["Mars moons for students."],
         "subclass": "500", "aliases": ["student visa"]},
        {"doc_id": "b", "title": "Visitor", "abstract": ["Mars moons for visitors."],
         "subclass": "600", "aliases": ["visitor visa"]},
    ]))
    index = PassageIndex.build(path, WordEncoder())
    assert index.topics_in("Can I work on a Student Visa?") == {"500"}
    assert index.topics_in("Mars moons") == set()
    assert [p["id"] for p in index.search("Mars moons", 0, 5, {"600"})] == ["b:0"]
    assert "topic" not in index.passages[0] and "aliases" not in index.passages[0]
