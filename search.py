import json
from pathlib import Path

import numpy as np

from pipeline import aggregate_verdict, extract_claims

PAGE_SIZE = 5
MAX_RETRIES = 0
CONFIDENT = 0.9
MAX_WINDOW_CHARS = 1200
RERANK_POOL = 30
NLI_LABEL = {"SUPPORTED": "entailment", "CONTRADICTED": "contradiction"}


class PassageIndex:
    def __init__(self, passages, embeddings, encoder, reranker=None):
        self.passages = passages
        self.embeddings = embeddings
        self.encoder = encoder
        self.reranker = reranker

    @classmethod
    def build(cls, corpus_path, encoder, cache=None, reranker=None):
        """Index two-sentence windows of a SciFact-format corpus (JSONL with
        doc_id, title, abstract as a list of sentences)."""
        cache = Path(cache) if cache else None
        if cache and cache.with_suffix(".json").exists():
            passages = json.loads(cache.with_suffix(".json").read_text())
            return cls(passages, np.load(cache.with_suffix(".npy")), encoder, reranker)
        passages = []
        with open(corpus_path) as f:
            for line in f:
                doc = json.loads(line)
                sentences = doc["abstract"]
                for i in range(max(len(sentences) - 1, 1)):
                    text = " ".join(sentences[i:i + 2])
                    if len(text) <= MAX_WINDOW_CHARS:
                        passages.append({
                            "id": f"{doc['doc_id']}:{i}", "text": text,
                            "source": f"SciFact document {doc['doc_id']}: {doc['title']}",
                        })
        embeddings = encoder.encode(
            [p["text"] for p in passages], batch_size=128,
            normalize_embeddings=True, show_progress_bar=False,
        )
        if cache:
            cache.with_suffix(".json").write_text(json.dumps(passages))
            np.save(cache.with_suffix(".npy"), embeddings)
        return cls(passages, embeddings, encoder, reranker)

    def search(self, query, start, count):
        vector = self.encoder.encode(query, normalize_embeddings=True)
        order = np.argsort(-(self.embeddings @ vector))
        if self.reranker is None:
            return [dict(self.passages[i]) for i in order[start:start + count]]
        pool = [dict(self.passages[i]) for i in order[:RERANK_POOL]]
        scores = self.reranker.predict([(query, p["text"]) for p in pool])
        ranked = [p for _, p in sorted(zip(scores, pool), key=lambda pair: -pair[0])]
        return ranked[start:start + count]


def confidence(claim):
    label = NLI_LABEL.get(claim["verdict"])
    if label is None:
        return 0.0
    return max(item["nli_probabilities"][label] for item in claim["evidence"]
               if item["verdict"] == claim["verdict"])


def verify_with_search(pipeline, index, question, answer, max_retries=MAX_RETRIES):
    results = []
    for claim in extract_claims(pipeline.nlp(answer)):
        passages = []
        rounds = 0
        for _ in range(max_retries + 1):
            found = index.search(claim["text"], len(passages), PAGE_SIZE)
            if not found:
                stopped = "index exhausted"
                break
            passages += found
            rounds += 1
            result = pipeline.verify(question, claim["text"], passages)["claims"][0]
            if result["verdict"] == "CONFLICTING_EVIDENCE":
                stopped = "conflicting evidence found"
                break
            if confidence(result) >= CONFIDENT:
                stopped = "confident verdict"
                break
        else:
            stopped = "retry budget used"
        results.append({
            **result, **claim,
            "search": {"rounds": rounds, "passages_searched": len(passages),
                       "stopped": stopped},
        })
    return {"verdict": aggregate_verdict(results), "claims": results}
