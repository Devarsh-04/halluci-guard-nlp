import json
import re
from pathlib import Path

import numpy as np

from pipeline import aggregate_verdict, extract_claims

PAGE_SIZE = 15
MAX_RETRIES = 0
CONFIDENT = 0.9
MAX_WINDOW_CHARS = 1200
NLI_LABEL = {"SUPPORTED": "entailment", "CONTRADICTED": "contradiction"}


def source_label(doc, part):
    label = f"{doc['title']} ({doc['doc_id']}, part {part + 1})"
    if "publisher" in doc:
        label = f"{doc['publisher']}: {label}"
    if "url" in doc:
        label += f", {doc['url']}, retrieved {doc['retrieved']}"
    return label


class PassageIndex:
    def __init__(self, passages, embeddings, encoder):
        self.passages = passages
        self.embeddings = embeddings
        self.encoder = encoder
        self.topics = [p.pop("topic", None) for p in passages]
        self.aliases = {}
        for p, topic in zip(passages, self.topics):
            for alias in p.pop("aliases", []):
                self.aliases.setdefault(alias, set()).add(topic)

    def topics_in(self, text):
        """Subclasses whose aliases appear in the text; empty when none do."""
        text = text.lower()
        return set().union(*(topics for alias, topics in self.aliases.items()
                             if re.search(rf"\b{re.escape(alias)}\b", text)))

    @classmethod
    def build(cls, corpus_path, encoder, cache=None):
        """Index two-sentence windows of a JSONL corpus with doc_id, title and
        abstract (a list of sentences), the SciFact format."""
        cache = Path(cache) if cache else None
        if cache and cache.with_suffix(".json").exists():
            passages = json.loads(cache.with_suffix(".json").read_text())
            return cls(passages, np.load(cache.with_suffix(".npy")), encoder)
        paths = [corpus_path] if isinstance(corpus_path, (str, Path)) else corpus_path
        passages = []
        for path in paths:
            with open(path) as f:
                lines = f.readlines()
            for line in lines:
                doc = json.loads(line)
                sentences = doc["abstract"]
                for i in range(max(len(sentences) - 1, 1)):
                    text = " ".join(sentences[i:i + 2])
                    if len(text) <= MAX_WINDOW_CHARS:
                        passages.append({
                            "id": f"{doc['doc_id']}:{i}", "text": text,
                            "source": source_label(doc, i),
                            "topic": doc.get("subclass"),
                            "aliases": doc.get("aliases", []),
                        })
        embeddings = encoder.encode(
            [p["text"] for p in passages], batch_size=128,
            normalize_embeddings=True, show_progress_bar=False,
        )
        if cache:
            cache.with_suffix(".json").write_text(json.dumps(passages))
            np.save(cache.with_suffix(".npy"), embeddings)
        return cls(passages, embeddings, encoder)

    def search(self, query, start, count, topics=()):
        vector = self.encoder.encode(query, normalize_embeddings=True)
        order = np.argsort(-(self.embeddings @ vector))
        if topics:
            order = [i for i in order if self.topics[i] in topics]
        return [dict(self.passages[i]) for i in order[start:start + count]]


def confidence(claim):
    label = NLI_LABEL.get(claim["verdict"])
    if label is None:
        return 0.0
    return max(item["nli_probabilities"][label] for item in claim["evidence"]
               if item["verdict"] == claim["verdict"])


def verify_with_search(pipeline, index, question, answer, max_retries=MAX_RETRIES):
    results = []
    asked = index.topics_in(question)
    for claim in extract_claims(pipeline.nlp(answer)):
        topics = index.topics_in(claim["text"]) or asked
        passages = []
        rounds = 0
        for _ in range(max_retries + 1):
            found = index.search(claim["text"], len(passages), PAGE_SIZE, topics)
            if not found:
                stopped = "index exhausted"
                break
            passages += found
            rounds += 1
            result = pipeline.verify(question, claim["text"], passages, retrieved=True)["claims"][0]
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
