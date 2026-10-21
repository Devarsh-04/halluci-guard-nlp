import json
import os
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from search import PassageIndex

DATA = Path(os.environ.get("SCIFACT_DIR", "scifact_data"))
POOL = 30
KS = (1, 3, 5, 10)


def main():
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    index = PassageIndex.build(DATA / "corpus.jsonl", encoder, cache=DATA / "passage_index")
    print("| Split | Labeled claims | Ranking | " + " | ".join(f"Hit@{k}" for k in KS) + " | MRR |")
    print("|---|---:|---|" + "---:|" * (len(KS) + 1))
    for split in ("dev", "train"):
        claims = [c for c in map(json.loads, open(DATA / f"claims_{split}.jsonl")) if c["evidence"]]
        ranks = {"bi-encoder": [], "bi-encoder + reranker": []}
        for claim in claims:
            gold = set(map(str, claim["evidence"]))
            pool = index.search(claim["claim"], 0, POOL)
            scores = reranker.predict([(claim["claim"], p["text"]) for p in pool])
            reranked = [p for _, p in sorted(zip(scores, pool), key=lambda x: -x[0])]
            for name, ordering in (("bi-encoder", pool), ("bi-encoder + reranker", reranked)):
                ranks[name].append(next(
                    (i + 1 for i, p in enumerate(ordering) if p["id"].split(":")[0] in gold), None))
        for name, rs in ranks.items():
            hits = " | ".join(f"{sum(r is not None and r <= k for r in rs) / len(rs):.1%}" for k in KS)
            mrr = sum(1 / r for r in rs if r) / len(rs)
            print(f"| {split} | {len(claims)} | {name} | {hits} | {mrr:.3f} |")


if __name__ == "__main__":
    main()
