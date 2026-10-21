import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from pipeline import NLI_MODEL, VerificationPipeline
from search import PassageIndex

DATA = Path(os.environ.get("SCIFACT_DIR", "scifact_data"))
GOLD = {"SUPPORT": "SUPPORTED", "CONTRADICT": "CONTRADICTED"}
POOL = 15
BATCH = 15


def main(split):
    pipe = VerificationPipeline(NLI_MODEL)
    index = PassageIndex.build(DATA / "corpus.jsonl", pipe.encoder, cache=DATA / "passage_index")
    claims = [json.loads(line) for line in open(DATA / f"claims_{split}.jsonl")]
    out, start = [], time.time()
    for n, claim in enumerate(claims):
        gold = (GOLD[next(iter(claim["evidence"].values()))[0]["label"]]
                if claim["evidence"] else "INSUFFICIENT_EVIDENCE")
        vector = index.encoder.encode(claim["claim"], normalize_embeddings=True)
        scores = index.embeddings @ vector
        order = np.argsort(-scores)[:POOL]
        pool = [index.passages[i] for i in order]
        rerank = pipe.reranker.predict([(claim["claim"], p["text"]) for p in pool])
        inputs = pipe.tokenizer([p["text"] for p in pool], [claim["claim"]] * len(pool),
                                return_tensors="pt", padding=True, truncation=True, max_length=512)
        with pipe.torch.inference_mode():
            probs = pipe.nli_model(**inputs).logits.softmax(dim=-1)
        rows = []
        for rank, (i, p, rr, pr) in enumerate(zip(order, pool, rerank, probs)):
            con, ent, neu = (float(pr[j]) for j in pipe.label_indices)
            rows.append({"id": p["id"], "doc": p["id"].split(":")[0], "rank": rank,
                         "sim": float(scores[i]), "rr": float(rr), "ent": ent, "con": con, "neu": neu})
        out.append({"gold": gold, "gold_docs": [str(d) for d in claim["evidence"]], "rows": rows})
        if n % 100 == 0:
            print(split, n, round(time.time() - start), flush=True)
    (DATA / f"open_features_{split}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
