import json
import os
from pathlib import Path

from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

from experiments.tune_scifact import FEATURES, NLI_MODEL, passages, read

DATA = Path(os.environ.get("SCIFACT_DIR", "scifact_data"))


def main():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    corpus = {d["doc_id"]: d for d in read("corpus")}
    for split in ("train", "dev"):
        path = DATA / f"features_{split}{FEATURES}.json"
        features = json.loads(path.read_text())
        for claim, entry in zip(read(f"claims_{split}"), features):
            _, texts = passages(claim, corpus, tokenizer)
            assert len(texts) == len(entry["rows"])
            for row, score in zip(entry["rows"], reranker.predict([(claim["claim"], t) for t in texts]) if texts else []):
                row["rr"] = float(score)
        path.write_text(json.dumps(features))
        print(split, "done", flush=True)


if __name__ == "__main__":
    main()
