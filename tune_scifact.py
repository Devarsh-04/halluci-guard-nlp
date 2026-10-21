import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import (VerificationPipeline, entity_number_score,
                      named_entities_overlap)

DATA = Path(__file__).resolve().parent.parent / "work" / "scifact"
GOLD = {"SUPPORT": "SUPPORTED", "CONTRADICT": "CONTRADICTED"}
LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]


def read(name):
    with open(DATA / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f]


def passages(claim, corpus, tokenizer):
    if claim["evidence"]:
        gold = GOLD[next(iter(claim["evidence"].values()))[0]["label"]]
        texts = [" ".join(corpus[int(doc)]["abstract"][i] for i in rationale["sentences"])
                 for doc, rationales in claim["evidence"].items()
                 for rationale in rationales]
        return gold, texts
    texts = []
    for doc in claim["cited_doc_ids"]:
        sentences = corpus[doc]["abstract"]
        whole = " ".join(sentences)
        if len(tokenizer(whole, claim["claim"])["input_ids"]) <= 512:
            texts.append(whole)
        else:
            texts.extend(sentences)
    return "INSUFFICIENT_EVIDENCE", texts


def extract(split, pipe, corpus):
    out = []
    for n, claim in enumerate(read(f"claims_{split}")):
        gold, texts = passages(claim, corpus, pipe.tokenizer)
        claim_doc = pipe.nlp(claim["claim"])
        claim_emb = pipe.encoder.encode(claim["claim"], normalize_embeddings=True)
        rows = []
        for text, doc, emb in zip(texts, pipe.nlp.pipe(texts),
                                  pipe.encoder.encode(texts, normalize_embeddings=True)):
            inputs = pipe.tokenizer(text, claim["claim"], return_tensors="pt",
                                    truncation=True, max_length=512)
            with pipe.torch.inference_mode():
                probs = pipe.nli_model(**inputs).logits.softmax(dim=-1)[0]
            con, ent, neu = (float(probs[i]) for i in pipe.label_indices)
            rows.append({
                "con": con, "ent": ent, "ner": entity_number_score(doc, claim_doc),
                "sim": float(np.clip(emb @ claim_emb, -1, 1)),
                "overlap": bool(named_entities_overlap(doc, claim_doc)),
            })
        out.append({"gold": gold, "rows": rows})
        if n % 100 == 0:
            print(split, n, flush=True)
    return out


def passage_verdict(r, p):
    if (r["con"] >= p["con"] and r["con"] > max(r["ent"], 1 - r["con"] - r["ent"])
            and (r["overlap"] or not p["gate_con"])):
        return "CONTRADICTED"
    combined = 0.2 * r["ner"] + 0.5 * r["ent"] + 0.3 * r["sim"]
    if (r["ent"] >= p["ent"] and r["ent"] > max(r["con"], 1 - r["con"] - r["ent"])
            and combined >= p["comb"] and (r["overlap"] or not p["gate_sup"])):
        return "SUPPORTED"
    return "INSUFFICIENT_EVIDENCE"


def claim_verdict(rows, p):
    verdicts = [(passage_verdict(r, p), r) for r in rows]
    sup = [r["ent"] for v, r in verdicts if v == "SUPPORTED"]
    con = [r["con"] for v, r in verdicts if v == "CONTRADICTED"]
    if sup and con:
        if min(max(sup), max(con)) >= p["margin"]:
            return "CONFLICTING_EVIDENCE"
        return "SUPPORTED" if max(sup) > max(con) else "CONTRADICTED"
    return "SUPPORTED" if sup else "CONTRADICTED" if con else "INSUFFICIENT_EVIDENCE"


def score(data, p):
    pairs = [(c["gold"], claim_verdict(c["rows"], p)) for c in data]
    counts = Counter(pairs)
    f1s = []
    for label in LABELS:
        tp = counts[(label, label)]
        pred = sum(v for (g, q), v in counts.items() if q == label)
        gold = sum(v for (g, q), v in counts.items() if g == label)
        precision, recall = tp / pred if pred else 0, tp / gold if gold else 0
        f1s.append(2 * precision * recall / (precision + recall) if tp else 0)
    return {
        "acc": sum(g == q for g, q in pairs) / len(pairs),
        "macro_f1": sum(f1s) / 3,
        "recall": [counts[(l, l)] / sum(1 for g, _ in pairs if g == l) for l in LABELS],
        "false_support": counts[("CONTRADICTED", "SUPPORTED")] + counts[("INSUFFICIENT_EVIDENCE", "SUPPORTED")],
        "false_refute": counts[("SUPPORTED", "CONTRADICTED")] + counts[("INSUFFICIENT_EVIDENCE", "CONTRADICTED")],
        "conflicts": sum(q == "CONFLICTING_EVIDENCE" for _, q in pairs),
    }


def load(split, pipe_corpus):
    path = DATA / f"features_{split}.json"
    if not path.exists():
        pipe, corpus = pipe_corpus()
        path.write_text(json.dumps(extract(split, pipe, corpus)))
    return json.loads(path.read_text())


ABLATION = {
    "NLI argmax only": dict(gate_sup=False, gate_con=False, ent=0.0, comb=0.0, con=0.0, margin=0.0),
    "+ conflict margin 0.95": dict(gate_sup=False, gate_con=False, ent=0.0, comb=0.0, con=0.0, margin=0.95),
    "+ contradiction >= 0.6": dict(gate_sup=False, gate_con=False, ent=0.0, comb=0.0, con=0.6, margin=0.95),
    "+ entity gate on refutes (final)": dict(gate_sup=False, gate_con=True, ent=0.0, comb=0.0, con=0.6, margin=0.95),
    "+ entity gate on supports": dict(gate_sup=True, gate_con=True, ent=0.0, comb=0.0, con=0.6, margin=0.95),
    "+ entailment >= 0.6 and combined >= 0.6 (previous)": dict(gate_sup=True, gate_con=True, ent=0.6, comb=0.6, con=0.6, margin=0.0),
}


def ablation():
    pipe_corpus = lambda: (VerificationPipeline(), {d["doc_id"]: d for d in read("corpus")})
    splits = {name: load(name, pipe_corpus) for name in ("train", "dev")}
    print("| Configuration | Split | Accuracy | Macro F1 | Supported recall | Contradicted recall | False support | False refute | Conflicts |")
    for name, p in ABLATION.items():
        for split, data in splits.items():
            s = score(data, p)
            print(f"| {name} | {split} | {s['acc']:.1%} | {s['macro_f1']:.3f} | {s['recall'][0]:.1%} | {s['recall'][1]:.1%} | {s['false_support']} | {s['false_refute']} | {s['conflicts']} |")


def main():
    if sys.argv[1:] == ["ablation"]:
        return ablation()
    def pipe_corpus():
        return VerificationPipeline(), {d["doc_id"]: d for d in read("corpus")}

    train = load("train", pipe_corpus)
    current = dict(gate_sup=True, gate_con=True, ent=0.6, comb=0.6, con=0.6, margin=0.0)
    print("current train", score(train, current))
    grid = itertools.product(
        [True, False], [True, False], [0.3, 0.4, 0.5, 0.6], [0.0, 0.3, 0.5, 0.6],
        [0.6, 0.8, 0.9, 0.95], [0.0, 0.8, 0.9, 0.95, 0.99],
    )
    results = sorted(
        ((score(train, p), p) for p in (
            dict(gate_sup=gs, gate_con=gc, ent=e, comb=c, con=k, margin=m) for gs, gc, e, c, k, m in grid)),
        key=lambda x: -x[0]["macro_f1"],
    )
    for s, p in results[:8]:
        print(p, {k: (round(v, 3) if not isinstance(v, list) else [round(x, 3) for x in v]) for k, v in s.items()})
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        dev = load("dev", pipe_corpus)
        print("dev current", score(dev, current))
        print("dev best-on-train", results[0][1], score(dev, results[0][1]))


if __name__ == "__main__":
    main()
