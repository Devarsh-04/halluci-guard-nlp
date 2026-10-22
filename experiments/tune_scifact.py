import itertools
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import (CONFIDENCE_TEMPERATURE, NLI_MODEL as DEFAULT_NLI_MODEL,
                      VerificationPipeline, calibrated, entity_number_score)

DATA = Path(os.environ.get("SCIFACT_DIR", "scifact_data"))
NLI_MODEL = os.environ.get("NLI_MODEL", DEFAULT_NLI_MODEL)
FEATURES = "" if NLI_MODEL.endswith("small") else "_" + NLI_MODEL.split("/")[-1]
GOLD = {"SUPPORT": "SUPPORTED", "CONTRADICT": "CONTRADICTED"}
LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]


def named_entities_overlap(source_doc, claim_doc):
    numeric_labels = {
        "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "DATE", "TIME", "PERCENT",
    }

    def names(doc):
        entities = {re.sub(r"^\W+|\W+$", "", entity.text.lower()) for entity in doc.ents
                    if entity.label_ not in numeric_labels}
        entities.discard("")
        return entities or {token.lower_ for token in doc if token.pos_ == "PROPN"}

    claim_names = names(claim_doc)
    return not claim_names or bool(claim_names & names(source_doc))


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
            and (r["overlap"] or not p["gate_con"]) and r["sim"] >= p.get("sim_con", 0.0)
            and r.get("rr", 99.0) >= p.get("rr_con", -99.0)):
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
        relevant = lambda r: r.get("rr", 99.0) >= p.get("rr_conf", -99.0)
        sup_rel = [r["ent"] for v, r in verdicts if v == "SUPPORTED" and relevant(r)]
        con_rel = [r["con"] for v, r in verdicts if v == "CONTRADICTED" and relevant(r)]
        if sup_rel and con_rel and min(max(sup_rel), max(con_rel)) >= p["margin"]:
            return "CONFLICTING_EVIDENCE"
        return "SUPPORTED" if max(sup) > max(con) else "CONTRADICTED"
    return "SUPPORTED" if sup else "CONTRADICTED" if con else "INSUFFICIENT_EVIDENCE"


def score(data, p):
    return metrics([(c["gold"], claim_verdict(c["rows"], p)) for c in data])


def metrics(pairs):
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
    path = DATA / f"features_{split}{FEATURES}.json"
    if not path.exists():
        pipe, corpus = pipe_corpus()
        path.write_text(json.dumps(extract(split, pipe, corpus)))
    return json.loads(path.read_text())


ABLATION = {
    "NLI argmax only": dict(gate_sup=False, gate_con=False, ent=0.0, comb=0.0, con=0.0, margin=0.0),
    "+ conflict margin 0.95": dict(gate_sup=False, gate_con=False, ent=0.0, comb=0.0, con=0.0, margin=0.95),
    "+ contradiction >= 0.6": dict(gate_sup=False, gate_con=False, ent=0.0, comb=0.0, con=0.6, margin=0.95),
    "+ entity gate on refutes": dict(gate_sup=False, gate_con=True, ent=0.0, comb=0.0, con=0.6, margin=0.95),
    "+ similarity >= 0.5 instead of entity gate": dict(gate_sup=False, gate_con=False, sim_con=0.5, ent=0.0, comb=0.0, con=0.6, margin=0.95),
    "+ conflict needs a relevant opposing passage (**current**)": dict(gate_sup=False, gate_con=False, sim_con=0.5, ent=0.0, comb=0.0, con=0.6, margin=0.95, rr_conf=4.0),
    "+ entity gate on supports": dict(gate_sup=True, gate_con=True, ent=0.0, comb=0.0, con=0.6, margin=0.95),
    "+ entailment >= 0.6 and combined >= 0.6 (previous)": dict(gate_sup=True, gate_con=True, ent=0.6, comb=0.6, con=0.6, margin=0.0),
}


def ablation():
    pipe_corpus = lambda: (VerificationPipeline(NLI_MODEL), {d["doc_id"]: d for d in read("corpus")})
    splits = {name: load(name, pipe_corpus) for name in ("train", "dev")}
    print("| Configuration | Split | Accuracy | Macro F1 | Supported recall | Contradicted recall | False support | False refute | Conflicts |")
    for name, p in ABLATION.items():
        for split, data in splits.items():
            s = score(data, p)
            print(f"| {name} | {split} | {s['acc']:.1%} | {s['macro_f1']:.3f} | {s['recall'][0]:.1%} | {s['recall'][1]:.1%} | {s['false_support']} | {s['false_refute']} | {s['conflicts']} |")


FINAL = ABLATION["+ conflict needs a relevant opposing passage (**current**)"]


def calibration():
    edges = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0001]
    label = {"SUPPORTED": ("ent", "entailment"), "CONTRADICTED": ("con", "contradiction")}
    for split in ("train", "dev"):
        for temperature in (1.0, CONFIDENCE_TEMPERATURE):
            committed = []
            for c in load(split, None):
                verdict = claim_verdict(c["rows"], FINAL)
                if verdict in label:
                    key, name = label[verdict]
                    committed.append((max(calibrated(
                        {"entailment": r["ent"], "contradiction": r["con"],
                         "neutral": max(1 - r["ent"] - r["con"], 0.0)}, name, temperature)
                        for r in c["rows"]), c["gold"] == verdict))
            print(f"\n{split}, temperature {temperature}: {len(committed)} committed SUPPORTED/CONTRADICTED verdicts")
            print("| Confidence | Claims | Mean confidence | Accuracy |\n|---|---:|---:|---:|")
            ece = 0.0
            for lo, hi in zip(edges, edges[1:]):
                b = [(p, ok) for p, ok in committed if lo <= p < hi]
                if b:
                    conf, acc = sum(p for p, _ in b) / len(b), sum(ok for _, ok in b) / len(b)
                    ece += len(b) / len(committed) * abs(conf - acc)
                    print(f"| {lo:.2f}-{min(hi, 1):.2f} | {len(b)} | {conf:.3f} | {acc:.1%} |")
            print(f"ECE (5 bins): {ece:.3f}")


def latency():
    import time
    corpus = {d["doc_id"]: d for d in read("corpus")}
    claims = read("claims_dev")
    start = time.perf_counter()
    pipe = VerificationPipeline(NLI_MODEL)
    print(f"model load: {time.perf_counter() - start:.1f}s")
    sentences = [s for d in list(corpus.values())[:40] for s in d["abstract"]]
    pipe.verify("q", claims[0]["claim"], [{"id": "w", "text": sentences[0]}])
    for n in (3, 10, 20):
        times = []
        for i, claim in enumerate(claims[:30]):
            evidence = [{"id": f"p{j}", "text": sentences[(i * n + j) % len(sentences)]} for j in range(n)]
            start = time.perf_counter()
            pipe.verify("q", claim["claim"], evidence)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        print(f"{n} passages: p50 {times[len(times) // 2]:.0f} ms, p95 {times[int(len(times) * 0.95) - 1]:.0f} ms (30 requests)")


def open_retrieval():
    import time
    from search import PassageIndex, verify_with_search

    pipe = VerificationPipeline(NLI_MODEL)
    index = PassageIndex.build(DATA / "corpus.jsonl", pipe.encoder, cache=DATA / "passage_index")
    claims = read("claims_dev")
    print(f"{len(index.passages)} indexed passages")
    labeled = [c for c in claims if c["evidence"]]
    for k in (5, 10, 15):
        hits = sum(
            any(p["id"].split(":")[0] in map(str, c["evidence"]) for p in index.search(c["claim"], 0, k))
            for c in labeled
        )
        print(f"gold document in top {k} passages: {hits / len(labeled):.1%} of {len(labeled)} labeled claims")
    print("| Evidence | Accuracy | Macro F1 | Supported recall | Contradicted recall | False support | False refute | Conflicts | Mean rounds | Seconds/claim |")
    for retries in [int(n) for n in os.environ.get("RETRIES", "0,2").split(",")]:
        pairs, rounds, start = [], [], time.perf_counter()
        for c in claims:
            gold = GOLD[next(iter(c["evidence"].values()))[0]["label"]] if c["evidence"] else "INSUFFICIENT_EVIDENCE"
            result = verify_with_search(pipe, index, "", c["claim"], max_retries=retries)
            pairs.append((gold, result["verdict"]))
            if len(pairs) % 25 == 0:
                print(f"retries={retries}: {len(pairs)}/{len(claims)} claims, {time.perf_counter() - start:.0f}s", flush=True)
            rounds.append(sum(x["search"]["rounds"] for x in result["claims"]) / len(result["claims"]))
        s = metrics(pairs)
        print(f"| Retrieved, {retries} retries | {s['acc']:.1%} | {s['macro_f1']:.3f} | {s['recall'][0]:.1%} | {s['recall'][1]:.1%} | {s['false_support']} | {s['false_refute']} | {s['conflicts']} | {sum(rounds) / len(rounds):.2f} | {(time.perf_counter() - start) / len(claims):.2f} |", flush=True)


def main():
    if sys.argv[1:] == ["retrieval"]:
        return open_retrieval()
    if sys.argv[1:] == ["calibration"]:
        return calibration()
    if sys.argv[1:] == ["latency"]:
        return latency()
    if sys.argv[1:] == ["ablation"]:
        return ablation()
    def pipe_corpus():
        return VerificationPipeline(NLI_MODEL), {d["doc_id"]: d for d in read("corpus")}

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
