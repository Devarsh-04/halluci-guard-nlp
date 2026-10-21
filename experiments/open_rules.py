import itertools
import json
import os
import sys
from pathlib import Path

from experiments.tune_scifact import metrics

DATA = Path(os.environ.get("SCIFACT_DIR", "scifact_data"))
SMALL = {"order": "bi", "k": 5, "con": 0.6, "sim_con": 0.5, "rr_con": -99.0,
         "rr_sup": -99.0, "agg": "any", "margin": 0.95}


def passage_verdict(r, p):
    if r["con"] >= p["con"] and r["con"] > max(r["ent"], r["neu"]):
        if r["sim"] >= p["sim_con"] and r["rr"] >= p["rr_con"]:
            return "CONTRADICTED"
    elif r["ent"] > max(r["con"], r["neu"]) and r["rr"] >= p["rr_sup"]:
        return "SUPPORTED"
    return "INSUFFICIENT_EVIDENCE"


def claim_verdict(rows, p):
    rows = sorted(rows, key=lambda r: r["rank"] if p["order"] == "bi" else -r["rr"])[:p["k"]]
    verdicts = [(passage_verdict(r, p), r) for r in rows]
    if p["agg"] == "top":
        return next((v for v, _ in verdicts if v != "INSUFFICIENT_EVIDENCE"), "INSUFFICIENT_EVIDENCE")
    if p["agg"] == "topc":
        committed = [(v, r) for v, r in verdicts if v != "INSUFFICIENT_EVIDENCE"]
        if not committed:
            return "INSUFFICIENT_EVIDENCE"
        first, lead = committed[0]
        strength = lambda v, r: r["ent"] if v == "SUPPORTED" else r["con"]
        if strength(first, lead) >= p["margin"] and any(
                v != first and strength(v, r) >= p["margin"] and r["rr"] >= p["rr_con"]
                for v, r in committed[1:]):
            return "CONFLICTING_EVIDENCE"
        return first
    sup = [r["ent"] for v, r in verdicts if v == "SUPPORTED"]
    con = [r["con"] for v, r in verdicts if v == "CONTRADICTED"]
    if sup and con:
        if min(max(sup), max(con)) >= p["margin"]:
            return "CONFLICTING_EVIDENCE"
        return "SUPPORTED" if max(sup) > max(con) else "CONTRADICTED"
    return "SUPPORTED" if sup else "CONTRADICTED" if con else "INSUFFICIENT_EVIDENCE"


def score(data, p):
    return metrics([(c["gold"], claim_verdict(c["rows"], p)) for c in data])


def show(name, s):
    r = s["recall"]
    return (f"{name}: acc {s['acc']:.1%} F1 {s['macro_f1']:.3f} rec {r[0]:.0%}/{r[1]:.0%}/{r[2]:.0%} "
            f"FS {s['false_support']} FR {s['false_refute']} conf {s['conflicts']}")


def load(split):
    return json.loads((DATA / f"open_features_{split}.json").read_text())


GRID = dict(
    order=["rr"], k=[15], con=[0.9, 0.95], sim_con=[0.5, 0.6, 0.7],
    rr_con=[2.0, 4.0, 6.0], rr_sup=[-99.0, -2.0, 0.0], agg=["topc"],
    margin=[0.95, 0.99],
)


def main():
    train = load("train")
    print(show("current rules (train)", score(train, SMALL)))
    results = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        p = dict(zip(keys, values))
        results.append((score(train, p), p))
    results.sort(key=lambda x: -x[0]["macro_f1"])
    print(f"{len(results)} configs; top 8 by macro F1 on train")
    for s, p in results[:8]:
        print(show("", s), p)
    cautious = [x for x in results if x[0]["false_refute"] <= 20]
    print("best with false refutes <= 20:")
    for s, p in cautious[:5]:
        print(show("", s), p)
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        dev = load("dev")
        print(show("current rules (dev)", score(dev, SMALL)))
        for name, (s, p) in (("best F1", results[0]), ("best cautious", cautious[0])):
            print(show(f"{name} on dev", score(dev, p)), p)


if __name__ == "__main__":
    main()
