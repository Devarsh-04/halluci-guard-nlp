import json
from collections import Counter
from pathlib import Path

import pipeline
from pipeline import VerificationPipeline, resolve_evidence

LABELS = ["SUPPORTED", "CONTRADICTED", "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE"]


def main():
    items = [json.loads(line) for line in open(Path(__file__).with_name("conflict_set.jsonl"))]
    pipe = VerificationPipeline()
    assessed = []
    for item in items:
        evidence = [{"id": f"p{i}", "text": text, "source": "s"} for i, text in enumerate(item["passages"])]
        assessed.append(pipe.verify("", item["claim"], evidence)["claims"][0]["evidence"])
    print("| Conflict margin | Accuracy | Conflict precision | Conflict recall | Support recall | Contradiction recall |")
    for margin in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        pipeline.CONFLICT_MARGIN = margin
        pairs = [(item["gold"], resolve_evidence(ev)["verdict"]) for item, ev in zip(items, assessed)]
        counts = Counter(pairs)
        predicted = sum(n for (_, q), n in counts.items() if q == "CONFLICTING_EVIDENCE")
        gold = sum(n for (g, _), n in counts.items() if g == "CONFLICTING_EVIDENCE")
        tp = counts[("CONFLICTING_EVIDENCE",) * 2]
        recall = lambda label: counts[(label, label)] / sum(n for (g, _), n in counts.items() if g == label)
        print(f"| {margin} | {sum(g == q for g, q in pairs) / len(pairs):.1%} | {tp}/{predicted} | {tp}/{gold} | {recall('SUPPORTED'):.0%} | {recall('CONTRADICTED'):.0%} |")
        if margin == 0.95:
            print("confusion at 0.95 (gold -> predicted):", dict(counts))


if __name__ == "__main__":
    main()
