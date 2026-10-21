import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline import VerificationPipeline
from search import PassageIndex, verify_with_search

HERE = Path(__file__).parent / "visa_v5"


def report(name, rows):
    """rows: (difficulty, label, verdict); label 1 = grounded, 0 = hallucinated."""
    flagged = lambda verdict: verdict != "SUPPORTED"
    tp = sum(1 for _, y, v in rows if y == 0 and flagged(v))
    fp = sum(1 for _, y, v in rows if y == 1 and flagged(v))
    fn = sum(1 for _, y, v in rows if y == 0 and not flagged(v))
    correct = sum(1 for _, y, v in rows if flagged(v) == (y == 0))
    precision, recall = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn)
    print(f"\n{name}: accuracy {correct / len(rows):.1%}, hallucination precision {precision:.1%}, "
          f"recall {recall:.1%}, grounded answers wrongly flagged {fp}/{sum(y == 1 for _, y, _ in rows)}")
    by = defaultdict(list)
    for difficulty, y, v in rows:
        by[difficulty].append((y, v))
    print("| Difficulty | Pairs | Hallucinations caught | Grounded passed |\n|---|---:|---:|---:|")
    for difficulty, items in sorted(by.items(), key=lambda kv: -len(kv[1])):
        bad = [v for y, v in items if y == 0]
        good = [v for y, v in items if y == 1]
        print(f"| {difficulty} | {len(items)} | {sum(flagged(v) for v in bad)}/{len(bad)} | {sum(not flagged(v) for v in good)}/{len(good)} |")
    print("hallucinated-answer verdicts:", dict(Counter(v for _, y, v in rows if y == 0)))


def main():
    pairs = [json.loads(line) for line in open(HERE / "hallucination_pairs.jsonl")]
    pipe = VerificationPipeline()
    oracle = []
    for p in pairs:
        evidence = [{"id": "source", "text": p["source"], "source": "v5 source passage"}]
        oracle.append((p["difficulty"], p["label"], pipe.verify("", p["answer"], evidence)["verdict"]))
    report("Given the source passage", oracle)
    index = PassageIndex.build(Path(__file__).parent.parent / "data" / "visa_knowledge_base.jsonl", pipe.encoder)
    print(f"\n{len(index.passages)} knowledge-base passages indexed")
    searched = [(p["difficulty"], p["label"], verify_with_search(pipe, index, "", p["answer"])["verdict"]) for p in pairs]
    report("Retrieved from the 120-passage knowledge base", searched)


if __name__ == "__main__":
    main()
