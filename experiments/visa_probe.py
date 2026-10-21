"""Score search mode on hand-written visa claims against the official knowledge base.
python -m experiments.visa_probe experiments/visa_probe_heldout.jsonl"""
import json
import sys

import api
from pipeline import VerificationPipeline, extract_claims
from search import PassageIndex, verify_with_search


def main(path):
    cases = [json.loads(line) for line in open(path)]
    pipe = VerificationPipeline()
    index = PassageIndex.build(api.DEFAULT_CORPUS, pipe.encoder)
    correct = 0
    for case in cases:
        got = verify_with_search(pipe, index, case["question"], case["claim"])["verdict"]
        correct += got == case["gold"]
        mark = "ok  " if got == case["gold"] else "MISS"
        print(f"{mark} {case['gold']:<22} -> {got:<22} {case['claim']}")
    print(f"{correct} of {len(cases)} correct")


if __name__ == "__main__":
    main(sys.argv[1])
