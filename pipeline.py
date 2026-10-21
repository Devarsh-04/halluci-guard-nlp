"""Inference extracted from v5 research notebook cells 25 and 37."""

import re


NLI_LABELS = ("contradiction", "entailment", "neutral")
CONFLICT_MARGIN = 0.95
MIN_CONTRADICTION_SIMILARITY = 0.5


class VerificationInputError(ValueError):
    pass


class EvidenceTooLong(VerificationInputError):
    pass


def split_sentence(sentence):
    root = sentence.root
    coordinated = any(token.dep_ == "cc" for token in sentence)
    scope_words = {
        "no", "not", "never", "neither", "nor", "without", "or", "either",
        "both", "may", "might", "could", "would", "should", "can", "must",
        "will", "shall", "only", "always", "sometimes", "probably", "than",
        "respectively", "reportedly", "allegedly", "apparently", "possibly",
    }
    conditional = any(token.lower_ in {"if", "unless", "whether"}
                      or token.dep_ == "advcl" for token in sentence)
    unsafe_scope = conditional or (
        coordinated and any(token.lower_ in scope_words or token.is_quote
                            for token in sentence)
    )
    held = {
        "start_char": sentence.start_char,
        "end_char": sentence.end_char,
        "text": sentence.text.strip(),
        "review_required": unsafe_scope,
        "extraction_note": (
            "Kept together to preserve negation, alternatives, comparison, "
            "quotation, or conditional scope. Review this statement as written."
            if unsafe_scope else None
        ),
    }
    if unsafe_scope or not sentence.doc.has_annotation("DEP"):
        return [held]

    subject = next((child for child in root.children
                    if child.dep_ in {"nsubj", "nsubjpass"}), None)
    units = [root, *(token for token in root.conjuncts
                     if token.pos_ in {"VERB", "AUX"})]
    adjective_mode = False
    if len(units) == 1 and root.lemma_ == "be":
        adjective = next((child for child in root.children
                          if child.dep_ == "acomp" and child.pos_ == "ADJ"), None)
        if adjective is not None:
            adjectives = [adjective, *adjective.conjuncts]
            if len(adjectives) > 1 and all(t.pos_ == "ADJ" for t in adjectives):
                units = adjectives
                adjective_mode = True
    units.sort(key=lambda token: token.i)
    if len(units) < 2 or subject is None:
        if coordinated and any(t.dep_ == "conj" and t.pos_ != "PROPN"
                               for t in sentence):
            held.update(
                review_required=True,
                extraction_note=(
                    "The coordination could not be split safely; checked as written."
                ),
            )
        return [held]

    boundaries = []
    for previous, unit in zip(units, units[1:]):
        conjunctions = [token for token in sentence
                        if previous.i < token.i < unit.i and token.dep_ == "cc"]
        if not conjunctions or conjunctions[-1].lower_ not in {"and", "but"}:
            return [{
                **held, "review_required": True,
                "extraction_note": (
                    "The coordination could not be split safely; checked as written."
                ),
            }]
        boundaries.append(conjunctions[-1])

    subject_text = sentence.doc[subject.left_edge.i:subject.right_edge.i + 1].text
    subject_text = re.sub(r"^\W+", "", subject_text)
    auxiliaries = " ".join(child.text for child in root.children
                           if child.dep_ in {"aux", "auxpass"})
    pieces = []
    for index, unit in enumerate(units):
        start = (
            sentence.start_char if index == 0
            else boundaries[index - 1].idx + len(boundaries[index - 1])
        )
        end = sentence.end_char if index == len(units) - 1 else boundaries[index].idx
        raw = sentence.doc.text[start:end]
        text = raw.strip().rstrip(",;").rstrip()
        start += len(raw) - len(raw.lstrip())
        prefix = ""
        own_subject = any(child.dep_ in {"nsubj", "nsubjpass"}
                          for child in unit.children)
        if index and (adjective_mode or not own_subject):
            prefix = subject_text
            if adjective_mode:
                prefix += " " + " ".join(
                    part for part in (auxiliaries, root.text) if part
                )
            elif unit.tag_ in {"VBN", "VBG", "VB"} and auxiliaries:
                prefix += " " + auxiliaries
            prefix += " "
        pieces.append({
            "start_char": start, "end_char": start + len(text),
            "text": prefix + text + ("" if text.endswith((".", "!", "?")) else "."),
            "review_required": False,
            "extraction_note": (
                "Split from a compound sentence; the shared subject or "
                "auxiliary was restored."
                if prefix else "Split at a conjunction between separate statements."
            ),
        })
    return pieces


def extract_claims(answer_doc):
    claims = []
    for sentence in answer_doc.sents:
        for piece in split_sentence(sentence):
            if not any(char.isalnum() for char in piece["text"]):
                continue
            source = answer_doc.text[piece["start_char"]:piece["end_char"]]
            start = piece["start_char"] + len(source) - len(source.lstrip())
            source = source.strip()
            claims.append({
                **piece, "id": f"claim_{len(claims) + 1}",
                "start_char": start, "end_char": start + len(source),
                "source_text": source,
            })
    if not claims:
        raise VerificationInputError("The answer contains no verifiable sentences.")
    if len(claims) > 20:
        raise VerificationInputError("Supply an answer with at most 20 claims.")
    return claims


def aggregate_verdict(claims):
    verdicts = {claim["verdict"] for claim in claims}
    if "CONFLICTING_EVIDENCE" in verdicts:
        return "CONFLICTING_EVIDENCE"
    if "CONTRADICTED" in verdicts:
        return "CONTRADICTED"
    if "INSUFFICIENT_EVIDENCE" in verdicts:
        return "INSUFFICIENT_EVIDENCE"
    return "SUPPORTED"


def resolve_evidence(evidence):
    supported = [item for item in evidence if item["verdict"] == "SUPPORTED"]
    contradicted = [item for item in evidence if item["verdict"] == "CONTRADICTED"]
    if supported and contradicted:
        support_strength = max(i["nli_probabilities"]["entailment"] for i in supported)
        contradiction_strength = max(
            i["nli_probabilities"]["contradiction"] for i in contradicted
        )
        if min(support_strength, contradiction_strength) < CONFLICT_MARGIN:
            if support_strength > contradiction_strength:
                contradicted = []
            else:
                supported = []
    inconclusive_count = len(evidence) - len(supported) - len(contradicted)

    def passage_count(count):
        return f"{count} passage{'s' if count != 1 else ''}"

    def passage_action(count, singular, plural):
        return f"{passage_count(count)} {singular if count == 1 else plural}"

    inconclusive_summary = passage_action(
        inconclusive_count, "is inconclusive", "are inconclusive"
    )

    if supported and contradicted:
        verdict = "CONFLICTING_EVIDENCE"
        explanation = (
            f"{passage_action(len(supported), 'supports', 'support')} this claim, "
            f"{passage_action(len(contradicted), 'contradicts', 'contradict')} it, "
            f"and {inconclusive_summary}. Review the "
            "sources; their authority is not assessed."
        )
    elif supported:
        verdict = "SUPPORTED"
        explanation = (
            f"{passage_action(len(supported), 'supports', 'support')} this claim; "
            f"none of the assessed passages contradicts it, and "
            f"{inconclusive_summary}."
        )
    elif contradicted:
        verdict = "CONTRADICTED"
        explanation = (
            f"{passage_action(len(contradicted), 'contradicts', 'contradict')} "
            "this claim; none of the assessed passages supports it, and "
            f"{inconclusive_summary}."
        )
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
        explanation = (
            "None of the supplied passages clearly supports or contradict this "
            f"claim; {inconclusive_summary}."
        )
    representative = (supported or contradicted or evidence)[0]
    return {
        **{key: representative[key] for key in (
            "combined_score", "ner_score", "embedding_score",
            "nli_label", "nli_probabilities",
        )},
        "verdict": verdict, "explanation": explanation,
        "verified_evidence_id": representative["id"],
    }


def score_verification(ner, probabilities, similarity):
    scores = dict(zip(NLI_LABELS, probabilities, strict=True))
    label = max(scores, key=scores.get)
    combined = 0.2 * ner + 0.5 * scores["entailment"] + 0.3 * similarity
    if label == "contradiction" and scores[label] >= 0.6:
        verdict = "CONTRADICTED"
    elif label == "entailment":
        verdict = "SUPPORTED"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
    return {
        "verdict": verdict,
        "combined_score": combined,
        "ner_score": ner,
        "embedding_score": similarity,
        "nli_label": label,
        "nli_probabilities": scores,
    }


def entity_number_score(source_doc, answer_doc):
    source_entities = {(e.text.lower().strip(), e.label_) for e in source_doc.ents}
    answer_entities = {(e.text.lower().strip(), e.label_) for e in answer_doc.ents}
    pattern = r"\b\d+(?:,\d+)*(?:\.\d+)?\b"
    source_numbers = set(re.findall(pattern, source_doc.text))
    answer_numbers = set(re.findall(pattern, answer_doc.text))
    total = len(answer_entities) + len(answer_numbers)
    unsupported = len(answer_entities - source_entities)
    unsupported += len(answer_numbers - source_numbers)
    return max(0.0, 1.0 - unsupported / total) if total else 1.0


class VerificationPipeline:
    def __init__(self):
        import spacy
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.nlp = spacy.load("en_core_web_sm")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )
        model_name = "cross-encoder/nli-deberta-v3-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).eval()
        self.label_indices = [
            next(i for i, name in self.nli_model.config.id2label.items()
                 if name.lower() == label)
            for label in NLI_LABELS
        ]

    def verify(self, question, answer, evidence):
        import numpy as np

        claims = extract_claims(self.nlp(answer))
        texts = [item["text"] for item in evidence]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        claim_embeddings = self.encoder.encode(
            [claim["text"] for claim in claims], normalize_embeddings=True
        )
        query_embeddings = self.encoder.encode(
            [f"{question}\n{claim['text']}" for claim in claims],
            normalize_embeddings=True,
        )
        scores = query_embeddings @ embeddings.T
        indices = np.argsort(-scores, axis=1)
        evidence_indices = {item["id"]: i for i, item in enumerate(evidence)}
        source_docs = list(self.nlp.pipe(texts))
        claim_docs = list(self.nlp.pipe([claim["text"] for claim in claims]))

        for claim_index, claim in enumerate(claims):
            candidates = [
                {**evidence[int(i)],
                 "retrieval_score": float(scores[claim_index, i])}
                for i in indices[claim_index]
            ]
            rerank_scores = self.reranker.predict(
                [(claim["text"], item["text"]) for item in candidates]
            )
            for item, score in zip(candidates, rerank_scores, strict=True):
                item["rerank_score"] = float(score)
            candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
            for candidate in candidates:
                inputs = self.tokenizer(
                    candidate["text"], claim["text"],
                    return_tensors="pt", truncation=False,
                )
                if inputs["input_ids"].shape[1] > 512:
                    raise EvidenceTooLong(
                        f"Passage {candidate['id']} and {claim['id']} exceed "
                        "the NLI token limit (512). Supply shorter passages "
                        "or a shorter statement."
                    )
                with self.torch.inference_mode():
                    probs = self.nli_model(**inputs).logits.softmax(dim=-1)[0]
                ordered_probs = [float(probs[i]) for i in self.label_indices]
                evidence_index = evidence_indices[candidate["id"]]
                ner = entity_number_score(
                    source_docs[evidence_index], claim_docs[claim_index]
                )
                similarity = float(np.clip(
                    np.dot(embeddings[evidence_index], claim_embeddings[claim_index]),
                    -1.0, 1.0,
                ))
                candidate.update(score_verification(ner, ordered_probs, similarity))
                candidate["explanation"] = {
                    "SUPPORTED": (
                        "This passage supports the claim under the model's checks."
                    ),
                    "CONTRADICTED": (
                        "The model finds this passage inconsistent with the claim."
                    ),
                    "INSUFFICIENT_EVIDENCE": (
                        "This passage does not establish sufficient support "
                        "or a clear contradiction."
                    ),
                }[candidate["verdict"]]
                if (candidate["verdict"] == "CONTRADICTED"
                        and similarity < MIN_CONTRADICTION_SIMILARITY):
                    candidate.update(
                        verdict="INSUFFICIENT_EVIDENCE",
                        explanation=(
                            "This passage is not similar enough to the claim to "
                            "count as a contradiction."
                        ),
                    )
            claim.update(**resolve_evidence(candidates), evidence=candidates)

        return {"verdict": aggregate_verdict(claims), "claims": claims}
