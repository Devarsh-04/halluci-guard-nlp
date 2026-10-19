import pytest

from pipeline import (
    VerificationInputError, aggregate_verdict, entity_number_score,
    extract_claims, score_verification,
    resolve_evidence,
)


@pytest.fixture(scope="module")
def parser():
    import spacy

    return spacy.load("en_core_web_sm")


def test_supported_evidence_retains_notebook_weighted_score():
    result = score_verification(1.0, [0.05, 0.9, 0.05], 0.8)
    assert result["verdict"] == "SUPPORTED"
    assert result["combined_score"] == pytest.approx(0.89)
    assert result["nli_label"] == "entailment"


def test_contradiction_cannot_be_outweighed_by_similarity():
    result = score_verification(1.0, [0.65, 0.3, 0.05], 1.0)
    assert result["combined_score"] == pytest.approx(0.65)
    assert result["verdict"] == "CONTRADICTED"


def test_neutral_evidence_is_not_a_false_claim():
    result = score_verification(1.0, [0.05, 0.05, 0.9], 0.9)
    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_ambiguous_nli_abstains():
    result = score_verification(1.0, [0.4, 0.35, 0.25], 0.9)
    assert result["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_nli_probability_names_preserve_model_label_order():
    result = score_verification(1.0, [0.8, 0.15, 0.05], 0.9)
    assert result["nli_probabilities"] == {
        "contradiction": 0.8, "entailment": 0.15, "neutral": 0.05,
    }


def test_numbers_absent_from_evidence_reduce_entity_score():
    import spacy

    nlp = spacy.blank("en")
    source, answer = nlp.pipe(["The code is 4729.", "The code is 1234."])
    assert entity_number_score(source, answer) == 0.0


def test_matching_entities_and_numbers_are_grounded():
    import spacy
    from spacy.tokens import Span

    nlp = spacy.blank("en")
    source, answer = nlp.pipe(["Mars has 2 moons.", "Mars has 2 moons."])
    for doc in (source, answer):
        doc.ents = [Span(doc, 0, 1, label="LOC")]
    assert entity_number_score(source, answer) == 1.0


def test_answer_without_entities_or_numbers_has_no_entity_mismatch():
    import spacy

    nlp = spacy.blank("en")
    source, answer = nlp.pipe(["It is red.", "It is blue."])
    assert entity_number_score(source, answer) == 1.0


def test_sentence_claims_preserve_text_and_offsets():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    answer = "Mars has two moons.  Jupiter is the largest planet."
    claims = extract_claims(nlp(answer))
    assert [claim["text"] for claim in claims] == [
        "Mars has two moons.", "Jupiter is the largest planet.",
    ]
    assert [claim["id"] for claim in claims] == ["claim_1", "claim_2"]
    for claim in claims:
        assert answer[claim["start_char"]:claim["end_char"]] == claim["text"]


def test_sentence_split_keeps_abbreviations_and_decimals():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    claims = extract_claims(nlp("Dr. Lee measured 3.5 cm. The sample is red."))
    assert [claim["text"] for claim in claims] == [
        "Dr. Lee measured 3.5 cm.", "The sample is red.",
    ]


def test_punctuation_is_not_a_verifiable_claim():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    with pytest.raises(VerificationInputError, match="verifiable"):
        extract_claims(nlp("... !!!"))


def test_claim_limit_rejects_excess_instead_of_dropping_sentences():
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    with pytest.raises(VerificationInputError, match="20"):
        extract_claims(nlp("Mars is red. " * 21))


@pytest.mark.parametrize("verdicts,expected", [
    (["SUPPORTED"], "SUPPORTED"),
    (["SUPPORTED", "SUPPORTED"], "SUPPORTED"),
    (["SUPPORTED", "INSUFFICIENT_EVIDENCE"], "INSUFFICIENT_EVIDENCE"),
    (["INSUFFICIENT_EVIDENCE", "SUPPORTED"], "INSUFFICIENT_EVIDENCE"),
    (["SUPPORTED", "CONTRADICTED"], "CONTRADICTED"),
    (["CONTRADICTED", "INSUFFICIENT_EVIDENCE"], "CONTRADICTED"),
])
def test_overall_verdict_does_not_average_away_bad_claims(verdicts, expected):
    assert aggregate_verdict([{"verdict": item} for item in verdicts]) == expected


@pytest.mark.parametrize("answer,expected", [
    ("Mars has two moons and Jupiter is the largest planet.",
     ["Mars has two moons.", "Jupiter is the largest planet."]),
    ("Mars is red and has two moons.", ["Mars is red.", "Mars has two moons."]),
    ("🪐 Mars is red and has two moons.", ["🪐 Mars is red.", "Mars has two moons."]),
    ("Mars is red and cold.", ["Mars is red.", "Mars is cold."]),
    ("Mars has rotated and cooled.", ["Mars has rotated.", "Mars has cooled."]),
    ("Mars is red but Jupiter is large.", ["Mars is red.", "Jupiter is large."]),
])
def test_clear_compounds_are_checked_as_separate_claims(parser, answer, expected):
    claims = extract_claims(parser(answer))
    assert [claim["text"] for claim in claims] == expected
    for claim in claims:
        assert answer[claim["start_char"]:claim["end_char"]] == claim["source_text"]
    assert all(a["end_char"] <= b["start_char"]
               for a, b in zip(claims, claims[1:]))


@pytest.mark.parametrize("answer", [
    "Mars does not have rings and supports life.",
    "If Mars has water, it could support life.",
    "Mars has two moons or three moons.",
    "Mars is larger than Mercury and smaller than Earth.",
    "Mars may be red and have two moons.",
    "Mars is not both red and cold.",
    "Mars and Jupiter are red and cold, respectively.",
    "Mars is reportedly red and has two moons.",
    "Mars is red and a planet.",
])
def test_ambiguous_scope_is_preserved_with_a_review_note(parser, answer):
    claims = extract_claims(parser(answer))
    assert len(claims) == 1
    assert claims[0]["text"] == answer
    assert claims[0]["review_required"] is True
    assert claims[0]["extraction_note"]


def test_entity_lists_are_not_mistaken_for_independent_claims(parser):
    text = "Mars has two moons, Phobos and Deimos."
    claims = extract_claims(parser(text))
    assert [claim["text"] for claim in claims] == [text]


def test_compound_claims_count_toward_the_check_limit(parser):
    with pytest.raises(VerificationInputError, match="20"):
        extract_claims(parser("Mars is red and has two moons. " * 11))


def test_conflicting_evidence_is_not_hidden_by_ranking():
    supported = {"id": "a", **score_verification(1, [0.01, 0.98, 0.01], 0.9)}
    contradicted = {"id": "b", **score_verification(1, [0.98, 0.01, 0.01], 0.9)}
    for evidence in ([supported, contradicted], [contradicted, supported]):
        result = resolve_evidence(evidence)
        assert result["verdict"] == "CONFLICTING_EVIDENCE"
        assert "1 passage supports this claim" in result["explanation"]
        assert "1 passage contradicts it" in result["explanation"]
        assert "0 passages are inconclusive" in result["explanation"]


def test_evidence_explanation_counts_each_verdict_without_averaging():
    supported = {"id": "a", **score_verification(1, [0.01, 0.98, 0.01], 0.9)}
    contradicted = {"id": "b", **score_verification(1, [0.98, 0.01, 0.01], 0.9)}
    inconclusive = {"id": "c", **score_verification(1, [0.01, 0.01, 0.98], 0.2)}

    result = resolve_evidence([supported, supported, contradicted, inconclusive])

    assert result["verdict"] == "CONFLICTING_EVIDENCE"
    assert "2 passages support this claim" in result["explanation"]
    assert "1 passage contradicts it" in result["explanation"]
    assert "1 passage is inconclusive" in result["explanation"]


def test_neutral_passage_does_not_erase_support():
    neutral = {"id": "a", **score_verification(1, [0.01, 0.01, 0.98], 0.2)}
    supported = {"id": "b", **score_verification(1, [0.01, 0.98, 0.01], 0.9)}
    result = resolve_evidence([neutral, supported])
    assert result["verdict"] == "SUPPORTED"
    assert result["verified_evidence_id"] == "b"


def test_overall_answer_discloses_any_evidence_conflict():
    assert aggregate_verdict([
        {"verdict": "CONTRADICTED"}, {"verdict": "CONFLICTING_EVIDENCE"},
    ]) == "CONFLICTING_EVIDENCE"
