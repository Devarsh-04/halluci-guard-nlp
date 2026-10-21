import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from pipeline import VerificationInputError, VerificationPipeline
from search import PassageIndex, verify_with_search


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Annotated[Text, Field(max_length=100)]
    text: Annotated[Text, Field(max_length=4000)]
    source: Annotated[Text, Field(max_length=2000)] | None = None


class VerifyRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "question": "Describe Mars and Jupiter.",
            "answer": "Mars has two moons and Jupiter is the largest planet.",
            "evidence": [
                {"id": "mars", "text": "Mars has two moons, Phobos and Deimos.",
                 "source": "Example evidence supplied by the user"},
                {"id": "jupiter", "text": "Jupiter is the largest planet.",
                 "source": "Example evidence supplied by the user"},
                {"id": "conflict", "text": "Mars has no moons.",
                 "source": "Deliberately conflicting synthetic example"},
            ],
        }},
    )

    question: Annotated[Text, Field(max_length=1000)]
    answer: Annotated[Text, Field(max_length=2000)]
    evidence: list[Evidence] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def unique_evidence_ids(self):
        ids = [item.id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence IDs must be unique.")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[Text, Field(max_length=1000)]
    answer: Annotated[Text, Field(max_length=2000)]


class RankedEvidence(Evidence):
    retrieval_score: float
    rerank_score: float


class VerificationScores(BaseModel):
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    combined_score: float = Field(
        description="Research score; not calibrated confidence."
    )
    ner_score: float
    embedding_score: float
    nli_label: Literal["entailment", "contradiction", "neutral"]
    nli_probabilities: dict[str, float]


class AssessedEvidence(RankedEvidence, VerificationScores):
    explanation: str


class SearchTrace(BaseModel):
    rounds: int
    passages_searched: int
    stopped: str


class ClaimResult(VerificationScores):
    id: str
    text: str
    source_text: str
    start_char: int
    end_char: int
    review_required: bool
    extraction_note: str | None
    verdict: Literal[
        "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE",
    ]
    explanation: str
    evidence: list[AssessedEvidence]
    verified_evidence_id: str = Field(
        description=(
            "Representative passage for the claim-level scores; "
            "all supplied passages are assessed."
        )
    )
    search: SearchTrace | None = None


class VerificationResult(BaseModel):
    verdict: Literal[
        "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE",
    ]
    claims: list[ClaimResult]
    scope: str = (
        "Clear coordinated clauses are split conservatively. Ambiguous compounds "
        "are kept intact with a review note. Each claim is checked against every "
        "supplied passage; disagreements are reported "
        "without deciding source authority. "
        "Character offsets refer to the answer after trimming outer whitespace."
    )


app = FastAPI(
    title="HalluciGuard 2.0",
    description=(
        "Research verification against user-supplied evidence. "
        "Does not establish whether the evidence is true or current. "
        "The first verification loads local models; no cloud LLM is used."
    ),
    version="0.3.0",
)
_pipeline = None
_index = None
_load_lock = Lock()
_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", include_in_schema=False)
def interface():
    return FileResponse(_static / "index.html")


def get_pipeline():
    global _pipeline
    with _load_lock:
        if _pipeline is None:
            _pipeline = VerificationPipeline()
    return _pipeline


def get_index():
    global _index
    corpus = os.environ.get("HALLUCIGUARD_CORPUS")
    if not corpus:
        raise HTTPException(
            status_code=503,
            detail="Evidence search is off: set HALLUCIGUARD_CORPUS to a SciFact-format corpus.jsonl.",
        )
    encoder = get_pipeline().encoder
    with _load_lock:
        if _index is None:
            _index = PassageIndex.build(
                corpus, encoder,
                cache=Path(corpus).parent / "passage_index",
            )
    return _index


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/verify", response_model=VerificationResult)
def verify(body: VerifyRequest):
    try:
        return get_pipeline().verify(
            body.question, body.answer,
            [item.model_dump() for item in body.evidence],
        )
    except VerificationInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/v1/verify-search", response_model=VerificationResult)
def verify_search(body: SearchRequest):
    index = get_index()
    try:
        return verify_with_search(get_pipeline(), index, body.question, body.answer)
    except VerificationInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
