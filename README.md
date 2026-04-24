# HalluciGuard: Domain-Specific QA with Multi-Layered Hallucination Detection

## NLP Assignment 3 — University of Technology Sydney (2026)

HalluciGuard is a Retrieval-Augmented Generation (RAG) based Question-Answering system for Australian visa information, integrated with a multi-layered hallucination detection pipeline.

## Architecture

```
User Query
    |
    v
[Query Processing]     -- POS Tagging + NER (Weeks 3-4)
    |
    v
[Document Retrieval]   -- Sentence-BERT + FAISS (Week 2)
    |
    v
[Answer Generation]    -- Flan-T5-small (Week 8)
    |
    v
[Hallucination Detection Pipeline]
  Layer A: NER Cross-Check      (Week 4)
  Layer B: NLI Entailment       (Week 7)
  Layer C: Embedding Similarity (Week 2)
  Combined Score -> Verdict
    |
    v
[Response + Trust Verdict]
```

## Syllabus Coverage (10/12 Weeks)

| Week | Topic | Application |
|------|-------|-------------|
| 2 | Word Embeddings | Word2Vec vs GloVe vs SBERT comparison |
| 3 | POS Tagging | Query analysis and key term extraction |
| 4 | NER | Entity cross-checking (Layer A) |
| 5 | Text CNNs | Baseline hallucination classifier |
| 6 | RNN/LSTM | BiLSTM baseline classifier |
| 7 | Transformers/BERT | DeBERTa NLI model (Layer B) |
| 8 | LLMs | Flan-T5 answer generation |
| 9 | Model Distillation | Full vs Distilled comparison |
| 10 | AI Agents | End-to-end agent pipeline |
| 11 | Model Robustness | Adversarial/OOD/paraphrase testing |
| 12 | Hallucination | Core project focus |

## Quick Start

### Google Colab (Recommended)
1. Open `HalluciGuard_NLP_A3.ipynb` in Google Colab
2. Set runtime to GPU (Runtime > Change runtime type > T4 GPU)
3. Run all cells sequentially

### Local Setup
```bash
git clone https://github.com/YOUR-REPO/halluci-guard-nlp.git
cd halluci-guard-nlp
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
jupyter notebook HalluciGuard_NLP_A3.ipynb
```

## Repository Structure

```
halluci-guard-nlp/
├── HalluciGuard_NLP_A3.ipynb    # Main notebook (all code)
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── report/
│   └── nlp_a3_XXXXXXXX.pdf      # Final report
└── presentation/
    └── HalluciGuard_slides.pptx  # Presentation slides
```

## Team

| Member | Contribution |
|--------|-------------|
| Member A | KB construction, RAG pipeline, baselines, notebook |
| Member B | NLI model, hallucination pipeline, distillation, report |

## Acknowledgements

AI tools (Claude, ChatGPT) were used to assist with code generation, debugging, and report drafting, as permitted by the assessment guidelines.
