# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **Knowledge Graph + RAG** (Retrieval-Augmented Generation) pipeline for diabetes Q&A. Given a natural-language question, the system pulls facts from a biomedical knowledge graph (Hetionet) **and** vector-searches document chunks (WHO PDF + MedlinePlus XML), then assembles a prompt that a downstream LLM can answer from.

## Setup

All scripts must be run from inside the `kg_rag_diabetes/` subdirectory (where `config.py` lives), because `ROOT = Path(__file__).resolve().parent` pins all data paths relative to that file.

```bash
cd kg_rag_diabetes/kg_rag_diabetes
pip install -r requirements.txt
```

## Pipeline — run these steps in order on a fresh clone

```bash
python download_data.py   # Step 1: downloads raw data into data/raw/
python ingest.py          # Step 2: chunks WHO PDF + MedlinePlus XML -> data/processed/chunks.jsonl
python build_kg.py        # Step 3: filters Hetionet triples -> data/processed/diabetes_triples.csv
```

## Running and evaluating

```bash
# Demo: runs 5 built-in questions (shows graph facts, retrieved chunks, cosine scores)
python demo_b.py

# Demo: run a single custom question (also prints the full LLM prompt)
python demo_b.py "What symptoms are associated with type 2 diabetes?"

# Evaluation: Recall@1/3/5 + correct-rejection rate
python evaluate.py
```

## Architecture

| File | Role |
|---|---|
| `config.py` | Single source of truth for all paths, URLs, and tunable parameters |
| `download_data.py` | Step 1 — downloads Hetionet TSV/GZ, MedlinePlus XML, WHO PDF |
| `ingest.py` | Step 2 — word-level sliding-window chunking of PDF and XML into `chunks.jsonl` |
| `build_kg.py` | Step 3 — filters Hetionet edges by `KEEP_METAEDGES` and "diabet" disease nodes into `diabetes_triples.csv` |
| `retriever.py` | Vector index (SBERT or TF-IDF fallback). Embeddings are MD5-keyed and cached as `.npy` files so the model only runs once |
| `kg.py` | Loads the NetworkX `MultiDiGraph`, does substring entity recognition against question text, returns 1-hop graph facts using `TEMPLATES` |
| `hybrid.py` | Combines graph facts + top-k vector chunks into `hybrid_context()` dict; computes `enough` flag; `build_prompt()` formats the final LLM prompt |
| `demo_b.py` | End-to-end demo driver |
| `evaluate.py` | Recall@k over `eval_set.csv`; also measures out-of-scope rejection rate |

### Data flow

```
question
  └─► kg.py: find_entities() + graph_facts()  ──┐
  └─► retriever.py: retrieve()                 ──┤ hybrid.py: hybrid_context()
                                                  └─► format_context() + build_prompt()
                                                       └─► (feed to your LLM)
```

The `enough` flag in `hybrid_context()` is `True` if any graph facts were found **or** the best cosine score exceeds `MIN_SCORE`. Downstream code should refuse to answer when `enough` is `False`.

## Key configuration knobs (all in `config.py` or env vars)

| Parameter | Default | Meaning |
|---|---|---|
| `EMBED_BACKEND` | `"sbert"` | Set env var to `"tfidf"` to skip model download |
| `MIN_SCORE` | `0.30` | Cosine threshold below which the system says "not enough info" |
| `TOP_K` | `3` | Number of document chunks returned per query |
| `CHUNK_WORDS` | `150` | Chunk size in words (SBERT max is 256 tokens) |
| `KEEP_METAEDGES` | `{"CtD","CpD","DpS","DrD","DlA","DaG"}` | Hetionet relation types to include in the KG |

## Evaluation dataset (`eval_set.csv`)

Each row has: `id`, `question`, `type` (`answerable`/`no_answer`), `expect` (substring that must appear in a retrieved chunk), `expect_graph` (substring that must appear in graph facts). Evaluation is content-based, not index-based, so it stays valid after re-chunking or data updates.
