# Karnataka & Central Schemes RAG - Project Notes

## Objective
Build a local-first knowledge base that ingests government scheme PDFs and prepares them for Retrieval-Augmented Generation (RAG).

### Primary goal
- Ingest text-based PDF documents into a local vector database.
- Preserve source metadata (especially page numbers) so future answers can include citations.
- Use this indexed repository later to answer user questions through a RAG system.

## Current Scope (MVP)
- Input format: text PDFs.
- Vector store: Chroma (local).
- Citation requirement: yes (page-level citations).
- Embedding strategy: local SentenceTransformer embeddings.

## Progress So Far
- Clarified project direction: PDF ingestion pipeline first, RAG answer layer later.
- Reviewed current script (`new reliance.py`) and identified embedding flow:
  - Embeddings are currently created implicitly by Chroma via `DefaultEmbeddingFunction()`.
  - Query embeddings are also generated implicitly during `collection.query(...)`.
- Created local feasibility script: `st_local_test.py`.
- Verified local model execution from terminal:
  - Model tested: `sentence-transformers/all-MiniLM-L6-v2`
  - Result: successful load + embedding generation + similarity ranking.
  - Observed embedding dimension: 384.

## Technical Notes
- Local SentenceTransformer setup is feasible on this machine.
- Current warnings seen in terminal are non-blocking for the feasibility test:
  - Hugging Face unauthenticated request warning.
  - Windows symlink cache warning.

## Next Planned Steps (No code changes until approval)
- Replace Chroma `DefaultEmbeddingFunction()` with an explicit SentenceTransformer embedding function.
- Keep citation metadata in stored chunks (`source`, `page`, `doc_id`).
- Add simple ingestion conventions for dedup/versioning later.

## Status
Project discovery and local embedding feasibility are complete.
Ready to move to integration changes after explicit approval.
