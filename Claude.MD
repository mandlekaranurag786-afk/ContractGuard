# ContractGuard

A system that ingests vendor/procurement contracts (PDFs), extracts structured
data from them (renewal terms, liability caps), and checks that data against
a company's internal policy rules — returning pass / fail / needs_review with
the exact clause as evidence.

This is a personal learning project. The goal is to understand every design
decision well enough to defend it in an interview, not just to have working
code. See docs/decisions.md for the reasoning behind each choice.

## Working rules (read before writing any code)

1. Before implementing anything, explain your plan in plain language and list
   any alternatives you considered (e.g. "PyMuPDF vs pdfplumber vs pypdf —
   here's why PyMuPDF"). Wait for explicit approval before writing code.
2. Do not add a new dependency without asking first and explaining why it's
   needed.
3. Keep changes scoped to one stage/module at a time. Don't refactor
   unrelated code as a side effect of a feature request.
4. Prefer the simplest implementation that solves the actual problem. Don't
   add abstraction (extra classes, config options, indirection) for
   hypothetical future needs. If you can't name the specific failure a piece
   of code prevents, don't add it.
5. After implementing something, run it against the real data in
   data/contracts/ yourself (not just the test suite) and report the actual
   output, not just "tests pass."
6. When something fails or looks wrong, report it plainly rather than
   quietly working around it.

## Project layout

src/contractguard/
ingestion/ - PDF -> structured text (ingest_pdf, ingest_directory)
chunking/ - splitting document text into retrievable chunks
retrieval/ - embedding + search (vector, hybrid, reranking)
extraction/ - structured field extraction (Pydantic schemas + confidence)
agent/ - compliance-check agent loop (tool-calling)
api/ - FastAPI app
scripts/ - CLI entry points that call into src/contractguard
tests/ - mirrors src/contractguard structure
evals/ - evaluation harness and labeled test sets
docs/ - decisions.md, requirements.md, architecture.md
data/ - contracts/ (gitignored PDFs), parsed/ (gitignored JSON)

## Commands

uv sync # install dependencies
uv run pytest # run tests
uv run ruff check . # lint
uv run python scripts/ingest_all.py [data/contracts] # batch-ingest PDFs

## Status

- [X] Project skeleton
- [X] Ingestion (single PDF -> ParsedDocument, batch directory ingestion)
- [X] Chunking
- [ ] Embeddings + vector search (naive RAG baseline)
- [ ] Eval harness (CUAD as answer key)
- [ ] Hybrid search + reranking
- [ ] Extraction with confidence scoring
- [ ] Compliance agent loop
- [ ] API layer
- [ ] Deployment
