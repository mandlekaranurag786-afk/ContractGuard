# Design decisions

Each entry: what we decided, and why. Written as the project is built, so
the reasoning is accurate to when the decision was actually made, not
reconstructed later.

## Project layout mirrors the pipeline stages

The src/contractguard/ subpackages (ingestion, chunking, retrieval,
extraction, agent, api) each correspond to one stage of the system. This
means a bug points to a specific folder (wrong extracted value -> extraction;
wrong retrieved clause -> retrieval), and one stage can be changed or
re-measured without touching the others. It also means the folder tree
doubles as an outline I can walk an interviewer through.

## Config lives in environment variables, not in code

Settings that differ between my laptop, tests, and a deployed server
(API keys, model names, DB URLs) shouldn't require a code change to update.
Keeping secrets out of source files also keeps them out of git history,
where a leaked key would stay forever even after being "removed." A
.env.example is committed to document which variables are needed; the real
.env is gitignored.

## data/ is gitignored

The contract PDFs are large and are freely re-downloadable (from CUAD), so
committing them would bloat the repo for no benefit, and some real-world
contracts could carry sensitivity concerns. The repo should only contain
what I actually wrote. data/README.md (to add) documents where to get the
files and which ones to use, so the setup is still reproducible.

## Ingestion keeps per-page text and page numbers, never one flat string

Citations ("page 35") and the retrieval eval ("is the right page in the top
k?") both depend on knowing which page a piece of text came from. Flattening
pages into one big string now would make this impossible to add back later
without re-parsing everything.

## doc_id is a hash of the file content, not the filename

CUAD filenames contain commas, spaces, and special characters that break
paths and URLs. A content hash also makes ingestion naturally idempotent:
re-ingesting the same file produces the same ID instead of creating a
duplicate, which matters once this becomes an upload API.

## Pydantic models for ParsedDocument / ParsedPage, not raw dicts

Every later stage (chunking, extraction, the API) depends on this shape.
A typed schema fails loudly (a validation error) if something's wrong,
instead of silently passing a malformed dict further down the pipeline
where the error becomes harder to trace.

## clean_text preserves line and paragraph structure; it does not flatten text

The first version collapsed all whitespace (including newlines) into single
spaces, which merged clause headings into the surrounding paragraph text
(e.g. "ARTICLE 15 TERM AND TERMINATION 15.1 Term. This Agreement..." all on
one line). Chunking needs to detect clause boundaries by recognizing
headings at the start of a line, so flattening the text would have made
clause-aware chunking impossible without re-parsing. Fix: collapse repeated
spaces/tabs within a line and strip trailing whitespace, but keep single
newlines (line breaks) and collapse 3+ newlines to exactly 2 (paragraph
breaks). Hyphens are left untouched — contracts contain many real hyphenated
words (non-voting, co-chairpersons), and end-of-line hyphenation in these
PDFs is not reliably a word-wrap artifact, so "fixing" it would silently
corrupt real words.

## needs_ocr counts non-whitespace characters, not raw string length

The first version counted len(clean_text), which includes the single spaces
left between words after cleaning, so a page with only a handful of real
characters (e.g. scattered page numbers) could still pass the 20-character
threshold and never get flagged. Counting only non-whitespace characters
matches what the flag is actually supposed to detect: pages PyMuPDF could not
meaningfully extract text from (blank pages, scans).

## A flagged page is retained, not silently dropped or treated as empty

If an unreadable page just produced an empty string indistinguishable from
"this page has no clause," downstream stages (and a human reading an
eval report) would have no way to tell the difference between "the clause
genuinely isn't here" and "we couldn't read this page." For a
compliance-adjacent tool, a confident-looking wrong answer is worse than
a visible gap, so the page is kept with needs_ocr=True instead of removed.

## Batch ingestion (ingest_directory) lives in the ingestion package, not in the script

scripts/ingest_all.py only parses arguments, calls ingest_directory, and
prints. This came after an earlier version put the whole loop inside the
script itself, which meant it could only be tested by mocking the real
ingest_pdf function — and that mocked test passed even when the actual
script, run against real data, found zero files. Testing against real
generated PDFs (tests/test_batch.py) catches that kind of bug; mocking the
one thing you're trying to verify does not.

## Heading detection must handle more than one numbering style, plus a fallback

A survey across all 30 contracts (regex counts for numbered headings,
ARTICLE headings, Section headings, and ALL-CAPS lines) showed most
contracts use simple numbered headings ("15.1 Term."), but several use
ARTICLE or Section-prefixed headings instead, and sub-items like "(b)",
"(c)", "i.", "j." look similar to headings but are not — they're list items
inside a clause and must not be treated as chunk boundaries. Any
clause-boundary chunker has to recognize multiple heading styles and
explicitly exclude lettered sub-items, plus fall back to another rule
(e.g. paragraph breaks with a size cap) for documents where no heading
pattern is detected, rather than assuming one style covers every contract.

## CUAD is used only to test extraction accuracy, never inside the live app

The real application takes two inputs: a small set of hand-written policy
rules and a user-uploaded vendor contract. CUAD's PDFs and
master_clauses.csv exist only during development, as a stand-in corpus and
an expert-verified answer key, to measure whether the extraction code
reads contracts correctly before it's trusted on a real upload. Once
extraction is validated this way, CUAD is not referenced by the running
system at all.


## Retrieval baseline — dense vector search (BAAI/bge-small-en-v1.5)

### Finding 1: Dense search ranks definitions above operative clauses

Tested on Bellicum (Miltenyi Biotec Supply Agreement), query "what is the renewal term":

- Rank 1 (score 0.6644): chunk :20, page 5 — a glossary line reading only
  `"Term" means the Initial Term and any Renewal Term thereof.` Names the
  term but doesn't explain it.
- Rank 3 (score 0.6439): chunk :191, page 38 — Section 15.1, the real
  answer (10-year initial term, 5-year renewals, 6 months' notice).

Dense embeddings match on topic/vocabulary density, not on whether a chunk
actually answers the question. A one-line cross-reference and a full
operative clause can score nearly identically if they share vocabulary.
BM25 wouldn't fix this (it's not a missing-keyword problem — both chunks
contain the literal terms). A cross-encoder reranker, which reads query +
chunk together, is the likely fix.

### Finding 2: True vocabulary mismatch (different failure mode)

AlliedEsports Sponsorship Agreement: CUAD's own answer key has an empty
Renewal Term list for this contract. Manually confirmed: the word
"renewal" does not appear anywhere in the document. This contract has no
renewal clause at all — a different problem from Finding 1, and one BM25
and reranking can't solve, since there's no answer to find. Must be
excluded from recall scoring, not counted as a miss.

### Finding 3: Close-but-real ranking miss

2TheMart Co-Branding Agreement, query "what is the renewal term":

- Rank 1 (score 0.6713): Disclaimer of Warranties clause — irrelevant.
- Rank 2 (score 0.6665): chunk :18 — the real answer (30-day renewal
  notice clause). Margin over the wrong top result: only 0.0048.

Confirms Finding 1's pattern on a second contract: the right chunk is
often retrieved, but can lose a close race to an unrelated chunk that
shares generic contract boilerplate vocabulary.

## Eval harness built (evals/eval_retrieval.py)

Scores retrieval automatically against CUAD's master_clauses.csv as the
answer key, across all 30 selected contracts, for Renewal Term and Cap On
Liability. Scoring method: word-overlap fraction (what % of the CUAD
answer's words appear in a retrieved chunk), not exact-match, because
exact match is too brittle against our chunk-boundary cuts. Recorded per
rank (top 10), not collapsed to pass/fail, so the threshold can be chosen
by inspecting real score distributions rather than guessed upfront.

Bug found and fixed: filename join between parsed JSON and
master_clauses.csv used exact string match, silently dropping any
contract whose filename had different punctuation (e.g. "A&R" in the CSV
vs "A_R" in our parsed files for the Pacira contract). Fixed by
normalizing both sides (lowercase, strip non-alphanumeric) before
joining. This matters because a silent join failure doesn't error — it
just quietly shrinks your eval set, and you'd report a baseline on 29
contracts believing it covered 30.

### Baseline results (30 contracts, 59 clause instances, overlap≥0.5 threshold)

- recall@1:  57.63% (34/59)
- recall@3:  86.44% (51/59)
- recall@5:  89.83% (53/59)
- recall@10: 98.31% (58/59)

Interpretation: the correct chunk is almost always retrievable within the
top 10 (98%), but lands at rank 1 only ~58% of the time. The ~30-point
gap between recall@1 and recall@5 is the ranking-confusion problem from
Findings 1 and 3, now confirmed across the full dataset, not just
hand-checked examples. This is the number hybrid search and reranking
will be measured against.

### Finding 4: CUAD labeling defect (OPERALTD Service Agreement)

Renewal Term query scored only 0.43 overlap at best rank (5) — the only
instance below the 0.5 threshold. Investigated by pulling CUAD's raw
answer span directly: it points to a data-protection/confidentiality
clause, not anything about renewal. Manually confirmed in the source PDF:
the word "renewal" does not appear anywhere in the contract. This is not
a retrieval failure — CUAD's own ground truth is mislabeled for this row
(non-empty span pointing at the wrong clause, rather than correctly
marked empty like AlliedEsports). Excluded from scoring with this
specific reason logged, distinct from Finding 2's exclusion.

## Next steps

- Hybrid search (BM25 + dense, RRF fusion) — targets the rank-1 vs rank-5
  gap shown in the baseline.
- Cross-encoder reranking — targets Finding 1's "definition vs operative
  clause" confusion specifically.
- Re-run eval_retrieval.py after each change; compare against this
  baseline, not against hand-read examples.
