
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
