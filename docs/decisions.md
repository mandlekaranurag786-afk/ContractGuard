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

##Hybrid Search

Hybrid (RRF k=60) vs dense, recall@1 at overlap≥0.5: Renewal Term net +2 (5 gained, 3 lost), Cap On Liability net −3 (0 gained, 3 lost). BM25 helps clauses with distinctive vocabulary (“renew”) and hurts clauses whose key word is common across the contract (“liability”) while the operative wording (“shall not exceed”) doesn’t match the query. Overall recall@1 hides this, so recall should be reported per clause type from now on.

## Correction: earlier numbers in this file used a buggy metric

The "Baseline results" and "Hybrid Search" sections above were measured
with the old recall@k, which used the rank of the HIGHEST-overlap chunk
instead of the FIRST chunk that passes the threshold. That undercounted
recall by about 14-17 points. The fix is in commit 46024f2. Commit 2aa0390's
message also has the old numbers. I left the old text above as history.

Corrected numbers (58 instances: AlliedEsports renewal = no clause,
OPERALTD renewal = wrong label, both excluded), overlap >= 0.5:

| Method | @1         | @3    | @5    | @10  | MRR@10 |
| ------ | ---------- | ----- | ----- | ---- | ------ |
| Dense  | 72.4% (42) | 94.8% | 98.3% | 100% | 0.823  |
| Hybrid | 74.1% (43) | 93.1% | 94.8% | 100% | 0.835  |

Hybrid vs dense at rank 1 (paired, from the CSVs):

- Renewal Term: 18 -> 23 (8 gained, 3 lost)
- Cap On Liability: 24 -> 20 (0 gained, 4 lost)

## MRR@10 and recall@20 added

Recall@10 reached 100%, so it can no longer show any improvement. The open
problem is ordering (rank 1 only ~73%), so I added MRR@10, which rewards
rank 1 more than rank 5. Before trusting it, I calculated the range MRR
must fall in from the recall counts (dense 0.807-0.848, hybrid
0.813-0.849). Measured 0.823 and 0.835, both inside.

Added recall@20 because a reranker reorders the top 20, so recall@20 is
its ceiling. It is 100% at overlap 0.5. Checked that retrieving 20
instead of 10 did not change recall@1/3/5/10.

## Bug: per-row CSV showed the old buggy rank

The totals were correct, but each CSV row still printed best_rank (rank
of the highest-overlap chunk). Found it by counting: only 14 dense Renewal
rows showed best_rank=1, but the table said 18. Fixed: rows now store
first_hit_rank at 0.5/0.7/0.9, best_rank renamed to best_overlap_rank
(diagnostic only). Added a check that stops the run if per-row counts
don't match recall@1 and recall@10. Codex's summary did not catch this;
comparing raw rows with totals did.

## Reranker experiment (cross-encoder)

Rules written BEFORE running any model:

- Metric: MRR@10 and rank-1 count per clause, overlap 0.5
- Latency budget: median <= 500 ms, p95 <= 1500 ms per query (20 pairs,
  CPU). Reason: ~10 rules per contract x 0.5 s = ~5 s per contract.
- Keep only if >= 3 more rank-1 hits than the BEST baseline (hybrid, 43).
  One instance = 1.7 points, so 1-2 is noise.

Hardware: i5-1135G7 (4 cores/8 threads), 15 GB RAM, Ubuntu, CPU only.

Results (overlap 0.5):

| Config          | Rank 1 | MRR@10 | Renewal | Cap | Rerank median / p95 |
| --------------- | ------ | ------ | ------- | --- | ------------------- |
| Dense           | 42     | 0.823  | 18      | 24  | -                   |
| Hybrid          | 43     | 0.835  | 23      | 20  | -                   |
| Dense + MiniLM  | 45     | 0.860  | 25      | 20  | 568 / 1288 ms       |
| Hybrid + MiniLM | 45     | 0.851  | 25      | 20  | 555 / 1275 ms       |
| Dense + bge     | 22     | 0.554  | 17      | 5   | 4640 / 6199 ms      |
| Hybrid + bge    | 23     | 0.568  | 18      | 5   | 4452 / 5637 ms      |

Recall@20 is the same for every config (a reranker only reorders), and
an automatic check confirms it. MiniLM gave the same numbers in two runs.

Decision: NO reranker. Default stays hybrid.

- MiniLM: +2 vs hybrid (noise) and median 568 ms (over budget).
- bge: -20 and 9x over budget.
  I did not change the rules after seeing the results. The code stays as an
  optional config.

Why MiniLM hurt Cap (6 lost, 2 gained): I printed the new rank-1 chunks.
It chose an insurance clause, a recall clause, a definition
("Environmental Liability means..."), a tariff paragraph, a definitions
list containing "Royalty Cap", and a table of contents. It matches the
WORDS "liability" and "cap", not the meaning. My first guess (CUAD labels
are wrong) was wrong: the labels were fine for these 6.

Sanity check: with 4 test sentences, BOTH models ranked a table-of-contents
line ("15. Limitations of Liability ....8") above a real cap clause. bge
loads correctly (right model class, sensible order for the other 3), so
this is a model weakness, not a bug. bge scores were all below 0.02, so
reranker scores cannot be used as confidence.

Tokens: my estimate "1000 chars ~ 250 tokens" was wrong for one chunk.
A table-of-contents chunk is 766 tokens (dot leaders become many tokens),
so it was cut off at 512. No correct chunk was cut off (211-260 tokens).
bge's tokenizer: max 408, no cut-off.

Latency notes: first MiniLM load took 98 s because of Hugging Face network
checks; with HF_HUB_OFFLINE=1 it took 121 ms. A re-run gave the same 568 ms,
so it was real cost, not memory pressure.

Practical lessons: load one model at a time; download models first, then
run offline (two downloads at once caused a lock wait); use
PYTHONUNBUFFERED=1 and tee to see progress.

## Open items

- Hybrid vs dense is still within noise (43 vs 42). Hybrid is the default
  for now. Decide with more eval data.
- Table-of-contents / heading chunks fool retrieval and rerankers.
  Possible fix: filter them at ingestion.
- Not measured: whether bge-small (embedder) also cuts off the 766-token chunk.
- Still open from before: clause-aware chunking, selected_30 filter check,
  Cap On Liability label mixes money caps and damages exclusions.

## Next: structured extraction with Gemini (Day 1 plan, 18 Oct 2026)

- LLM: Gemini via API key (AI Studio free tier). Main model: Gemini 3.5
  Flash Lite (500 requests/day). One provider for now.
- Eval runs use ONE fixed model (no automatic fallback), so results are
  not mixed. The product API may fall back to another model.
- Cache every LLM response on disk, so re-runs cost no API calls.

## Structured extraction: design (19 Oct 2026)

* Two Pydantic schemas, one per clause type:
  * Renewal Term: found, initial_term_months, auto_renews, renewal_term_months,
    notice_days, evidence_quote, evidence_chunk_id.
  * Cap On Liability: found, has_money_cap, cap_amount, currency, cap_formula,
    has_damages_exclusion, evidence_quote, evidence_chunk_id.
* Why two schemas: each clause type needs different facts, every field has a
  clear type for rules, and each prompt only contains its own definitions
  ("clause isolation").
* Separate has_money_cap and has_damages_exclusion: CUAD's "Cap On Liability"
  mixes caps and exclusions; a rule "cap >= $2M" must never pass because of an
  exclusion.
* Nullable fields: "not stated" is a real answer; forcing a number forces a guess.
* Evidence quote + chunk id: a person can verify it, and my code can check it.
* Input: top 5 hybrid chunks (rank 1 is right only ~3 times in 4). Temperature 0.
* Prompt says the contract text is data, not instructions (prompt injection).
* Field definitions live in one place (FIELD_DEFINITIONS); the labelling guide
  is generated from them, so the model and the labeller use the same meaning.
* Invalid JSON: retry once, then return an error result. Never crash.

## Gemini client, cache, fallback (20 Oct)

* Model: gemini-3.5-flash-lite (AI Studio free tier). New dependency: google-genai only.
* Pacing 4 s between calls (15 requests/min). Retries on 429/5xx: 5 s, 15 s, 45 s.
* LLMQuotaError STOPS an eval run. Recording failures would make accuracy look
  bad for a reason unrelated to the model.
* Disk cache key = sha256(model + schema name + schema JSON + prompt). Only valid
  JSON is cached (otherwise the retry gets the same broken answer).
* FallbackLLMClient exists for the product only. Eval runs use one model, so
  results never mix models.
* API key in .env (git-ignored). Before every commit: review staged files and
  grep the staged diff for the key prefix (must be 0).

## Labels (19-20 Oct, corrected 21 Oct)

* CUAD gives clause spans, not values, so I made my own labels: 30 contracts x 2
  clause types = 60 cases. First draft with AI help; every row reviewed by me
  plus a second review; CHECK rows discussed and fixed with written reasons.
* Labels v2: empty = not scored; NONE = "the correct answer is empty" and IS
  scored. Same v1 answers: fully correct 47/60 -> 44/60. The model didn't get
  worse; the test got stricter.
* The two abstention cases were wrong, the model was right:
  * AlliedEsports: no renewal wording, but a 5-year term is stated, and my
    definition of found includes the term length -> found = True.
  * OPERALTD: "may be extended or renewed by the Parties in a writing
    executed by..." exists. My Part 1 note "no renewal clause exists" was
    WRONG. CUAD's span is still the wrong clause, so it stays a bad label.
* Labels v2 changed 44 cells in 17 rows, each with a reason in label_notes.
  The original labels file is kept unchanged.
* After this, the set has 0 true "no answer" cases (abstention 0/0).

## First real run, prompt v1 (21 Oct)

* Fully correct (labels v2): Renewal 24/30, Cap 20/30 = 44/60.
* Weak fields: notice_days 24/29, cap_formula 25/30, has_damages_exclusion 22/30.
* cap_amount 7/7 sounds perfect but only 7 rows are scorable (3 real numbers +
  4 redacted). Too small to claim anything.
* Error split (per wrong field): retrieval_miss 5, extraction_error 19.
  Known flaw: decided per ROW (was CUAD's clause in the top 5?), but a field can
  live in another chunk. It blames the model for some retrieval misses.
* Cost: ~1,700 input tokens and ~200 output tokens per call; median 1.6 s.

## Faithfulness (21 Oct)

* Strict check: quote must be in the cited chunk (after cleaning). v1: 45/56.
* I diagnosed all 11 failures before changing anything: 5 crossed a chunk
  border, 3 had page numbers/headers inside, 1 cited the wrong chunk,
  1 whitespace only, 1 joined two passages with "...". Mostly my chunking and
  the PDF, not invented text.
* I did NOT loosen the strict check. I added levels: strict > in_context
  (touching retrieved chunks only, never the full contract) > composite
  ("..." parts all found) > not_found. Plus cited_chunk_correct (52/52).
* Zero-width space bug: the PDF had an invisible U+200B in "SECTION 8".
  normalize() now removes invisible characters. Lesson: print with repr().
* Faithful is not the same as correct; both are measured separately.
* Tooling bug: --fake wrote 120 fake answers into the real cache. Deleted them;
  --fake now uses a temporary folder that is always cleaned up.

## Prompt v2 (21 Oct)

* notice_days: ONLY notice to STOP an automatic renewal (incl. termination at
  the end of the term). Not notice to renew/extend/exercise an option, not
  termination for convenience. Null if not automatic. (Bellicum 180 and
  Bravatek 60 were wrong in v1.)
* Redacted amounts: number null, never copy "[***]". A fixed amount is not a
  formula.
* has_damages_exclusion: true if ANY chunk excludes these damages. This rule
  did not help (still 22/30), see the exclusion check below.
* v1 is protected by a hash test. The cache key includes the prompt, so v1
  and v2 answers never mix. Results files are versioned, never overwritten.

## Repeat runs and the v1 vs v2 decision (21 Oct)

* Temperature 0 is not deterministic. Repeat runs used identical settings and
  a separate empty cache folder.
* v1: 44 -> 45 fully correct (3 prediction changes).
  v2: 47 -> 50 (7 prediction changes).
* v2's first run showed 7 field regressions; the repeat fixed 5 of them.
  I called them "systematic" before the repeat. That was wrong.
* Decision: prompt v2 is the product default. More accurate in both runs, and
  its errors are mostly EMPTY values (-> review) instead of WRONG values
  (-> confident wrong decision). Cost: ~7% more tokens, a bit less stable on
  borderline contracts.
* v2's first run had median 7.1 s (p95 41 s); the repeat had 2.2 s. The API
  was slow that day, not the prompt.
* One repeat is evidence, not a full noise measurement.

## Exclusion misses (21 Oct)

* has_damages_exclusion stayed 22/30 in every run. For PACIRA, Pareteum and
  Xpresspa, the exclusion sentence was in a chunk NOT in the top 5
  (:99, :49, :44). Retrieval misses, wrongly counted as extraction errors.
* My earlier guess (a model pattern over 8 cases) was wrong. I only checked
  these 3.
* Possible fix: a second retrieval query just for exclusions.

## Confidence and field status (22 Oct)

* Confidence comes ONLY from code checks, never from the model:
  NOT_FOUND (model said found=false, always review), LOW (quote not found or
  required fields missing), MEDIUM (composite quote or a number not in the
  quote), HIGH (otherwise).
* Number support: number in the quote in any common form, unit within 20
  characters, field keyword within 120 characters. Fixed "three-year" /
  "one-year" (hyphen). Left Hyperion and ordinals as they are (safe: review).
* Result: HIGH-but-wrong 14.3% (v2) and 9.3% (repeat). Target was 5%: missed.
* Every wrong HIGH answer was the model saying false/null.
* Field status: supported / keyword_signal / explicit_negative /
  absence_claim / conflicting / unsupported.
  supported and keyword_signal: 0 wrong in both runs. absence_claim holds the
  errors (exclusion 2 correct / 8 wrong; Cap found=false 0/5).
* Negation bug: "SHALL NOT EXCEED" was treated as negative. In contracts a
  limit is written as a prohibition. Negation is now checked only for
  auto_renews.
* Limits: designed and measured on the same 60 rows; one quote can't support
  every field; no true "no answer" cases (so found=false can never be right
  in this data).

## Rule engine (24-25 Oct)

* Gemini extracts facts; deterministic Python makes the compliance decision.
  No LLM in the decision.
* Rules in JSON (no new dependency): field, operator (>=, <=, ==, is_true,
  is_false), threshold, currency, optional "when" condition.
  "==" only on numbers (True == 1 in Python).
* Decisions: PASS, FAIL, NEEDS_REVIEW, NOT_APPLICABLE.
* Order: extraction error -> NOT_FOUND (never "FAIL: no cap") -> LOW ->
  condition -> field status -> currency -> compare. MEDIUM does not block:
  trust is checked per field.
* Currency must be proven by the quote ($, USD, US Dollars); "Canadian
  dollars" etc. do not count. Unknown or different currency -> review.
* Three trust policies: STRICT (supported), KEYWORD (+ keyword_signal),
  BALANCED (+ explicit_negative). Pre-registered choice: move to a looser
  policy only with 0 unsafe in BOTH runs and >= 3 fewer reviews.
* Unsafe = any final decision (PASS, FAIL or NOT_APPLICABLE) that differs from
  the true decision. A wrong NOT_APPLICABLE silently skips a check.
* Truth = the same rules applied to the labels.
* Bug in my own spec: the truth function checked NONE before the condition,
  so non-auto-renewing contracts got NEEDS_REVIEW instead of NOT_APPLICABLE
  (truth NOT_APPLICABLE showed 0). It made BALANCED look like 4 unsafe; the
  real number was 1 (2Themart, v2 run only).
* Result (scored rule-rows, v2 / repeat):
  STRICT 0 unsafe, 95% review (useless in practice).
  KEYWORD 0 / 0 unsafe, 58% / 57% review. SELECTED.
  BALANCED 1 / 0 unsafe -> blocked.
* Every automatic decision under KEYWORD was correct (42 / 43), but 37-38 of
  them rest on a weak keyword signal.
* The engine PASSes easily with proof but rarely FAILs on missing things
  (2 engine FAILs vs 18 true FAILs): "there is no cap" can't be proven by one
  quote, so it goes to a person.
* Limits: designed on the same rows; keyword is a weak signal; only 3 real cap
  amounts; exceptions like "EXCEPT WITH RESPECT TO INDEMNIFICATION" are not
  handled.

## Contract checker and the 30-contract run (25-26 Oct)

* check_contract(PDF, policy) -> one JSON report + CLI. Each clause type is
  extracted once, only if a rule needs it.
* Overall status: any FAIL -> FAIL; else any NEEDS_REVIEW -> NEEDS_REVIEW;
  else at least one PASS -> PASS; all NOT_APPLICABLE -> NO_APPLICABLE_RULES
  ("this is not an approval"). An empty policy is an error.
* Quota error -> no partial report presented as complete.
* PRODUCT_PROMPT_VERSION = "v2" for checker/CLI/API. Eval code keeps v1
  defaults so old commands reproduce old results.
* Token counts: uncached tokens only; cached tokens reported separately.
* Plain-English reasons, e.g. "The cap is a formula (...), not a fixed amount".
  cap_is_formula only when cap_amount is empty (Hyperion has both a fixed
  amount and a formula).
* 30 contracts, cache only: 0 LLM calls, 0 decision differences vs the
  offline eval (150 rule results). The product path matches the eval.
* Result: 3 FAIL (Bellicum, Centrack, MrsFields), 27 NEEDS_REVIEW, 0 PASS.
  Rule-level review rate 64% (96/150, includes unscored rows).
* All 4 FAILs checked against the text: correct (3 with numeric proof).
* Hyperion: "the lesser of ... or $500,000" can never be >= $2M, so it should
  logically FAIL. The engine doesn't parse formulas -> review (safe miss).
* Time: median 3.9 s, p95 15.9 s per contract with 0 LLM calls (PDF parsing +
  embedding long contracts).
* Incident: Codex recreated .venv in its offline sandbox and Torch was missing.
  Fixed with uv sync in my terminal. Rule: Codex never touches .venv,
  pyproject.toml or uv.lock.

## Evidence follow-up loop: design (26 Oct)

* Goal: fewer unnecessary reviews with 0 unsafe decisions. Improve the
  evidence, don't lower the safety threshold.
* Deterministic controller + 3 tools: search (fixed query per field, skip seen
  chunks, top 3), extract ONE field with its own quote (LLM), verify (code).
* Targets: absence_claim, unsupported, condition_unverified. Not
  clause_not_found, low_confidence, cap_is_formula.
* A follow-up needs a PATTERN, not a keyword: we searched for those words, so a
  keyword proves little. E.g. exclusion wording + damage type in the same
  sentence, and no reversing wording ("shall be liable for", "shall not apply").
* Evidence per field as an overlay; the original extraction is never changed;
  each field is checked only against its own quote.
* Merge: unverified or untrusted -> discard; a follow-up "no" never changes
  anything; null -> accept; unproven false + proven true -> accept;
  explicit_negative + proven true -> conflict; same value -> accept as
  confirmation; different value or currency -> conflict -> review.
  A follow-up only changes NEEDS_REVIEW decisions, never PASS/FAIL.
* Limits: 1 follow-up per field, <= 4 extra LLM calls per contract, 3 new
  chunks. Quota error -> return the original report marked followup_skipped.
* Acceptance (written before the run): 0 unsafe on scored cases, >= 10 fewer
  reviews, every changed decision checked by hand.
* Caution: still the same 30 development contracts. Must be confirmed on
  held-out contracts.

## Open items (updated 26 Oct)

* Held-out set: label ~10 new CUAD contracts and run the full system once,
  without tuning.
* No true "no answer" contracts in the eval.
* Exclusion retrieval misses (follow-up loop or a second query).
* Error attribution per row, not per field.
* Page numbers/headers inside text still break some quotes.
* Formula caps ("lesser of") are not parsed.
* Labels drafted with AI help; labeller agreement not measured.
* Still open from before: hybrid vs dense, TOC/heading filtering,




## Evidence follow-up loop: build (27 Oct)

* Built in small checkpoints, safety core first: follow-up patterns, the merge
  table, the overlay in the rule engine, then the controller loop. All with fake
  LLM tests first.
* Pattern bugs found by reading the patterns against common legal phrases,
  before any real run:
  * "including but not limited to" was counted as a money cap ("limited to").
  * "to the maximum extent permitted by law" was counted as a cap ("maximum").
  * "shall not be limited to" was counted as a cap.
    Fixed with tests using the exact phrases.
* Merge guard moved first: a trusted field must never be followed up, even if
  the new evidence is weak (otherwise the bug would be hidden).
* Currency for cap_amount follow-ups is merged as its own field; a trusted
  currency is never replaced, only flagged as a conflict.
* Patterns don't read the contract, the LLM does. Patterns only decide whether
  an answer can be trusted automatically. Unknown wording -> NEEDS_REVIEW
  (costs review time, never a wrong decision). Wrong patterns are the real
  safety risk, so false positives are fixed first.

## Follow-up loop: live runs (27 Oct)

* Run 1 INVALID: all 60 follow-up calls failed with 400 INVALID_ARGUMENT
  ("Unknown name additional_properties"). The one-field schemas used
  extra="forbid", which adds additionalProperties: false; Gemini rejects it.
  387 fake tests passed because fakes accept any schema.
  Fixes: schemas without extra="forbid"; a contract test that no schema sent to
  Gemini contains additionalProperties; the trace now keeps the error message.
  New rule: one real call, then one contract, then the batch.
* Run 2 (loop v1): 0 unsafe, only 2 fewer reviews (59 -> 57). 57 of 60
  follow-ups returned nothing (no value). No verification bug.
  Design flaw found in the trace: v1 only read UNSEEN chunks, but for
  value_unsupported and condition_unverified the proof is usually in the chunks
  already seen (the single main quote just doesn't contain it).
* Added MergeOutcome NO_VALUE so "found nothing" is not reported as
  "unverified".
* Loop v2 (the ONE allowed revision): original chunks + up to 3 unseen chunks
  for value_unsupported and condition_unverified; unseen only for
  value_absence_claim.
  Result: 0 unsafe, 9 fewer reviews (59 -> 50), all 9 changed decisions checked
  by hand against the quotes and correct (8 were "unsupported" exclusions).
* Pre-registered rule: 0 unsafe AND at least 10 fewer reviews. 9 < 10 ->
  REJECT. I did not move the bar.
* Decision: the default product stays follow-up OFF. Follow-up v2 is kept as an
  opt-in option (--followup --followup-version v2). It can become the default
  only after passing the same rule on held-out contracts.
* Cost of v2: ~2 follow-up calls per contract (max 4), median time per
  contract 5.8 s -> 9.9 s.

## Verdict accuracy and root causes (28 Oct)

* Default (follow-up off) / opt-in (follow-up v2):
  * unsafe at rule and contract level: 0 / 0
  * rule checks decided automatically: 42/101 / 51/101 (all correct)
  * truth PASS decided automatically: 40/67 / 49/67
  * truth FAIL caught automatically: 2/18 / 2/18
  * contracts decided automatically: 3/30 / 9/30
* Why reviews happen (default, 50 unnecessary reviews):
  correct but unproven 30 (60%: unsupported 16, absence claim 11, explicit
  negative 3), retrieval miss 10, low confidence 6, extraction wrong 4 (8%).
* Main finding: the LLM is not the bottleneck. In only 4 of 50 reviews was
  Gemini wrong. Better EVIDENCE would help more than a better model.
* 11 of 16 missed FAILs are cap_required: "there is no cap" cannot be proven by
  one quote. Proving absence is risky: here a pattern miss would cause an
  unsafe FAIL (fail-closed reverses).
* Caveat: contract-level truth is partial (only 1 contract has all 5 rules
  scored).
* Next steps, ranked by data: per-field evidence quotes in the main extraction;
  number-proof fixes; proving absence (careful design); retrieval misses. All
  need a held-out test.

## LangGraph experiment (28 Oct)

* Rebuilt ONLY the follow-up controller in LangGraph (separate folder
  experiments/langgraph_followup/, optional dependency group langgraph-exp).
  Same tools, same cached answers. Nothing in src/ depends on it.
* Equivalence: 30 contracts, 150 rule results, 60 trace steps: 0 differences,
  0 live calls.
* Plain Python: 214 controller lines, 0 extra packages, median 43 ms per
  contract. LangGraph: 258 lines (+20%), 19 extra packages (incl.
  langchain-core, langsmith), median 54 ms. Speed difference doesn't matter
  next to ~2 s per Gemini call.
* Lesson: LangGraph's real strength is checkpoint/resume and human interrupts.
  To get it, ALL progress (budget, seen chunks, overlays) must live in the graph
  state. My version kept the budget in a closure, so it wasn't resumable.
* Decision: plain Python stays the product version. I would switch for a
  reviewer workflow (pause on NEEDS_REVIEW, then resume), long agents that must
  survive restarts, or many branching/parallel tools.

## Repo cleanup (28 Oct)

* Committed the follow-up LLM cache (88 answers) so all follow-up results can be
  reproduced offline. Deleted regenerable regression outputs and old logs.
* .gitignore: manual_check/ and evals/*.log.
* Never run "git clean -f" without "git clean -n" first: it would delete the
  cache.

## Fix #2 Part A: ordinal + anniversary (28 Oct) - REJECTED

* Goal: prove "fifteenth (15th) anniversary" = 180 months, so Penntex and
  RangeResources initial_term_max_60 (truth FAIL) become automatic FAILs.
* Pre-registered ACCEPT: 0 unsafe in both runs and the batch; both targets
  become FAIL; no earlier automatic decision changes; "supported" with wrong =
  0 for every field in both runs.
* Result: 0 unsafe, 0 decisions changed, supported-wrong 0, but 0 of 2 targets
  fixed -> REJECT. Not merged (git restore).
* Cause: my mistake. I designed from a summary, not the exact text. Both
  contracts say "until the end of the fifteenth (15th) Contract Year", not
  "anniversary".
* Not extended to "Contract Year": it is a defined term; the first Contract
  Year can be shorter or longer than 12 months. Proving 180 months would need
  reading the definition. These cases stay in review (safe).
* Lesson: always print the exact text before designing a pattern.
* Part B (formula bounds like "the lesser of X or $500,000") was not done: high
  risk of misreading formulas, only 1 case (Hyperion). Separate future
  experiment.

## Open items (updated 28 Oct)

* Held-out set: label ~10 new CUAD contracts and run the full system once,
  without tuning. Needed before any default change (e.g. follow-up v2).
* No true "no answer" contracts in the eval.
* Per-field evidence quotes in the main extraction (biggest expected gain:
  16 "unsupported" reviews).
* Proving absence ("no cap anywhere") without unsafe FAILs.
* "Contract Year" durations (need definition lookup).
* Formula caps ("lesser of" / "greater of") not parsed (Part B).
* Exclusion and clause retrieval misses.
* Error attribution per row, not per field.
* Page numbers/headers inside text still break some quotes.
* Labels drafted with AI help; labeller agreement not measured.
* Still open from before: hybrid vs dense, TOC/heading filtering,
  clause-aware chunking, selected_30 filter check, bge-small truncation.
* clause-aware chunking, selected_30 filter check, bge-small truncation.
