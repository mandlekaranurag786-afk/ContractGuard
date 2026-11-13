# Second held-out test: pre-registration

Written: 13 Nov 2026, before any contract for this test was selected, labelled or run.
Committed to git together with `scripts/select_heldout2.py` before the script is run;
the commit time is the proof.

## Purpose

The first held-out test (tag `heldout-v1`, 10 Nov) measured the frozen system once on 12
unseen contracts. Since then the held-out results were looked at, and fix A (cap amount
evidence check) was found partly from a held-out contract. Its held-out gain is therefore
post-hoc. This test measures the **current** system once on 12 contracts that neither the
system nor the developers have seen.

## Contracts

- Pool: CUAD v1 contracts (master_clauses.csv, 510 rows) that are NOT among the 30 design
  contracts in `data/contracts` and NOT among the 12 held-out contracts in `data/heldout`.
- Main set: 10 contracts. Eligibility exactly as in the first test: CUAD "Renewal Term" AND
  "Cap On Liability" both non-empty (empty = "" or "[]").
- No-answer set: 2 contracts where CUAD "Renewal Term", "Notice Period To Terminate Renewal"
  and "Cap On Liability" are all empty.
- Selection: sort eligible keys, shuffle with `random.Random(20261114)`, take the first 10
  (main) and first 2 (no-answer). Done once by `scripts/select_heldout2.py`, a copy of
  `scripts/select_heldout.py` with only these changes: the seed, the output paths
  (`data/heldout2`, `evals/heldout2_selection.csv`) and the exclusion of `data/heldout`.
  Name matching, the 510-name checks and the pre-flight are unchanged.
- Replacement: only for the pre-flight reasons of the first test (missing PDF, unreadable,
  empty, fully scanned, over 200 pages). Recorded with the reason; the next file in the
  shuffled order is used. No other reason is allowed.
- Until selection is committed, no CUAD contract outside the 42 already used is opened.

## Labels

- File: `evals/heldout2_labels.csv`, same columns and rules as `evals/heldout_labels.csv`
  and `docs/labeling_guide.md` (empty = unscored, NONE = expected empty).
- Labelled BEFORE the system is run on any of these contracts. Nobody (person or assistant)
  sees ContractGuard output on them before the labels are committed.
- Process (as Amendment 4 of the first test): an AI assistant (ChatGPT) proposes labels
  from the PDFs; a second assistant (Claude) reviews them against the labelling rules;
  Anurag checks the open or disputed fields in the PDFs and decides.
- found=False rows: other fields empty (as Amendment 3 of the first test).
- CUAD columns are a reading aid only; every field is decided from the text.
- Limits: AI-proposed labels may share errors with the system's LLM; one human labeller,
  so agreement cannot be measured.

## Frozen system

- The commit that is run gets the git tag `heldout2-v1`. It contains fix A.
  Prompt v2 (PRODUCT_PROMPT_VERSION), trust policy KEYWORD, `policies/eval_policy.json`.
- One pass only: follow-up off (the product default). Follow-up v2 is not re-tested
  (rejected twice by pre-registered bars; it stays opt-in).
- Run order: 1 Gemini call -> 1 contract -> all 12. Live once; answers cached and committed.
- No code, prompt, pattern or rule change after the run. Problems found are recorded as
  open items; a fix would need yet another unseen set, which is NOT planned (see Reporting).

## Success rules

Primary (safety): 0 unsafe decisions, counted as in the first test:
- scored rule checks: any PASS / FAIL / NOT_APPLICABLE that differs from the truth
  decision (same truth function as before);
- plus every automatic decision on a found=False row, checked by hand against the text;
  it counts as unsafe unless the text clearly supports it.
If the total is > 0, the "0 unsafe" claim is corrected everywhere (README, docs, interview
material), with each case explained.

Descriptive (no pass/fail), reported next to the first held-out test and the design set:
automation rate, truth FAILs caught, extractions fully correct, HIGH-but-wrong rate,
unnecessary-review root causes, contracts fully decided.

Fix A on fresh data (descriptive): the same cached answers are re-scored, cache-only
(0 Gemini calls), with the code of commit `385f0ae` (the last commit before fix A).
Rule rows whose decision differs between the two runs are attributed to fix A and listed,
each with its truth.

## Reporting

- Results go to `docs/decisions.md` and the README whatever they show.
- Small sample: report "0 of n" with the rule-of-three bound (about 3/n), never "never".
- This is the last held-out set before the interviews. A disappointing result is reported
  as it is; it is not re-tested on a new draw.
