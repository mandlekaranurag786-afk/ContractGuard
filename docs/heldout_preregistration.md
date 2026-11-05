
# Held-out test: pre-registration

Written: 4 Nov 2026, before any held-out contract was selected, labelled or run.
Committed to git before selection; the commit time is the proof.

## Purpose

All earlier results (0 unsafe, 42/101 automatic, 47–50/60 extraction fully correct)
come from the same 30 contracts the system was designed on. This test measures the
frozen system once on unseen contracts.

## Contracts

- Pool: CUAD v1 contracts (master_clauses.csv, 510 rows) that are NOT among the 30
  contracts in data/contracts.
- Main set: 10 contracts. Eligibility: CUAD "Renewal Term" AND "Cap On Liability"
  columns both non-empty (master_clauses.csv; empty = "" or "[]").
  This rule was RECONSTRUCTED FROM THE DATA: no written selection rule exists in
  code or docs (the 30 are simply the PDFs in data/contracts; a referenced
  selected_30_contracts.csv does not exist). Of the original 30: 28 meet the rule,
  1 (AlliedEsports) has an empty CUAD Renewal Term, 1 (Pacira) did not match by
  plain file name (trailing space). Pool: CUAD contracts meeting the rule, minus all
  30 originals (matched by file-name stem, stripped and lower-cased; the script
  stops if any original is not found).
- No-answer set: 2 contracts where CUAD's "Renewal Term", "Notice Period To Terminate
  Renewal" and "Cap On Liability" columns are all empty.
- Selection: sort eligible file names, shuffle with random.Random(20261103), take
  the first 10 (main) and first 2 (no-answer). Done by a committed script.
- Replacement: if a selected file is missing from the CUAD PDF folder, or the API
  pre-flight rejects it (unreadable, empty, fully scanned, over 200 pages), it is
  recorded with the reason and the next file in the shuffled order is used.
  No other reason for replacement is allowed.

## Labels

- File: evals/heldout_labels.csv, same columns and rules as
  evals/extraction_labels_v2.csv (empty = unscored, NONE = expected empty).
- Labelled by Anurag BEFORE the system is run on any held-out contract; system
  output is not viewed before labelling is complete and committed.
- CUAD columns are used only as a reading aid; every field is decided from the text.
- Limit: one labeller; agreement cannot be measured.

## Frozen system

- Git tag heldout-v1 on the tested commit. Prompt v2 (PRODUCT_PROMPT_VERSION),
  trust policy KEYWORD, policies/eval_policy.json.
- Pass 1: follow-up off (default). Pass 2: follow-up v2 on.
- Run order: 1 Gemini call -> 1 contract -> all 12. Live once; answers cached.
- No code, prompt, pattern or rule change after the run. Bugs found are recorded
  as open items; a fix would need a new held-out set.

## Success rules

Primary (safety): 0 unsafe decisions on scored held-out rule checks, pass 1.
"Unsafe" = any PASS / FAIL / NOT_APPLICABLE that differs from the truth decision
(same truth function as the 30-contract evaluation). If > 0, the "never decided
wrong" claim is corrected everywhere, with each case explained.

Descriptive (no pass/fail), reported next to the 30-contract numbers:
automation rate, truth FAILs caught, extractions fully correct, HIGH-but-wrong
rate, review root causes, contracts decided.

Follow-up v2 as default: accepted only if pass 2 has 0 unsafe AND at least
ceil(0.10 x scored held-out rule checks) fewer reviews than pass 1 AND every changed
decision is checked by hand against its quote. (Proportional form of the original
"at least 10 fewer of 101", chosen before seeing any held-out data.)

## Reporting

Results are written to docs/decisions.md whatever they show. Small sample: report
"0 of n" with the rule-of-three bound (about 3/n) instead of "never".
