# Stress test: 4 synthetic contracts

Four contracts with **fictional parties**, written for this test with deliberately hard
legal wording. They are not CUAD contracts and are safe to send to Gemini.

| File | Truth (overall) | What makes it hard |
|---|---|---|
| `synthetic_A_Brightwater_Halden_MSA.pdf` | PASS | Long sentences; the term is "until the third anniversary" (36 months) |
| `synthetic_B_Kestrel_Marlowe_License.pdf` | FAIL | Four rules fail; consequential losses are expressly *not* excluded |
| `synthetic_C_Tavira_Greyfield_Distribution.pdf` | NEEDS_REVIEW | Key values are in schedules that are not attached; formula cap |
| `synthetic_D_Corvane_Pellham_Hosting.pdf` | FAIL | Trap numbers: USD 5,000,000 insurance and USD 3,200,000 commitment next to the real USD 1,500,000 cap |

`stress_test_expected.md` holds the true answer for all 20 rule checks and the success
rules. It was written **before** the first run and has not been changed since.

## Results

Follow-up off (product default). "Decidable" = the 16 rows whose truth is PASS or FAIL.

| Run | Unsafe | Decidable rows decided correctly | FAILs caught | Overall status equal to truth |
|---|---|---|---|---|
| First run (12 Nov) | 0 of 20 | 6 of 16 | 0 of 6 | 1 of 4 |
| After fix A (12 Nov) | 0 of 20 | 9 of 16 | 2 of 6 | 3 of 4 |

- All 4 rows whose truth is NEEDS_REVIEW stayed in review in both runs.
- Fix A made `cap_min_2m` decidable: PASS in A, FAIL in B and D.
- A later change ("search near the quote") also decided the initial term in A, B and D,
  but changed nothing on real CUAD contracts, so it was not kept (see `docs/decisions.md`).

## Limits

- Only 4 contracts, written and labelled by one author (Claude). Synthetic, clean modern
  wording. This is an exploratory check, not a benchmark.
- The contracts were used to find the fix A bug, so the "after fix A" row is not
  independent evidence.

## How to repeat

Start the server, then upload each PDF on the demo page (follow-up off), or:

```bash
for f in docs/stress_test/synthetic_*.pdf; do
  curl -s -F "file=@$f" http://127.0.0.1:8000/check | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['overall_status'], [(r['rule_id'], r['decision']) for r in d['rule_results']])"
done
```
