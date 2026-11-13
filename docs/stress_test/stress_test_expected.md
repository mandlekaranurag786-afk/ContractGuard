# ContractGuard stress test: expected results (written BEFORE the run)

Date written: 12 Nov 2026. Four synthetic contracts with fictional parties, written for this
test. They are not CUAD contracts and not part of the held-out set. This is an exploratory
test: it does not change any reported number.

## Success rules (fixed before the run)

1. **Safety (must hold): 0 unsafe decisions in 20 rule checks.**
   A decision is *unsafe* when the system gives an automatic PASS, FAIL or NOT_APPLICABLE that
   is different from the truth below. Where the truth is NEEDS_REVIEW, any automatic decision
   is unsafe.
2. **NEEDS_REVIEW is always safe.** If the truth is PASS or FAIL and the system says
   NEEDS_REVIEW, that is a missed automation, not an error.
3. **Automation (reported, no bar):** how many of the 16 rows with truth PASS or FAIL the
   system decides correctly by itself.
4. **No tuning on these files.** If something is fixed because of this test, the fix must be
   checked on new contracts, not on these four.

## Truth per contract

### A - synthetic_A_Brightwater_Halden_MSA.pdf (truth overall: PASS)

| Rule | Truth | Where | Why |
|---|---|---|---|
| cap_required | PASS | 11.2 | Aggregate cap stated |
| cap_min_2m | PASS | 11.2 | USD 5,000,000 |
| excludes_consequential | PASS | 11.1 | Indirect, special, consequential damages excluded |
| renewal_notice_min_30 | PASS | 4.2 | Renews automatically; notice not fewer than 90 days |
| initial_term_max_60 | PASS | 4.1 | 36 months (until the third anniversary) |

Distractors: monthly fee USD 48,000 (3.1); insurance USD 3,000,000 (12.1); payment 45 days
(3.2); cure period 60 days (5.1); dispute negotiation 30 days (14.2); confidentiality 5 years (6.3).

### B - synthetic_B_Kestrel_Marlowe_License.pdf (truth overall: FAIL)

| Rule | Truth | Where | Why |
|---|---|---|---|
| cap_required | PASS | 9.1 | A cap exists |
| cap_min_2m | FAIL | 9.1 | USD 500,000 |
| excludes_consequential | FAIL | 9.3 | Says consequential losses are NOT excluded and are recoverable |
| renewal_notice_min_30 | FAIL | 3.2 | Renews automatically; notice only 15 days |
| initial_term_max_60 | FAIL | 3.1 | 84 months (seven years) |

Distractors: annual fee USD 310,000 (4.1); invoices 30 days (4.2); warranty 90 days (8.1);
cure period 30 days (11.1); renewal periods of 24 months (3.2).

### C - synthetic_C_Tavira_Greyfield_Distribution.pdf (truth overall: NEEDS_REVIEW)

| Rule | Truth | Where | Why |
|---|---|---|---|
| cap_required | PASS | 8.2 | A money cap exists (the amount is not knowable) |
| cap_min_2m | NEEDS_REVIEW | 8.2 | "Greater of" a Schedule amount (not attached) and 150% of charges |
| excludes_consequential | NEEDS_REVIEW | 8.1 | Depends fully on Schedule 4, which is not attached |
| renewal_notice_min_30 | NEEDS_REVIEW | 2.2 | Continues if parties keep performing; "reasonable" notice, no number |
| initial_term_max_60 | NEEDS_REVIEW | 2.1 | Initial Period is in the Commercial Schedule, not attached |

Distractors: forecast 6 months (4.1); payment 60 days (5.2); defect notice 10 business days
(6.1); cure period 21 days (9.1); 12 months look-back in the cap formula (8.2).

### D - synthetic_D_Corvane_Pellham_Hosting.pdf (truth overall: FAIL) - the trap contract

| Rule | Truth | Where | Why |
|---|---|---|---|
| cap_required | PASS | 11.2 | A cap exists |
| cap_min_2m | FAIL | 11.2 | USD 1,500,000 |
| excludes_consequential | PASS | 11.1 | Excludes losses not arising "directly and naturally" (the meaning of consequential loss), without using the word |
| renewal_notice_min_30 | PASS | 3.2 | Renews automatically; notice not less than 30 days (exactly on the limit) |
| initial_term_max_60 | PASS | 3.1 | 60 months (exactly on the limit) |

Traps: insurance USD 5,000,000 and USD 10,000,000 (9.1); minimum commitment USD 3,200,000
(4.1); non-payment notice 10 days (4.2); termination for convenience 180 days (3.3); records
kept 84 months (7.1); migration help 90 days (12.2). Picking any of these as the cap, the
notice or the term would give a wrong automatic decision.

## Judgement calls in the labels (read before running)

- C `cap_required` = PASS: a formula cap is still a money cap.
- D `excludes_consequential` = PASS: the clause describes consequential loss without naming it.
- B `excludes_consequential` = FAIL: the clause expressly keeps consequential loss recoverable.

## Totals

20 rule checks. Truth: 10 PASS, 6 FAIL, 4 NEEDS_REVIEW.
