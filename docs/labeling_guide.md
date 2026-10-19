# Extraction labeling guide

These labels are the answer key for the extraction eval.

## Renewal Term

FIELD DEFINITIONS
- found: true if the text states the contract's term length or how it renews
- initial_term_months: length of the FIRST term in months (1 year = 12). null if the term is only given as an end date; do not calculate from dates
- auto_renews: true if the contract renews automatically unless a party gives notice; false if renewal needs a new agreement or a party's option; null if not stated
- renewal_term_months: length of EACH renewal period in months
- notice_days: notice needed to STOP AUTOMATIC RENEWAL (non-renewal notice), in days (1 month = 30 days). NOT notice for termination for breach or for convenience

## Cap On Liability

FIELD DEFINITIONS
- found: true if the text limits or excludes a party's liability
- has_money_cap: the total amount a party can be liable for is limited (a fixed amount OR a formula)
- cap_amount: only a fixed number written in the text; else null
- currency: ISO code such as "USD", else null
- cap_formula: the cap in words if it is not a fixed number (e.g. "fees paid in the 12 months before the claim")
- has_damages_exclusion: indirect / consequential / special damages are excluded. This alone is NOT a money cap
- If both a cap and an exclusion appear, set both true and quote the cap sentence. If there are several caps, use the general overall cap.

## Labeling rules

- Label from the contract text, not from what the model might say.
- If CUAD's answer looks wrong, label what the contract says and explain in `label_notes`.
- Leave a field empty if the contract does not state it.
- Booleans: True / False. Numbers: digits only (12, not "twelve").
