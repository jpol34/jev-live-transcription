# Accuracy benchmark results

Full 100-call corpus, `gliner_jev` pipeline, scored with `scripts/score_recall.py` against the
checked-in Phase 0 baseline (`benchmarks/score_recall/baseline.json`). Numbers below are from a
real GLiNER + real jev/typesafe.ai API run on a RunPod A100 via `jlt gpu-run --call-concurrency
100`, using the GPU batching infrastructure validated in #71-75, against an image rebuilt from
this ticket's own commit. Raw snapshot: `benchmarks/accuracy/gpu_run_results.json`.

A local CPU run (`uv run jlt batch`, `full_run.sqlite3`/`full_run_results.json` in this directory,
not committed) was scored independently as a cross-check before the GPU run landed. Both runs
land in the same ballpark per field (GLiNER's span extraction is deterministic; jev's LLM-side
resolution has only small run-to-run variance) -- e.g. `permission_to_enter` recall 36.84% on both
runs, `budget_amount` recall 33.33% on both, `price_quoted` recall 28.00% on both.

The Phase 0 baseline was scored with a buggy `is_match` regex that could never match a
`$`-prefixed value preceded by whitespace (fixed in #90/#91). This run uses the fixed matcher, so
part of `price_quoted`/`budget_amount`'s recall movement below is attributable to that scorer fix,
not only to the pipeline changes -- the two effects aren't separated further here, since both
fields also carry genuine, independently-verified pipeline changes (the per-field threshold
override, confirmed in the jev call-volume columns below).

## Results

| field | taxonomy | recall (base→now) | precision (base→now) | jev calls (base→now) | Choice calls (base→now) |
|---|---|---|---|---|---|
| caller_name | span | 91.92%→94.95% | 95.79%→97.92% | 1394→1397 | 919→828 |
| email | span | 54.55%→66.67% | 69.23%→88.00% | 489→488 | 266→251 |
| phone_number | span | 86.67%→86.67% | 100.00%→100.00% | 488→425 | 76→53 |
| unit_number | span | 91.80%→91.80% | 82.35%→81.16% | 744→737 | 414→385 |
| amenities_requested | list_span | 57.14%→50.00% | 20.00%→17.95% | 891→971 | 464→521 |
| pet_info | list_span | 63.64%→63.64% | 87.50%→93.33% | 106→85 | 77→69 |
| permission_to_enter | determination | 0.00%→36.84% | n/a→93.33% | 0→192 | 0→6 |
| work_order_issue | span | 15.22%→19.57% | 50.00%→56.25% | 42→42 | 8→6 |
| move_in_date | span | 66.67%→62.50% | 48.48%→46.88% | 274→261 | 85→71 |
| price_quoted | span | 16.00%→28.00% | 50.00%→29.17% | 62→285 | 18→149 |
| budget_amount | span | 0.00%→33.33% | n/a→36.36% | 30→217 | 3→88 |

`n/a` precision means the Phase 0 baseline committed zero values for that field, so no precision
was computable. No field currently carries the `multi_fact` taxonomy tag -- `work_order_issue` is
the only field that was ever a candidate for it, and #79 reclassified it to `span` (see
`work_order_issue`'s ceiling section below).

## permission_to_enter: recall win, no precision cost

Recall went from 0.00% to 36.84% (0/38 → 14/38 matched) and precision landed at 93.33% (14/15
committed values correct) -- #81's local rule-based determination classifier is doing real work
with a low false-positive rate. Choice-call volume rose from 0 to 6 calls total, a small absolute
increase against the corpus's ~38 relevant calls. The field was completely uncapturable before
#81: GLiNER's zero-shot span extraction reports no signal at all for a yes/no judgment phrased
indirectly ("I guess it's fine if I'm not there"), so recall could only move by adding a
classification path outside GLiNER's span-extraction shape.

## budget_amount: recall win, real precision and jev-volume cost

Recall went from 0.00% to 33.33% (0/12 → 4/12 matched). Precision landed at 36.36% (4/11 committed
values correct) and jev call volume rose sharply: 30→217 total jev calls, 3→88 Choice
(multi-candidate) calls specifically -- a ~29x increase in Choice-call volume for this one field.
#79's own PR described a 10-call subset finding of ~33% precision and Choice volume rising 2→18;
the real full-corpus numbers confirm the precision estimate almost exactly (36.36% vs. ~33%) but
show a substantially larger Choice-call increase (3→88, not 2→18) -- the 10-call subset
undersold the volume cost. Lowering `budget_amount`'s post-hoc confidence floor from the default
0.30 to 0.15 (#79) is what unlocked recall on this field (GLiNER already found the correct span in
most cases, just scored it below the old global cutoff), and the corresponding jev-volume/precision
cost is the direct, expected tradeoff of that same change, not a separate regression.

## price_quoted: recall win, but precision dropped and jev volume rose sharply

This tradeoff was not mentioned in #79/#81/#84's own PR descriptions, so it's called out here in
full: recall rose from 16.00% to 28.00% (4/25 → 7/25 matched), but precision *dropped* from 50.00%
to 29.17% (7/24 committed values correct, out of 24 -- three times the 8 committed at baseline),
and jev call volume rose from 62 to 285 total calls, with Choice calls rising from 18 to 149 (an
~8x increase). `price_quoted` shares #79's threshold change with `budget_amount` (0.30 → 0.12), and
the same mechanism applies: a lower floor surfaces more low-confidence candidates, which recovers
some previously-missed correct spans but also feeds jev many more wrong or ambiguous candidates to
adjudicate. The original global 0.30 threshold was deliberately chosen to favor recall over
precision project-wide; this field's real cost from loosening it further is larger than the
recall gain it bought, and is flagged here as a tradeoff to weigh, not a hidden win.

## amenities_requested: small recall regression on a field nobody touched

Recall dropped slightly (57.14% → 50.00%, 8/14 → 7/14 matched) and Choice-call volume rose (464 →
521), even though no ticket changed `amenities_requested`'s threshold, label description, or
taxonomy. The root cause is #79's change to `config.GLINER_ZERO_SHOT_THRESHOLD` -- the floor
confidence passed to the single joint GLiNER model call across *all* eleven labels at once, lowered
from 0.30 to 0.05 so `price_quoted`/`budget_amount`'s real per-field cutoffs (applied post-hoc)
could see candidates that used to be discarded before ever reaching that per-field filter. Because
GLiNER scores all eleven labels jointly in one forward pass per tick, that lower shared floor
changes which candidate spans the model considers and reports at all for *every* field, not just
the two whose post-hoc cutoff moved -- `amenities_requested`, `caller_name`, `unit_number`,
`pet_info`, and `move_in_date` all show small (a few points either way) shifts in this same run for
the same reason. `amenities_requested`'s shift happens to land as a small regression; the others
land flat or slightly positive. This is a side effect of the shared-floor mechanism `PER_FIELD_THRESHOLDS`
relies on, not a change specific to this field.

## work_order_issue: accepted recall ceiling (structurally uncapturable calls)

`work_order_issue` recall is 19.57% (9/46), up from the Phase 0 baseline's 15.22% (7/46) --
`work_order_issue` was reclassified from `multi_fact` to `span` in #79's `FIELD_TAXONOMY` after #78's
diagnosis found the compound-fact pattern too rare (~3-5 of 46 calls) to justify dedicated
resolver machinery (#80 was closed as not-applicable on that basis).

Five calls carry a compound ground-truth string synthesizing two distinct facts raised in separate
transcript turns, which plain span extraction cannot produce as a single committed value:

| call | ground truth | this run's committed value |
|---|---|---|
| 018 | "Leak under kitchen sink and noise when cold water is turned on" | "maintenance issue" (generic, matches neither fact) |
| 032 | "Leaky faucet; bathroom light flickering" | "leaky faucet" (only the first fact) |
| 046 | "leaky pipe under kitchen sink, possible mold or additional damage" | *(nothing committed)* |
| 075 | "No heating for two weeks; work order incomplete" | *(nothing committed)* |
| 092 | "Shrubs need trimming, possible dampness in living room, check window seals." | "damp issue" (only the second fact, paraphrased) |

018 and 032 were confirmed multi-fact cases per #78's diagnosis (032 also had an unrelated,
separately-fixed ground-truth bug in #81); 046 and 075 were borderline. In every one of these five
calls, the pipeline either commits nothing or commits only a fragment of the compound fact, never
the full joined string -- confirming this is a structural ceiling of span extraction, not a bug: a
`span`-taxonomy field can hold one candidate span per commit, and no candidate the pipeline ever
sees spans both turns at once. This is an accepted limitation, not a regression to chase further at
this field count.

## Smoke checks

Re-ran the three originally-diagnosed examples end-to-end against this run's capture DB:

- **Call 005** (`permission_to_enter`): committed `"yes"` (confidence 0.9) -- matches ground truth.
  Confirms #81's determination classifier fix.
- **Call 007** (`budget_amount`): committed `"$1,900"` (confidence 1.0) -- matches ground truth
  exactly. Confirms #79's threshold fix.
- **Call 018** (`work_order_issue`): committed `"maintenance issue"` -- does not match the compound
  ground truth. Confirmed as an expected, documented failure (see the ceiling table above), not a
  surprise regression; the multi-fact commit-join path was never built (#80 closed as
  not-applicable).

## Tests

`uv run pytest tests/`: 337 passed, 0 failed (scoped to `tests/` to avoid a known, separately
tracked bug in a bare `uv run pytest` from the repo root involving unrelated benchmark files with
duplicate basenames -- issue #86).
