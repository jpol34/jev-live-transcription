# `gliner_only` benchmark results

Full 100-call corpus, real GLiNER + real jev/typesafe.ai API run on a RunPod A100 via `jlt
gpu-run`-equivalent (`--call-concurrency 100`), against an image whose baked-in `src/` was verified
byte-identical (per-file SHA-256) to this repo's current `master` tip before the run -- the pod was
already up from #108's tuning session and reused rather than rebuilt, since the hash check confirms
it, not an assumption, is running the checked-in `GLINER_ONLY_*` constants (see this project's own
prior stale-pinned-image lesson in `benchmarks/accuracy/RESULTS.md`).

Two runs were collected:

- **Concurrent** (`benchmark.sqlite3` via `jlt batch --enable-gliner-only`, jev still enabled):
  `gliner_jev` and `gliner_only` both resolve off the same GLiNER extraction per tick, for an
  apples-to-apples accuracy diff off identical candidates. Used for every recall/precision/
  decision-reason number below.
- **Solo** (`jlt batch --enable-gliner-only --disable-jev`): zero `jev`-stage rows recorded (verified
  directly against the DB), for the real, measured `gliner_only` latency number, free of any
  resource contention with jev's own network calls. Used for every latency number below.

Both runs completed 100/100 calls with zero pipeline errors. `gliner_only`'s own recall/precision
numbers are identical between the two runs (GLiNER extraction is deterministic and
`GlinerOnlyResolver` doesn't depend on whether jev is running), confirming the concurrent run's
accuracy numbers aren't an artifact of jev/gliner_only contention. Raw snapshots:
`benchmarks/gliner_only/gliner_only_snapshot.json`, `gliner_jev_snapshot.json` (`score_recall.py
--save` output from the concurrent run). Raw capture DBs are not committed (`*.sqlite3` is
gitignored, same convention as `benchmarks/accuracy/`).

`benchmarks/accuracy/RESULTS.md` is unchanged by this ticket.

## Success bar (from the `gliner_only` plan)

No field regresses more than 5 points recall/precision vs. `gliner_jev`'s current numbers;
aggregate stays within 3 points; latency drops substantially.

## Verdict: FAIL on accuracy, PASS on latency

Latency drops substantially, as intended. But recall and/or precision regress past the 5-point bar
on **9 of 11 fields** -- several by 15-50 points -- and aggregate recall/precision both regress by
more than 4x the 3-point tolerance. `gliner_only` is not a drop-in replacement for `gliner_jev` as
currently tuned; see per-field sections below for where and why.

## Accuracy results (concurrent run, same GLiNER candidates both arms)

| field | taxonomy | gliner_only recall | gliner_jev recall | Δ recall | gliner_only precision | gliner_jev precision | Δ precision | within 5pt bar? |
|---|---|---|---|---|---|---|---|---|
| caller_name | span | 55.56% | 94.95% | -39.39 | 55.00% | 97.92% | -42.92 | no |
| email | span | 45.45% | 66.67% | -21.21 | 36.59% | 88.00% | -51.41 | no |
| phone_number | span | 86.67% | 86.67% | 0.00 | 82.54% | 100.00% | -17.46 | no |
| unit_number | span | 78.69% | 91.80% | -13.11 | 66.67% | 81.16% | -14.49 | no |
| amenities_requested | list_span | 50.00% | 50.00% | 0.00 | 14.00% | 17.50% | -3.50 | **yes** |
| pet_info | list_span | 50.00% | 63.64% | -13.64 | 84.62% | 93.33% | -8.72 | no |
| permission_to_enter | determination | 39.47% | 36.84% | +2.63 | 68.18% | 93.33% | -25.15 | no |
| work_order_issue | span | 4.35% | 19.57% | -15.22 | 22.22% | 56.25% | -34.03 | no |
| move_in_date | span | 45.83% | 62.50% | -16.67 | 42.31% | 46.88% | -4.57 | no |
| price_quoted | span | 52.00% | 28.00% | +24.00 | 52.00% | 29.17% | +22.83 | **yes** |
| budget_amount | span | 16.67% | 33.33% | -16.67 | 7.41% | 36.36% | -28.96 | no |
| **aggregate (micro-avg)** | | **53.23%** | **67.74%** | **-14.52** | **51.56%** | **74.43%** | **-22.87** | **no** |

`gliner_jev`'s numbers in this run land within run-to-run LLM variance of the checked-in
`benchmarks/accuracy/RESULTS.md` numbers (e.g. `caller_name` 94.95%/97.92% both places,
`permission_to_enter` 36.84%/93.33% both places), confirming this run is a valid basis for the diff
above rather than an outlier run.

## None-of-these / margin-too-close rates (the `gliner_only` analogue of jev's Choice-call volume)

`score_recall.py`'s "jev calls / choice calls / avg distinct" columns hardcode `WHERE stage =
'jev'` and print `0` for every field under `--pipeline gliner_only` -- not a bug, just not
meaningful for this arm (omitted from the table above for that reason). The real signal for
`gliner_only`'s decision cost is `decision_reason`, parsed from `gliner_only_commit` rows via the
new `scripts/score_decision_reasons.py`. "none-of-these" here is `margin_too_close` (2+ candidates,
settled, but no candidate's margin over the runner-up clears the threshold) plus `below_floor` (2+
candidates, settled, but even the top-ranked one doesn't clear its commit floor) -- both are
`_resolve_settled` outcomes that actively reject every candidate for that tick, mirroring
`is_none_of_these=True`. `single_below_floor` (the 1-candidate case that fails to commit) is
excluded from this rate -- a lone low-confidence candidate never triggers settle-gated arbitration,
so it isn't a confident rejection the way a settled none-of-these decision is.

| field | total decisions | none-of-these rate | margin-too-close rate |
|---|---|---|---|
| caller_name | 872 | 66.97% | 62.16% |
| email | 144 | 38.89% | 36.11% |
| phone_number | 220 | 6.82% | 6.82% |
| unit_number | 339 | 28.91% | 21.24% |
| amenities_requested | 368 | 36.68% | 20.11% |
| pet_info | 47 | 38.30% | 31.91% |
| permission_to_enter | 24 | 0.00% | 0.00% |
| work_order_issue | 38 | 15.79% | 0.00% |
| move_in_date | 158 | 23.42% | 12.03% |
| price_quoted | 131 | 32.82% | 29.77% |
| budget_amount | 63 | 57.14% | 57.14% |

"Total decisions" counts every tick `GlinerOnlyResolver` actually returned a resolution for (a
settled commit or a settled none-of-these), not every tick or every call -- it's the same unit
`score_recall.py`'s jev call counts use for the fields they cover.

## Per-field tradeoffs

### caller_name: the field the margin heuristic handles worst

66.97% of `caller_name`'s decisions land as none-of-these, almost all (62.16%) `margin_too_close`
-- GLiNER's zero-shot extraction surfaces filler words and greeting fragments ("Caller", "Hi", "um")
as low-confidence `caller_name` candidates alongside the real name, and once 2+ distinct candidates
are present, `GlinerOnlyResolver`'s purely score-based margin check has no way to recognize that one
candidate is semantically a name and the others aren't -- exactly the discrimination jev's LLM-based
arbitration was doing for this field (a ~39-point recall regression, ~43-point precision
regression). This is the single largest driver of the aggregate regression above.

### price_quoted and amenities_requested: the two fields gliner_only handles as well or better

`price_quoted` recall and precision both improve (+24.00, +22.83) -- `gliner_jev`'s Choice-call
volume for this field was already flagged in `benchmarks/accuracy/RESULTS.md` as the field where a
lowered confidence floor bought recall at a steep precision cost (148 Choice calls, 29.17%
precision); `gliner_only`'s local margin gate rejects more of those low-confidence candidates
outright (32.82% none-of-these) rather than asking jev to arbitrate among them, landing on a better
precision/recall tradeoff for this specific field. `amenities_requested` is flat on recall and
within the 5-point bar on precision -- the only other field that clears the bar.

### permission_to_enter: recall holds, precision collapses

Recall improves slightly (+2.63) but precision drops 25.15 points (7 of 22 committed values wrong,
vs. 1 of 15 for `gliner_jev`). `GLINER_ONLY_DETERMINATION_COMMIT_FLOOR` (0.55) lets more of the
classifier's mid-confidence, hedged-phrasing commits through than jev's own LLM-based Noul/Choice
judgment did for the same borderline cases -- committing more, but with a real accuracy cost.

### budget_amount and work_order_issue: the two worst-performing fields

`budget_amount` regresses on both recall (-16.67) and precision (-28.96), with a 57.14%
none-of-these rate that is itself split entirely between rejecting genuinely ambiguous candidates
and still landing on wrong ones when it does commit (2 of 27 committed values correct) --
`GLINER_ONLY_COMMIT_THRESHOLD`'s per-field override for this field was already left at its
pre-existing 0.15 in #108 specifically because "its correct and wrong commits overlap too heavily
for any floor to separate them"; this run's numbers confirm that diagnosis rather than contradicting
it. `work_order_issue` recall craters to 4.35% (2/46, down from 19.57%): 23 of its 38 decisions are
`single_below_floor` -- GLiNER most often surfaces exactly one `work_order_issue` candidate per tick
and its score usually doesn't clear even this field's already-lowered 0.40 override, so the field
mostly never commits at all under `gliner_only`. This field also carries the structural multi-fact
ceiling already documented in `benchmarks/accuracy/RESULTS.md` (compound ground-truth strings no
single span can match) -- `gliner_only` performs worse than `gliner_jev` even within that shared
ceiling.

### email, phone_number, unit_number, pet_info, move_in_date: smaller but still out-of-bar regressions

Each of these regresses past the 5-point bar on at least one metric (email and unit_number on both;
phone_number on precision only; pet_info on both; move_in_date on recall). None shows a distinct
new failure mode beyond the general pattern above: a purely confidence/margin-based commit policy
is less discriminating than jev's LLM-based arbitration whenever a field's candidate set contains
more than one plausible-looking span.

## Latency

Per-tick total latency (`gliner_standard` extraction + resolution), via `scripts/score_latency.py`:

| pipeline | run | n_ticks | mean_ms | median_ms | p95_ms | error_rate |
|---|---|---|---|---|---|---|
| gliner_jev | concurrent (jev + gliner_only both resolving that tick) | 15472 | 501.7 | 356.1 | 1132.5 | 0.00% |
| gliner_only | concurrent (same run, still shares GPU/network load with jev) | 15472 | 330.9 | 307.4 | 664.8 | 0.00% |
| gliner_only | **solo** (`--disable-jev`, real zero-jev-calls run) | 15472 | **294.9** | **324.0** | **403.9** | 0.00% |

The solo run is the real, measured claim: mean latency drops 41.2% (501.7ms -> 294.9ms) and p95
drops 64.3% (1132.5ms -> 403.9ms) vs. `gliner_jev` on the same corpus. The p95 drop is the larger
effect -- `gliner_jev`'s tail is dominated by jev's network round trips (visible in the concurrent
run's own `gliner_only` row already running faster than `gliner_jev` at the same time, before even
removing jev's contention entirely). Median latency drops more modestly (9.0%), since both arms pay
the same shared GLiNER extraction cost as their floor.

## Tests

`uv run pytest tests/`: 412 passed, 0 failed, unaffected by this ticket (no production code changed
-- this ticket only adds the benchmark writeup and `scripts/score_decision_reasons.py`).
