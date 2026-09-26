# jev-live-transcription — ticket orchestration status

## NEXT PHASE (2026-09-26): RunPod GPU migration + demo webapp — plan ready, not yet built

Tickets #1-26 below are all done (see table). Everything in this file below this note describes
that already-completed CPU-based work. The next phase of this project — moving the benchmark to a
RunPod GPU pod, plus a separate stakeholder-demo web page — has been fully researched, grilled,
and adversarially reviewed (`/plan-review`, findings incorporated) in a separate session, and will
be **built in a fresh session**, not this one.

The full plan lives at `C:\Users\jorda\.claude\plans\reactive-finding-yeti.md` (two parts: Part 1
is the GPU migration, Part 2 is the demo webapp). That plan file is the source of truth for the
next phase — read it before doing anything else if you're picking this project back up.

## LIVE STATE (as of 2026-09-26 ~07:15 UTC, before a /compact) — historical, CPU-phase only

- **Ticket #9's real 100-call benchmark run is IN PROGRESS right now**, as a detached OS
  process (pid 43810, launched via nohup/disown so it survives independent of Claude Code's
  own process tracking) in worktree
  `C:\Users\jorda\projects\jev-live-transcription\.claude\worktrees\agent-ab79729275eda3c05`.
  DB: `data\benchmark.sqlite3` inside that worktree. Command:
  `uv run jlt batch --db-path data/benchmark.sqlite3` (defaults: call_concurrency=1,
  gliner_concurrency=1, enable_llm_baseline=False — sequential, GLiNER+jev only, no GPT-5.1).
  ~24/100 calls done as of this note, healthy, zero errors, ~144 jev calls/call, ~$0.002/call
  jev cost (~$0.20 total projected). Expect ~3-4 hours total wall-clock.
- Ticket #9's supervising Claude agent is named/id **`ab79729275eda3c05`** — it goes dormant
  between check-ins (normal, not a crash); resume it via SendMessage to check status or to
  proceed with PR/code-review/rebase/merge once the run completes. **All of its code changes
  are already safety-committed** to its worktree branch as commit `cf22ca1` (not pushed, not
  merged) — 103 tests passing, includes: batch_runner.py, cli.py, sequential-by-default,
  --enable-llm-baseline gate, GLiNER independent-timing fix, jev latency-capture fix, warm-up
  step, and regression tests for the confident-rejection carry-forward clearing.
- Research on the GLiNER→jev handoff design question (4 forks dispatched): 3/4 converged on a
  clear recommendation (distinctUntilChanged + 2-tick settle debounce), already turned into
  ticket #20's spec. The 4th (entity-linking prior art) hit an agent-routing/context-confusion
  glitch and was DROPPED as supplementary/non-blocking — do not re-dispatch it, ticket #20 is
  already fully spec'd without it.
- Do NOT dispatch tickets #19 or #20 until #9 actually merges to master (both touch files #9
  also touched — #19: gliner_pipeline.py, #20: pipeline_core.py — dispatching before #9 merges
  risks conflicts). #19 and #20 CAN run in parallel with each other once #9 lands (no shared
  files between them), though #20 should land before/alongside #19 per the context-window
  caching interaction documented on ticket #20.
- Cost so far in this session: jev is negligible (~$0.20 for the full 100-call run). GPT-5.1
  is currently disabled by design — do NOT enable it (`--enable-llm-baseline`) without Jordan's
  explicit approval; he wants it run once, deliberately, as an approved "final benchmark," not
  as routine data collection.

## Progress check (2026-09-26, post-compact)

- Run confirmed alive (uv process, ~1h09m elapsed at check time), 25/100 calls complete, zero
  LLM baseline rows (correct — disabled by default).
- **Transient jev outage, self-resolved, NOT a code bug**: calls 9-11 hit a wave of
  `403 RBAC: access denied` from `api.typesafe.ai` (430 failed `pipeline_runs` rows, each
  correctly logged with `error` populated per the plan's "never swallow jev failures" rule — no
  silent data loss). Recovered cleanly mid-call-11; calls 12-25 all succeeded normally. This will
  show up as reduced jev field coverage for calls 9-11 in the eventual scoring report — that's
  expected/correct, not something to patch. Worth a one-line callout in the final benchmark
  report as a real-world reliability data point.
- **GLiNER latency confirms the ticket #19 motivation** (measured on real data, not a smoke
  test): `gliner_standard` avg 1118ms / max 5358ms; `gliner_stream_pii` avg 658ms / max 3696ms —
  both well over the sub-400ms target. `jev` itself is healthy: avg 162ms / max 6725ms (one
  outlier, likely a retry). No action needed now — #19 already scoped to fix this, queued behind
  #9 merging.
- **Verified single process, no contention**: confirmed via `Get-CimInstance Win32_Process` that
  only one `jlt batch` process tree exists (uv pid 25488 → jlt.exe pid 13896 → child python pid
  19192, started 2026-09-26 01:09:41 AM). Two other process groups seen in `ps aux` (8:39 PM,
  9:06 PM starts) are unrelated `windows-mcp`/`mcp-google-vision` MCP servers for this Claude Code
  session, not stray batch runs — ruled out as a contamination source. Revised pace: 25/100 calls
  in ~76 min (~3.05 min/call) → projected ~5hr total wall-clock (slower than the original 3-4hr
  estimate, consistent with GLiNER's known-high latency, not a new problem).


Repo: https://github.com/jpol34/jev-live-transcription
Concurrency cap: floor((RAM_GB-4)/4) = floor((16-4)/4) = 3 (per ~/.claude/CLAUDE.md formula, RAM_GB=16)
in-flight agents: 1/3

NOTE: corpus regen (ticket 1) came in at 369 words/call avg, still short of the 450-600 target
for a 3-4 min call. Not blocking so far; revisit prompt tuning before ticket 9's real batch run.

| # | Title | Status | Blocked by |
|---|-------|--------|------------|
| 1 | Project scaffolding, dependency isolation, and secrets loading | done (PR #11, b62eea6) | None |
| 2 | Live-replay pacer | done (PR #12, e19f217) | 1 |
| 3 | SQLite capture schema and single-writer store | done (PR #13, 5c7cb02) | 1 |
| 4 | GLiNER hybrid candidate extraction | done (PR #14, aee1a72) | 1 |
| 5 | jev resolver pipeline | done (PR #15, 57fd1d7) | 1 |
| 6 | GPT-5.1 baseline pipeline | done (PR #16, d011984) | 1 |
| 7 | Single-call pipeline orchestration and capture | done (PR #17, b518613) | 2,3,4,5,6 |
| 8 | Live single-call TUI viewer | done (PR #18, f7304f1) | 7 |
| 9 | Headless batch runner across the full 100-call corpus | **MERGED** (PR #21, squash commit 9f0051d). Code review found 6 real issues (TUI silently never running GPT-5.1 despite its own docstring/key-loading implying it did; batch_runner cleanup not isolated; concurrency=0 hangs forever; GLiNER error-path latency asymmetric vs success-path; PII streaming model's cross-call-id thread-safety unverified; a duplicated-branch simplification) -- all fixed and verified (111 tests passing) before merge. | 7 |
| 10 | Scorer and benchmark report | blocked | 9 (done), still needs 19/20/22 for valid data |
| 19 | Bound zero-shot GLiNER latency with a sliding context window | **MERGED** (PR #25, merge commit c8bbf32) -- verified independently. 135 tests passing. `GLINER_ZERO_SHOT_WINDOW_CHARS=200` (deviated from the ticket's ~800-1200 example after real benchmarking on this machine -- see caveat below). Code review found 6 real issues (mid-word truncation risk, a too-loose latency test, 2 stale comments, an overclaiming config comment, a suggested reuse), all fixed. Real-run verified on the corpus's 2 longest transcripts: latency stays flat regardless of call length; no field's detection structurally lost. | 4, 22 |
| 20 | Gate jev re-confirmation calls behind a genuine candidate-set change (SettleGate) | **MERGED** (PR #23, squash commit 86c5e75) -- verified independently via `gh pr view`/`git log`. 121 tests passing (started 111). New `settle_gate.py` module, `JEV_RECONFIRM_SETTLE_TICKS=2`. Code review found 3 real issues, all fixed: a smoke-test regression, a real design gap (uncommitted Choice results were being marked "resolved" forever instead of retried, now fixed+regression-tested), a stale docstring. | 5, 9 |
| 22 | Cache jev candidate context snippets so they survive ticket 19's GLiNER windowing | **MERGED** (PR #24, merge commit 5d259fc) -- verified independently. 131 tests passing (started 111; #20 added ~8, this added 12 incl. a real end-to-end aging-out integration test). Had to reconcile a real conflict against #20's concurrently-merged SettleGate changes in `jev_pipeline.py` (resolved by hand, kept both). Code review found one real design flaw (a candidate's field-level fallback context could get permanently cached instead of a later precise per-candidate window), fixed before merge. #19's agent notified it's clear to proceed. | 5, 9 |
| 26 | Reduce zero-shot GLiNER latency further below 200-char-window's ~400-600ms band | **Investigation complete, no code changed, awaiting Jordan's call.** Real profiling on this machine: the model's transformer forward pass is 226ms/~98% of total latency (tokenization/collation/decoding are negligible). Label-embedding caching is architecturally impossible on this checkpoint (`encode_labels`/`predict_with_embeds` raise `NotImplementedError` -- it's a joint-attention model, not a bi-encoder); the one available "precompute" mechanism (`compress_prompt_embeddings`) gave zero latency win and collapsed recall to 0%. Thread tuning already at the least-bad setting (4 threads; fewer makes it worse). Shrinking the window further trades real recall (~13% relative drop at 100-150 chars vs. 200) without reliably closing the remaining gap to sub-400ms, which is dominated by concurrent-model CPU contention, not window size. See PR-less branch `ticket-26-gliner-label-cache` (untouched) for the investigation. | 19 |

## Dispatch (2026-09-26)

All three post-#9 tickets dispatched in parallel as fresh worker agents in worktrees (concurrency
cap 3/3, per floor((16-4)/4) formula -- fully used). #20 and #22 both touch `jev_pipeline.py` from
different angles (expected -- merges serialize, conflicts resolved at merge time by whichever
lands second). #19 was explicitly told to wait for #22 to merge before merging itself, since #19's
windowing is what breaks jev's context visibility without #22's caching already in place -- this
is now a real instruction to the dispatched agent, not just a tracker-level blocking edge.

**Window-size caveat (2026-09-26)**: #19 landed with `GLINER_ZERO_SHOT_WINDOW_CHARS=200`, not the
ticket's example ~800-1200 range -- real on-machine benchmarking showed 800-1200 chars cost
800ms-1.5s+/tick, nowhere near the 400ms target. 200 chars lands mostly in a ~400-600ms band (down
from the old unbounded baseline's avg 1118ms/max 5358ms) -- a large real improvement, but NOT a
strict <400ms guarantee on every tick, especially under any CPU contention. Flagged to Jordan for a
decision on whether this is good enough to proceed to the real benchmark run or worth pushing
further (e.g. a smaller window, or optimizing the model call itself) before spending the ~5hr run.

**Near-miss caught 2026-09-26**: #19's agent (aa029cffb692ea2a6) checked #22's status itself,
correctly saw #22 wasn't built yet, but wrongly concluded nobody owned it and started building #22
itself -- which would have raced/duplicated the dedicated #22 agent (a109fd76a6b5f88fc) already
working on it. Caught via its status report and corrected via SendMessage: told to stand down from
building #22, just poll for #22's merge and proceed with #19 afterward, escalate if the wait
exceeds ~30-45 min. Lesson for future dispatches: tell each agent explicitly which other tickets
are already claimed/in-flight and by whom, not just "this ticket is blocked by that one" -- an
agent seeing an unblocked-but-unbuilt blocker has no way to know it's already assigned elsewhere
unless told.

Stale-worktree cleanup attempted for tickets 1-8's already-merged worktrees (9 total sitting in
`.claude/worktrees/`, never cleaned up) but blocked by the auto-mode permission classifier
("Interfere With Workloads", likely due to several being marked `locked`) -- not pursued further.
Flagged to Jordan as a separate hygiene item, not blocking current work.

## Ticket re-scoping (2026-09-26)

Per Jordan's "scope this fix properly": the context-window-caching requirement (originally a
comment-only addendum on #20) is now its own ticket, **#22** (NOT #21 -- #21 was consumed by ticket
9's PR number, since PRs and issues share one sequence on GitHub). #19 now has a real tracked
blocking edge on #22 (must merge no later than #19), replacing the soft "should land
before/alongside" note. #20's scope is now just the SettleGate re-confirmation gate.

Dispatch order once #9 merges: #19, #20, #22 can all be dispatched in parallel (worktrees) --
#19 touches `gliner_pipeline.py`+`config.py`; #20 and #22 both touch `jev_pipeline.py` (some
build-time conflict risk between those two, resolved at merge time per the normal rebase-before-
merge step) but #19 is genuinely blocked by #22 merging first, not by #20. Correct merge order:
#22 (or #20) first, then #19 last, so #19 never lands without #22's caching already in place.
