# Task E1d — ablation test of the identity-stuck-rule / cross-mode-contamination hypothesis

Issued by the principal investigator session, 2026-09-15. Worker: Sonnet. This task is exactly the hypothesis your
own E1c report proposed (`docs/tasks/E1c-report.md`, final section) — adopting it directly rather than rewriting it.
No training: pure forward-pass ablation analysis on `T2_large` bundles already on disk, same method
`metrics.path_faithfulness` already uses for single-node ablation, applied here to whole rules instead.

## Hypothesis, restated precisely (yours, from E1c)
The value-transform tables for `succ` (and structurally, `pred`/`pls2` — tables never moved off identity init) are
causally responsible for their modes' deterministic-step errors. The same identity-stuck rules, when they cross-fire
during `cpy2`/`cpy3`/`rept` steps (because their firing condition is a weighted sum an unrelated feature can satisfy
alone), inject a spurious identity-echo vote that degrades those modes' otherwise-correct copy mechanism.

**Prediction, stated before testing**: ablating the identified identity-stuck rules from the readout and re-running
inference should measurably *improve* deterministic-step accuracy on `succ`, `cpy2`, `cpy3`, `rept` — and should show
little or no improvement on `pred`/`pls2`, since E1c's firing-rate evidence didn't support the same causal story for
those two.

## What to compute
For each of the 3 `T2_large` seeds (re-identify the specific rules per seed the same way E1c did — don't assume
seed 0's rule indices `L0R144/265/363/296` carry over to other seeds):

1. Identify that seed's identity-stuck rules gated on `succ`/`pred`/`pls2` (static table check: still ≈identity,
   per E1c's method) that also show the firing-rate discrimination pattern E1c found for succ (fires much more on
   wrong than right steps, for at least one mode). This may be a different rule count/set per seed — report what you
   find, don't force 3-rules-per-seed if the real count differs.
2. **Ablation**: zero out those specific rules' contribution to the output register only (set their `W2`'s vote
   columns to 0 for the ablated rules — do not remove their effect on the state/routing, only their vote, so this
   tests exactly the "spurious vote" mechanism, not a broader intervention). Re-run inference on the same iid split
   used throughout E0/E1/E1b/E1c. Report deterministic-step accuracy per mode, before vs after ablation, all 3 seeds
   (mean±std).
3. **Control**: ablate the same *number* of randomly-chosen active rules (not the identified ones) and report the
   same before/after comparison — this is the same "random control" pattern used everywhere else in this project's
   metrics (`metrics.py`'s magnitude/faithfulness audits); don't skip it, it's what makes the result mean something
   rather than "ablating any rules changes things."
4. Report overall iid NLL before/after too (both the targeted and random-control ablation) — a real fix should not
   tank the modes that were already working.

## Decision rule
* **Confirms**: targeted ablation improves accuracy on succ/cpy2/cpy3/rept (report the actual deltas — "improves" 
  means a real, seed-consistent delta, not noise) and does not on pred/pls2; random-rule-count control shows no such
  pattern. → this becomes a concrete training-time fix to scope next (e.g. stricter conjunctive gating instead of
  weighted-sum thresholds, or a regularizer discouraging cross-mode rule firing) — propose it precisely in the
  report, don't just say "fix the gating."
* **Refutes**: ablation doesn't move accuracy on any mode, or moves it on modes it shouldn't (e.g. hurts pred/pls2
  too, or the random control shows the same pattern). → say so plainly. Per your own E1c report's fallback: the
  identity-rule firing-rate correlation was non-causal, and the next step should be per-example
  `path_faithfulness`-style ablation tracing on individual wrong predictions rather than static/firing-rate proxies.
* **Partial** (confirms for some of succ/cpy2/cpy3/rept but not all): report exactly which, and your judgment on
  whether that's still enough to motivate a training fix or needs more characterization first.

## Constraints
Standing rules unchanged: sole launcher for anything you start (shouldn't need any — this is forward-pass only),
bounded polling if anything runs a while, no changes to existing `metrics.py`/`rules.py`/`models_g.py` function
behavior for existing callers — implement the ablation as new code (a `feat_masks`-style intervention on `W2`, or a
copy of the params with those entries zeroed, whichever is cleaner) alongside the existing functions, not by
modifying them in place. Commit/push incrementally.

**On reporting progress**: same as before — this should be fast (no training). Name anything long-running once, then
don't ping again until you have the real numbers.

## Report (`docs/tasks/E1d-report.md`, commit and push)
The per-seed rule identification, the before/after accuracy table (targeted vs random control, per mode), the NLL
comparison, the decision-rule outcome stated plainly, and — if confirmed — a precise proposed training-time fix.
Append the research-log entry per the worker README. Any deviations, and why.

When done: commit, push, then give me the concise summary — the verdict and, if confirmed, the proposed fix; if
refuted, the recommended next step.
