# Task E1c — characterize the deterministic argmax errors directly (routing-margin theory is dead)

Issued by the principal investigator session, 2026-09-15. Worker: Sonnet. Read `docs/tasks/E1b-report.md` first: the
routing-tie-break mechanism is refuted, not weakly — it reverses on the `cpy3`-near-switch case it was built to
explain. Read `docs/PLAN-v3-observability-frontier.md` §7 for the surrounding context. This task is the three cheap
angles E1b's report proposed as its own recommendation, combined into one task since all three are analysis-only on
bundles already on disk (`T2_large`, all 3 seeds).

No training. No new mechanism theory to test — the point of this task is to describe the errors accurately before
proposing another explanation, per the "characterize before fixing" lesson from E1b.

## What to compute
Reuse E0's deterministic-step bucket and its argmax-wrong/argmax-right split (same `make_splits` call as E0/E1/E1b
for consistency — state explicitly if you regenerate vs. reuse saved indices).

1. **Mode-wise breakdown**: for each of the 6 modes, what fraction of that mode's deterministic-rule steps are
   argmax-wrong? Report as a table, all 3 seeds (mean±std). Is the error concentrated in specific modes or spread
   evenly? Cross-reference against T1's original per-mode near-switch numbers (`docs/tasks/T1-report.md`) — is this
   the same pattern (copy modes struggling near switches) or a different one (e.g. errors spread across all modes
   even far from switches)?
2. **Layer-wise origin**: for a sample of wrong steps (aim for ~200 per seed, stratified across modes so no mode
   dominates the sample), determine whether the error is already present after layer 0's contribution or only
   emerges at the final readout. Concretely: compare the layer-0 output state's best-matching feature (via
   `metrics.legibility`'s per-feature predicate fit, reusing predicates already in `metrics.predicate_matrix`) against
   what a "clean" execution would need at that point — i.e. does the state after layer 0 already lack the feature the
   correct answer depends on, or is the right information present in the state but the final rule/readout step
   discards or misreads it? State clearly what counts as "present" (a concrete, falsifiable operational definition —
   don't leave this fuzzy) before computing it.
3. **Rule-table inspection**: for 20–30 sampled wrong steps (spread across modes, prioritize the modes step 1 flags
   as worst), print which rule(s) fired (reuse `metrics.rulebook` / the model's own rule-firing trace) and manually
   characterize each into one of: (a) **no rule fired that covers this input** (a capacity gap — literally nothing in
   the rulebook handles this case), (b) **a rule fired but its condition or its value-table entry is wrong for this
   specific input** (a mislearned rule — optimization found a bad entry, not a missing one), (c) **the right rule
   fired but a later step overwrote or out-voted its output** (a persistence/interference problem, a third category
   E0/E1/E1b haven't considered). Report the count in each bucket. This is the most labor-intensive part — budget
   time for it, but 20-30 examples is enough to see whether one bucket clearly dominates; don't try to exhaustively
   classify every wrong step in the dataset.

## What this decides
Write a plain verdict: does one of (a)/(b)/(c) clearly dominate the sampled wrong steps? Does the mode-wise pattern
point at a specific known weak point (e.g. does it match T1's `cpy3`-near-switch finding, or is it broader)? Given
all three angles together, what's the most concrete, testable next hypothesis — stated as precisely as the ones
already tested and refuted (E0's calibration theory, E1b's routing-margin theory), so it can be confirmed or refuted
just as cleanly. Do not propose training a fix without a hypothesis this specific; a vague "try more capacity again"
is not an acceptable conclusion given T2 already showed that saturates.

## Constraints
Same standing rules: sole launcher for anything you start (though this task shouldn't need any training or long
jobs), bounded polling not recurring monitors if anything does run a while, no changes to existing `metrics.py`/
`rules.py` function behavior for existing callers, commit/push incrementally. If step 3's manual classification is
ambiguous for a given example, say so and put it in whichever bucket is the closest fit rather than inventing a
fourth category — note the ambiguous count separately if it's non-trivial.

**On reporting progress**: same as E1b — this should be a fast task (no training). If something takes a while, name
it once and then don't ping again until you have the real numbers; I'll watch the named process directly rather than
needing hand-backs on every forced check-in.

## Report (`docs/tasks/E1c-report.md`, commit and push)
The mode-wise table, the layer-wise origin finding (with your operational definition of "present" stated), the
rule-table inspection's bucket counts with a few representative examples printed verbatim (rule + input + why it's
wrong), the plain verdict, and the next concrete hypothesis. Append the usual short entry to `docs/RESEARCH-LOG.md`
per the worker README's instructions. Any deviations, and why.

When done: commit, push, then give me the concise summary — the verdict, the next hypothesis, anything blocking.
