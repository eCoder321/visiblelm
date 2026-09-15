# Task E1b — does a narrow routing margin predict the deterministic argmax errors?

Issued by the principal investigator session, 2026-09-15. Worker: Sonnet. Read `docs/PLAN-v3-observability-frontier.md`
§7 first — this task tests the specific mechanism your own E0/E1 report proposed: that deterministic-step argmax
errors come from routing's attention argmax breaking under quantization (a tie-break problem), not from the rule
layer lacking magnitude precision generally. No training; existing `T2_large` bundles only.

## What to compute
On `T2_large`, all 3 seeds, iid split (same `make_splits` call as E0/E1 for consistency):

1. For every **deterministic rule step** (reuse E0's bucket definition: `heldout==0`, oracle max prob ≥ 0.99),
   identify which routing head/layer/position actually determined the outcome. The cleanest way: reuse
   `metrics.path_faithfulness`'s circuit-tracing machinery (or `rules.py`'s attention arrays from
   `apply_rulenet_g(..., opts={'return_internals': True})`) to get, per step, the attention weight vector at the
   layer/head whose output feeds the eventual winning output feature — if tracing the exact causal head per-step is
   too slow at full scale, a reasonable proxy is: for each layer/head, take the attention distribution at the
   position in question, compute its **margin** = (top attention weight − second-highest attention weight), and use
   the *minimum* margin across heads/layers at that position (the most-contested routing decision) as that step's
   margin. State clearly which method you used.
2. Split steps into **argmax-wrong** and **argmax-right** (already computed in E0 — reuse that split, don't
   recompute the buckets from scratch).
3. Compare the margin distributions: report mean/median margin for each group, and a held-out check (fit a threshold
   on half the data that best separates wrong from right by margin, report its F1/accuracy on the other half).
4. Repeat the same comparison restricted to steps in `cpy3` mode within 4 positions of a mode switch (the specific
   case T1 originally diagnosed) — smaller sample, report it separately, don't merge into the aggregate.
5. As a control: repeat the same margin computation and comparison on `ctrl_dense_sae`'s attention (if accessible
   from its bundle) or, if that's awkward, skip the control and say so — the main comparison (T2_large only) is what
   the decision below hinges on.

## Decision rule (from the plan, apply exactly)
* **Confirms the mechanism**: margin is significantly smaller (report the actual gap and the held-out separability)
  on argmax-wrong steps, especially pronounced on the `cpy3`-near-switch subset. → recommend the next task be either
  (a) an E2 redesign that gives routing's attention scores their own finer quantization level count, decoupled from
  the rule layer's, or (b) E3 (distillation), stating which you think is the better next experiment and why.
* **Refutes it**: margin doesn't meaningfully separate the two groups. → say so plainly; the routing-tie-break story
  from the E0/E1 report doesn't hold up under a direct test, and the next task should go back to characterizing the
  deterministic argmax errors a different way (e.g. is it specific modes, specific rule-application steps, specific
  layers) rather than assuming a routing fix.
* **Ambiguous** (some separation but weak/noisy): report the actual numbers and give your judgment on whether it's
  worth a follow-up or whether the signal is too weak to act on.

## Constraints
Same standing rules as before: sole launcher for anything you start, bounded polling not recurring monitors, no
change to existing `metrics.py`/`rules.py` function behavior for existing callers, commit/push incrementally. This
is pure analysis — if any single computation takes more than ~15 minutes, look for a vectorized approach before
letting it run long; the bundles and splits are the same ones E0/E1 already used, so nothing here should require
new training or new large data generation.

**On reporting progress**: this task should be fast (no training, similar scope to E0). If you do hit a genuinely
long-running step, tell me once what you're running and roughly how long you expect it to take, then don't ping
again until it's done or you have the real numbers — I'll be watching the process directly rather than waiting on
interim hand-backs, so there's no need to check in without new information.

## Report (`docs/tasks/E1b-report.md`, commit and push)
The margin comparison (aggregate and `cpy3`-near-switch), the decision-rule outcome, and your recommendation for
what runs next (E2-redesign vs E3, or neither if refuted). Any deviations from this file, and why.

When done: commit, push, then give me the concise summary — the decision outcome and your recommendation.
