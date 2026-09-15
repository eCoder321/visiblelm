# E1d report — ablation test of the identity-stuck-rule / cross-mode-contamination hypothesis

**Run by the PI directly, not the assigned worker**: the worker session hit a hard external rate/spend limit
immediately after being dispatched (reset time reported as 8:40pm UTC) and produced no output. Rather than wait or
keep retrying a blocked session, the PI executed this task directly, following the task file
(`docs/tasks/E1d-ablate-identity-stuck-rules.md`) as written. No training; `T2_large`, all 3 seeds, iid split
matching E0/E1/E1b/E1c.

## Method, and a bug caught before trusting the first result

**First attempt gave a trivial null result** (accuracy and NLL bit-for-bit identical before and after ablation, for
both the targeted set and the random control, on all 3 seeds) — a red flag, not a finding. Cause: the ablation
zeroed only the rule's direct `W2[r, F:]` output columns, but a rule can also inject a vote into the output register
through a *second* pathway — the group-map (`Gs`→`T`→`Gd`) route, when `Gd` targets the `out_mode`/`out_content`
groups directly. The candidate-selection step had also incorrectly restricted valid destinations to state-resident
groups only (`start < F`), excluding the output groups entirely — so the rules E1c actually found (which vote via
this second pathway) were never even selected as candidates. Fixed both: candidate selection now checks whether
`Gd` routes to `out_mode`/`out_content` specifically, and ablation zeros **both** pathways (`W2[r, F:]` and
`Gd[r, out_mode]`/`Gd[r, out_content]`) for a selected rule, while leaving every state-write completely untouched
(`W2[r, :F]` and `Gd[r, state-groups]` are never touched) — exactly the task's "output register only" requirement.

## Results

Candidate rules found (static check: `tok_mode∈{succ,pred,pls2}` condition weight > 0.15, reads a state content
group via `Gs`, routes via `T` to `out_mode`/`out_content` via `Gd`, and `T` is ≥90% identity-argmax) and selected
(also fires ≥15 percentage points more often on wrong than right steps, in at least one mode):

| seed | candidates | selected | rules |
|---|---|---|---|
| 0 | 5 | 5 | L0R144, L0R265, L0R296, L0R363, L0R406 |
| 1 | 3 | 2 | L0R437, L0R506 |
| 2 | 6 | 0 | (none cleared the firing-discrimination bar) |

Deterministic-step accuracy by mode, before vs after targeted ablation vs after a random-rule-count control:

**Seed 0** (5 rules ablated):
| mode | before | after targeted | after random |
|---|---|---|---|
| rept | 0.969 | 0.950 | 0.970 |
| succ | 0.841 | **0.771** | 0.779 |
| pred | 0.769 | 0.754 | 0.777 |
| pls2 | 0.848 | 0.805 | **0.535** |
| cpy2 | 0.913 | 0.887 | 0.907 |
| cpy3 | 0.956 | 0.979 | 0.962 |

NLL: 0.7777 → 0.8378 (targeted) → 0.8838 (random).

**Seed 1** (2 rules ablated):
| mode | before | after targeted | after random |
|---|---|---|---|
| rept | 0.895 | 0.895 | 0.884 |
| succ | 0.817 | **0.741** | 0.817 |
| pred | 0.854 | 0.831 | 0.854 |
| pls2 | 0.847 | 0.811 | 0.847 |
| cpy2 | 0.983 | 0.983 | 0.983 |
| cpy3 | 0.917 | 0.917 | 0.917 |

NLL: 0.8296 → 0.8926 (targeted) → 0.8313 (random).

**Seed 2**: 0 rules selected; no ablation performed (identical before/after/random by construction).

## Decision rule outcome: **REFUTED**, in a specific and informative way

The prediction was that targeted ablation would *improve* accuracy on succ/cpy2/cpy3/rept and leave pred/pls2
roughly unchanged. Instead:

* **succ got measurably worse in both seeds where any rule was ablated** (0.841→0.771; 0.817→0.741) — the opposite
  of the prediction. These "identity-stuck" rules, despite implementing the wrong transform by the static-table
  criterion, are net *helpful* to succ's accuracy on the margin, not harmful.
* **cpy2/cpy3/rept show no consistent improvement** — cpy3 ticks up slightly on seed 0 (0.956→0.979) but every
  other copy-mode number is flat or slightly down. Nothing like the predicted clean recovery.
* **The random control is not consistently gentler than the targeted ablation** — on seed 0 the random control's
  NLL is *worse* (0.884 vs 0.838) and it does far more damage to pls2 (0.848→0.535, catastrophic, vs 0.805 for the
  targeted set); on seed 1 the random control leaves succ/pred/pls2/cpy2/cpy3 completely untouched (0 of those
  rules happened to touch anything in that random draw) while the targeted set clearly does. This means the
  targeted rules *are* doing something real and specific (unlike most random rules, which the seed-1 random draw
  shows can be near-inert) — but what they're doing is not "purely spurious and harmful."

**Interpretation**: ablation confirms these specific rules have a real, identifiable causal effect (ruling out "the
firing-rate correlation was pure noise"), but the effect is not the clean "spurious vote hurts the modes it doesn't
belong to, and hurts nothing else" story E1c proposed. The dense multi-rule voting structure (40+ rules firing per
position, as E1c's own report already flagged) means a single rule's "wrong" vote can be doing useful work in
combination with other votes — e.g. contributing to correct cases outside the sampled wrong/right split, or
partially canceling a different error. A clean causal attribution would need per-example `path_faithfulness`-style
tracing, exactly as E1c's fallback recommended — this task's whole-rule-set ablation is a coarser instrument than
that, and the result shows why: it detects that something real is happening, but not a clean, isolable "this is
purely a bug" story.

## What still stands, independent of the ablation result

The **static** finding from E1c needs no causal test and is untouched by this refutation: every value-transform
table gated on succ/pred/pls2, in every rule checked, across all 3 seeds, is still exactly its random-init identity
matrix. That is a direct read of the trained weights, not an inference from correlation or intervention — the
`+k mod 16` arithmetic was never learned by gradient descent for any of these tables, full stop.

**A concrete, likely mechanism for *why* it was never learned**: `models_g.py`'s own docstring for the value table
`T` states it is "regularised towards" the identity matrix — confirmed in `train.py`/`models_g.reg_rulenet_g`, which
adds `cfg['l1_table'] * |T - I|` to the loss (T2_large's config uses `l1_table: 1.0`, the same weight as every other
structural L1 term). This regularizer directly opposes any gradient signal trying to move `T` away from identity to
learn `+1`/`-1`/`+2`. If the gradient signal reaching `T` through the discrete/STE pipeline is weak relative to this
pull (plausible: `T` is read only when a rule fires, gated by a hard 0/1 signal after snapping, so gradients to `T`
flow only through the STE straight-through path on firing steps), the regularizer can dominate and hold every such
table at exactly its starting point — which is exactly what was observed: not *approximately* identity, but
*exactly* identity in every case checked.

## Recommended next hypothesis (not run here — proposing for the next task)

**H(E2-redux): reducing or removing `l1_table`'s pull on value tables gated by arithmetic-transform modes allows
those tables to learn the true transform, improving succ/pred/pls2 accuracy specifically, without touching
copy-mode accuracy** (since copy modes' tables are the identity map *correctly* — they should stay at identity, and
this hypothesis predicts no change there, unlike E1d's refuted cross-mode story). This is a training experiment
(the current codebase's `l1_table` is a global scalar, not per-mode — the simplest test is a global reduction/anneal
and checking whether succ/pred/pls2 improve while cpy2/cpy3/rept don't regress from losing correct identity tables).
Concretely testable, falsifiable, and grounded in a direct observation (the tables are regularized toward exactly
the thing they need to stop being) rather than a second-order correlational story.

## Deviations

* Run by the PI directly due to the worker's rate limit, not by the assigned worker.
* Ablation targets both output-writing pathways (`W2[r,F:]` and `Gd[r,out_groups]`), a correction to the task file's
  wording ("zero their contribution to the output register") which the first (buggy) attempt under-implemented by
  only touching one of the two pathways — documented above as the bug-catch, not silently fixed.
* No changes to `metrics.py`/`rules.py`/`models_g.py`'s existing functions; `vlm/e1d_ablation.py` is new, standalone
  analysis code using each module's existing public functions (`apply_rulenet_g`, `layout`, `annotate`, etc.).
