# E1b report — does a narrow routing margin predict the deterministic argmax errors?

No training. `T2_large`, all 3 seeds, iid split (same `make_splits(n_train=20000, n_test=1000)` call as E0/E1).
Reuses E0's deterministic-step bucket (`heldout==0`, oracle max prob ≥0.99) and its argmax-wrong / argmax-right split.

## Method (stated up front, since it matters)

**Margin definition**: per deterministic-rule position, for every (layer, head), the routing score `S` (the same
formula as `models_g.route_g`) is recomputed from the model's real trained weights (`A`, `u`, `rel`) and the real,
hard-cascaded intermediate state at that point in the sequence (taken from the model's own forward pass, not
resimulated) — then softmaxed, and margin = (top attention weight − runner-up). The **minimum** margin across all
(layer, head) at that position is the step's margin (the most-contested routing decision feeding that step), per the
task's own stated proxy.

**A bug caught before trusting any number**: the first attempt used the `attn` array `apply_rulenet_g` already
returns via `return_internals=True`. That array is *already STE-hardened* — for a fully-snapped model (`alpha=1.0`),
the forward *value* of attention is exactly `(1-alpha)*softmax(S) + alpha*one_hot(argmax(S))`, which at `alpha=1.0`
degenerates to pure one-hot regardless of `S`. Margin computed from that array was exactly 1.0 for every single step
in every group, on all 3 seeds — trivially uninformative, not a real result. Fixed by recomputing the raw pre-STE
`softmax(S)` directly from the trained weights and the real cascaded states (method above), which is what the
numbers below use.

**A performance note, not a correctness one**: the naive dense `einsum('btgv,hgk,bskv->bhts', ...)` for this
recomputation is what made `rules.py`'s existing `execute_hard_g` slow on `T2_large` earlier in this project (~5-7
min per call) — it computes over all G×K≈5000 group pairs when `A` is actually pruned to 7-8 nonzero pairs per head.
Restricting the computation to `A`'s actual nonzero `(g,k)` pairs (a `np.argwhere` plus a small loop) turned a
computation that hadn't finished after 5 minutes into one that finishes in under a second per seed — a ~1000x
speedup from exploiting sparsity the dense einsum ignores. Analysis-only script, no shared code touched.

## Results

**Aggregate** (n=2772 wrong, n=16878 right, pooled over 3 seeds):

| | mean margin | median margin |
|---|---|---|
| argmax-wrong | 0.0375 | 0.0169 |
| argmax-right | 0.0380 | 0.0192 |

Essentially identical — if anything argmax-right has the (very slightly) *larger* median margin, the opposite of the
predicted direction. Held-out separability (threshold fit on half, scored on the other half): **F1 = 0.292, accuracy
= 0.490** — worse than the accuracy from just always predicting "right" (0.859, the majority class). The margin
threshold is not predictive.

**Per seed** (not pooled — the direction is not even consistent):

| seed | wrong mean | right mean | direction |
|---|---|---|---|
| 0 | 0.0196 | 0.0413 | wrong smaller (predicted direction) |
| 1 | 0.0120 | 0.0278 | wrong smaller (predicted direction) |
| 2 | 0.0561 | 0.0459 | **wrong larger** (opposite) |

Two of three seeds show the predicted direction individually, but the third reverses it strongly enough that pooling
erases the effect entirely — this is not "weak but real," it's inconsistent across independently-trained seeds, which
is a stronger disqualifier than a small pooled effect size would be on its own.

**`cpy3`-near-switch subset** (T1's original diagnosed case: mode=`cpy3`, `dist_mode` 1–4; n=362 wrong, n=2635 right;
reported separately, not merged into the aggregate per the task's instruction):

| | mean margin | median margin |
|---|---|---|
| argmax-wrong | 0.1118 | 0.0573 |
| argmax-right | 0.0429 | 0.0314 |

**This is the opposite of the predicted direction, and more pronounced than the aggregate**: in the specific case the
mechanism was built to explain, wrong steps have routing margins *2.6x larger* than right steps, not smaller. Held-out
accuracy of "margin < threshold ⇒ wrong" on this subset: **11.1%** — far below the 87.9% majority-class baseline,
because the fit-half threshold (0.279, itself absurdly high relative to the median margins above — a symptom of no
real separation to fit) flags nearly everything as "wrong" and is wrong most of the time doing so.

**Control (`ctrl_dense_sae`)**: skipped, as the task allows. `apply_dense`'s attention (`attn_dense` in `models.py`)
computes softmax attention internally but never returns it — there is no `return_internals` path for the dense
model, unlike `apply_rulenet_g`. Exposing it would require changing `models.py`'s shared `apply_dense` signature,
which the task's constraints ask to avoid; the main `T2_large` comparison is what the decision below hinges on, so
this was skipped rather than risk touching shared model code for a non-decisive control.

## Decision rule outcome: **REFUTED**

Per the plan's decision rule, this requires "margin significantly smaller on argmax-wrong steps, especially
pronounced on the `cpy3`-near-switch subset" to confirm. Neither holds:
* Aggregate: no meaningful separation (means differ by <2%, held-out accuracy *worse* than the majority baseline).
* Per-seed: the direction is not even consistent (2 of 3 seeds show it, 1 reverses it).
* `cpy3`-near-switch, the case the mechanism was specifically built to explain: **the effect reverses and gets
  larger**, not confirms and gets sharper.

**The routing-tie-break story proposed in the E0/E1 report does not hold up under this direct test.** Stated
plainly, as asked: I was wrong about the mechanism. E0's finding stands (deterministic-step argmax errors, 58.6% of
the gap, dominate over stochastic calibration) — that part isn't touched by this result — but *why* those steps are
wrong is not "the routing decision was close and coarse precision picked the wrong side." Per the plan's refutation
branch, the next task should go back to characterizing the deterministic argmax errors a different way rather than
assuming a routing fix. Concrete angles worth trying, in rough order of cheapness (not run here — this task's scope
was the margin test only):
1. **Mode-wise breakdown** of argmax-wrong share (is it concentrated in `cpy3`/`cpy2` specifically, or spread evenly
   across all 6 modes?) — already-available data (`ann['mode']`, E0's `wrong` mask), no new computation needed.
2. **Layer-wise**: does the error originate at layer 0 or layer 1 (check whether the *state* after layer 0 already
   diverges from what a clean execution would produce, vs. only the final readout being wrong)?
3. **Rule-table inspection**: for a sample of wrong steps, print the printed rulebook's matching rule and check
   whether its condition or its output map (`T`) is simply wrong for that input — i.e. whether this is a capacity
   problem (no rule exists for this case) or a specific mislearned rule (optimization found a bad local minimum for
   one entry), which the E0/E1 report already flagged as the two remaining candidate causes (a)/(b) and this task
   was meant to help separate — it didn't, so this may need to happen at that grain to make progress.

## Deviations

* Recomputes routing margins outside `apply_rulenet_g`'s existing STE-hardened `attn` output (see "A bug caught"
  above) — a new, analysis-only computation; no existing function's behavior changed.
* Performance optimization (exploiting `A`'s sparsity instead of a dense contraction) is local to this task's
  analysis script; `rules.py`/`metrics.py` were not modified.
* Skipped the `ctrl_dense_sae` control, as explicitly permitted, because exposing its attention would require
  changing shared model code (`models.py`'s `apply_dense`/`attn_dense`).
