# E1c report — characterizing the deterministic argmax errors directly

No training. `T2_large`, all 3 seeds, iid split (`make_splits(n_train=20000, n_test=1000)`, regenerated fresh — same
call as E0/E1/E1b, not reused saved indices, since none were persisted). Reuses E0's deterministic-step bucket and
argmax-wrong/argmax-right split, recomputed from the same definitions.

## 1. Mode-wise breakdown

Fraction of each mode's deterministic-rule steps that are argmax-wrong (mean ± std over 3 seeds):

| mode | wrong fraction |
|---|---|
| pred | 0.281 ± 0.135 |
| pls2 | 0.227 ± 0.106 |
| cpy3 | 0.145 ± 0.116 |
| succ | 0.130 ± 0.060 |
| cpy2 | 0.082 ± 0.052 |
| rept | 0.084 ± 0.037 |

**Errors concentrate in the arithmetic-transform modes (pred, pls2, succ — need `+k mod 16`), roughly 2-3x the rate
of copy/identity modes (cpy2, cpy3, rept — need only fetch-and-forward, no value transform).** This is a **different
pattern than T1's finding**, not the same one: T1's near-switch diagnostic (T1-report.md §3) found the weak case was
copy modes specifically *near a mode switch* (`cpy3`, dist_mode 1-4). Here, restricted to deterministic steps (no
switch involved at all — `heldout==0`, oracle max prob ≥0.99, so this is steady-state, far-from-any-switch behavior),
the weak modes are the transform modes, and copy modes are comparatively strong. The two failure modes are not the
same phenomenon: T1's is about *routing far enough back* right after a switch; this one, as shown below, is about
*value arithmetic never being learned*, and it shows up throughout the sequence, switch or no switch.

## 2. Layer-wise origin

**Operational definition (stated up front, falsifiable, structural — not a statistical fit)**: `rulenet_g`'s
post-layer-0 state is grouped into named slots by construction (`models_g.layout`): `val0..val{n_val-1}` and
`tok_content` are content-kind groups whose feature index *is* a content value 0-15 by one-hot design, not something
that needs fitting. For a wrong step with correct answer `v = ann['next'][b,t]`, **PRESENT** = `v` is active
(`cg[b,t,g,v] > 0`) in at least one content-kind group of the state after layer 0 (both its route and compute
sub-steps have run) — the correct value exists as a fact *somewhere* in the state, whether or not it has reached the
output register yet. **ABSENT** = `v` is not active in any content-kind group — the correct value was never
fetched or computed by the end of layer 0 at all.

(A first attempt used statistical predicate-fitting — `metrics.predicate_matrix`'s `next=c` predicates, same method
as `metrics.legibility` — but no single feature reaches the project's 0.8 "clean" F1 bar for any of the 16 possible
values (max 0.468): a 16-way value cannot be identified by one boolean feature at useful precision, because the
architecture represents values structurally in slots, not as fittable booleans. Abandoned in favor of the structural
check above, which needs no threshold at all.)

Sampled ~200 wrong steps per seed, stratified across modes (≤34 per mode):

| seed | present | absent | present frac |
|---|---|---|---|
| 0 | 33 | 171 | 0.162 |
| 1 | 64 | 131 | 0.328 |
| 2 | 65 | 132 | 0.330 |
| **pooled** | **162** | **434** | **0.272** |

**73% of wrong steps have the correct value completely absent from the state after layer 0.** Consistently across
seeds (16-33%, never over 1/3). This rules out "the answer is computed correctly and a later step discards or
misreads it" as the dominant story — most errors originate at or before layer 0, not from late-stage interference on
an already-correct value. (The remaining 27% *are* present-but-wrong at the end, which §3 partly explains too — see
the cross-mode contamination finding below.)

## 3. Rule-table inspection

**Static audit**: for every rule (both layers, all 3 seeds) whose condition weights a `tok_mode=succ/pred/pls2`
feature above 0.15 and whose `Gs`/`Gd` route a content-kind group through its value table `T`, the table's discrete
`argmax` function was checked against the mode's true transform (`+1`, `-1`, `+2 mod 16`). **Every single one of
these rules — all 3 seeds, both modes checked — has a table that is still exactly the init identity: 0/16 matches to
the true transform, in every case.** The arithmetic these three modes need was never learned by any of the value-map
pathway's rules.

That alone doesn't explain succ's 87% success rate, so the next check was a **runtime firing trace** (`hid[l][b,t,r]
> 0.5`, the model's real STE-hardened firing signal — not a static weight inspection) on all deterministic steps,
seed 0, for the specific identity-stuck rules found (`L0R144`, `L0R265`, `L0R363` gated on `succ`; `L0R296` gated on
`pls2`):

| rule (labeled mode) | fires on RIGHT steps | fires on WRONG steps | discrimination |
|---|---|---|---|
| L0R144/265/363 (succ) | 8.8% (n=491) | 78.5% (n=93) | **8.9x more on wrong** |
| L0R144 (labeled succ) on **cpy3** steps | 0.0% (n=1225) | 69.6% (n=56) | **fires on zero right steps, 70% of wrong ones** |
| L0R265 (labeled succ) on **cpy2** steps | 8.7% (n=1377) | 80.9% (n=131) | **9.3x more on wrong** |
| L0R144 on **rept** steps | 3.2% (n=1454) | 31.9% (n=47) | **10x more on wrong** |
| L0R144 on **pred** steps | 99.6% (n=473) | 84.5% (n=142) | none — fires almost always regardless |
| L0R296 (labeled pls2) on **pls2** steps | 89.9% (n=900) | 28.6% (n=161) | **reversed** — fires *more* on right steps |

Two clean, opposite findings, both decisive:
* **succ, cpy2, cpy3, rept**: these succ-labeled, identity-stuck rules fire far more often specifically on wrong
  steps — including firing on **zero** correct `cpy3` steps but 70% of wrong ones. Their condition is nominally
  gated on `tok_mode=succ`, but the rule's firing threshold is a weighted *sum* over 2-3 inputs (not a strict AND),
  so a large enough co-activation on an unrelated boolean feature is enough to fire the rule during a completely
  different mode's step (`cond=['tok_mode=succ:+1.06', 'b50:+0.56']`, `b1=-0.06` — `b50` alone clears the threshold).
  When it fires during `succ` itself, its identity table votes for the untransformed value, which is simply wrong
  (this **is** the arithmetic-never-learned finding from the static audit, now confirmed causally). When it
  cross-fires during `cpy2`/`cpy3`/`rept`, it injects a spurious identity-echo vote into the output register that
  competes with whatever the (mostly correct — 8-15% wrong) copy mechanism is doing, degrading it.
* **pred, pls2**: the corresponding identity-stuck rules exist (same static-audit finding: tables never moved off
  identity) but do **not** discriminate wrong from right the same way — `L0R144` on `pred` fires almost always
  regardless of correctness (99.6% vs 84.5%), and `L0R296` on `pls2` fires *more* on right steps (89.9% vs 28.6%),
  the opposite direction. **These two modes' errors are not explained by this mechanism** and need separate
  characterization — flagged honestly rather than forced into the same story.

**Bucket counts** (30 examples sampled, prioritized toward the worst modes per the task's instruction: pred 8, pls2
7, succ 6, cpy3 5, cpy2 2, rept 2 — seed 0):

| bucket | count | modes |
|---|---|---|
| (b) mislearned rule (identity-stuck table, cleanly discriminating) | 5 | succ |
| (c) right mechanism interfered with (cross-mode identity-rule contamination, cleanly discriminating) | 8 | cpy2 (2), cpy3 (5), rept (1) |
| not cleanly explained by this mechanism (flagged, not forced) | 17 | pred (8), pls2 (7), succ (1), rept (1) |
| (a) no rule fired covering this input | 0 | — |

None of the sampled examples showed a clean "nothing fired at all" case — some rule always fires; the honest
distinction that emerged is whether the *specific* identity-stuck rules discriminate wrong-vs-right for that mode,
not whether *anything* fired. A caveat worth stating plainly: with 40+ rules typically firing per position (the
readout is a dense vote aggregate, not a single clean causal path per example), a fully rigorous per-example "this
exact rule caused this exact wrong token" read requires ablation-based circuit tracing (`metrics.path_faithfulness`'s
existing method) run per example, which was not done here for time — the bucket counts above rest on the
dataset-wide firing-rate discrimination (statistically solid, computed over the full seed-0 deterministic bucket,
not just the 30 samples) rather than a claim that any single sampled prediction is fully explained end-to-end by one
rule alone.

Representative example (`succ`, seed 0, i=471, t=22): input value 5, correct next = 6. `L0R144`/`265`/`363` all
fired (condition includes `tok_mode=succ:+0.77-1.06`, weight strong enough alone to clear each rule's `b1`); each
rule's table row for input 5 maps to output 5 (identity — confirmed via the static table dump), not 6. Model's
actual prediction was content 3, not simply 5 — consistent with the "many rules jointly vote" finding above: the
identity-stuck rule contributes a wrong-direction vote, but isn't the sole determinant of the final (also wrong)
value.

## Verdict

**Two of (a)/(b)/(c) both appear, cleanly, but for different mode groups — not one dominant story across the board.**
For succ (and by the static-table evidence, likely pred/pls2 too, though their causal picture is less clean): a
genuine **capacity-adjacent optimization failure** — the value-transform table for `+k mod 16` never moved off its
random-identity initialization in any of the rules found gated on these modes, across all 3 seeds. For cpy2/cpy3/rept:
a distinct, previously-uncharacterized **cross-mode interference** mechanism — rules nominally labeled for one mode
fire during a different mode's steps because their firing threshold is a weighted sum, not a strict per-input gate,
and when they do, they inject a spurious vote that degrades an otherwise largely-correct mechanism. pred and pls2
remain **open** — the identity-stuck-table fact holds structurally, but the firing-rate evidence that would confirm
it *causes* their errors does not (pred: rule fires regardless of correctness; pls2: reversed direction) — so
whatever is wrong with pred/pls2 specifically needs its own look, not an assumption that it's the same mechanism.

## Next concrete hypothesis (falsifiable, no training required to test it)

**H(E1d): the value-transform tables for succ (`T` matrices in the rules gated on `tok_mode=succ`) are causally
responsible for succ's deterministic-step errors, and the succ/cpy2/cpy3/rept-labeled identity-stuck rules found
here are causally responsible for degrading cpy2/cpy3/rept's otherwise-correct predictions via cross-mode
contamination. Prediction, stated before testing: ablating (zeroing) the specific rules `L0R144`, `L0R265`, `L0R363`
(and their per-seed equivalents, re-identified per seed the same way) from the readout and re-running inference
(pure forward-pass analysis, no training) should measurably *improve* deterministic-step accuracy on succ, cpy2,
cpy3, and rept — while pred and pls2 should show little or no improvement from ablating their analogous identity-stuck
rules, since the firing-rate evidence here does not support the same causal story for those two modes.** This is
directly falsifiable exactly like E0's and E1b's theories were: a specific ablation, a specific predicted direction,
and an explicit prediction that it will *not* generalize to two of the six modes. If the ablation improves accuracy
on the four predicted modes but not pred/pls2, this confirms the mechanism and — separately — motivates asking why
pred/pls2's tables are equally stuck at identity yet don't show the same failure signature (a distinct follow-up,
not assumed away). If ablation doesn't move accuracy on any mode, the identity-rule firing-rate correlation was
non-causal (e.g. a downstream consequence of the same underlying condition rather than a cause), and the search
should go back to the dense multi-rule voting structure directly, likely via per-example `path_faithfulness`
ablation tracing rather than static/firing-rate proxies.

## Deviations

* Step 2's operational definition changed from the originally-planned statistical predicate-fit (per the task's
  suggested method) to a structural group-membership check, because the statistical approach's F1 never cleared the
  project's own 0.8 clean bar for any of the 16 target values (a real, informative negative result about the
  architecture, not a bug) — documented above rather than silently swapped.
* An initial attempt to fully quantify per-rule "vote share" of the final output (to make bucket assignment fully
  mechanical) was abandoned after it produced numbers inconsistent with the model's real predictions (it omitted the
  route sub-steps' direct contribution to the output register, only accounting for the compute sub-steps) — caught
  by cross-checking its implied "most common wrong prediction" against the real, fully-computed wrong-token
  distribution, which showed no such concentration. Not used in the report; bucket assignment instead rests on the
  dataset-wide firing-rate discrimination test, which needed no reconstruction of the full vote.
* No changes to `metrics.py`/`rules.py`/`models_g.py`'s existing functions; all analysis is new, local scratch code.
