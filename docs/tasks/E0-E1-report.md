# E0 + E1 report — attribute the lost nats; scan how coarse the magnitude channel is

Analysis only, no training (per the task; the optional soft `T2_large` twin was skipped — see §4). iid split via
`testbed.make_splits(n_train=20000, n_test=1000)` (run.py's own seed), same for E0 and E1. Input bundles verified
present and non-trivial before starting: `ctrl_dense_sae/seed{0,1,2}.pkl`, `rng_v1/seed0.pkl`, `T2_large/seed{0,1,2}.pkl`.

## 1. E0 — attribute the lost nats

Four buckets over every predicted position (t ≤ T-2) on iid: **seed** (t≤2, unfixable), **deterministic** (switch not
possible, oracle max prob ≥0.99), **stochastic** (switch possible, oracle max prob <0.99), **held-out**. Held-out is
0% on iid by construction (`make_splits`'s iid generator rejects held-out events; only `shift`/`long` contain them) —
expected, not a bug, and confirms the four buckets are otherwise a true partition (share_of_positions sums to 1).

| bucket | share of positions | ctrl_dense_sae gap share | rng_v1 gap share | T2_large gap share (mean±std, 3 seeds) |
|---|---|---|---|---|
| seed | 13.0% | 31.0% | 1.7% | 5.2% ± 5.6% |
| deterministic | 28.5% | -12.5% | 72.1% | **58.6% ± 12.5%** |
| stochastic | 58.5% | 81.5% | 26.1% | 36.2% ± 8.4% |
| held-out | 0% | n/a (0 count) | n/a (0 count) | n/a (0 count) |

(`ctrl_dense_sae`'s total gap is *negative*, -536.6 — its overall NLL, 0.593, is below the 0.62 oracle floor; a known,
already-documented artifact of exploiting the rejection-sampled training distribution, not a real generalization
advantage. Included for completeness per the task spec; not part of the binary model's decision rule.)

**Deterministic-bucket sub-split** (T2_large, mean over 3 seeds): of the gap sitting in deterministic rule steps,
**69.4% ± 3.2%** is the model's argmax being outright wrong; only **30.6% ± 3.2%** is argmax-right-but-underconfident.

**`switch_possible` legibility** (new field added to `testbed.annotate`; T2_large seed 0, final state `state4`,
384,000 (position) samples, half/half fit/held split, same method as `metrics.legibility`): best-matching feature is
index 152, held-out F1 = **0.827** (base rate 58.5%) — above this project's 0.8 "clean" bar. It is not alone: 4 more
features clear 0.8 too (indices 118, 78, 40, 123; F1 0.805–0.824). A feature *does* track `switch_possible`, and
redundantly so.

**E0 decision rule (plan §E0), applied exactly:** "if ≥50% of gap is in stochastic-step calibration AND no feature
tracks `switch_possible`, cause (d) dominates." **Neither half holds** — stochastic share is 36.2% (<50%), and 5
features clear the legibility bar for `switch_possible`. The alternative branch is what the data shows instead: **the
gap is concentrated in deterministic-step argmax errors (58.6% of the total gap, 69.4% of that from outright wrong
predictions) → causes (a) capacity and (b) optimization dominate, not (d) calibration.** This is decisive, not
close: on the disqualifying test alone (stochastic share 36.2% vs the 50% bar) there's a 14-point margin, and the
deterministic argmax-wrong share (69.4%) is more than double its underconfidence counterpart.

## 2. E1 — post-hoc quantization scan (`rng_v1`, no training)

**Smoke test (correctness gate, run before any scan number was trusted):** `execute_quantized_g(n_levels=2)` exactly
reproduces `execute_hard_g(binarize=True)` — confirmed on `T2_large` seed 0 (max abs diff 0.0) and on `rng_v1` seed 0
(`np.allclose` True; agreement with the model 0.4953, matching `north_star_any`'s own reading exactly, which in turn
matches REPORT.md's committed "discrete rulebook agrees with model: 49.5%" — the *combined* binarize+hard-attention
figure, not the "binarization alone: 30%" figure quoted during debugging, which is a different setting
(hard_attn=False). See §4 for the bug this smoke test caught on the first attempt.

| L | agreement with soft model (rule steps) | NLL |
|---|---|---|
| 2 | 49.53% | 2.388 |
| 3 | 52.38% | 2.118 |
| 4 | 50.71% | 2.255 |
| 8 | 61.72% | 2.502 |
| 16 | 60.58% | 2.598 |
| ∞ (no rounding, hard_attn still on) | **91.10%** | 1.278 |

**Correcting the task file's assumption**: the task file expected L=∞ to be "~100% agreement, a sanity ceiling." It
is not — it's 91.1%, and that is not a flaw in the run, it is the task file's assumption being wrong. The executor
uses hard (argmax) attention at every L, including L=∞/no-rounding; hard attention alone, independent of any
magnitude rounding, already caps agreement at ~91% (this exactly reproduces PLAN §1 fact #3's already-documented
"hard attention alone: 91%" — a useful internal-consistency check that the executor is correct, not a new finding).
So the L-scan's real ceiling is **~91%, not 100%**, and every agreement number above should be read against that
ceiling, not against perfect reproduction — magnitude precision alone can close at most ~91 points, not 100.

Agreement never gets close to even that 91% ceiling at any *finite* L up to 16 — it drifts in a 49–62% band, and
**not monotonically**: L=4 (50.7%) dips *below* L=3 (52.4%), and L=16 (60.6%) dips *below* L=8 (61.7%). This is a
real result, not noise to smooth over — uniform-spacing quantization (L evenly-spaced levels across `(thr0, max]`)
does not interpolate smoothly toward the ceiling as L grows; one honest guess is that uniform bins are a poor scheme
for this specific channel (most probability/activation mass may sit unevenly across the range, so an *evenly spaced*
grid can genuinely resolve fewer functionally-distinct cases at L=4 than at L=3 depending on exactly where the extra
boundary lands relative to where mass concentrates); quantile-based bins would likely behave more monotonically, but
that's a follow-up worth trying inside E2/E3, not something to chase further here. The flat-then-jump shape (49–62%
at every finite L, then 91% only when rounding is removed entirely) points at a specific mechanism, not generic "not
enough levels": attention scoring is a dot product over the *code*, so quantizing the code changes which source
position wins the argmax, and unlike the rule/compute stage, routing needs enough resolution to preserve fine value
*differences* between competing positions, not just enough resolution to represent one feature's own magnitude. 16
evenly-spaced levels are nowhere near enough for that; only exact (unrounded) values are.

**E1 decision rule (plan §E1), applied exactly:** "if L≤8 gives ≥95% agreement → magnitude channel is coarse, go to
E2 with L∈{4,8}. If agreement stays <90% until L≥16, magnitudes carry fine-grained information → E3 (distillation)
and E4 (noise dial) become necessary." **Agreement stays below 90% through L=16** (max 61.7% at L=8, against a ~91%
ceiling, so it's not even close to *that* ceiling either) → this lands on the second branch: **E3 and E4 are
necessary, not optional**, and specifically because of routing precision, not general "how many brightness levels a
feature needs."

**Optional soft `T2_large` twin (E1 point 3): skipped**, by agreement with the PI — explicitly optional and
non-blocking per the task, requires a full training run (~35–60 min) against a task scoped as no-training analysis,
and doesn't change either decision rule above (both are fully determined by the required `rng_v1` scan).

## 3. What this means for E2

Cross-referencing both verdicts: E0 says the binary model's gap is dominated by causes (a)/(b) — capacity and
optimization, concentrated in deterministic-step argmax errors — not (d) missing information (`switch_possible` is
already legible). E1 says that under post-hoc quantization, what breaks first and hardest is specifically *routing's*
argmax, not the rule layer's magnitude precision generally, and it doesn't recover until precision is exact. Put
together: the deterministic-step errors E0 found are plausibly the same failure E1 just characterized mechanistically
— routing needs to correctly select the right upstream position (e.g. the 3rd-back token for `cpy3`, T1's original
diagnosis) and a coarse alphabet corrupts exactly that selection, which then shows up downstream as "picked the wrong
argmax," i.e. an (a)/(b)-looking symptom with a routing-precision cause.

**My judgment, since you asked directly: E2 as literally scoped (train `T2_large` at L∈{4,8}, uniform levels, same
routing/compute treatment) is a weaker bet than E3 right now, and I'd reorder rather than run it as written.**
Three reasons, each from a number above, not intuition: (1) E1's own decision rule already says E3/E4 are necessary,
not merely worth trying after E2 — the plan gates E3 behind E2's outcome, but E1 supplies the evidence for that gate
*before* E2 runs, and it says go straight to E3. (2) The uniform-level scheme E2 would train under is the same shape
that gave non-monotonic, plateaued results in E1 (§2's L=4<L=3, L=16<L=8 dips) — training might compensate for this
somewhat (STE lets gradients reshape what a level *means*), but E2 as scoped doesn't change the binning shape, so
it's betting that training alone fixes a scheme that already looked poorly suited to the channel it's quantizing.
(3) E2 doesn't differentiate routing's alphabet from the rule layer's, and E1's clearest signal is that these two
need different treatment (routing needs far more precision than 16 levels gave it; nothing in E1 suggests the rule
layer needs nearly that much). **Concretely, I'd suggest**: either give E2 a finer/separate `n_levels` for the
routing score computation specifically (untested here — E1 only quantized the state, not a routing-specific
alphabet), or run E3 first/in parallel at L=2 and the best E1 L, and let *its* result (not E2's) decide whether
quantization-aware training on a shared alphabet is worth scoping at all. Either is a real change to E2 as written,
not a green light to launch it unchanged.

## 4. Deviations

* **`vlm/rules.py`**: added `execute_quantized_g` (generalizes `execute_hard_g`'s binarize step to `n_levels`
  evenly-spaced levels; `n_levels=2` is byte-identical to today's `execute_hard_g(binarize=True)` for both ste and
  non-ste models, `n_levels=None` reproduces the continuous re-execution). **Bug found and fixed during this task,
  not before**: the first version anchored levels above L=2 (and, due to a scoping error, L=2 itself for non-ste
  models) at `clampv/2` where `clampv` fell back to the batch's own observed max for a model with no `cfg['clamp']`
  (`rng_v1`). Since most real activations sit well below half of an outlier max, this zeroed out nearly every
  genuinely-active feature and additionally wrote back the *raw* magnitude (up to ~15) instead of a normalized value,
  feeding weights that were never trained for that scale. Measured effect: L=2 agreement on `rng_v1` came out 0.055%
  — far below chance (~4.5% for a 22-token vocab), not just "quantization hurts." Fix: the level-0/1 boundary is now
  always anchored at `thr0` (0 for non-ste, the trained `ste_thr` for ste — the same threshold `execute_hard_g`
  already uses for both cases), so L=2 is provably `z > thr0` regardless of ste/clamp status; levels above 2
  subdivide `(thr0, clampv]`; values written back are always the normalized level index `/(n_levels-1)` in [0,1],
  never a raw magnitude. Re-verified after the fix on both `T2_large` (unaffected — its code path was unchanged by
  the fix) and `rng_v1` (now matches the committed 49.5% baseline exactly). This is analysis-only code with no
  existing callers, so nothing else was affected.
* **`vlm/testbed.py`**: added a `switch_possible` key to `annotate()`'s output (`State.switch_possible(t+1)` at each
  position) — a pure addition, no existing key renamed, removed, or changed in meaning. `python vlm/testbed.py` still
  runs clean.
* **`metrics.py` / `rules.py`'s existing functions**: not touched in any way that changes output for existing
  configs — `execute_hard_g`, `north_star_any`, `full_analysis`, `legibility`, etc. are byte-for-byte unchanged;
  `execute_quantized_g` is new code alongside them.
* Optional soft `T2_large` twin (E1 point 3 of the task): skipped by agreement with the PI (§2 above).
