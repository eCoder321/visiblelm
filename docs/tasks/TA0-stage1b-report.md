# T-A0 stage 1b: results (2026-10-26)

**Question:** does faithful-replay pressure make explanations shorter than the free alternative of using fewer active
units, at equal performance?

**Answer: no.** At equal performance, using fewer units gives explanations **1.7–2.9× more faithful** than the
pressure. The result is robust. Following the pre-registered decision rule, the pressure "adds nothing over choosing
K" (in fact it is dominated), so stage 2 is not triggered with this pressure.

- **Brief and predictions:** `TA0-stage1b.md`.
- **Method:** `docs/redesign/FAITHFUL-REPLAY-PRESSURE.md` v3.1.
- **Raw results:** `results/lm/seed*/ta0s1b_*/` (compacted on Modal; the full files are on the volume).
- **Summary:** `python lm/report1b.py`.
- **Compute:** phase A, 15 runs, **14.9 GPU-hours**, against 6.5 estimated. The new evaluator (4 replay modes × 324
  budgets × 2 categories, plus bootstrap) took 35–55 min per run, longer than training (≈22 min). It must be slimmed
  before more runs.
- **Phase B was not run** (see "Decision" below).

## Results (validation bpb; free-selection 8-bit strict replay; KL p90 = 90th-percentile per-position KL)

| job | seeds | bpb | KL p90 @ 30k bits (prose / code) | KL p90 @ 60k bits | bits for KL p90 ≤ 0.1 (prose / code) |
|---|---|---|---|---|---|
| softmax K=16 (base) | 2 | 1.567 | 1.62 / 2.36 | 0.47 / 0.67 | not reached |
| softmax K=12 | 2 | 1.583 | 1.13 / 1.52 | 0.23 / 0.28 | not reached |
| softmax K=8 | 2 | 1.591 | 0.63 / 0.79 | **0.10 / 0.15** | 59k / not reached |
| softmax K=4 | 2 | 1.660 | 0.18 / 0.20 | 0.07 / 0.08 | 39k / 36k |
| **pressure λ=0.1** (K=16) | 1 | 1.589 | 0.77 / 1.07 | 0.23 / 0.32 | not reached |
| pressure λ=0.3 | 1 | 1.703 | 0.67 / 0.98 | 0.23 / 0.28 | not reached |
| pressure λ=1 | 1 | 1.800 | 0.65 / 0.62 | 0.21 / 0.24 | not reached |
| sigmoid K=8 | 2 | 1.752 | 0.13 / 0.14 | 0.11 / 0.12 | not reached |
| sigmoid, θ frozen at 0 (K=16) | 1 | 1.647 | 0.23 / 0.21 | 0.19 / 0.15 | not reached |
| softmax(s − θ) (K=16) | 1 | 1.522 | 1.56 / 2.49 | 0.43 / 0.52 | not reached |

**Seed effects are large and consistent.** Seed 1 is 0.04–0.06 bpb worse than seed 0 in every configuration (e.g.
K=16: 1.539 vs 1.595). So **all comparisons below are paired by seed.**

K=16 seed 0 reproduces stage 1's softmax seed 0 exactly (1.539), so the setup is stable.

## The decisive comparison (paired, seed 0)

| | bpb | KL p90 @ 60k (prose / code), with bootstrap 90% CI |
|---|---|---|
| softmax K=16 (its base) | 1.539 | 0.45 [0.31, 0.63] / 0.60 [0.49, 0.65] |
| **pressure λ=0.1** | 1.589 (+0.050) | 0.23 [0.15, 0.37] / 0.32 [0.25, 0.38] |
| **softmax K=8** | **1.570** (+0.031) | **0.086 [0.057, 0.141] / 0.115 [0.094, 0.136]** |

- **The pressure does improve on its own base:** about 2× lower KL p90 at 60k bits.
- **Its tax (+0.050 bpb paired) is outside the ≤ 0.02 band.** The unpaired mean (+0.022) understated it, because
  the pressure ran only on the better seed.
- **K=8 dominates it:** a lower tax *and* 2.7× / 2.8× lower KL, with non-overlapping CIs.
- Interpolated against the seed-0 k-sweep at the pressure's bpb, the pressure reaches only **0.34–0.60×** of the
  sweep's faithfulness (`report1b.py`, paired).

## Ruling out the ambiguous explanations (the critics' guards)
- **Train/eval rounding mismatch: no.** Noise-replay and strict-replay KL agree within 3–14% for every arm.
- **Pressure too weak or sparse: no.** λ_eff went from 0.4 to 4 (10×), and KL p90 @ 60k barely moved (0.23 → 0.21)
  while the tax rose to +0.23 bpb. The effect saturates; it isn't starved.
- **Selection (teacher-forced training, free evaluation): no.**
  - Graded with *given* selection (how it was trained), the pressure still trails K=8: 0.53–0.63× at 60k.
  - The replay re-selects the same units as the full run about 95% of the time, mass-weighted.
- **Seed noise: no.** The decisive comparison is paired, and the CIs don't overlap.

## Other findings
1. **Activation sparsity (fewer active units) is a strong, cheap lever.**
   - K 16 → 8 costs +0.031 bpb paired and gives ≈5× more faithful explanations at 60k bits.
   - K → 4 costs +0.11 bpb for diminishing extra gain.
2. **Independent (sigmoid) gates are not a better trade than fewer softmax units.**
   - Sigmoid K=8 is worse than softmax K=4 in bpb (1.752 vs 1.660) without being more explainable.
   - The stage-1 "sigmoid is ~13× cheaper to explain" was a comparison against K=16 only. Against equal-performance
     alternatives it disappears.
   - Pre-registered rule: no pressure arms on sigmoid.
3. **The per-unit offset θ is not the cause of the sigmoid tax.**
   - Sigmoid without θ: 1.647 vs 1.656 with θ, seed 0 (−0.009), still ≈0.11 above softmax.
   - Softmax with θ: 1.522 vs 1.539 (−0.017, one seed). A possible small gain that needs seeds.
   - That is the fourth refuted explanation (output scale, coupling, saturation, θ). The cause is open.
4. **Dead units over time:** 44–80% unused in the first logged window; 11–44% at the end (K=4 highest).

## Predictions scorecard (`TA0-stage1b.md`)

| # | Prediction | Outcome |
|---|---|---|
| 1 | k-sweep monotone in bpb and bits | **Holds** (bpb 1.567 / 1.583 / 1.591 / 1.660; KL falls with K) |
| 2 | θ probes would show whether θ causes the sigmoid tax | **Refuted as the cause** (both moved < 0.03 paired) |
| 3 | λ=0.1 tax ≤ 0.02 bpb | **Fails paired** (+0.050). λ=1 > 0.05 ✓ |
| 4 | Pressure beats the k-sweep at equal bpb | **Refuted**: 0.34–0.60× |
| 5 | If it works, it works via peaky routing | n/a (it did not beat the sweep) |
| 6 | Noise and strict replays within 2× | **Holds** (within 1.14×) |

## Decision
**Pre-registered rule:** the gain disappears against the k-sweep, so the pressure adds nothing over choosing K. Here it
is worse than choosing K.

**Phase B** (2nd/3rd seeds, an eager arm, STE vs noise, a free-selection phase) was planned to *confirm a pass*. With
a robust fail on every guard above, those runs would not change the decision, so they were not run.

## First-principles reading
- A local explanation's size is dominated by:
  - the number of **active units** that must be named;
  - the reads per unit;
  - the edges per read;
  - plus the coupling the softmax normalizer creates among active units.

  Reducing K attacks the dominant term directly, *by construction*.
- The pressure tries to reach the same sparsity indirectly, by making a K=16 model robust to truncation. That costs
  more performance than training a K=8 model in the first place. It is the familiar pattern in which **an
  architectural constraint beats a soft penalty for the same sparsity** (as top-k sparse autoencoders beat L1
  ones).
- **Implication:** the cheapest observability lever at this scale is designed-in activation sparsity. The natural next
  step is to make it **adaptive**: few active units on easy tokens, more on hard ones. Examples: top-p routing, or a
  learned null slot in the softmax. Description length is averaged over positions, so adaptive K could beat the
  fixed-K curve rather than just move along it.
