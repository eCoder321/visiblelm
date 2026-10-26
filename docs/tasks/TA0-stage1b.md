# T-A0 stage 1b — does faithful-replay pressure beat simply using fewer units?

**Design:** `docs/redesign/FAITHFUL-REPLAY-PRESSURE.md` v3.1 (signed off by both critics, critique round 6).
**Code:** `lm/frontier2.py` (evaluator and replay core), `lm/pressure.py` (training pressure), `lm/run.py`.
**Jobs:** `lm/jobs/ta0s1b/`.

This brief is written **before any stage-1b result exists**.

## The question
Faithful-replay pressure trains the model so that its own short, 8-bit, recomputed-from-kept-edges explanation
reproduces its output. Does it make explanations shorter **than the free alternative of activating fewer units**,
at equal performance?

## Phase A (launched 2026-10-26; 15 runs on Modal L4s)

| job | seeds | role |
|---|---|---|
| softmax_k4 / k8 / k12 / k16 | 2 each | the **k-sweep**: the bar the pressure must beat (r_heads = 4 fixed, everything else equal) |
| sigmoid_k8 | 2 | a second point for independent gates in the (bpb, bits) plane |
| softmax_theta | 1 | probe: PEER's softmax plus the per-unit offset θ |
| sigmoid_theta0 | 1 | probe: sigmoid gates without θ |
| press_l0.1 / l0.3 / l1 | 1 each | the pressure λ sweep on softmax K=16, teacher-forced selection, noise rounding, hinge weight 0.1 |

λ_eff = 4λ, because the pressure runs every 4 steps on one window. It ramps in from 10% to 30% of training.

**Phase B** (after phase A, picking the best λ):
- a 2nd seed for the best two λ;
- an eager arm (every step);
- STE vs noise;
- a free-selection-phase arm;
- 3rd seeds for the decisive comparison.

## Measurement (identical for every job)
`frontier2.evaluate` runs on 8 validation windows of 128 bytes per category (prose and code), with four replay modes:
- **free-strict-8**: primary;
- **given-strict-8**: upper bound with selection given;
- **free-noise-8**: the training surrogate;
- **free-strict-6**: a variant the model never trains against.

Primary numbers, per category:
- bits at KL p90 ≤ 0.1, interpolated;
- KL p90 at fixed budgets of 30k and 60k bits;
- explained fraction against an order-2 byte n-gram null.

Each comes with a bootstrap CI over windows. Also measured: val bpb, dead fraction over time, selection agreement.

## Predictions (written before results)
1. **The k-sweep is monotone.** bpb rises as K falls (K=4 worst), and bits fall roughly in proportion to K, because
   unit ids and edges dominate the description.
2. **θ probes.** If the offset θ is what makes sigmoid gates costly:
   - softmax_theta degrades toward ≈1.68 bpb;
   - sigmoid_theta0 improves toward softmax (≈1.57).

   If neither moves by more than 0.03 bpb, θ is ruled out too.
3. **Pressure tax.** λ=0.1 costs ≤ 0.02 bpb (≈10% compute). λ=1 costs more than 0.05.
4. **Pressure gain.**
   - At the best λ inside the tax band: KL p90 at 30k and 60k bits is lower than the k-sweep curve's at equal
     bpb, by a margin outside the bootstrap CIs.
   - bits@p90 ≤ 0.1 is ≥ 1.5× lower than the k-sweep's at equal bpb.
   - **Expected size (performance critic's estimate): 1.5–3×.** 10× would be surprising at this scale.
5. **Mechanism.** If the pressure works, it works mostly by making routing peaky: an effective K below 16. That
   shows up as concentration in the coefficients (top-4 share) and cancellation ratios that don't rise.
6. **Surrogate gap.** free-noise-8 and free-strict-8 KL are within a factor of 2 of each other. A larger gap means
   any null may be caused by the train/eval mismatch.

## Decision rules
- **Pass (proceed to stage 2):** prediction 4 holds on both prose and code, with 3 seeds in phase B, the strict-6
  variant agrees within 1.5×, and the cancellation ratio doesn't rise.
- **The pressure is just "fewer units in disguise":** the gain disappears against the k-sweep. Report it. The
  pressure adds nothing over choosing K.
- **Ambiguous null:** the eager arm and the noise-vs-strict gap decide between "pressure fails" and "pressure too
  sparse or mismatched".
