# T-A0 stage 1: results (2026-10-25)

> **Correction (2026-10-26). Two numbers below are wrong; see "Correction" at the end.**
> 1. Evaluation selected PEER units without the load-balancing bias they were trained with. With the bias:
>    - the sigmoid-gate tax is **+0.11 bpb (≈1.7× compute)**, not +0.27 (≈4×);
>    - sigmoid dead units are **11%**, not 59%.
> 2. The frontier replay used float-precision edge values while charging 8 bits each. Re-scored with an 8-bit
>    replay, the cancellation finding is *strengthened*.


**Scale:** 18 trainings (9 jobs × 2 paired seeds) on Modal L4 GPUs, 5–25 minutes each, ≈ 4.6 GPU-hours in total.
**Brief and pre-registered predictions:** `TA0-stage1.md`.
**Raw results:**
- `results/lm/seed*/ta0s1_*/seed*.json` (full metrics);
- `results/lm/ta0s1_mechanism.json` (follow-up mechanism checks);
- `python lm/report.py` (the summary table).

## Summary table (means over 2 seeds; validation bits per byte, lower is better)

| job | params | MFLOP/byte | bpb | Δ vs base | dense-equiv. compute | bits @ KL≤0.1 (prose / code) | bits @ KL≤0.3 | best KL in grid (prose, seed 0) | dead units |
|---|---|---|---|---|---|---|---|---|---|
| dense_d128 | 0.9M | 1.9 | 1.802 | | 0.98× | not reached | not reached | | |
| dense_d192 | 1.9M | 4.0 | 1.642 | | 1.06× | not reached | not reached | | |
| dense_d256 | 3.3M | 6.9 | 1.552 | | 0.97× | not reached | not reached | 0.61 (455k bits) | |
| dense_d256 + λ0.1 | 3.3M | 6.9 | 1.563 | | 0.92× | not reached | not reached | 0.37 (457k bits) | |
| **peer_softmax** (published PEER) | 35M | 4.6 | **1.576** | | **1.32×** | 204k / 205k | 56k / 56k | 0.041 (206k bits) | 18% |
| peer_sig λ0 (**base**) | 35M | 4.6 | 1.842 | — | 0.33× | 19.5k / 19.9k | 7.8k / 10.2k | 0.015 | 59% |
| peer_sig λ0.03 | 35M | 4.6 | 1.854 | +0.013 | 0.31× | 16.3k / 20.0k | 7.4k / 7.8k | | 62% |
| peer_sig λ0.1 | 35M | 4.6 | 1.875 | +0.033 | 0.28× | 23.7k / 22.0k | 9.0k / 9.8k | 0.077 | 65% |
| peer_sig λ0.3 | 35M | 4.6 | 1.887 | +0.045 | 0.26× | not reached | not reached | 1.58 | 66% |

**How to read the columns:**
- **Dense fit:** bpb = 8.30 − 0.194·ln(training FLOPs), from 3 widths × 2 seeds. So a 10% compute tax ≈ 0.020 bpb at
  this scale.
- **Full-budget replay check:** passed for every specialists (PEER) model (|KL| < 1e-8). For dense models it is
  0.02–0.05: dense keeps 256 candidate neurons of 1,024, and that cap costs a little.

## Predictions vs outcomes

| # | Prediction | Outcome |
|---|---|---|
| 1 | Per-unit thresholds are cheap: sigmoid within 0.02 bpb of softmax PEER | **Refuted.** +0.27 bpb, about 4× compute. 59% of units are never used, and the top-k cap still decides firing 30–40% of the time, so the coupling wasn't even removed. |
| 2 | PEER (λ=0) ≥ 1× dense per FLOP | **Holds for published PEER (1.32×)**, fails for the sigmoid variant (0.33×) |
| 3 | The tax rises smoothly with λ, within bands | **Holds.** 0.6×, 1.6× and 2.2× the 10%-compute unit, all inside the 30% band. |
| 4 | Pressure buys explanation length: ≥ 2× fewer bits at KL≤0.1 at the best in-band λ | **Refuted.** At best ~1.2× (λ=0.03, prose only). Higher λ makes explanations *worse*: λ=0.3 never reaches KL 0.3. |
| 5 | The architecture matters: PEER λ0.1 needs ≥ 3× fewer bits than dense λ0.1 at equal KL | **Holds strongly.** Dense never reaches KL 0.3 anywhere on the grid. At KL ≈ 0.37, PEER needs < 10k bits vs dense's 457k. *Caveat:* the dense control is raw neurons, the weak control the critics warned about. A transcoder-based control is still owed. |
| 6 | Dead units < 50% at λ=0, rising < 15 points under pressure | **Mixed.** Softmax PEER 18% ✓. Sigmoid PEER 59% ✗. The rise under pressure is +7 points ✓. |

**Decision rule triggered:** prediction 4 failed, so **redirect before stage 2** (per the brief).

## Why the pressure failed: two hypotheses, one refuted, one confirmed

**What the evaluator showed.** Breaking the frontier down by budget dimension, the pressured models became
extremely fragile to *any* truncation. At λ=0.3, keeping 1 attention source gives KL 15.9, against 0.18 at λ=0, and
even the most generous grid point gives KL 2.3.

**Hypothesis 1 (refuted): the information moved into attention.** The pressure covers per-position writer
concentration but not attention spread, so information might have spread across more source positions. Measured
from the checkpoints, at every position:
- attention entropy and the mass outside the top-16 sources are **unchanged** across λ (layer 0: 2.6–2.7 nats,
  16–18% outside the top 16);
- meanwhile the pressured quantities themselves did fall (outside-top-8 share: unit reads 0.17 → 0.05, value reads
  0.20 → 0.10, logit 0.43 → 0.03).

So the pressure hit its target, and attention didn't change.

**Hypothesis 2 (confirmed): the pressure was gamed by cancellation.** The pressure measured each read's *gross*
contribution share outside its top-8 writers. It is scale-free, so it can be satisfied by making a few contributions
**huge and opposite in sign**, with the signal carried by their small difference. Measured on the next-byte logit:

| model | gross Σ\|c\| | net \|Σc\| | cancellation ratio | top-8 share of net |
|---|---|---|---|---|
| peer_sig λ0 | 16.5 | 13.6 | 1.19 | 0.69 |
| peer_sig λ0.03 | 20.5 | 16.1 | 1.25 | 0.90 |
| peer_sig λ0.1 | 47.0 | 21.5 | 2.14 | 0.99 |
| peer_sig λ0.3 | 112.6 | 20.4 | **5.36** | 1.00 |
| dense λ0 | 35.3 | 14.3 | 2.42 | 0.39 |
| dense λ0.1 | 234.5 | 32.1 | **7.27** | 0.95 |

Unit reads in the last layer show the same pattern: cancellation 7.2 → 17.6 at λ=0.3.

Large canceling terms make the computation **fragile**. Any small error upstream (one dropped source, one truncated
edge) is a small fraction of a huge term, but a large fraction of the small difference that carries the signal. That
explains the truncation fragility.

This is the "materiality fallacy" and cancellation exploit the observability critic predicted in round 1. They
recommended reporting the cancellation ratio for exactly this reason. Now it is measured, not hypothesized.

## What we learned
1. **Wide sparse pools are more explainable than a dense MLP, and the published version is also more efficient per
   FLOP.**
   - At the most generous budget, softmax PEER's replay reaches KL 0.04.
   - The dense model's best is 0.61 at twice the description length.
   - This is the first real-text evidence that the "many specialists, few active" substrate helps observability
     *without* costing performance.
   - It is still a comparison against raw neurons, so the transcoder control is owed.
2. **Per-unit sigmoid thresholds are not a free way to remove unit coupling.** They cost ~4× compute here. The
   explanations they gave were cheaper partly because the model was weaker and used fewer units: a weaker model is
   easier to explain. The comparison that counts is at equal performance.
3. **Any proxy for "explainable" that the model can satisfy by rescaling will be gamed.** "Concentrated
   contributions" was satisfied by inflating opposing contributions. The pressure must be on the quantity we actually
   care about: whether a short description reproduces the output.
4. **The instrument earned its keep twice.**
   - The full-budget replay check caught a GPU compiler bug before any results (research log step 17).
   - The budget-dimension breakdown plus the mechanism checks turned "the pressure didn't work" into a specific,
     tested cause, after one plausible story was refuted (lesson L8 again).

## Proposed redirect: stage 1b (not yet run)
- **Base:** published softmax PEER. Its softmax normalizer becomes a *described* node in the explanation: the
  coupling is real, so the description must state it. This replaces the sigmoid gate.
- **Pressure on the real target: truncation self-distillation.**
  - On a subsample of each batch, run the model's own top-m truncated replay.
  - Penalize KL(full ‖ truncated).
  - This optimizes the frontier quantity directly. Cancellation no longer pays: big canceling terms make truncation
    *worse*, so they are penalized.
- **Cheaper variant for comparison:** the old share pressure plus an explicit cancellation penalty, (Σ|c| − |Σc|) /
  Σ|c|.
- **Fair control:** dense + per-layer transcoders, described in the same way, with and without the same pressure.
- **Always reported:** cancellation ratios, alongside the frontier.

---

## Correction (2026-10-26): evaluation-bias bug, and an 8-bit re-score

**Found by:** the performance critic in critique round 4, reading the code. **Confirmed** in `lm/run.py` and
`lm/frontier.py`. **Fixed** in commit `1a302b1`.

**The bug.** PEER training selects units with a load-balancing bias (aux-loss-free, updated outside the gradient).
`eval_loss`, `stability` and the frontier replay selected units *without* it, so they evaluated a different routing
than the one trained. Every checkpoint was re-scored with the bias (`results/lm/ta0s1_rescore_q8_stage1.json`).

**Corrected performance (val bpb, mean of 2 seeds):**

| model | without bias (reported above) | with bias (correct) |
|---|---|---|
| peer_softmax | 1.576 | **1.570** |
| peer_sig λ0 | 1.842 | **1.677** |
| peer_sig λ0.03 | 1.854 | **1.681** |
| peer_sig λ0.1 | 1.875 | **1.702** |
| peer_sig λ0.3 | 1.887 | **1.719** |

Dense models are unaffected (they have no routing bias).

**What changes:**
- **Prediction 1** is still refuted, but by +0.11 bpb (≈1.7× compute), not +0.27.
- **Dead units:** sigmoid 11% and softmax 14% (with bias), so prediction 6 now holds. The earlier "59% dead" was the
  bug.
- **Train/val gap:** the sigmoid model's large gap disappears (train 1.70, val 1.68).
- **The pressure tax** is essentially unchanged: +0.004, +0.025, +0.041 bpb.

**8-bit re-score of the frontier.** Edge values, attention weights and logit contributions are rounded to 8 bits
relative to each read's largest kept term, which is what the description length already charged for.

Full-budget KL (nothing truncated, only rounded):

| model | full-budget KL |
|---|---|
| honest models (softmax λ0, sigmoid λ0) | 0.001 |
| sigmoid λ0.1 | 0.077 |
| sigmoid λ0.3 | **0.289** |
| dense λ0.1 | **2.039**, against 0.038 for dense λ0 |

So the cancellers collapse exactly where the cancellation analysis predicted.

Corrected bits per position to reach KL ≤ 0.1 (8-bit replay, with bias, prose/code):

| model | bits |
|---|---|
| softmax λ0 | 204k / 204k |
| sigmoid λ0 | 15.0k / 19.5k |
| sigmoid λ0.03 | 12.3k / 11.5k |
| sigmoid λ0.1 | not reached / 27.3k |
| sigmoid λ0.3 | not reached |
| dense | not reached |

**What changes:**
- **Prediction 4** is still refuted. The best in-band λ (0.03) gives 1.2× fewer bits on prose and 1.7× on code,
  short of the ≥ 2× predicted, and stronger pressure is worse.
- **The sigmoid-gate model is ~13× cheaper to explain than softmax PEER** at ≈1.7× compute. That is a real
  trade-off. Whether it beats simply using fewer active units in softmax PEER (the k-sweep control) is the open
  question.
