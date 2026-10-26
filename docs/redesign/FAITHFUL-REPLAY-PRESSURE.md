# Faithful-replay pressure: design, red-team, and critique rounds

**Context.** In T-A0 stage 1, the description pressure was gamed by cancellation
(`docs/tasks/TA0-stage1-report.md`). It asked whether each read's *gross* contribution came mostly from its top-8
writers. The model satisfied that by inflating a few opposing contributions.

**Goal of the pressure (C4's local frontier, DESIGNS.md v2.2):** *a short description of a forward pass, executed
from the input, reproduces the model's output distribution.* The pressure should push that quantity directly, and
nothing that can be satisfied without it.

Version history (details below):
- v0: the stage-1 report's proposal;
- v1: after my own red-team;
- v2: after critique round 4 (both critics), plus two empirical checks run in response;
- v3: after critique round 5. Both critics: no fatal holes, majors fixed below.
- **v3.1: after critique round 6. Both critics sign off** with the additions in the last section.

---

## Empirical checks run during round 4 (2026-10-26)

### 1. The quantized-replay "teeth" test (proposed by the observability critic)
The evaluator replayed edges at the model's float precision while *charging* 8 bits per edge. So canceling giants
survived truncation. We re-scored all stage-1 checkpoints with a **quantized replay**: every kept edge value is
rounded to 8 bits relative to the largest kept term of its read.

Even at full budget (nothing truncated), the quantized replay exposes cancellation:

| model | full-budget KL, 8-bit replay |
|---|---|
| PEER softmax, λ0 | 0.001 |
| PEER sigmoid, λ0 | 0.001 |
| PEER sigmoid, λ0.03 | 0.010 |
| PEER sigmoid, λ0.1 | 0.077 |
| PEER sigmoid, λ0.3 | **0.289** |
| dense, λ0 | 0.038 |
| dense, λ0.1 | **2.039** |

The cancellers blow up and the honest models don't. **The quantized evaluator has teeth.** It is now the default
for both evaluation and training (v2 §1).

### 2. Evaluation-bias bug (found by the performance critic)
Training selects units with the load-balancing bias. Evaluation, stability and the frontier replay selected without
it. Re-scored with the bias:

| | before fix | after fix |
|---|---|---|
| PEER sigmoid λ0 val bpb | 1.842 | 1.677 |
| PEER softmax val bpb | 1.576 | 1.570 |
| sigmoid tax vs softmax | +0.27 bpb, "~4×" | **+0.11 bpb ≈ 1.7× compute** |
| sigmoid dead units | 59% | 11% |
| sigmoid train/val gap | large | none |

Corrected frontier (8-bit replay, with bias), bits per position to reach KL ≤ 0.1, prose/code:

| model | bits |
|---|---|
| softmax λ0 | 204k / 204k |
| sigmoid λ0 | 15.0k / 19.5k |
| sigmoid λ0.03 | 12.3k / 11.5k |
| sigmoid λ0.1 | not reached / 27.3k |
| sigmoid λ0.3 | not reached |
| dense | not reached |

---

## v2 method (input to critique round 5)

1. **Quantized replay, in training and evaluation.** Kept edge values, attention weights and logit contributions
   are rounded to b=8 bits relative to their read's largest kept term (straight-through gradient in training).
   Precision is the currency of cancellation: a description that states its numbers at 8 bits cannot carry a small
   net signal on top of canceling giants.
2. **Nothing given except charged ids.** The replay recomputes, from kept edges only:
   - unit routing and **selection**;
   - activations;
   - the softmax normalizer, as a described node;
   - head **query and key** reads and attention weights;
   - norm gains;
   - logits.

   Charging and checks:
   - Derived selections cost log₂(candidate set) bits per selected item. Kept attention sources cost log₂(t).
   - **Selection agreement** (mass-weighted overlap between replay-selected and full-run-selected units) is
     reported, and is a pass criterion.
   - A given-selection replay is reported as an upper bound.
   - The routing bias is a described constant.
3. **Training schedule.**
   - The pressure is applied **lazily**: every 8 steps, on 2 windows. A replay is effectively a second forward pass
     plus pair contributions, about 4–10× a forward on those windows, so amortized overhead is ≈ 10% or less.
     It is counted in the tax.
   - Budgets (m, n, m_a) are sampled per application.
   - Start with teacher-forced selection: the replay uses the full run's selected set. Then switch to free
     selection, with a **margin hinge** on replayed selection and attention scores (k-th vs (k+1)-th), so selection
     is robust to truncation.
   - KL(stop-grad(model) ‖ replay).
4. **Necessity: evaluation only, never trained on** (training on it would reward ablation-insensitivity, i.e.
   hydra):
   - group-ablation curves (remove the top-j kept nodes, j = 1…m), predicted vs actual degradation;
   - a pair test (node plus its most correlated twin);
   - resample-patching from matched counterfactual inputs instead of zeroing;
   - **necessity coverage**: the share of output KL carried by nodes whose ablation exceeds a noise floor.
5. **Variant replays the model never trains against:** resample-replace instead of drop; 6-bit values;
   additive noise at 2⁻ᵇ of each read's scale; given-selection; a fresh greedy-KL description search vs magnitude
   ranking. If greedy is far shorter, the model overfit the ranking rule. Later, the independent 𝓛 interpreter
   reading a serialized description.
6. **Normalized fidelity** (KL can be gamed by blurring): explained fraction EF = 1 − (NLL_replay − NLL_model) /
   (NLL_null − NLL_model), with NLL_null from byte unigram frequencies. Pass on the **p90** and per token type, not
   only the mean.
7. **Global opacity check:** amortized edge-table size (distinct writer→reader pairs used across 10⁵ tokens) and
   per-unit consistency of source types across contexts. Add a global term only if these drift under pressure.
8. **No selection bias in the frontier:** pick budgets on a calibration split, report on a held-out split.
9. **Controls** (performance critic):
   - a **k-sweep**: the same substrate with fewer active units (topk 8 and 4) and no pressure;
   - **unit-dropout**;
   - **dense + per-layer transcoders**, with and without the same pressure.

   If the pressure doesn't beat the k-sweep curve at equal bpb, it is just a roundabout way of reducing k.
10. **Substrate** pending the gate ablations (`sigmoid_scaled`, `sigmoid_norm`): softmax (optionally with a null
    slot so a position can abstain) vs normalized sigmoid.

**Pass criteria (v2).** At a λ with tax ≤ 0.02 bpb (≈10% compute) vs its λ=0 base, measured with the 8-bit,
free-selection replay on the held-out split:
- bits to reach p90 EF ≥ 0.9 are **≥ 2× fewer than its λ=0 base** and **≥ 1.5× fewer than the best k-sweep /
  unit-dropout control at equal bpb**;
- the advantage holds within 1.5× under the variant replays (§5);
- the cancellation ratio does not rise;
- selection agreement ≥ 0.95;
- the necessity group-ablation curves agree (predicted vs actual R² ≥ 0.9, coverage reported);
- ≥ 3× fewer bits than the pressured dense + transcoders control.

---

## Red-team history

### My own red-team of v0 (before critique)
| # | Attack | Status in v2 |
|---|---|---|
| A1 | Computation pushed into "given" side information (attention weights, norm gains, which units fire) | §2: recomputed and charged. Selection agreement is a pass criterion. |
| A2 | Backup circuits (hydra): sufficiency without necessity | §4: evaluation-only group, pair and resample necessity tests |
| A3 | Overfitting the truncation rule or budget | §3 randomized budgets, §5 variant replays and greedy search, §8 held-out split |
| A4 | Exactly sparse but canceling | §1 quantized replay (shown to have teeth) |
| A5 | Average-case only | §6 p90 and per-type criteria; canary protocol (DESIGNS.md §11) |
| A6 | Full model dragged toward the replay | stop-grad target; tax measured |
| A7 | Cost | §3 lazy application, counted in the tax |

### Critique round 4

**Observability critic.**
- Cancellation survives exact replay → §1.
- Recomputed selection opens hidden self-repair, interchangeable families and uncharged scans → §2.
- Single-node necessity is ill-posed → §4.
- Train and eval share replay semantics → §5.
- Absolute KL is gameable by blurring → §6.
- Local-only pressure allows globally opaque units → §7.
- Grid-min selection bias → §8.

**Performance critic.**
- Evaluation-bias bug → fixed and re-scored.
- Sigmoid diagnosis: missing competition most likely, output scale least likely. The ablations test it.
- The replay is a full second forward and pair terms are 4–10× → §3 lazy application.
- No gradient through top-k flips → §3 margin hinge, teacher-forced then free.
- It is a disguise of slimmable networks, LayerDrop, nested dropout and end-to-end SAEs on KL. The literature
  predicts small full-budget loss and weaker small budgets.
- The k-sweep and unit-dropout controls are required → §9.
- Expected magnitude: ~2× is plausible; 10–100× would mean ~8 units per layer, a tax of maybe 15–40%. That is a
  guess, not evidence.

---

## v3 changes (critique round 5). These override v2 where they differ.

**Quantization (the observability critic's anchor hole, the performance critic's STE hole).**
- **Evaluation (strict):** each read's scale is a **power of two** at or above its largest kept term: 5 bits,
  charged in description length. **Every** term is rounded on that grid, including the largest, so no exact anchor
  remains. Node coefficients feeding described edges are 8-bit floats.
- **Training (surrogate):** straight-through rounding hides cancellation from the gradient. A giant term gets the
  same gradient as unrounded. So training adds **uniform noise of ±½ step**, where the step is set by the read's
  *continuous* largest term, making it differentiable. Expected KL then grows with the scale, so the gradient
  pushes giants down.
- Unit test: on a synthetic canceling pair, the gradient on the giants is nonzero and shrinks them.
- **Ablation arm:** straight-through vs noise at the best λ. Noise must lower the cancellation ratio; STE is
  expected not to.

**Fidelity metric.**
- The explained fraction is computed as a **ratio of sums within token-type bins**, not per position.
- The null is a **fixed order-2 byte n-gram model** fitted once on the training split, the same for every model.
- **Co-primary pass condition:** absolute KL p90 ≤ 0.1. EF alone may not loosen the bar.

**Selection hinge.** A **ranking hinge**: the full run's selected units' replayed scores must exceed the best
non-selected replayed score by a fixed margin **measured in the read's quantization step**. That makes it
scale-free, so inflating scores doesn't help. The same rule applies to attention logits.

**Selection agreement.** Report mass-weighted *and* unweighted agreement over the kept top-n.

**Necessity.** Group-ablation curves are judged by relative error at each step plus rank correlation, not R² of
cumulative curves (both rise monotonically, which inflates R²).

**Schedule.**
- The pressure is applied **every 4 steps on 1 randomly chosen window**, with the effective λ stated (λ_eff = λ ×
  4), and ramped in.
- **One eager arm** (every step) at the best λ, so a null result can be told apart from too-sparse pressure.
- The pressured fraction is reported.

**Measurement.**
- A finer budget grid. Bits at the threshold are **interpolated** in log-bits along the Pareto set; a model that
  never reaches the threshold is capped and flagged.
- Bootstrap confidence intervals over windows.
- **3 seeds** for the decisive arms.

**Comparison plane.**
- Every claim is made in the (bpb, bits) plane against a **pre-registered k-sweep** on the same substrate:
  - topk K ∈ {4, 8, 12, 16} per retrieval head, r_heads = 4 fixed;
  - the same steps, lr, bias and tokens; 2 seeds each;
  - a monotone fit of log(bits) vs bpb, with no extrapolation.
- The pressured model's bpb must fall inside the k-sweep's range.
- Reducing K cuts bits roughly in proportion for free, and the pressure has to beat that.

**Reachability, stated honestly.** At this architecture and scale the bit floor is ~5–10k bits per position (≈8
units per layer, 8 edges per read). The design's 2-kbit C4 target is **not reachable here**, so stage 1b is judged
on relative criteria only.

**Stage 1b run list (≈15 runs, ≈5 GPU-hours on L4):**
- **Substrate:** chosen from the gate ablations by position in the (bpb, bits) plane. Rule: `sigmoid_norm` is used
  if it lands within +0.03 bpb of softmax with ≤ 30k bits@KL0.1; otherwise the best-positioned gate, with its tax
  stated.
- **k-sweep on it:** K ∈ {4, 8, 12} × 2 seeds (K = 16 exists).
- **Pressure:**
  - λ ∈ {0.1, 0.3, 1} (noise rounding plus ranking hinge, teacher-forced selection), 1 seed each;
  - the best two λ get a 2nd seed;
  - the eager arm at the best λ;
  - STE vs noise at the best λ.
- **Decisive comparison:** the best pressured arm vs the nearest k-sweep point, with a 3rd seed each.
- **Deferred to stage 2:** the dense + transcoder control and unit-dropout.

---

## Gate ablations: results, and the substrate decision (2026-10-26)

All four gates re-scored with the routing bias and the 8-bit replay (2 seeds each):

| gate | val bpb | bits@KL≤0.1 (prose/code) | dead units |
|---|---|---|---|
| softmax (published PEER) | **1.570** | 204k / 204k | 14% |
| sigmoid (independent, unnormalized) | 1.677 | **15.0k / 19.5k** | 11% |
| sigmoid ÷ K (independent, output scale like softmax) | 1.780 | 14.2k / 16.9k | 21% |
| sigmoid normalized over selected (coupled) | 1.711 | 108k / 156k | 54% |

**Why does sigmoid cost performance? Three explanations tested, all refuted:**
1. *Output scale.* Scaling the gates down made performance worse, not better.
2. *Missing competition.* Normalizing the sigmoid gates did not recover softmax's performance.
3. *Saturation / flat gates.* With the bias, sigmoid saturation is only 5–7%. Gate-share entropy and top-4
   coefficient share do not separate softmax from the sigmoids.

**The cause is open.** End-of-training snapshots can't see training dynamics: how units specialize early, how the
bias interacts with each gate. A dead-fraction-over-time log is the next cheap probe.

**A clean explainability finding.**
- **Normalizing over the selected units is what makes explanations expensive.** Both normalized gates need
  100–200k bits; both independent gates need ~15k.
- Normalization spreads weight across all selected units and couples them. So a faithful local explanation must
  name nearly all of them, and dropping one rescales the rest.

**Substrate decision (per the v3 rule).**
- `sigmoid_norm` fails the rule (+0.14 bpb, 108k+ bits).
- In the (bpb, bits) plane the Pareto points are softmax and plain sigmoid; scaled and normalized sigmoid are
  dominated.
- Stage 1b uses **softmax** as the pressure substrate, the published design with the best performance. The k-sweep
  is on softmax.
- Plain sigmoid is plotted as an extra architectural point in the same plane. The question it answers: is
  "independent gates" itself a better trade than fewer softmax units?

---

## v3.1: critique round 6, final sign-off (2026-10-26)

Both critics: **no fatal issues; sign-off conditional on these additions**, all of which are adopted.

**Training noise matched to strict evaluation** (observability critic).
- Evaluation's power-of-two step is 1–2× the training step, and deterministic rounding kills kept terms below ½ step.
  Both mismatches are conservative: they can create a false *null*, never a false pass.
- Fix: train with step = (continuous largest term) × r, r ~ log-uniform[1, 2).
- Report KL under both the noise replay and the strict replay for every arm. A large gap flags a mismatch-driven
  null.
- Optionally, use deterministic rounding with STE on a minority of pressured steps.

**Free selection is trained, not just tested** (both critics). The best-λ arm adds a **free-selection phase** (last
25% of training) with the ranking hinge active. Selection agreement ≥ 0.95 and free-selection KL gate the pass for
that arm only. For teacher-forced arms they are reported, not gating.

**Fixed-budget comparison** (observability critic). Softmax may never reach KL p90 ≤ 0.1 within the grid, which
would leave bits-to-threshold capped. So **KL p90 and EF at fixed budgets of 30k and 60k bits** are a primary
comparison too, in the (bpb, bits) plane against the k-sweep.

**A second architectural point** (performance critic).
- Plain sigmoid may dominate the softmax k-sweep at equal bpb, and one point can't show that. So **sigmoid K=8 × 2
  seeds** is added.
- Pre-registered rule: if the sigmoid points lie below the softmax k-sweep fit, the pressure arms are also run on
  sigmoid, with its ≈1.7× compute flagged against the 30% band and traded off explicitly.

**The sigmoid tax: the performance critic's leading hypothesis.**
- All sigmoid variants share a per-unit offset θ_u and a bounded gate. Selection ranks by s, but the gate acts on
  s − θ_u: a gauge ambiguity between the shared half-key score and θ_u weakens the ranking signal.
- Cheapest probes, one seed each:
  - softmax(s − θ_u): if θ is the culprit, it degrades toward 1.68;
  - sigmoid with θ frozen at 0: if θ is the culprit, it improves.
- Free probes: the fraction of selected units with raw gate < 0.1; dead fraction over time.

**Final stage 1b run list (≈19 runs, ≈6.5 GPU-hours on L4):**
- **Softmax k-sweep:** K ∈ {4, 8, 12} × 2 seeds (K=16 exists).
- **Sigmoid K=8** × 2 seeds.
- **Pressure on softmax:**
  - λ ∈ {0.1, 0.3, 1}, 1 seed each; the best two get a 2nd seed;
  - the eager arm at the best λ;
  - STE vs noise at the best λ;
  - one free-selection arm at the best λ.
- **Decisive comparison:** the best arm vs the nearest k-sweep point, 3rd seed each.
- **θ probes:** softmax(s − θ), sigmoid with θ=0, 1 seed each.
- Conditionally, the pressure arms on sigmoid.
