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
- **v2: after critique round 4** (both critics), plus two empirical checks run in response.

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
