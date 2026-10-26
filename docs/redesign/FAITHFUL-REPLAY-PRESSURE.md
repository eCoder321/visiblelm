# Faithful-replay pressure: design and red-team (draft v0 → v1, before critique)

**Context.** In T-A0 stage 1, the description pressure was gamed by cancellation
(`docs/tasks/TA0-stage1-report.md`). It asked "does each read's *gross* contribution come mostly from its top-8
writers?". The model satisfied that by inflating a few opposing contributions:
- the next-byte logit's cancellation ratio went from 1.2 to 5.4;
- explanations got *worse* under truncation, not better.

**The goal of the pressure, stated precisely.** C4's local frontier (DESIGNS.md v2.2): *a short description of a
forward pass, executed from the input, reproduces the model's output distribution.* The pressure should push that
quantity directly, and nothing that can be satisfied without it.

---

## v0: the proposal in the stage-1 report
On a subsample of each batch:
1. Run the model's own **truncated replay** at a budget (m edges per read, n nodes per layer, m_a attention sources).
   This is the same replay `lm/frontier.py` evaluates.
2. Add λ·KL(model ‖ replay) to the loss.

Why it beats cancellation: huge opposing terms make any upstream truncation error large relative to the net, so
they raise the KL and are penalized. The pressure is the evaluation target itself, not a proxy for it.

## Red-team of v0: how a model under this pressure could still cheat

| # | Attack | Why it works on v0 | Severity |
|---|---|---|---|
| A1 | **Push computation into "given" side information.** The v0 replay *states* attention weights (from the full run), norm gains, and *which* units fire. It recomputes only reads' values. | Attention patterns computed by full-fidelity QK circuits, and unit selection computed from full product-key scores, enter the explanation as unexplained facts. The model can route arbitrary computation through them, paying only id and position bits. | **Fatal**. This is the same move as cancellation: satisfy the measure by hiding work where it isn't checked. |
| A2 | **Backup circuits (the "hydra effect").** | Truncation-KL measures *sufficiency* of the kept set, not *necessity*. Duplicating information across writers makes truncation harmless while the explanation ("A causes X") is wrong under intervention: ablate A and backups B and C still do X. Dropout-like pressure is known to breed redundancy. | Major |
| A3 | **Overfit the specific truncation rule.** | Edges are ranked by contribution magnitude at one fixed budget. The model can be exactly m-sparse at that m and fall off a cliff at m−1, or put "important but small" signal just inside the rank cut. | Major |
| A4 | **Exactly sparse but still canceling.** | If every read becomes exactly ≤ m-sparse, the replay is exact (KL 0) even if the kept terms are giant and opposite. That is faithful and short, but not *legible* ("+1,000,000 and −999,980"). | Moderate (legibility, not fidelity) |
| A5 | **Average-case only.** | KL is averaged over training positions, so rare inputs can still use dense hidden paths. | Major (C3) |
| A6 | **Drag the full model down to the replay.** | KL gradient flows into the full model too. It can become simpler by becoming worse. | This is the honest tax, but it must be measured, not hidden. |
| A7 | **Cost.** | A replay is roughly one extra forward pass on the subsample. | Counted in the tax. |

## v1: refined method (input to critique round 4)

1. **Nothing is given except ids (fixes A1).** The replay recomputes, from the kept edges only:
   - unit routing scores, then unit **selection** (product-key top-k on the replayed query, so a different set may
     fire, and its ids are what the description pays for);
   - unit activations;
   - the softmax normalizer, which is an explicit described node;
   - head **query and key** reads, then attention scores, then weights, then the top-m_a sources;
   - value reads;
   - **norm gains**, from the replayed residual;
   - the logits.

   The only description content not derived from kept edges is node ids and source positions, and those are
   charged in bits.
2. **Randomized budgets (fixes A3).** Each step samples (m, n, m_a) from a range, e.g. m ∈ {4, 8, 16}. At
   evaluation, add held-out budgets and a held-out edge-ranking rule: gradient × input, and greedy KL selection.
   These serve as the 𝓛′ check.
3. **Necessity is measured and, if needed, penalized (fixes A2).**
   - Evaluation: ablate a kept node in the full model; the replay predicts the effect; report agreement (C2).
   - Optional training term: on a few sampled kept nodes, penalize the gap between the full model's ablation
     effect and the replay's prediction.
4. **Cancellation is reported, with an optional mild penalty (A4).**
5. **Worst case (A5).** Report worst-decile KL by token type. Keep the trigger-searched inputs and the canary
   protocol from DESIGNS.md §11.
6. **KL(stop-grad(model) ‖ replay) (A6).** The replay is pulled toward the model, not the reverse, except through
   shared weights. Tax is measured as always.
7. **Substrate:** published softmax PEER, with its normalizer described (sigmoid gates cost about 4× compute in
   stage 1; diagnosis in progress).

**Pass criteria (falsifiable).** At a λ with tax ≤ 10%:
- bits@KL≤0.1 is ≥ 2× lower than at λ=0, on the trained budget range **and** on held-out budgets and ranking rules;
- the cancellation ratio does not rise;
- necessity agreement is ≥ 90%;
- the improvement over the fair dense + transcoders control (same pressure) is ≥ 3×.
