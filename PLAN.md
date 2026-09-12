# Plan (revised 2026-09-12)

Living document. The goal is fixed; the design and the phases change as the measurements come in.
Prior work and the reasons for this revision: `docs/REVIEW-2026-09-12.md`. Prior report: `docs/interpretable_lm_design.md`.

## Goal

A language model whose answer to "why did it produce this output?" is the computation itself, in a vocabulary a
person can read, verifiable by intervention, with no hidden channel, so that behaviour on inputs never tested can be
predicted from the mechanism.

The goal is scored, not asserted. Every design is measured on the same harness:

| Property | Metric (all in `vlm/metrics.py`) | Null model must lose on it |
|---|---|---|
| P1 Faithful | Path faithfulness: the circuit found by ablation is necessary and sufficient for the prediction, versus a random node set of the same size | dense + post-hoc SAEs |
| P2 Legible | Held-out predicate F1 per feature and per rule; fraction of *circuit nodes* that are clean | same |
| P3 Complete | Completeness gap (run only through the legible states); magnitude audit (binarize the states) | same |
| P4 Compositional | The rulebook: printed rules and routing tables; description length; and the north-star test below | same |
| Generalization | NLL and accuracy on held-out (rule, argument) events and held-out mode switches, against the exact oracle | dense transformer |

**North-star test (P4).** Extract the rulebook from the trained model, execute it symbolically on the shift split,
and compare its predictions to the neural model's and to the truth. If the rulebook agrees with the model, the
explanation *is* the mechanism. If both agree with the truth on held-out events, the mechanism generalizes.

## What changed from the previous plan and why

1. **New testbed first, replication later.** The old synthetic language had 180 distinct sequences and every test
   sequence was in training. `vlm/testbed.py` has ~10^5 distinct sequences per mode, mid-sequence rule switches,
   held-out (rule, argument) events, held-out switch pairs, a length-extrapolation split, and an exact oracle.
2. **Controls are gates.** Every result is compared with (a) a dense transformer with post-hoc sparse autoencoders at
   every layer (what "by construction" must beat) and (b) the previous design (bottleneck v3) and its
   "named-neuron" variant with an identity dictionary. Absolute numbers on their own are not evidence.
3. **Faithfulness is measured on the whole path**, at every state and every position, not on the last layer's
   linear readout. Sufficiency is compared with a random circuit of the same size.
4. **Legibility is held-out**: predicates are fitted on half the data and scored on the other half, over an
   enumerated vocabulary (single variables and pairs). "Clean" means held-out F1 > 0.8.
5. **Hidden channels are audited**, not assumed away: the completeness gap and the magnitude audit are reported for
   every model.
6. **The dense vector is gone from the new design.** The previous design still decoded the sparse code into a dense
   d-dimensional vector inside every block. The new design (`rulenet`) has no dense space: the state is only ever a
   sparse code over named features, plus a separate signed output register.

## The design under test: rulenet

State at each position = (`code`: at most k active non-negative features out of F, the first V of which are
"current token is v"; `out`: V signed votes, one per vocabulary item). Nothing else exists.

* **Embedding**: a sparse non-negative table token → ≤ 3 features (readable; lets the model define "is a mode token").
* **Route** (per head): score(t, s) = Σ code_t[f]·A[f,g]·code_s[g] + Σ u[g]·code_s[g] + rel[t−s], strictly causal,
  with a null source; the attended features are copied through a sparse relabel table P[g → f']. Reads as
  "LOOK FOR g (prefer distance d), MATCH (f,g), COPY g AS f'". A, u, P are pruned to a few entries per head.
* **Compute**: R rules; rule r reads ≤ 3 features through W1[:, r], applies a threshold, writes ≤ 2 features through
  W2[r, :]. Reads as "IF w1·f1 + w2·f2 + w3·f3 + b > 0 THEN write g1 +v1, g2 +v2".
* **Persistence**: every write is a signed delta into the state, `code ← topk(relu(code + Δ))`; features stay until
  erased. Votes accumulate in `out` and are never read back.
* **Readout**: logits[v] = out[v]·scale[v] + bias[v]. Exact per-rule attribution by construction; the "which token
  does this feature vote for" search that killed the old v4 sparse readout does not exist.
* **Training**: dense masks at start, L1 pressure from step 0, progressive magnitude pruning to the structural caps,
  then fine-tuning under the hard structure. The printed rulebook is what runs; there is no soft/hard gap to verify.

Open design questions, to be settled by measurement: whether magnitudes carry hidden information (audit; if so,
clamp activations to [0, 1]); whether magnitude pruning can find sparse mechanisms or gates must be learned
(hard-concrete / regrowth); how description length scales with task complexity and with real text.

## Phases and gates

* **Phase 0 (done in this session)**: testbed, oracle, harness, all four model families, metrics, job runner.
* **Phase 1 — trainability of rulenet on the testbed.** Gate: iid NLL within 0.1 nats of the dense model at matched
  steps, over 3 seeds, with the full structural caps in force.
* **Phase 2 — the controls.** Dense+SAE null, bottleneck v3, named-neuron, rulenet: all metrics, 3 seeds. Gate:
  rulenet beats the null on path faithfulness, circuit-clean fraction, completeness and magnitude gaps; its
  generalization to held-out events is at least the dense model's.
* **Phase 3 — the north-star test.** Rulebook extraction and symbolic execution. Gate: rulebook/model agreement
  > 95% on the shift split, and a written prediction of held-out-event behaviour from the rulebook alone before
  the model is run on them.
* **Phase 4 — hidden-channel closure.** If the magnitude audit shows a gap, clamp activations and re-run Phase 2.
* **Phase 5 — real text.** Character-level text with the small config; report tax at plateau, description length
  per prediction, and intervention-defined legibility. Gate: a description-length budget agreed in advance.
* **Phase 6 — scale probe.** 3–5 sizes: does the tax and the description length per prediction shrink or grow?

## Working rules

* Every number in the report is from ≥ 3 seeds unless marked as a single-seed pilot.
* Every new constraint gets a gradient smoke test before a full run.
* Jobs are restartable and write results to `results/<name>/`; the report `REPORT.md` is updated per phase with a
  dated changelog.
* No result counts until it beats the null model on the same harness.
