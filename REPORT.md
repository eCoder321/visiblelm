# Report: an LM whose mechanism can be read (revised project, 2026-09-12 onward)

Living report. Numbers are single-seed pilots unless marked otherwise. Plan and gates: `PLAN.md`. Why the project was
re-based: `docs/REVIEW-2026-09-12.md`. Prior work: `docs/interpretable_lm_design.md`.

## 1. Goal and how it is scored

Build a next-token model where "why this output?" is answered by the computation itself, in a vocabulary a person can
read, verifiable by intervention, with no hidden channel, so behaviour on untested inputs can be predicted from the
mechanism. Four properties (faithful, legible, complete, compositional) plus generalization, each with a metric, each
compared against a null model (a dense transformer with post-hoc sparse autoencoders). See `PLAN.md` for the table.

## 2. Testbed

`vlm/testbed.py`. 6 rules (repeat, +1, −1, +2, copy-2-back, copy-3-back) over 16 content values, rule switches
mid-sequence, 24 tokens, exact oracle. Splits: train (19k distinct sequences), iid test, shift test (contains events
never seen in training: one held-out argument value per rule, and four held-out ordered switch pairs), long test
(32 tokens). Oracle floors: iid 0.62 nats/token; held-out switch steps 0.00; held-out argument steps 0.32.

## 3. Findings so far

### 3.1 The null model (dense transformer + post-hoc SAEs, 1 seed)

| metric | value | reading |
|---|---|---|
| iid NLL | 0.59 | at the floor |
| held-out switch pairs | 0.14 | generalizes: the "find the latest mode token" mechanism transfers |
| held-out argument (arithmetic rules) | 0% accuracy | cannot generalize: nothing in the data links value 7 in "+1" to value 7 elsewhere |
| held-out argument (copy rules) | 50–80% accuracy | partial |
| completeness gap (drop the SAE error term) | 0.002 | the SAEs reconstruct the residual almost exactly |
| magnitude audit (binarize SAE codes) | +3.2 nats | almost all the information is in *magnitudes*, not in which features are on |
| circuit sufficiency (retained log-prob fraction) | 0.58 vs 0.00 random | the ablation-found circuit is meaningful |
| argmax kept by the circuit alone | 33% | but not sufficient to reproduce the decision |
| clean features per state (held-out F1 > 0.8) | 18 / 7 / 2 of 96 | legibility collapses with depth |
| circuit nodes that are clean | 30% (iid), 14% (shift) | most of the mechanism is not nameable |

### 3.2 Propositional rule network (rulenet)

State = sparse named features + signed vote register; sparse lookup routing; fan-in-3 rules; no dense vector.
* Learns as fast as the dense model before pruning (0.74 at step 250).
* Pruning the **rules** to fan-in 3 is free (0.68 after recovery). Pruning the **routing tables** costs ~0.3 nats and
  neither L1, gradient-based regrowth, nor larger caps recover it (best 0.89–0.96). The dense phase learns a pairwise
  lookup table for attention; pruning cannot turn a table into a mechanism.
* **Propositional rules cannot generalize across values.** "IF mode=succ AND cur=7 THEN next=8" is a table entry.
  Held-out argument accuracy: repeat 77–88%, copy2 69–78%, arithmetic 0%.

### 3.3 Structured rule network (rulenet_g) — first pilot (`rng_v1`, 1 seed)

Features are grouped into slots holding one value; routing and rule maps are expressed at the slot level ("copy their
tok_content into my val0", one parameter for all 16 values). Arithmetic is an explicit per-rule value table
initialised to the identity. Routing tables shrink from ~10⁴ to ~10² entries per head.

| metric | null model | rulenet_g | reading |
|---|---|---|---|
| iid NLL (floor 0.62) | 0.59 | 0.66 | 0.07 nats tax, fully pruned (4+4+6 routing entries per head, fan-in 3) |
| held-out switch pairs (floor 0.00) | 0.14 | 1.37 | worse; to investigate |
| held-out argument: copy2 / copy3 / repeat | 80 / 50 / 62% | 92 / 86 / 100% | the slot copy transfers to unseen values |
| held-out argument: arithmetic | 0% | 0–26% | as expected for a value table |
| completeness gap | 0.002 | 0.000 | by construction |
| magnitude audit | +3.2 | +2.3 | magnitudes still carry information |
| circuit sufficiency vs random | 0.58 / 0.00 | 0.55 / 0.00 | similar |
| clean features per state | 18 / 7 / 2 | 41 / 40 / 57 / 47 / 47 | far more nameable, at every depth |
| circuit nodes that are clean | 30% | 50% | |
| description length | — | 3,326 nonzero parameters | |
| **north star**: discrete rulebook agrees with model | — | 49.5% | binarization alone: 30%; hard attention alone: 91% |

Diagnosis: attention is already nearly discrete, but the rules use continuous activation *magnitudes*, so reading
features as on/off loses half the predictions. Fix under test: clamp activations to [0,1] and, for the last quarter of
training, run the forward pass as the discrete program (binary features, hard attention) with straight-through
gradients, so the trained weights are the discrete program.

## 4. Changelog

* 2026-09-12 — testbed, harness, four model families, null-model baseline, propositional rulenet diagnostics,
  structured rulenet built and run once; north-star diagnosis; snap phase implemented; 3-seed controls queued.
