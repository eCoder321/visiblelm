# Redesign: a language model that is both performant and observable

**Status: v2 (2026-10-24).** This revision follows critique round 1: an independent performance skeptic and an
observability skeptic. Every change from v1 is listed, with the finding that forced it, in the refinement history
(§12). v1 is in git history (commit `e45149f`). The reasoning trail is in `THINKING-LOG.md`. Prior-art claims
marked *(verify)* are from memory, not yet checked against the papers.

**Scope.** "Observable" means **mechanism visibility**: what the model actually computed and why. **Epistemic
visibility** (what it believes and on what evidence; Graepel, MIT Tech Review, 2026-10-02) is parked as C7. Nothing
here may rule it out.

---

## 0. The short version

Dense transformers are already *exactly decomposable*: between nonlinearities everything is linear. What they lack is
three things: **(i) a named basis, (ii) few terms, (iii) a guarantee that the named terms are the whole story.**

v1 tried to get these by making the raw computation *small*: about 200 active features, cut back every layer. The
critics showed two problems:
- **(a)** that starves capacity;
- **(b)** v1's "by construction" checks are passed by *any* model that logs its state, so they don't measure
  observability.

v2's answer:
- **Let the raw computation be as large as performance needs**, but keep it in a named basis, exactly accountable,
  with every nonlinearity logged on a declared grid.
- **Train it under pressure to admit short descriptions.**
- **Measure observability as a curve, the explanation frontier.** How faithfully do descriptions of a given size
  reproduce the model? The comparison point is a dense transformer described by the best post-hoc tools.

**Observability is a curve, not a yes/no property.** A design is better when its curve sits closer to "small and
exact".

---

## 1. Criteria (v2)

### C1 — Performant
*Compute-equivalent tax* = 1 − (dense compute that reaches our loss) / (our compute), read off IsoFLOP curves.
- **Baselines:** both a dense transformer **and** a mixture-of-experts / PEER model with a dense residual. Frontier
  models are MoE, and MoE gives roughly a 2–4× compute multiplier *(verify)*, so beating only dense is too easy.
- **Report separately:** active FLOPs, total parameters, wall-clock. Use ≥ 3 seeds.
- **Bands:**
  - **success:** ≤ 10%;
  - **redirect:** 10–30% (find the responsible step in the ablation ladder, §11);
  - **kill that configuration:** > 30%.
- **Scale:** the tax must not grow across ≥ 3 sizes at fixed ratios (k/d, N/d). Ratios must match the target
  regime, not absolute sizes.

### Observability, level 1: hygiene
Necessary, **not sufficient**. A dense model that logs its state also passes these.

| | Requirement |
|---|---|
| H1 | **Bit-exact replay.** An independent executor reproduces the decision log bit-exactly. The log is canonically ordered and hash-chained, and **every** top-k / threshold decision in it is re-checked exhaustively (about one forward pass of cost). |
| H2 | **Books balance exactly.** Every logit and every feature value equals the sum of its logged contributions. Any imbalance is an implementation bug. |
| H3 | **No full-precision scalar anywhere in the log.** Every logged number (feature values, gates, attention weights, retrieval scores, norm gains) sits on its declared grid **at inference**. |

### Observability, level 2: discriminating
A dense model with logging must **fail** these, and every test is run on that control to prove it does.

- **The fixed description language 𝓛.** A description of a unit or feature is a short program over named features.
  Its pieces are: a selection rule with b-bit weights, a 1-D activation, a write list, and typed operators. 𝓛 is
  fixed before training and run by an **independent interpreter** with its own implementation.
- **Description length.** DL is measured in bits under a fixed code, and counted *to full grounding*: a description
  that uses a learned feature pays for that feature's own description, recursively.

| # | Criterion | Test (thresholds are provisional; calibrate on the first real-text run) |
|---|---|---|
| C2 | **Faithful** | Specify an intervention in *description* terms (e.g. "set `is-question` to 0"). Its effect, as predicted by the interpreter running the descriptions, must match the model's actual effect. Use random single, **pairwise** and off-distribution interventions. Target ≥ 95% agreement. |
| C3 | **Complete** | *Description-simulated replay (DSR) gap*: the interpreter runs only the descriptions, at full budget, and must match the model within 0.01 nats on iid **and** shifted data. *Canary:* an adversary co-trained to smuggle k bits through undeclared structure must be detected with power ≥ 0.9. |
| C4 | **Legible** | *Explanation frontier.* **Global:** fidelity vs total description bits. Target: dominates the dense + post-hoc control by ≥ 10× fewer bits at equal fidelity. **Local:** fidelity of a single decision vs bits read, to full grounding. Target: for ≥ 80% of tokens, ≤ 2 kbits (about a page) reproduces ≥ 90% of the decision's gross margin over the runner-up. Always report the cancellation ratio Σ\|c\|/\|Σc\| and the tail mass. Natural-language labels are optional and are audited by an independent simulator that is **never** used in training. |
| C5 | **Checkable** | H1–H3, plus a written **trust set**. A checker trusts: the weights, the 𝓛 interpreter, the test data, floating-point determinism, and log integrity. It does **not** trust the model's own explanations or labels. |
| C6 | **Predictive** | A third party makes random edits to **any** unit, entry or description, not ones we chose. The change predicted from the descriptions holds ≥ 90% of the time. Failures among units removed by clonal selection are reported too. |
| C7 | *Epistemic (parked)* | Hooks only: explicit propositions with provenance (§8). |

**Why this fixes the vacuity problem.** Every hygiene test reduces to "the log is the computation", which costs
nothing to pass. Level 2 asks something a logged dense model can't give: a *short* program, in a fixed language,
that reproduces it. Ciphers, polysemantic features and joint codes are not hidden by this measure. They show up as
a bad frontier, which is exactly where they belong.

---

## 2. First principles (unchanged from v1, condensed)

**Why LLMs perform:**
- capacity per FLOP (superposition and dense matmuls);
- benign optimization;
- generic inductive bias;
- graded representations;
- scale.

**Why they're opaque:**
- unnamed superposition;
- everything active;
- no completeness guarantee;
- unbounded magnitudes;
- too many operations to read.

The redesign keeps the first list and fixes the second, via (i)–(iii) in §0.

## 3. The tension and its four resolutions (revised)

| Performance needs | Observability fears | Resolution (v2) |
|---|---|---|
| N ≫ d features per position | features unnamed and entangled | **Carry the sparse code explicitly** (ids + values). Each feature has its own learned threshold (JumpReLU-style) against a target L0; there is no global top-k. Features only compete with their own threshold, so nothing has to "win" a slot every layer. The bus is **additive and persistent**: features stay until a logged decay or erase. Raw k is as large as performance needs, around d/8–d/4. Short explanations come from the frontier, not from a tiny raw k. |
| nonlinearity | nonlinearity mixes everything | **Every nonlinearity is a logged decision on a declared grid; everything else is exact accounting.** |
| graded magnitudes | magnitudes smuggle bits | **Hard quantization at inference.** b = 8 bits by default, with **per-feature scales**: each feature has its own range, so outliers in residual coordinates don't matter. Training uses quantization-aware training (straight-through estimator). Noise injection is at most a training aid with **fixed** scales, never the bound itself. The bound is representational: a value has 2^b possible states, full stop. |
| millions of contributions | unreadable | **Explanation frontier + hierarchical summaries with exact remainder.** Reported by margin over the runner-up, gross (\|c\|), with cancellation ratio and tail mass, so many small terms can't hide a decision. |

The three capacity strategies are unchanged:
- **S1** many named units, few active;
- **S2** an external library;
- **S3** a bounded fluency channel.

v2 folds S2 and S3 *into* Design A instead of building them as separate taxed stages (§8–§10).

---

## 4. Requirement tree (v2)

```
G  Performant AND observable LM
├─ R1 Performant (C1)
│  ├─ capacity: N ≫ d named features ........ BB1 bus (per-feature thresholds, additive, raw k ≈ d/8–d/4)
│  ├─ capacity: many units, few active ....... BB2 PEER-style pools reading the named bus
│  ├─ knowledge capacity ..................... BB2 (parametric, *visible*) + BB12 library (external, provenance)
│  ├─ sublinear selection .................... product keys over units and over the dictionary (re-encoding)
│  ├─ trainability ........................... BB5 top-k / thresholds (exact gradients on chosen items), QAT,
│  │                                            AuxK-style revival, aux-loss-free load balancing
│  ├─ attention expressiveness ............... BB4 general low-rank OV + re-encode; per-head sparsity incl. entmax
│  └─ generalization on structured sub-domains BB6 typed operators (numbers, dates, counts, lists)
├─ R2–R4 Faithful, Complete, Legible (C2–C4)
│  ├─ hygiene: log = computation, on grids ... BB9 decision log, BB7 hard quantization, BB11 exact replay/audit
│  ├─ short descriptions exist ............... BB10 clonal selection on fidelity-per-description-bit; DL penalty
│  ├─ short descriptions are measured ........ BB17 frontier evaluator with independent 𝓛 interpreter
│  ├─ the measure can't be fooled ............ BB18 canary adversary; BB19 dense + post-hoc control
│  └─ content vs style ....................... BB15 information-flow typing + non-interference test inside A
├─ R5 Checkable (C5) ......................... BB11 exhaustive checks, hash-chained log, trust set
├─ R6 Predictive (C6) ........................ third-party random edits; BB6 / BB12 editability
└─ R7 Epistemic (C7, parked) ................. BB12 entries with provenance; content-typed features
```

---

## 5. Lessons from our experiments (unchanged inputs)

| # | Lesson | Design consequence |
|---|---|---|
| L1 | Measure generalization, not memory. | Held-out and OOD splits in every test. |
| L2 | Sparsity is nearly free; **binarizing** costs 0.29–0.5 nats (T1/T2). | Discrete *choices*, multi-bit *values*. |
| L3 | Undeclared magnitudes carried 2.3 nats (E1). | Declared, *enforced* grids (now hard quantization at inference). |
| L4/L5 | Lookup tables can't generalize arithmetic to unseen arguments, and **neither can dense** (all models ~0% held-out succ/pred/pls2; E2redux). | Equivariant typed operators for structured sub-domains. |
| L6 | Weighted-sum gating fired on accidental mixes (E1c). | Test AND-gating against a linear threshold with a margin (§11, T-A2). |
| L7/L8 | With dense voting, whole-rule ablation misleads, and stories are cheap and often wrong (E1b, E1d). | Interventions are first-class and are checked against an independent interpreter. |
| L9 | Structure already beats dense OOD (shift 1.41 vs 1.84 NLL) but loses iid (0.78 vs 0.59). | Locate the iid tax by ablation ladder. |
| L10 | The iid oracle ignores rejection-sampling conditioning, so it is slightly loose. | Use the dense NLL as the practical floor. |
| L11 *(new)* | Post-hoc quantization to ≤ 16 levels destroyed agreement (49–62%) while full precision gave 91% (E1). 8-bit was never tested. | Default b = 8 with QAT. Treat 4-bit as an experiment, not an assumption. |

---

## 6. Building blocks (v2)

v1 blocks keep their parallels and prior art (see the v1 table in git history). Changed or new entries:

| BB | Block | v2 specification | Status |
|---|---|---|---|
| 1 | **Explicit sparse bus** | (id, value) list. Per-feature learned thresholds with a target L0, so k varies per token. Additive and persistent across layers. After each layer the dense update is **re-encoded** into the named list by a thresholded encoder, searched in O(√N) with product keys. Whatever the encoder drops is gone: a *measured performance cost*, not a hidden channel. Canonical ordering by id. | new |
| 2 | **Unit pools reading the named bus** | PEER-style units. Unit u's score `z_u = Σ_{f∈S_t} a_f w_{u,f}` is *defined* over named features, so there is one source of truth and no separately declared conditions that could disagree with selection. Selection is top-k_u (logged). The activation is a 1-D function (logged, on a grid). The write is a dense update, re-encoded as in BB1. An L1 penalty on w_u in feature space and the DL pressure push toward short descriptions. Near-miss candidates (the next m after the top-k) are logged. | partial |
| 3 | **Coincidence gating** *(now optional, tested)* | Log-domain product, `log g = Σ_i log σ(β(c_i − θ))`. Its gradient on an unsatisfied condition is large, not vanishing. Cold start from low θ, annealed upward. Adopted only if T-A2 shows it beats a linear threshold with a margin penalty. | test first |
| 4 | **Relay attention** | Scores split exactly over feature pairs. A **general low-rank OV map** (not renaming-only) whose output is re-encoded; renaming is the legible special case and is encouraged. Per-head sparsity: top-k for copy/induction heads, entmax/sparsemax for averaging heads. Weights quantized and logged. **Dropped attention mass** goes to the remainder ledger. | partial |
| 7 | **Declared-precision values** | Hard quantization at inference, b = 8 by default, per-feature scales, QAT in training, fixed (not learnable) noise scales if noise is used at all. | exists (QAT) / new use |
| 10 | **Clonal selection** | Fitness = **fidelity contributed per bit of description**, computed automatically from the frontier evaluator, with no human or LLM in the loop. Feature splitting when a feature's description fails on a coherent subset. | partial |
| 11 | **Audit** | Exhaustive re-check of every logged decision, bit-exact replay, hash-chained logs (like certificate-transparency logs), and the written trust set. | new for NNs |
| 12 | **Library** | An external extension of A's unit pool: entries = (text, key/value in A's feature space, provenance). **The dictionary is frozen (versioned) before indexing**, and indices are rebuilt per dictionary version. | partial |
| 15 | **Content/style typing** | Dictionary features carry a declared *content* or *style* type. Non-interference test: hold the content features fixed, vary everything else, and check that the content of the output doesn't change. Done inside A, not as a separate voice. | new |
| 17 | **Frontier evaluator** *(new)* | Independent 𝓛 interpreter, evaluator-defined truncation (keep the highest-attribution terms up to B bits, mean-ablate the rest), and global and local fidelity curves. | new |
| 18 | **Canary adversary** *(new)* | Co-train a smuggler that must pass k hidden bits through undeclared structure, and measure whether the audits catch it. This tests the *test*. | new |
| 19 | **Dense + post-hoc control** *(new)* | The same dense model described with the best post-hoc tools (transcoders / SAEs at every layer) in the same 𝓛, which gives the reference frontier. | exists (tools) / new use |
| 20 | **Blinded clerk** *(new, optional)* | Entity identities are re-randomized per context in training (placeholder ids), so entity facts *cannot* be memorized in weights and must come from the library. Its decisive test: closed-book entity recall at chance. | new |

---

## 7. Design A v2 — Glass Box (the core)

**Principle:** *let the computation be as big as it needs to be; keep every piece of it named, accountable and on
a declared grid; and make it compete to be short.*

### 7.1 Anatomy (per layer)
1. **Bus** `S_t = {(f, a_f)}`. Ids come from a dictionary of N (target N/d ≈ 250). a_f is on a b-bit grid with a
   per-feature scale. |S_t| is set by per-feature thresholds (target L0 ≈ d/8–d/4). The hardware form
   `h_t = Σ a_f E[f]` is just an encoding: every read `W h_t = Σ a_f (W E[f])` splits exactly.
2. **Norm gain** `s_t`: one logged scalar on a grid.
3. **Relay attention** (BB4): exact pair decomposition of scores, logged sparse weights, low-rank OV, re-encode.
4. **Unit pool** (BB2): exact per-feature selection scores, logged top-k_u, 1-D activations on a grid, dense update
   re-encoded into the named list. Gating (BB3) only if T-A2 earns it.
5. **Persistence**: features carry over between layers. Decay and erase are logged writes.
6. **Readout**: `logit_v = Σ_f a_f U[f,v]`. The output softmax defines the distribution.

**Nonlinearity inventory:** thresholds (bus and re-encode), unit top-k, unit activation, attention selection and
weights, norm gain. Each is a logged value on a declared grid, computed from exactly decomposed inputs.

### 7.2 Why it should perform (C1), and what the critics changed
- **No depth-wise starvation.** There is no global top-k and the bus is persistent. Raw k is large. This closes v1's
  likely-fatal bus bottleneck, *if* the re-encoding loss is small. The re-encoding loss is the one quantity that
  decides A, and diagnostic D1 (§11) bounds it cheaply before anything is built.
- **Unit pools are established.** Product-key memory, PEER and memory layers report matching or beating dense FFNs
  per FLOP *(verify; custom kernels were needed)*. The control baseline is MoE/PEER with a *dense* residual. The
  ladder (§11) then separates the pool tax from the bus tax.
- **Values stay multi-bit** (L2). 8-bit activations are routinely near-lossless in LLMs (W8A8-style results,
  *verify*). Per-feature scales sidestep the coordinate-outlier problem.
- **Honest costs.** Q/K/V/O projections stay dense. Gathers are memory-bandwidth-bound. Top-k attention needs a
  materialized-score or approximate path, so FlashAttention-class kernels need adapting. Expect a **wall-clock**
  tax on current hardware even if the FLOP tax is small. It is reported separately and treated as engineering, not
  as a property of the design.

### 7.3 How A meets each criterion

| Criterion | Mechanism | Status |
|---|---|---|
| C1 | Big persistent named bus, PEER pools, multi-bit values, general OV, MoE-level baselines | **Open, empirical.** D1 bounds it first. |
| H1–H3 | Decision log, canonical order, hard grids, exhaustive re-checks, hash chain | **Guaranteed** (hygiene only). |
| C2 | Interventions in description terms, judged by the independent interpreter | **Measured.** Strong where descriptions are short; the frontier says where. |
| C3 | DSR gap + canary. No full-precision channel exists (H3). Combinatorial codes show up as DSR gap, not as hidden bits. | **Measured.** The bound on state bits per position, k·(log₂N + b), is a guaranteed *cardinality* bound. |
| C4 | Frontier with DL pressure and fidelity-per-bit selection. Margin/gross/cancellation/tail reporting. | **Open, empirical.** The main scientific question after C1. |
| C5 | H1–H3 plus the trust set | **Guaranteed for consistency with the weights.** *Not* a guarantee that names are right (that is C4's job). |
| C6 | Third-party random edits judged by descriptions | **Measured.** |
| C7 | Content-typed features and library entries as future propositions | Hook. |

### 7.4 Riskiest assumption, decisive test, kill criterion
- **Assumption:** re-encoding into an explicit named list at every layer, with per-feature thresholds, raw
  k ≈ d/8–d/4 and N/d ≈ 250, costs ≤ 10% compute-equivalent when trained from scratch.
- **Decisive tests:**
  - D1 (post-hoc splice, a cheap upper bound);
  - then the ladder step "+explicit bus" in T-A0.
- **Kill:** the from-scratch bus tax exceeds 30% at the target ratios, **or** its L(N,k) fit shows an irreducible
  floor above 10% (loss floors set by k that a bigger N can't remove).

---

## 8. Design B v2 — Library (knowledge with provenance)

**What changed and why.** The critics showed that forbidding parametric memory works against the loss, that it is
evadable (the clerk can learn to detect edited entries), and that it is unnecessary in v2. **A's unit pool already
is a readable key→value memory**: FFN layers act as key-value memories *(verify)*, and here both keys and values are
named features. So knowledge in weights is no longer *hidden*. It sits in visible units.

B's job is therefore the things only an external store gives:
- **provenance**: where a fact came from;
- **editability without retraining**;
- **freshness**;
- **capacity beyond the parameters**.

### Design
- **Entries:** (text, key/value in A's frozen, versioned feature space, provenance). Retrieval uses the same logged
  selection mechanism as units, over a larger store.
- **Parametric knowledge is allowed and measured, not capped.**
- **Citation must be causal when shown.** A cited entry has to actually drive the output. Test it with
  type-matched, *plausible* edits that include provenance fields. Edits must not be detectable: a probe on clerk
  activations classifying edited vs unedited entries should sit at chance. Add a distractor test: answer from a
  wrong but well-formed entry.
- **"Hidden memory" audit.** Use a relearning test, not blank-library recall: how fast do withheld facts come back
  under a few fine-tuning steps?
- **Optional blinded clerk (BB20)** for settings that need *guaranteed* provenance: entity identities are randomized
  per context, so entity facts can only come from the library.

| Criterion | B's contribution | Status |
|---|---|---|
| C1 | Capacity beyond parameters. Gains are reported on a near-duplicate-deduplicated test set (13-gram filter), split by token type. | Measured. Retrieval gains are partly train/test overlap *(verify)*, hence the dedup. |
| C2 / C6 | Citations proven causal by edit tests | Measured |
| C3 | Retrieval is a logged selection on grids. The retrieval choice channel is bounded by log₂(store size) per entry. Signatures are recomputed by a fixed encoder, so they can't carry extra bits. | Hygiene guaranteed; DSR measured |
| C4 | Entries are text with provenance | Strong |
| C7 | Entries are explicit propositions with sources | **Best hook for epistemic visibility** |

**Kill:** citations aren't causal (reliance < 90% under plausible edits), **or** the edit-detection probe is well
above chance even after training against it.

---

## 9. Design C v2 — Content and style (inside A; a separate voice only as fallback)

**What changed and why.**
- The performance critic showed a separate planner→voice pipeline can at best *tie* a dense model. It also
  degenerates on content-dense text (code, math, names), where the plan becomes the token stream.
- The observability critic showed a dense voice and its parser can **collude** on a cipher, and a dense voice can
  inject facts from its own weights.

### Design
- **Inside A (primary):** dictionary features are typed *content* or *style*.
  - **Non-interference test:** hold the content-typed features fixed, vary the style features and any hidden
    context, and check that the content of the output (as parsed by an independent parser) doesn't change.
  - **Bound:** the information reaching the text beyond content features is at most style bits plus choice entropy.
    It is estimated, with paraphrase-equivalent plans for the choice entropy.
  - This gives "what it is going to say" visibility at **no architectural tax**.
- **Separate dense voice (fallback only).** Used only if fluency turns out to be A's dominant tax. Requirements:
  - an **independent** parser, with different architecture and data, anchored on human-written text;
  - **two-directional** equality, parse(text) == plan, with the *unparseable residue* rate as a first-class
    metric;
  - a paraphrase-invariance test: an independent paraphraser rewrites the text and the plan must still parse;
  - plan values checked at b bits;
  - canonical plan ordering.
  - *Principle:* a checker may be opaque if it is **independent and calibrated**, like test equipment. The system
    being checked is what must be observable.

**Kill (fallback only):** round-trip error > 5%, **or** the oracle-plan test (D4) shows planner + voice NLL above
the token-level model.

---

## 10. The composite (v2): one model, three views

The v1 composite stacked three separately taxed stages, so taxes compounded. **v2 is one Design-A model**:
- **B** is its unit memory extended outward with provenance;
- **C** is a typing discipline on its dictionary plus a non-interference test.

A trace of one answer:

> **"Paris"**: decision margin 4.1 nats over "Lyon".
> - 2 kbits of description reproduce 93% of that margin (gross). Cancellation ratio 1.3; tail 7% across 61 terms.
> - The main path: unit #88 412, whose short description is "answer-capital-question". It fired because its exact
>   selection score was carried 84% by the features {question-about-capital, entity=France}, each described to
>   grounding.
> - That unit read `answer=Paris` from library entry #4 411 (source: encyclopedia, 2025-03).
> - Edit that entry to "Lyon" and the output follows, as predicted by the interpreter ✓.
> - Content-typed features fixed ⇒ the answer is invariant to style ✓.

**Design D, "Open Notebook"** (parked, C7): an explicit belief state with an independent evaluator. Content-typed
features and provenance-tagged entries are its ingredients.

---

## 11. Test plan (v2)

**Ordering principle.** Test the riskiest assumption first and cheapest first. Use **one trained dense model to
answer several questions**. Add constraints **one at a time** so every tax can be attributed.

**Compute reality.**
- 4 CPU cores. HuggingFace is blocked by the egress policy; GitHub raw and PyPI are reachable.
- The corpus will be assembled from reachable sources: public-domain books mirrored on GitHub, CPython stdlib
  source as a code slice, and tiny-shakespeare for a sanity check against published baselines. Target 20–50M
  tokens, byte-level or small-BPE.
- Small-scale numbers can mislead in either direction:
  - tiny corpora need few features, so N-sweeps go flat for the wrong reason;
  - wide sparse pools are data-starved when small.
- So: match ratios (k/d, N/d) to the target regime, run ≥ 3 seeds, fit dense IsoFLOP curves, and never read a
  single-scale result as a scaling claim. Real scaling trends (≥ 3 sizes) will need a GPU. That is flagged for
  later, not assumed.

| # | Test | Question | Setup | Prediction (written before running) | Redirect / kill |
|---|---|---|---|---|---|
| **D1** | **SAE / transcoder splice** | Upper bound on the bus tax; L(N,k) floor | Train a small dense LM (d≈256, 4 layers) on the corpus. Fit TopK/JumpReLU SAEs at every layer on a grid of k/d ∈ {1/32, 1/16, 1/8, 1/4} and N/d ∈ {16, 64, 256}. Splice **all layers at once**. Report the tax as compute-equivalent and fit L(N,k). | Tax ≤ 30% at k/d = 1/4 post-hoc, falling with N. The floor is set by k. | Post-hoc tax > 2× dense-at-half-compute even at k/d = 1/4 → the bus needs a rethink before T-A0. |
| **D2** | **Value quantization** | Precision tax | Same model and SAEs, feature values at b ∈ {4, 6, 8}: post-hoc, then brief QAT | b = 8 near-free; b = 4 costs | b = 8 costs > 5% → per-feature scale scheme rethink |
| **D3** | **Top-k attention swap** | Attention tax | Same dense model, top-k attention post-hoc, k ∈ {4…64}, context ≥ 512 | Copy/induction heads fine at small k; averaging heads need entmax or large k | — (informs BB4 per-head policy) |
| **D0** | **Control frontier** | The bar A must beat | D1's spliced model described in 𝓛 | Gives the dense + post-hoc frontier | — |
| **T-A0** | **Ablation ladder, from scratch** | Where any tax lives | dense → +PEER pool (dense residual) → +top-k/entmax attention → +b-bit logged scalars → +explicit persistent bus → +DL pressure / clonal selection. 3 seeds, 2–3 sizes at fixed ratios. Frontier at every step. | The bus step costs less than in D1 (it's trained for it). Total ≤ 30% on CPU scale. The frontier improves sharply at the bus and DL steps. | Any single step > 15% → redesign that block. |
| T-A2 | Gating toy | Does AND-gating beat a linear threshold with a margin? | m = 4 AND task plus the switching testbed; cross-firing, convergence | Log-domain product ≥ linear threshold on cross-firing at equal NLL | If not, drop BB3 |
| T-A3 | Typed operators | Equivariance for generalization (L5) | Switching testbed, cyclic code + shift(+n). Control: dense with the same code. Also numeric tokens in the real corpus. | Held-out arithmetic ~0 → ≥ 0.9 | < 0.5 → inspect operator selection |
| T-B | Library causality and hidden memory | Are citations causal; does memory hide? | Synthetic biographies (bioS-style, *verify*), plausible edits, edit-detection probe, relearning test | Reliance ≥ 90%, probe at chance | B kill criteria (§8) |
| D4 | Oracle-plan test | Is the C fallback ever worth it? | Plan = content-lemma ids per sentence. Planner + voice vs token-level at equal FLOPs, on prose and a code slice. | Prose: tie or small loss; code: loses | Planner + voice > token-level + 10% → drop the C fallback |
| Canary | Test the tests | Do the audits catch a smuggler? | Co-train an adversary to pass k bits through undeclared structure in a T-A0 model | Detected with power ≥ 0.9 | Undetected → fix the audit before trusting any C3 claim |

**Shared metrics for every test:**
- compute-equivalent tax (vs dense **and** MoE/PEER);
- per-token-type loss (entity / function word / numeric / code);
- long-context and induction probes;
- dead-unit fraction and utilization entropy;
- global and local frontier;
- DSR gap;
- quantization bits used;
- cancellation ratio and tail mass;
- replay exactness.

---

## 12. Refinement history

### v1 → v2 (critique round 1, 2026-10-24)

| # | Finding (critic) | Change | Why |
|---|---|---|---|
| 1 | "By construction" C2/C3/C5 are passed by any logged dense model, so they are vacuous (observability) | Split criteria into **hygiene** (H1–H3) and **discriminating** criteria judged by an independent interpreter in a fixed description language. New central metric: the **explanation frontier**, with a dense + post-hoc control that must fail. | The real content of observability is *short* faithful descriptions. That is measurable and can't be faked by logging. |
| 2 | Global top-k with re-sparsification every layer starves depth-wise capacity; likely fatal (performance) | Per-feature thresholds, an additive persistent bus, raw k ≈ d/8–d/4, re-encoding loss as the decisive quantity, D1 as a cheap upper bound | Features shouldn't compete with unrelated features for slots. Legibility should come from the frontier, not from a tiny k. |
| 3 | Noise injection doesn't bound information at inference; full-precision scalars elsewhere in the log are channels (observability) | **Hard quantization** of every logged scalar at inference (H3), QAT, fixed noise scales, canonical ordering | Turns a statistical hope into a cardinality bound. |
| 4 | 4-bit has contrary evidence (E1) and activation outliers (performance) | Default b = 8, per-feature scales, D2 test | 8-bit activations are routinely near-lossless; per-feature scales avoid coordinate outliers. |
| 5 | "Terms-to-90%" is gameable: many small canceling terms hide the margin (observability) | Margin-based, gross, depth-to-grounding local frontier, plus cancellation ratio and tail mass | The materiality fallacy: immaterial items can sum to a material one. |
| 6 | Clonal selection Goodharts an F1-on-firing legibility score (observability); an LLM judge in the training loop is impractical (performance) | Fitness = fidelity per description bit from the automatic frontier evaluator. NL labels audited only by an independent simulator outside training. | Optimizing "short and faithful" *is* the goal, so Goodharting it is success. Keep human-language labels out of the loop so they can't be gamed. |
| 7 | Product-key selection and declared conditions are two sources of truth that can disagree (observability); learning a few conditions out of 10^6 is unspecified structure learning (performance) | Selection score **defined** over named bus features, with no separate declared conditions. Shortness is pushed by L1/DL and measured by the frontier. | One source of truth, and no impossible structure-learning step. |
| 8 | Product gates have vanishing gradients and cold-start problems (performance) | Log-domain gate, low-θ start, adopted only if it beats a linear threshold (T-A2) | Don't adopt a mechanism on analogy alone. |
| 9 | Renaming-only OV is too restrictive; top-k starves averaging heads (performance) | General low-rank OV with re-encoding, per-head entmax/top-k, dropped mass logged, D3 | Keeps attention's expressiveness; renaming stays as the legible special case. |
| 10 | Forbidding parametric memory is anti-loss and evadable by edit detection (both) | B becomes a provenance/editability extension of A's visible unit memory. Parametric knowledge allowed and measured. Plausible edits, edit-detection probe, relearning test, frozen versioned dictionary, optional blinded clerk. | In v2 knowledge in weights is no longer hidden, so there's no need to fight it. |
| 11 | Planner/voice can at best tie dense and degenerates on content-dense text (performance); voice and parser can collude (observability) | C moves *inside* A as content/style typing + non-interference. A dense voice is only a fallback, with an independent parser, two-directional checks and paraphrase invariance. D4 decides whether the fallback is ever worth it. | No compounded tax. A checker can be opaque if it is independent and calibrated. |
| 12 | Sampled decision checks leave holes; the checker's trust set is unstated (observability) | Exhaustive checks, hash-chained logs, written trust set | Exhaustive checking costs about one forward pass. Say what the guarantee does *not* cover. |
| 13 | Compositional names explode with depth (observability) | DL counted to full grounding; aliases allowed but audited; name-length distribution reported | The cost of meaning must be paid somewhere visible. |
| 14 | The baseline was too easy (dense only), "compute" was ill-defined, and small-scale results can mislead (performance) | MoE/PEER baseline, active FLOPs + params + wall-clock reported separately, ratio-matched sizes, ≥ 3 seeds, IsoFLOP curves, success/redirect/kill bands, ablation ladder | Make every tax attributable and every claim scale-aware. |
| 15 | T-A3 says nothing about real-text C1 (performance) | Moved after the real-text diagnostics. Kept for generalization and numeric sub-domains. | Riskiest assumption first. |
| 16 | No test of the tests (observability) | Canary adversary | An audit that has never caught a planted smuggler proves nothing. |

**Findings not adopted, with reasons:**
- **Use QuaRot-style rotations for outliers.** Rotating the residual mixes named features, which defeats the named
  basis. Per-feature scales solve the same problem in our representation.
- **Cap leakage at 10% (from v1).** Dropped, not strengthened. Visible parametric memory is acceptable, and only
  *hidden* memory is a problem. The relearning test measures exactly that.

## 13. What remains open after v2 (honest list)

Every criterion now has a mechanism and a test that can fail. Two are **empirical questions no paper argument can
settle**:
1. **C1:** the from-scratch re-encoding (bus) tax at target ratios. D1 bounds it from above within days on CPU, and
   T-A0 measures it.
2. **C4:** whether the frontier really sits ≥ 10× better than dense + post-hoc, and whether ~80% of real-language
   decisions fit in a page.

Three are engineering or resource limits:
3. wall-clock on current accelerators (custom kernels);
4. scale transfer (needs a GPU for ≥ 3 sizes);
5. C7 is parked.
