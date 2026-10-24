# Redesign: a language model that is both performant and observable

**Status: v2.1 (2026-10-24), after two critique rounds** by an independent performance skeptic and an observability
skeptic. Every change, with the finding that forced it, is in the refinement history (§12).
- Earlier versions: v1 (`e45149f`), v2 (`742af31`).
- Reasoning trail: `THINKING-LOG.md`.
- Prior-art claims marked *(verify)* are from memory.

**Scope.** "Observable" means **mechanism visibility**: what the model computed and why. **Epistemic visibility**
(what it believes and on what evidence; Graepel, MIT Tech Review, 2026-10-02) is parked as C7. Nothing here may rule
it out.

---

## 0. The short version

**What dense models already have.** Dense transformers are exactly decomposable: between nonlinearities everything
is linear.

**What they lack.**
- **(i) a named basis;**
- **(ii) few terms;**
- **(iii) a guarantee that the named terms are the whole story.**

**What the critique rounds taught us.**
- *An explanation is a description shorter than what it explains that still reproduces it.* A complete log is not
  shorter, so it explains nothing, and any logged dense model passes "complete log" checks. Observability is
  therefore a **curve** (description length vs fidelity): the **explanation frontier**. It is not a yes/no property.
- *Squeezing the computation into a small named list* (v1), or *re-encoding it into one every layer* (v2), costs
  capacity. It also makes names ambiguous, because many lists fit the same vector.

**The v2.1 design: the Ledger Transformer.** The residual stream stays dense, as in a transformer, but **every write
into it comes from a logged decision**: a unit firing, or an attention head selecting sources. Each logged value
sits on a declared 8-bit grid. So the stream is, by construction, an **exact ledger of named writes**. Nothing is
decoded, so there are no ambiguous names and no bottleneck. Architecturally this is close to known-performant
wide-sparse models (PEER / mixture-of-experts). On top of it sit:
- training pressure toward short descriptions;
- a fixed description language run by an independent interpreter;
- an audit that has to catch planted smugglers.

---

## 1. Criteria (v2.1)

### C1: Performant
*Compute-equivalent tax* = 1 − (dense compute reaching our loss) / (our compute), read from IsoFLOP curves.
- **Baselines:** dense **and** PEER/MoE with a dense residual and no logging or pressure. Frontier models are MoE
  *(verify ≈ 2–4× multiplier)*.
- **Report separately:** active FLOPs, total parameters, wall-clock. Use ≥ 3 seeds. **All** training compute counts,
  including clonal selection and description-pressure overheads.
- **Bands:** success ≤ 10%; redirect 10–30%; kill that configuration > 30%. The tax must not grow across ≥ 3
  ratio-matched sizes.
- **Stability:** loss-spike rate and dead-unit fraction are reported from step 1.
- C1 against dense is a FLOP claim. Against the PEER baseline, wall-clock is comparable by construction (same
  gathers). A GPU kernel microbenchmark is flagged for when a GPU is available.

**The tax is a function of description pressure λ.** The deliverable is the Pareto curve of (tax, frontier) over λ:
"at ≤ 10% tax, how much better is the frontier than the control?"

### Level 1: hygiene
Necessary, **not sufficient**. A logged dense model passes these too.

| | Requirement |
|---|---|
| H1 | **Bit-exact replay.** An independent executor reproduces the log bit-exactly. Canonical order, hash-chained log, **exhaustive** re-check of every logged selection. Retrieval and product-key search are deterministic algorithms that are part of the spec, so there is no approximate-nearest-neighbour slack. |
| H2 | **The ledger balances exactly.** Every residual vector equals the sum of its logged writes, in fixed accumulation order. Every pre-activation score equals the sum of its per-write terms. Grid rounding is recorded in a rounding ledger. |
| H3 | **Every logged scalar is on its declared grid at inference.** That covers unit activations, attention weights, norm gains and retrieval scores. The dense residual is *derived*, not logged. Given the log, every residual vector and every score is a fixed linear combination of logged scalars times weight vectors, with **no further nonlinearity**, and that is what makes attribution exact. (Saying it "carries nothing beyond the input and the log" would be vacuous: that is true of any deterministic model.) |
| H4 | **Interpreter correctness.** At full description budget the independent 𝓛 interpreter reproduces the model within the description-weight quantization error. This is a check on the *interpreter*, not on observability. |

**Not claimed:** the log's cardinality as a meaningful bound. At realistic sizes the log has as many bits as a
dense residual. The guarantee is **provenance**: everything is attributable to named decisions. How *short* the
attribution is, is measured by C2–C4.

### Level 2: discriminating
A dense model with logging must **fail** these, and every test runs on that control.

**The description language 𝓛 (fixed before training, priced in bits).** A description is a sparse weighted graph:
- **Nodes:**
  - token embeddings and position bins;
  - units, per layer;
  - attention heads, per layer;
  - output logits.
- **Unit description:**
  - an activation function from a fixed menu, with a b-bit threshold;
  - in-edges `(source node [, via head h], b-bit weight)`.
- **Head description:**
  - score rule: `(query node, key node, weight)` pairs, plus a piecewise-constant kernel on log-distance bins;
  - selection rule: top-k or entmax-α.
- **Output:** logit edges.
- **Pricing.**
  - Each edge costs log₂(#possible sources) + b bits.
  - Tables are priced per entry, so a 256-entry table is not cheaper than "understanding".
  - Local DL is **unamortized**, counted to full grounding.
  - A dropped edge is replaced by a per-node constant, fit on a calibration split and counted in DL.
- **Interpreter.** An independent implementation executes a description *from the input tokens*. It never reads the
  model's state.
- **Held-out language 𝓛′.** A second language with different primitives (conjunctive ≤ 3-condition rules with
  lookup tables) and a different code. It is used only at evaluation, to detect "teaching to 𝓛".

| # | Criterion | Test (thresholds provisional; calibrate on the first real-text run) |
|---|---|---|
| C2 | **Faithful** (sufficiency *and* necessity) | Specify interventions in description terms: random single, **pairwise**, and off-distribution. The effect predicted by the interpreter must match the model's: ≥ 95% agreement in top-1, and KL within tolerance. Necessity is tested directly: remove a node the description says matters, and the predicted degradation must occur. |
| C3 | **Complete at the reader's budget** | *Residual information* R(B): the output-relevant bits the description at budget B does not expose. Estimated as the cross-entropy gap between a decoder trained on the **full log** and one trained on the **description-visible state at budget B**. Target R(page budget) ≤ 0.05 nats per token. Plus the **canary protocol** (§11). |
| C4 | **Legible** | **Explanation frontier.** Fidelity is KL (and TV) between the model's and the interpreter's full output distributions. Margin-only is out. **Global:** fidelity vs total description bits. **Local:** fidelity of one decision vs bits read. Which edges to keep is chosen by the interpreter's own KL reduction on a calibration split; for local explanations, by patching from matched contexts. Selection-order variance is reported. Report the **median and worst decile** by token type, plus cancellation ratio Σ\|c\|/\|Σc\| and tail mass. Targets: the global frontier dominates the control by ≥ 10× fewer bits at equal KL, **in both 𝓛 and 𝓛′**. The **control gets the same description-pressure fine-tuning**, so A must beat a dense model that was also taught to 𝓛. The local median is ≤ 2 kbits (about a page) at ≤ 0.1 nats KL. **Required report:** an independent natural-language simulator, never used in training, predicts node activity from labels alone. DL is not the same as comprehension, so this is measured too. |
| C5 | **Checkable** | H1–H4, plus a written trust set. A checker trusts the weights, the interpreter, the test data, floating-point determinism and log integrity. It does **not** trust the model's explanations or labels. |
| C6 | **Predictive** | Third-party random edits to **any** node, edge or entry. The change predicted from the descriptions holds ≥ 90% of the time. Failures among pruned units are reported too. |
| C7 | *Epistemic (parked)* | Hooks: content-typed units, provenance-tagged entries. |

**What "Goodharting the frontier" buys, honestly.** Optimizing fidelity-per-bit on the training distribution
produces real compression, which is what we want. It does **not** certify:
- *necessity* (C2 tests that);
- *off-distribution behavior* (shifted splits);
- *independence from 𝓛* (𝓛′ tests that).
Those are measured, not assumed.

---

## 2. First principles (unchanged; condensed)

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

The redesign keeps the first list and fixes the second via (i)–(iii) in §0.

## 3. The tension and its resolutions (v2.1)

| Performance needs | Observability fears | Resolution |
|---|---|---|
| N ≫ d features per position (superposition) | unnamed, entangled features | **Ledger.** Keep the dense residual: superposition is allowed. Every *write* is named by its writer (unit, head, embedding) and logged. A unit's reading of the residual splits exactly over writes: `z_u = Σ_writes value · ⟨key_u, v_write⟩`. "Features" are writes, identified by provenance, never decoded, so they are unambiguous. |
| nonlinearity | mixing | **Every nonlinearity is a logged decision on a declared grid.** Linear algebra in between is exact accounting. |
| graded magnitudes | smuggled bits | **Quantize the decisions, not the residual.** 8-bit logged activations and weights with per-unit scales, trained with QAT. The residual is accumulated exactly in a fixed order. Small updates are small activations, not sub-LSB residual increments. |
| millions of contributions | unreadable | **Explanation frontier + description pressure.** Short descriptions are trained for and measured, with KL fidelity, worst-decile reporting, and a held-out language. |

---

## 4. Requirement tree (v2.1)

```
G  Performant AND observable LM
├─ R1 Performant (C1)
│  ├─ capacity, superposition kept .......... dense residual (ledger) — no bottleneck by construction
│  ├─ many units, few active ................ BB2 PEER-style pools (product keys, O(√N_u)), AuxK-style revival,
│  │                                            aux-loss-free load balancing
│  ├─ attention expressiveness .............. BB4 dense low-rank OV; per-head top-k or entmax; weights logged
│  ├─ precision ............................. BB7 8-bit logged decisions, per-unit scales, QAT
│  ├─ knowledge capacity .................... visible parametric units + BB12 external entries with provenance
│  └─ structured sub-domains ................ BB6 typed operators (numbers, dates, counts)
├─ R2–R4 Faithful, Complete, Legible
│  ├─ provenance of every write ............. BB1 ledger + BB9 decision log + H1–H3
│  ├─ short descriptions exist .............. BB10 description pressure + clonal selection (order-robust credit)
│  ├─ short descriptions measured ........... BB17 𝓛 + independent interpreter; 𝓛′ held-out language
│  ├─ hidden information measured ........... C3 residual-information decoders; BB18 canary protocol
│  ├─ the bar to beat ....................... BB19 dense + post-hoc, *with the same pressure fine-tuning*
│  └─ content vs style ...................... BB15 typed units + architectural style budget + non-interference
├─ R5 Checkable ............................. H1–H4, trust set
├─ R6 Predictive ............................ third-party random edits
└─ R7 Epistemic (parked) .................... content-typed units, provenance entries
```

---

## 5. Lessons from our experiments

The v1/v2 table stands (L1–L11). The ones that shaped v2.1 most:
- **L2:** binarizing is what costs; multi-bit values are nearly free.
- **L3:** undeclared magnitudes carried 2.3 nats.
- **L5:** lookup tables, and dense models, can't generalize arithmetic to unseen arguments.
- **L7/L8:** stories are cheap and often wrong; test against an independent interpreter.
- **L11:** post-hoc ≤ 4-bit quantization was destructive; 8-bit was untested.

---

## 6. Building blocks (v2.1)

Parallels and prior art for unchanged blocks are in the v1 table (git `e45149f`).

| BB | Block | v2.1 specification | Status |
|---|---|---|---|
| 1 | **Ledger residual** *(replaces the explicit re-encoded bus)* | Dense residual = the exact sum of logged writes, in fixed order. Each write = (writer id, logged scalar on a grid, writer's fixed output vector), or a relayed write via (head, source position, logged weight). The ledger is not materialized. It is *exactly derivable* from the log plus weights, on demand. | new (combination) |
| 2 | **Unit pools** | PEER-style single-neuron or small units. Query linear in the residual, so the product-key score `⟨q, [k1_i; k2_j]⟩` is exactly decomposable over writes. Product-key top-k is **exact** for this factorized score, so it is a *defined* selection, not an approximate prefilter. Activations are 8-bit and logged. Near-misses (the next m candidates) are logged. | exists (PEER) + logging |
| 4 | **Attention** | Dense low-rank QK/OV. Scores split exactly over (query write, key write) pairs. Per-head top-k (copy/induction) or entmax (averaging). Weights 8-bit and logged; dropped mass logged. | exists + logging |
| 7 | **Declared precision** | Hard 8-bit grids with per-unit scales for every logged scalar, trained with QAT. 4-bit only as an experiment. | exists (QAT) |
| 10 | **Description pressure + clonal selection** | Loss term: L1/entropy on the per-write contribution distribution of each *active* score. This is computed only for active pairs, so it is cheap, and it pushes the interaction graph to be sparse. Clonal selection: units with low fidelity-per-bit are recycled. Credit is averaged over random removal orders, so redundant-but-causal units aren't killed by order effects. | partial |
| 15 | **Content/style typing** | Units typed *content* or *style*. **Architectural style budget:** at most L0_style style-unit firings and B_style logged bits per sentence, enforced by a capped top-k and logged. Non-interference test: vary style units and hidden context with content units fixed; the content of the output, as parsed by an independent parser, must not change. Leakage through style is bounded by B_style. | new |
| 12 | **External entries** | Units with provenance (text, source, date), stored outside the parameters. Retrieved by the same deterministic product-key selection. Frozen, versioned keys. | partial |
| 17 | **𝓛, 𝓛′, interpreter, frontier** | As §1. An independent implementation. | new |
| 18 | **Canary protocol** | §11 | new |
| 19 | **Control** | Dense model → per-layer transcoders → 𝓛 description, **plus** the same description-pressure fine-tuning applied to the dense model | exists (tools) |
| 20 | **Blinded entities** (optional) | Per-context entity id randomization in training, so entity facts can only come from entries | new |

---

## 7. Design A v2.1: the Ledger Transformer (core)

**Principle:** *keep the transformer's dense superposed residual. Make every write into it a logged, named,
grid-valued decision. Train the resulting interaction graph to be short. Prove it with an independent interpreter
and a planted smuggler.*

### 7.1 Anatomy (per layer)
1. **Norm:** RMSNorm gain `s_t`, one logged 8-bit scalar.
2. **Attention:** per head, scores (dense, exactly decomposable over write pairs), then top-k or entmax, then
   logged 8-bit weights, then the OV write. Each relayed write is named (head, source position, the source's
   writes).
3. **Unit pool:** product-key query, then exact top-k_u (logged ids), then 8-bit activations (logged), then writes
   `g_u · v_u`.
4. **Residual:** the exact sum of all writes, in a fixed order.
5. **Readout:** logits = unembedding of the residual, decomposable over writes.

### 7.2 Why it should perform (C1)
- **Architecturally it is a PEER/MoE transformer.** There is no information bottleneck: the residual is dense. The
  bus problems from v1/v2 (starvation, re-encoding cost, L0 ambiguity, sub-LSB loss) **do not arise**. Wide
  product-key pools have reportedly matched or beaten dense FFNs per FLOP *(verify)*.
- **The additions are cheap:**
  - logging: negligible;
  - 8-bit activations with QAT: near-free in LLMs, *(verify)* W8A8-type results;
  - sparse attention: measured by D3.
- **The real tax is description pressure λ.** It is tunable, and the deliverable is the (tax, frontier) curve.
- **Wall-clock:** the same gathers as the PEER baseline, so comparable to it. Against dense, the C1 claim is a FLOP
  claim until kernels exist.

### 7.3 How A meets each criterion

| Criterion | Mechanism | Status |
|---|---|---|
| C1 | PEER/MoE-class architecture + cheap logging + tunable λ | **Open, empirical**, but the risk is now concentrated in one knob (λ), not in the architecture |
| H1–H4 | Ledger, decision log, grids, exhaustive re-checks, hash chain, interpreter check | **Guaranteed** (hygiene) |
| C2 | Description-level interventions judged by the interpreter, sufficiency + necessity | **Measured** |
| C3 | Residual-information decoders at reader budget + canary protocol | **Measured** |
| C4 | λ pressure + clonal selection. KL frontier in 𝓛 and 𝓛′ against a control that also gets pressure. NL-simulator report. | **Open, empirical**: the main scientific question |
| C5 | H1–H4 + trust set | **Guaranteed for consistency with the weights**; names are C4's job |
| C6 | Third-party random edits | **Measured** |
| C7 | Content-typed units; provenance entries | Hook |

### 7.4 Riskiest assumption and kill
- **Assumption:** there is a λ at which the tax is ≤ 10% vs PEER/MoE *and* the frontier beats the
  pressure-fine-tuned control by ≥ 10× in both 𝓛 and 𝓛′.
- **Kill:** at every λ with tax ≤ 30%, the frontier advantage over the control is < 3× in either language. That
  would mean the architecture adds nothing over "dense + pressure + post-hoc".

---

## 8. Library (B, v2.1): entries with provenance

Parametric knowledge sits in visible units and is allowed. External entries add four things:
- provenance;
- editing without retraining;
- freshness;
- capacity.

| Requirement | How it's met |
|---|---|
| Citations must be **causal** | Plausible, type-matched edits including provenance fields; an edit-detection probe at chance; a distractor test |
| Hidden memory measured | Relearning test (how fast withheld facts return under brief fine-tuning) |
| No hidden retrieval channel | Deterministic exact retrieval |
| Clean performance claim | Gains on a 13-gram-deduplicated test set, split by token type |
| Guaranteed provenance (optional) | Blinded entities (BB20) |

**Kill:** reliance < 90% under plausible edits, or edit detection well above chance.

---

## 9. Content and style (C, v2.1)

Inside A, units are typed content or style.
- **The style budget is architectural**: L0_style, B_style per sentence. It is not just a tag.
- The **non-interference test** uses an independent parser and paraphrase-invariance checks.
- Content-unit firings are themselves logged and described. "Which facts to mention" is a visible content decision,
  not a covert channel.
- A separate dense voice remains a fallback only: two-directional parse equality, unparseable-residue rate, an
  independent parser. D4 decides whether it is ever worth it.

---

## 10. The composite: one model, three views

One Ledger Transformer:
- **B** is its external, provenance-tagged entries;
- **C** is its content/style typing with an architectural style budget.

Example trace:

> **"Paris"** (KL fidelity of a 2-kbit local description: 0.04 nats; worst-decile fidelity for entity tokens
> reported separately).
> - Main path: unit L14#88 412 "answer-capital-question" fired at 0.82.
> - Its score: 84% from writes by {L9#2 001 question-about-capital, L6#77 310 entity=France (relayed by head
>   L8.3 from position 4)}.
> - It read entry #4 411 "Paris is the capital of France" (encyclopedia, 2025-03).
> - Edit the entry to "Lyon" and the output follows, as predicted by the interpreter ✓.
> - Style units varied: content unchanged ✓.

**D, "Open Notebook"** (parked, C7): an explicit belief state plus an independent evaluator, built from
content-typed units and provenance entries.

---

## 11. Test plan (v2.1)

**Compute reality.**
- 4 CPU cores. HuggingFace is blocked; GitHub raw and PyPI are reachable.
- Corpus: public-domain books mirrored on GitHub, a CPython-stdlib code slice, tiny-shakespeare as a sanity
  baseline. Target 20–50M tokens.
- Ratio-matched sizes, ≥ 3 seeds, IsoFLOP fits. Never read a single scale as a scaling claim. Real scaling needs a
  GPU (flagged).

| # | Test | Question | Setup | Prediction (before running) | Redirect / kill |
|---|---|---|---|---|---|
| **D0** | **Control frontier** | The bar A must beat | Small dense LM (d≈256, 4 and 8 layers). Per-layer transcoders, converted to 𝓛 and 𝓛′. Also the same model after description-pressure fine-tuning. | Gives the control curve | — |
| **D2** | **8-bit decisions through depth** | Precision tax, cumulative | PEER-ized small LM: activations, attention weights and gains on 8-bit grids with QAT, at 4/8/12 layers | ≤ 2% tax at b=8; grows slowly with depth | > 5% at b=8 → rethink scales |
| **D3** | **Sparse attention** | Attention tax | Dense LM, post-hoc then fine-tuned top-k/entmax, k ∈ {4…64}, context ≥ 512 | Copy/induction heads fine at small k; averaging heads need entmax | Informs the per-head policy |
| **T-A0** | **Ladder, risk-first** | Where the tax lives, and what λ buys | dense → +PEER pool (dense residual) → +8-bit logged decisions → +sparse attention → **+λ sweep** {0, λ₁, λ₂, λ₃} + clonal selection. 3 seeds, 2 depths, 2–3 sizes. Frontier (𝓛 and 𝓛′), R(B) and stability at every step. | Steps before λ: ≤ 10% combined vs PEER. The λ sweep gives a curve where at ≤ 10% tax the frontier is ≥ 10× better than D0's pressured control. | Kill as §7.4 |
| **Canary** | **Test the tests** | Do the audits catch smugglers? | See protocol below | Naive smugglers are caught at ≥ 16 bits/token. Audit-aware smugglers set the real detection floor, which we report. | Floor > 16 bits/token → fix the audit before any C3 claim |
| T-A2 | Gating toy | AND-gating vs linear threshold | m=4 AND task + switching testbed | Log-domain product ≥ linear on cross-firing | Drop BB3 if not |
| T-A3 | Typed operators | Equivariance (L5) | Switching testbed + numeric tokens in the corpus | Held-out arithmetic ~0 → ≥ 0.9 | < 0.5 → inspect selection |
| T-B | Library | Citation causality; hidden memory | bioS-style synthetic biographies *(verify)*, plausible edits, edit-detection probe, relearning test | Reliance ≥ 90%, probe at chance | §8 kill |
| D4 | Oracle plan | Is the dense-voice fallback ever worth it? | Content-lemma plans; prose and code slices | Prose ≈ tie; code loses | Planner+voice > token-level + 10% → drop the fallback |

### Canary protocol (BB18)
1. **Channel catalogue, one smuggler per channel:**
   - unit-id combinations;
   - low bits of 8-bit grids;
   - near-miss and tie choices;
   - attention weights and selection;
   - write-vector components outside the described edges (the "unexplained tail");
   - retrieval choice;
   - style units (capacity should be ≤ B_style).
2. **Capacity ladder.** Random payloads independent of the data, k ∈ {1, 4, 16, 64, 256} bits/token.
3. **Audit-aware smugglers.** Trained to minimize KL loss, frontier degradation and R(B) while still transmitting.
   A naive smuggler proves little.
4. **Stated detector.** The R(B) decoder gap: full-log decoder vs description-visible decoder. Plus a per-channel
   decoder. The false-positive rate is fixed at 1% on a clean null model.
5. **Payload recovery + host cost.** The smuggler must actually recover its payload. Report the host's NLL cost.
   Result = the **minimum undetectable payload per channel** at a fixed false-positive rate. That is the honest
   C3 number.

**Shared metrics for every test:**
- tax vs dense and vs PEER/MoE;
- per-token-type loss;
- long-context and induction probes;
- dead fraction, utilization entropy, loss-spike rate;
- KL frontier, global and local, in 𝓛 and 𝓛′, median and worst decile;
- R(B);
- cancellation ratio and tail mass;
- NL-simulator score;
- replay exactness.

---

## 12. Refinement history

### v1 → v2 (round 1)
16 changes; see git `742af31` §12. In brief:
- hygiene vs discriminating criteria, and the explanation frontier;
- a persistent per-feature-threshold bus;
- hard quantization of everything logged;
- margin/gross/tail reporting;
- automatic fidelity-per-bit fitness;
- one source of truth for selection;
- B and C folded into A;
- MoE baseline and an ablation ladder;
- the canary.

### v2 → v2.1 (round 2)

| # | Finding (critic) | Change | Why |
|---|---|---|---|
| 1 | Re-encoding every layer is a sparse-coding inference problem. It is expensive (≈ 40× a dense layer for a full encoder) and lossy at k/d = 1/4, N/d = 250, and D1 didn't test the deployable encoder (performance) | **Removed re-encoding.** The residual is dense; every write is a logged decision (**ledger**). | The ledger is exact by provenance, so it never needs decoding. This deletes the costliest and riskiest block. |
| 2 | Dense writes + re-encoding give gauge-dependent, unidentifiable names: many lists fit one vector (observability) | Writes are named by their **writer**, not by decoding | Provenance is unique, and decoding never is (outside the compressed-sensing regime). |
| 3 | Persistent-bus L0 was ambiguous, with erase-starvation (performance) | Gone with the bus. The ledger count = logged firings + relays, explicit per layer. | — |
| 4 | Sub-LSB updates vanish on an 8-bit grid (performance) | Quantize **decisions**, not the residual. Exact fixed-order accumulation. | Small updates are small activations, which the grid represents. |
| 5 | The cardinality bound is ≈ 29k bits per position, so it isn't meaningful (observability) | Cardinality no longer claimed. The guarantee is **provenance**; shortness is measured. | Say what's actually guaranteed. |
| 6 | DSR at full budget is tautological, because 𝓛 can express the weights (observability) | Full-budget DSR demoted to H4 (interpreter check). **C3 redefined** as residual information R(B) at the reader's budget, via decoders. | Hidden information is what the *reader* can't see, not what the log lacks. |
| 7 | Frontier ambiguity: interpreter vs truncate-and-mean-ablate (observability) | Fidelity is **interpreter-only**, from input tokens. Dropped edges become per-node constants counted in DL. | Mean-ablating the model rewards redundancy, not simplicity. |
| 8 | Attribution for edge selection unspecified (observability) | Selection by the interpreter's own KL reduction on a calibration split, or by matched-context patching locally. Order variance reported. | Gradient × activation misranks saturating units. |
| 9 | Margin over the runner-up ignores the distribution; "≥ 80% of tokens" hides a fifth (observability) | KL/TV on full distributions; median **and worst decile** by token type | — |
| 10 | Teaching to 𝓛: the control isn't trained against 𝓛 (observability) | The **control gets the same pressure fine-tuning**. Held-out language 𝓛′. Tables priced per entry. Local DL unamortized. | Otherwise "10× fewer bits" partly measures overfitting to the language. |
| 11 | DL ≠ comprehension (observability) | NL-simulator report **required** | — |
| 12 | "Goodharting fidelity-per-bit is success" is only partly true (observability) | Claim narrowed. Necessity, off-distribution behavior and 𝓛-independence are measured separately. Clonal credit averaged over removal orders. | Sufficiency on one distribution isn't mechanism. |
| 13 | Moving C inside A lost the architectural style cap (observability) | **Architectural style budget** (L0_style, B_style) enforced and logged | Tags don't bound channels; capacity limits do. |
| 14 | Product-key prefilter vs named selection inconsistent (observability) | Product-key top-k is *exact* for the factorized score, and the query is linear in the residual. Penalties apply only to active pairs. | One defined selection rule, no prefilter. |
| 15 | Canary underspecified (observability) | Full protocol: channel catalogue, capacity ladder, audit-aware smugglers, stated detector, fixed false-positive rate, payload recovery, host cost | An audit is only as good as the strongest smuggler it has caught. |
| 16 | D1's kill threshold was vacuous; the ladder wasn't risk-first; depth was untested; stability was missing (performance) | D1 dropped, since there's no bus left to bound. Ladder ordered by risk, with λ as the final, decisive knob. Depth 4/8/12 in D2. Stability metrics from step 1. | — |
| 17 | Gather cost is inherent; there's no GPU benchmark (performance) | Stated plainly: comparable to the PEER baseline, a FLOP claim vs dense. GPU microbenchmark flagged. | No GPU in this environment. |
| 18 | Approximate-NN slack at library scale (observability) | Retrieval is a deterministic algorithm in the spec, re-run by the checker | — |

**Not adopted:**
- **Sparse-write units with s ≈ 8–16 named features each** (performance round 2, fix (a)). The ledger goes further:
  writes are already named by their writer, with no feature-assignment structure to learn.
- **A coherence limit on co-active dictionary features** (observability round 2). There is no dictionary decoding
  left for it to protect.

## 13. What remains open (honest list)

Every criterion has a mechanism and a falsifiable test, and two critique rounds' findings are resolved on paper.
What paper can't settle:
1. **The λ curve.** Does a λ exist with ≤ 10% tax and ≥ 10× frontier advantage over a *pressured* dense control,
   in both 𝓛 and 𝓛′? That is the single scientific question the whole design now reduces to. T-A0 answers it at
   small scale.
2. **The canary floor.** How many bits/token can an audit-aware smuggler hide? Unknown until measured.
3. **Scale transfer and wall-clock.** These need a GPU.
4. **C7** is parked.
