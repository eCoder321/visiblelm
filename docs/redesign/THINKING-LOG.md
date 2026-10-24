# Thinking log — redesign for "performant AND observable"

A chronological record of the reasoning behind `DESIGNS.md`. It includes the parallels explored, the approaches
rejected (with reasons) and the points where the direction changed. Entries are appended, never rewritten.

---

## 2026-10-24 — Session 1: from the goal backwards

### 1. Restating the goal so it can be tested
The user wants *both* current-model performance *and* visibility into the inner workings. "Visibility" was split
into two kinds:
- **mechanism visibility**: what was computed and why (our project);
- **epistemic visibility**: what the model believes and on what evidence (Graepel's article).

The second is parked, but no design may rule it out. Turned the goal into seven criteria, C1–C7, each with a number
attached (`DESIGNS.md` §1). The test for each criterion: could a skeptic run it and get a pass/fail? If not, it
isn't a criterion yet.

### 2. Why are LLMs good, and why are they dark? (first principles, not literature)
Listed the causes of performance: capacity per FLOP via superposition, benign optimization, generic inductive bias,
graded representations, and scale. Listed the causes of opacity: unnamed superposition, everything active, no
completeness guarantee, unbounded magnitudes, too many operations.

**The turn that mattered.** I asked, "is a dense transformer *decomposable*?" Yes. Between nonlinearities it is
linear, so any logit splits exactly into per-coordinate terms. That is what attribution graphs use. So
decomposability isn't what's missing. What's missing is:
- (i) a *named* basis;
- (ii) *few* terms;
- (iii) a *guarantee* that the named terms are everything.

This reframing cut the design space. We don't need to make every operation sparse or discrete, which was the
`rulenet` approach and costly (L2). We need a named basis, few active terms, and logged nonlinearities. Dense fan-in
is acceptable as long as its decomposition is exact and concentrated.

### 3. Is the tension fundamental? Four resolutions
- **Superposition vs naming.** Superposition is a *compression*, and compression is fine if decoding is exact. The
  cleanest version carries the sparse ids explicitly, so the dense vector is just an encoding. Compressed-sensing
  math gives the dimension needed if layers read the dense form: d ≳ C·k·log(N/k).
- **Nonlinearity vs accountability.** All information mixing happens at nonlinearities, so put the observability
  burden there. Every nonlinearity becomes a logged choice (top-k) or a logged scalar computed from
  exactly-decomposed inputs. Slogan: *all nonlinearity is a logged decision; everything else is exact accounting.*
- **Magnitudes vs hidden channels.** We measured 2.3 nats hiding in magnitudes (E1). But quantizing to binary cost
  0.29–0.5 nats (T1/T2). Resolution: neither "free" nor "binary", but **declared** precision. Information theory
  gives a clean enforcement tool: additive noise of known variance bounds a channel's capacity, so training with
  noise injection makes "b bits per value" a property, not a hope. The hidden channel becomes a declared one.
- **Scale of reading.** No human reads 10^11 operations. But auditors don't read every transaction either. They work
  to a *materiality threshold* and the books must balance. Hence hierarchical explanations with an exact remainder.

### 4. Parallels explored, and what each contributed

**Biology**
- **Cortical sparse coding** (V1; Olshausen & Field). Brains reach high performance with ~1–5% of neurons active
  over an overcomplete dictionary, so sparse codes are not a performance handicap in the one general intelligence
  we know of. → BB1.
- **Insect mushroom body.** About 50 projection neurons expand to about 2,000 Kenyon cells, with roughly 5–10%
  active through winner-take-all; a learned readout follows. This is a biological product-key / LSH memory ("fly
  hashing"). → BB2.
- **Dendritic coincidence detection.** A dendritic branch fires an NMDA spike only when several inputs co-occur, so
  each branch acts as an AND and the soma as an OR over branches. This mapped directly onto our E1c bug: rules
  fired because a *weighted sum* of unrelated conditions crossed threshold, which is an accidental OR. → BB3.
- **Spikes.** Neurons communicate with all-or-none events (logged decisions) and carry analog quantities in rates
  or timing, whose precision is limited by noise. That is exactly "discrete choices + declared-precision values".
  → BB5, BB7.
- **Grid cells / path integration.** Position is coded as phases, and movement shifts the phase: an equivariant
  operator. Combined with the E2redux finding below, this suggested typed operators. → BB6.
- **Immune clonal selection.** Generate diverse receptors and expand the ones that bind. Delete the self-reactive
  ones (negative selection), and refine the survivors (affinity maturation). Our prune+regrow optimizes usefulness
  only. Adding *legibility* to the fitness function is the immune-system move. → BB10.
- **Complementary learning systems.** The hippocampus is a fast episodic store and the cortex a slow structured
  learner. → Design B's library + clerk.
- **Levelt's speech production.** Conceptualizer (preverbal message), then formulator, then articulator. Humans
  already separate "what" from "how". → Design C.
- **Neuromodulators.** Global low-bandwidth gain signals. → BB8 (one logged gain scalar).
- **Gene regulation** (considered, not adopted as a block). Discrete promoter logic (AND/OR of transcription
  factors) controls continuous expression levels. It reinforces BB3 + BB7 but adds nothing new.

**Industry and other fields**
- **Double-entry bookkeeping.** Every amount is recorded twice and the books must balance. An imbalance *is* the
  detector of a hidden flow. → BB11 books-balance audit: Σ contributions = logit, exactly, at every layer.
- **Auditing.** Sampling plus materiality thresholds, so you verify without redoing the business. → C5 spot checks;
  hierarchical summaries.
- **Aviation flight recorders + deterministic replay debuggers** (rr, FoundationDB-style simulation). Record cheaply
  always and investigate later. → BB9 decision log.
- **Chip design-for-test** (scan chains). Hardware is built observable on purpose at a known area tax (~5–10%), and
  that is industry standard. It is evidence that "observability by construction at a small tax" is a normal
  engineering trade, not a fantasy. → framing for C1 targets.
- **Compilers.** Source maps; translation validation (check each compilation instead of proving the compiler).
  → Design C's round-trip parser.
- **Law.** Precedent and citation; reasons-giving opinions; appellate review. → precedent entries (BB13), cited
  library entries (BB12).
- **OS security information-flow control / noninterference.** The output must not depend on secret inputs except
  through declared channels, and this can be tested by varying the secret and checking the output. → BB15: the
  exact formal shape of "no hidden channel" for Design C.
- **Distributed tracing** (Dapper/OpenTelemetry). Always-on spans at under 1% overhead, expanded on demand.
  → on-demand decomposition.
- **Blackboard architectures** (Hearsay-II speech understanding, 1970s). Knowledge sources read and write a shared,
  inspectable blackboard. Design A's bus *is* a blackboard, with units as knowledge sources.
- **Markets/auctions** (considered). Units "bid" for a position, and the auction log is the decision log. That is
  the same thing as product-key top-k, so it was folded into BB2.
- **Music: score vs performer.** The score fixes content and the performer adds bounded expression. → Design C's
  style code.

**Information theory**
- Compressed sensing/RIP bounds the dimension. Gaussian channel capacity bounds the bits in magnitudes. The data
  processing inequality frames completeness: if output ⫫ input | trace, the trace is complete, and the completeness
  gap estimates that conditional information. MDL frames legibility: short descriptions that predict.

### 5. Approaches considered and rejected (with reasons)
- **Post-hoc interpretability of a dense model** (SAEs, attribution graphs, probes). Error terms remain, so
  completeness can't be guaranteed. The tools also chase a model not built to be explained. *Kept as the null
  baseline* (we already have `dense_sae`).
- **Fully discrete program learning** (our rulenet; "Learning Transformer Programs"). We paid the tax ourselves
  (L2), and published discrete-program learners also underperform on harder tasks *(verify)*. Rejected as the
  substrate. Its *virtue* (an executable rulebook) is kept via the lockstep executor.
- **Read the chain of thought.** CoT is often post-hoc and unfaithful (Turpin 2023, Lanham 2023 *(verify)*; this is
  Graepel's point). It is not mechanism visibility.
- **Concept-bottleneck models with human-chosen concepts.** Soft concept values leak information (the known
  "leakage" problem *(verify)*), and no fixed concept list covers language. Partially reborn as Design C, with
  learned typed features and a *tested* leak bound.
- **Kolmogorov–Arnold networks.** Legible 1-D functions, but no evidence at LM scale, and the combinatorics stay
  dense.
- **A dense model plus a co-trained sparse "shadow" replacement** (train both and penalize divergence). Tempting,
  because it keeps dense performance. But the output still comes from the dense model, so the explanation describes
  a different model unless divergence → 0, and at that point it *is* Design A. Rejected as a design; kept in mind as
  a cheap way to estimate a lower bound on A's tax.
- **Just make the model small enough to read.** Fails C1 by definition.
- **Neural cellular automata, Hopfield/energy-based attractors.** Interesting readable dynamics, but no performance
  evidence for language. The associative-memory idea survives inside Design B's retrieval.
- **Distilling to trees or rules after training.** Lossy; same flaw as post-hoc methods.

### 6. Re-reading our own data: E2redux changes a key assumption
Checked the unfinished E2redux runs (2 of 3 seeds) against the dense baseline before writing the designs.
- **Removing the identity regularizer did not fix arithmetic.** iid NLL was 0.785 vs 0.778 (seed 0) and 0.970 vs
  0.830 (seed 1). The rulebook got 8× bigger, because the tables were now free to move and spent bits.
- **Held-out arithmetic accuracy (succ/pred/pls2 on an unseen argument) is ~0% for every model, dense
  included.** I had been treating "arithmetic not learned" as a defect of the observable design. It isn't. It's
  a property of *lookup representations*: an unseen (operator, value) cell never gets a gradient. Dense models fail
  identically.
- **Consequence.** Only a representation where the operator is *shared across values* (a shift on a cyclic code)
  can generalize. That operator is also the most legible thing imaginable ("+1"). This is the clearest case so far
  of **legibility and performance pointing the same way**: the readable structure *is* the right inductive bias.
  Grokking studies found networks discovering exactly this rotation structure for modular arithmetic, and LLMs
  reportedly use helix codes for numbers *(verify)*. → Test T-A3 goes first.
- **Side finding.** The structured model already generalizes better than dense out of distribution (shift NLL 1.41
  vs 1.84, held-out cpy3 accuracy 0.96 vs 0.50). The iid tax comes from specific mechanisms, not from observability
  per se.
- **Side finding.** Dense iid NLL (0.593) is *below* the "oracle" (0.617) on the same split. The oracle ignores
  that training-distribution sequences are rejection-sampled to exclude held-out events, which makes some
  continuations impossible. The floor is a little loose. Logged as L10.

### 7. Forming the designs
Three capacity strategies fell out of the requirement tree (R1.1):
- named wide-sparse pools;
- an external library;
- a bounded fluency channel.

Each became a design whose *other* properties are built around that choice:
- **A, Glass Box:** everything in one model; the substrate.
- **B, Library & Clerk:** knowledge outside the weights; addresses Graepel's "entanglement" gap; best C7 hook.
- **C, Planner & Voice:** content decided in the open; wording bounded and tested for non-interference.

They compose, so the composite ("Layered Glass Box") is the target system. Each design is also testable alone.
**D, Open Notebook** (an explicit belief state) was parked as C7's eventual home.

### 8. Ordering the tests
Initially planned T-A3 → T-A2 → T-A1 → B → C → T-A4. Then I asked: *which single assumption, if false, kills the
most?* The answer is A's "an explicit k-sparse named residual is cheap on real language". The synthetic testbed
cannot answer that because it is too simple to need capacity. So I added **T-A0** (a bus-lite version on a tiny real
corpus) as the second test. It is cheap and tests the riskiest assumption before we build the rest of A on it.
T-A3 stays first because it is cheaper still and fixes the one mechanism we *know* is broken.

### 9. Next
Two independent critics review v1:
- a performance skeptic: will these match transformers?
- an observability skeptic: find hidden channels, unfaithful explanations and illegibility.

Each finding either changes the design (logged here and in the `DESIGNS.md` refinement history) or gets a stated
reason why not.

---

## 2026-10-24 — Critique round 1 and the v2 turn

Two independent critics reviewed v1 cold. One attacked performance (C1) and one attacked observability (C2–C6).
Full change table: `DESIGNS.md` §12.

### What each found that I hadn't
- **Observability critic.** My "by construction" guarantees (faithful, complete, checkable) are passed by a
  *dense* transformer that simply logs its residual stream. The books balance and replay is exact. So these tests
  can't tell an observable model from an opaque one. Second, noise injected during training doesn't bound
  information at *inference*, and the attention weights, gates and norm gain were still full-precision floats.
- **Performance critic.** A global top-200 bus, cut back every layer, is a depth-wise bottleneck. Post-hoc sparse
  autoencoders at k=32 on GPT-4 cost about "10× less compute" worth of loss *(verify)*. Also: the right baseline is
  MoE, not dense.

### The turn
The two findings point the same way. v1 tried to make the *raw computation* small so it would be readable. That:
- kills capacity (performance critic);
- doesn't even buy observability, since raw logs are trivially "complete" (observability critic).

First-principles question: *what is an explanation?* It is a **description shorter than the thing it describes
that still reproduces it**. A full log is complete but no shorter than the model, so it explains nothing. So
observability isn't a property of the substrate. It is the **trade-off curve between description length and
fidelity**: the *explanation frontier*.

This changes the design objective:
- The substrate can be as large as performance needs (large raw k, persistent bus, general OV maps).
- It must be (a) in a named basis, (b) exactly accountable with every logged number on a declared grid, and
  (c) *trained to admit short descriptions*.
- Observability is then measured as the frontier, against a dense model described by the best post-hoc tools.
- A cipher, a polysemantic feature or a joint code is no longer a "hidden channel" we must forbid. It shows up as
  a bad frontier, which is measurable.

An older idea returns in a better form: our early `north_star` executor and "rulebook description length" were
crude versions of this frontier.

### Smaller turns
- **B (library) shrinks.** In v2, A's unit pool *is* a readable key→value memory, so knowledge in weights is no
  longer hidden. The fight to forbid parametric memory (anti-loss, and evadable by edit detection) is unnecessary.
  B keeps what only an external store gives: provenance, editing, freshness. New optional idea: a **blinded clerk**,
  with entity ids re-randomized per context, so entity facts *cannot* live in weights.
- **C (planner/voice) moves inside A.** A separate voice can at best tie a dense model, and a dense voice plus
  parser can collude on a cipher. Instead, type dictionary features as content or style and test non-interference
  inside A. A separate dense voice is a fallback, and it must have an **independent** parser.
- **Principle from the collusion finding:** a *checker* can be opaque if it is independent and calibrated, like
  test equipment. Only the system under test must be observable.
- **Test order.** One small dense LM on real text answers three cheap questions first:
  - the post-hoc bus-tax upper bound (D1);
  - the precision tax (D2);
  - the attention-sparsity tax (D3).
  Then a from-scratch ablation ladder adds one constraint at a time, so every tax is attributable. The synthetic
  typed-operator test moves later, because it says nothing about real-text performance.
- **Not adopted:** rotation-based outlier handling, because it mixes named features. Per-feature scales do the job
  instead.

### Next
Round 2: the same critics check whether v2 actually resolves their findings, and look for new fatal issues
introduced by the fixes.

---

## 2026-10-24 — Critique round 2 and the ledger turn (v2.1)

**The new problem.** Both critics independently found that v2's weak point was the same block: *re-encoding*. In
v2, after each layer the dense update was compressed back into a named sparse list.
- Performance critic: re-encoding is a sparse-coding inference problem, ~40× a dense layer for a full encoder, and
  lossy in our regime.
- Observability critic: re-encoding makes names *gauge-dependent*. Many different lists fit the same vector, so
  "which features are active" is partly arbitrary, and a cipher could hide in the choice of list.

**The first-principles question:** *why decode at all?* We decoded because we wanted the state "in named terms".
But a name can come from **provenance** instead of decoding. If every write into the residual comes from a logged
decision (unit u fired with value g; head h moved position j's content with weight w), then the residual is
*exactly* the sum of named writes, by bookkeeping, with no inference step. Many lists fit one vector, but only one
**history** produced it. The ledger is exact, unique, and costs nothing to keep, because it's derivable on demand
from the log.

**What that does to the architecture.** It becomes a wide-sparse (PEER/MoE-style) transformer whose nonlinearities
are all logged and grid-valued. That is much closer to designs known to perform, so v1/v2's bottleneck risks
disappear.

**Is this still "radically different"?** The radical content has moved from the substrate into the **objective and
the audit**:
- description pressure on the interaction graph;
- a fixed description language with an independent interpreter and a held-out second language;
- completeness defined as information the *reader* can't see at their budget;
- a planted-smuggler protocol that tests the tests.

Two critique rounds pushed us here for a principled reason. Performance constraints pull the substrate toward what
already works, so **the cheapest place to buy observability is at the nonlinearities and in the loss, not in the
residual.** The whole program reduces to one empirical curve: *tax vs frontier as the pressure λ varies*, measured
against a dense model that gets the same pressure.

**A self-correction before round 3.** I had written that the dense residual "carries nothing beyond (input, log)"
by the data-processing inequality. That is true of *any* deterministic model, so it is vacuous. The real property
is narrower: given the log, every residual and score is a **fixed linear combination of logged scalars**, with no
further nonlinearity. That is what makes attribution exact. Fixed in the doc.

**Round-2 findings not adopted, with reasons:**
- *Sparse-write units* (each unit writes 8–16 named dictionary features). Superseded: writer provenance names
  writes with no feature-assignment structure to learn.
- *A coherence limit on co-active dictionary features.* Nothing is decoded any more, so there's nothing for it to
  protect.

---

## 2026-10-24 — Critique round 3 (v2.2): nothing fatal; the stopping point on paper

Both critics independently reported **no fatal issues** in the Ledger Transformer. Their major items were specific
and fixable, and every one was adopted (`DESIGNS.md` §12, v2.1→v2.2). Three of them taught something general.

- **Hidden couplings live wherever a decision depends on *other* units' values, not on edges.** Top-k competition
  (whether u fires depends on everyone else's scores) and normalization (the gain depends on every write's
  cross-terms) are both like that. A description language built from edges can't express either. So:
  - firing is decided by each unit's own threshold, with top-K kept only as a logged, described compute cap;
  - the norm becomes a described node.
  Lesson: **check every operation for "depends on the whole state" couplings**, not just for nonlinearity.
- **Any learned decoder is the wrong judge of hidden information.** A weak one sees nothing. A strong one learns
  the model. Neither can read a keyed code. The only complete decoder of what affects behavior is *the model
  itself*. So completeness becomes causal: resample everything the reader can't see, from matched contexts, and
  measure how much the output moves. Done per sequence, because the realistic smuggler is slow and cumulative
  (0.1 bit/token in the KV cache is about 400 bits over 4k tokens), not loud.
- **Bound the flow, not the work.** Capping style-unit *firings* would have re-created round 1's planner/voice tax,
  since wording is most of the compute. The risk is style *leaking into content*, so only that flow is capped. This
  mirrors integrity models in computer security: low-integrity sources may act freely, but may influence
  high-integrity decisions only through a narrow, audited channel.

**Why stop critiquing here.** The rounds converged:
- round 1: fatal flaws in the framing;
- round 2: major flaws in one block (re-encoding), removed by the ledger;
- round 3: no fatal flaws, and specific major refinements.

What remains is empirical and can't be settled on paper:
- the λ curve (is there a pressure level with a small tax and a big explanation gain over a dense model given the
  same pressure?);
- the smuggler floor.

Further paper rounds would mostly re-check edits. The next information has to come from experiments, starting
with T-A0 stage 1.
