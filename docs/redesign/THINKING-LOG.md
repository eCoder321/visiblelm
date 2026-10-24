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
