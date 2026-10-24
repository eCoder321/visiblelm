# Redesign: a language model that is both performant and observable

**Status: v1 (2026-10-24).** Written before any critique. A refinement history at the bottom records each change,
so the reasoning stays visible. The step-by-step reasoning behind this file (parallels explored, approaches rejected,
dead ends) is in `THINKING-LOG.md`. Prior-art claims marked *(verify)* are from memory and have not been checked
against the papers yet.

**Scope.** "Observable" here means **mechanism visibility**: we can see what the model actually computed and why.
The second kind, **epistemic visibility** (what the model believes and on what evidence; Graepel, MIT Tech Review,
2026-10-02), is parked as criterion C7. Every design must leave room for C7 later and must not rule it out.

---

## 1. The goal, as testable criteria

| # | Criterion | Plain meaning | Operational test (threshold) |
|---|---|---|---|
| C1 | **Performant** | As good as a normal transformer for the same compute. | *Compute-equivalent tax*: the dense compute needed to match our loss, divided by our compute. Target: tax ≤ 10% at small scale, and **not growing with scale**. Wall-clock tax is reported separately. Out-of-distribution (OOD) accuracy must be ≥ dense. |
| C2 | **Faithful** | The explanation *is* the computation, not a story told about it. | Intervene on a logged variable (feature value, unit firing, attention choice, library entry). The change the trace predicts must match the actual change: ≥ 99% top-1 agreement, KL < 0.01. |
| C3 | **Complete** | No hidden channel. Everything the output depends on is in the explanation. | *Completeness gap* = NLL(executor that sees only declared variables at declared precision) − NLL(model): ≤ 0.01 nats. Plus a magnitude audit and a non-interference test (§9). |
| C4 | **Legible** | A person can read it in human-sized pieces, and the names mean what they say. | (a) Explanation size: median number of terms needed to account for 90% of a decision, ≤ 20 per level. (b) Held-out legibility: a unit's description predicts its firing on unseen data at ≥ 0.9 F1. (c) A human-simulability spot check. |
| C5 | **Checkable** | A third party can verify the explanation without trusting the model. | An independent executor replays the decision log and reproduces the logits (to 1e-6) and the "books balance" (§7.8). Cost ≤ 1 forward pass. Individual decisions can be spot-checked. |
| C6 | **Predictive** | The descriptions predict behavior on new inputs and after edits. | Using only the descriptions, predict the output under input edits. After editing a unit or library entry, the predicted change holds ≥ 90% of the time. |
| C7 | *Epistemic (parked)* | We can see beliefs and their evidence. | No test yet. The design must expose a natural place for explicit propositions with provenance. |

Why these and not fewer. C2 without C3 lets the real work happen off the books. C3 without C4 gives a complete
explanation nobody can read; a dense model's weights already are one. C4 without C2 gives a readable story that
may be false, which is the chain-of-thought problem. C5 makes the other properties verifiable rather than claimed.
C6 is the payoff: the reason to want an explanation is to predict and control behavior.

---

## 2. First principles: why today's LLMs are performant, and why they are opaque

**Why they perform** (each item is something a redesign must keep, or replace with something just as good):
1. **Capacity per FLOP.** Dense matmuls run at near-peak hardware speed. *Superposition* lets a d-dimensional vector
   carry N ≫ d features as overlapping directions, so the model holds far more concepts than it has dimensions.
2. **Benign optimization.** Everything is differentiable, gradients reach every parameter, and overparameterization
   smooths the loss landscape.
3. **Generic inductive bias.** Attention plus MLPs can learn whatever structure the data has.
4. **Graded representations.** Similarity, uncertainty, register and style are continuous, and fluency lives in
   those fine gradations.
5. **Scale.** Data and compute.

**Why they are opaque** (each item is a requirement to fix):
1. **Unnamed superposition.** The information is there but not addressable. No coordinate means "France".
2. **Everything is active.** Every output is a sum of millions of small contributions, so even an exact
   decomposition is unreadable.
3. **No guarantee of completeness.** Post-hoc tools (sparse autoencoders, attribution graphs) leave an error
   term ("dark matter"). Nobody can say the named part is the whole story.
4. **Unbounded magnitudes.** A continuous value can carry arbitrarily many bits. We measured this directly: 2.3 nats
   of undeclared information rode on magnitudes (E1).
5. **Too many operations to read.** About 10^11 operations per token, so reading the trace needs to be on demand
   and hierarchical. It cannot be exhaustive.

**The sharpest version (the key insight of this redesign).** *Decomposability is cheap and dense models already
have it.* Between nonlinearities, a transformer is linear, so every logit already splits exactly into per-coordinate
contributions. Attribution-graph methods exploit this. What a dense model lacks is three things:
- **(i) a named basis**, so the terms of the decomposition are about something;
- **(ii) few active terms**, so the decomposition is readable;
- **(iii) a guarantee that the named terms are the whole story**, so nothing is off the books.

Every design below is a way to get (i)–(iii) without giving up items 1–4 of the "why they perform" list.

---

## 3. The core tension, and why it is not fundamental

Observability *seems* to want a few named, discrete, separate things. Performance *seems* to want many overlapping,
continuous, shared things. Our own experiments felt this: forcing binary values cost 0.29–0.5 nats (T1/T2). But
each half of the tension has a resolution that keeps what the other half needs:

| Performance needs | Observability fears | Resolution |
|---|---|---|
| N ≫ d features per position (superposition) | features are unnamed and entangled | **Carry the sparse code explicitly.** The state is a list of (feature id, value) pairs, k of them from a dictionary of N. The d-dim vector is only a hardware encoding of that list, so nothing has to be "decoded" and nothing is ambiguous. Compressed-sensing math says d ≳ C·k·log(N/k) suffices even for dense reads: k=200 and N=10^6 give d ≈ 3.5k–7k. |
| nonlinearity (expressiveness) | nonlinearity mixes everything | **Every nonlinearity is a logged decision; everything else is exact accounting.** Nonlinearities become top-k selections or 1-D functions of exactly-decomposed scores, and each one is written to a log. Between decisions the math is linear, so attribution is exact. |
| continuous magnitudes (graded meaning) | magnitudes smuggle bits | **Declared precision.** Each value carries a stated number of bits, enforced by training with injected noise. A noisy channel's capacity is bounded (½·log(1+SNR)), so the hidden channel becomes a *declared, measured* channel. Binarizing is not needed; top-k keeps the values exact (L2). |
| millions of contributions | unreadable | **Hierarchical summaries with an exact remainder**, like a financial audit's "materiality threshold". For example: "these 6 terms make up 91% of the logit, and the other 9% is these 40", with recursion on demand. Nothing is ever dropped silently. |

Three **capacity strategies** follow, one per design:
- **S1 Wide-sparse pools (Design A).** Hold capacity in *many named units, few active*, selected in O(√N) by
  product keys.
- **S2 External library (Design B).** Hold *knowledge* in readable entries outside the weights.
- **S3 Bounded fluency channel (Design C).** Put the capacity-hungry, low-stakes part (surface wording and style)
  in a component whose influence on *content* is bounded and audited.

---

## 4. Working backwards: the requirement tree

Each node answers "what must be true for the parent to hold?" The leaves are building blocks (BB), catalogued in §6.

```
G  Performant AND observable LM
├─ R1 Performant (C1)
│  ├─ R1.1 capacity per parameter and per FLOP comparable to dense
│  │   ├─ R1.1a represent N ≫ d features per position ............ BB1 explicit sparse feature bus
│  │   ├─ R1.1b many specialized units, few active ................ BB2 product-key unit pools
│  │   ├─ R1.1c knowledge capacity without parameters ............. BB12 library with provenance
│  │   └─ R1.1d selecting among N items costs sublinear in N ...... BB2 product keys (O(√N)), approximate-NN retrieval
│  ├─ R1.2 trainable at scale with no discrete-snap cliff
│  │   ├─ values stay continuous; only *choices* are discrete ..... BB5 top-k everywhere (gradients exact for the chosen items)
│  │   ├─ no dead or stuck units ................................... BB10 clonal selection (prune + regrow by fitness)
│  │   └─ load balance across units ................................ standard mixture-of-experts load balancing
│  ├─ R1.3 inductive bias that generalizes to unseen combinations
│  │   └─ operators shared across values (equivariance) ........... BB6 typed operator library  ← from E2redux (L5)
│  └─ R1.4 fluency: graded similarity, style, idiom
│      ├─ graded values ........................................... BB7 declared-precision magnitudes
│      ├─ idiom by precedent ...................................... BB13 precedent entries (kNN-LM style)
│      └─ style without content leakage ........................... BB15 bounded style channel
├─ R2 Faithful (C2)
│  ├─ the explanation is produced by the same forward pass ........ BB9 decision log (flight recorder)
│  └─ explanation variables are the causal variables .............. BB1 + an intervention API
├─ R3 Complete (C3)
│  ├─ R3.1 every bit used downstream passes a declared variable
│  │   ├─ state has no leftover dense remainder ................... BB1 (state *is* the list)
│  │   ├─ magnitudes carry only declared bits ..................... BB7 noise-bounded precision
│  │   ├─ normalization is not a side channel ..................... BB8 logged norm gain
│  │   └─ attention mixes only logged sources ..................... BB4 relay attention (sparse, logged)
│  └─ R3.2 completeness is verified, not assumed
│      ├─ books balance: Σ contributions = logit exactly .......... BB11 double-entry audit
│      └─ declared-info-only replay ............................... BB11 lockstep executor
├─ R4 Legible (C4)
│  ├─ names that are correct by construction ...................... BB14 compositional naming
│  ├─ explanations that are short ................................. BB3 coincidence gating, contribution-concentration penalty
│  ├─ names that predict held-out behavior ........................ BB10 clonal selection with a legibility fitness
│  └─ human-scale reading at LLM scale ............................ BB11 hierarchical summaries with exact remainder
├─ R5 Checkable (C5)
│  └─ verification without trusting the model ..................... BB11 independent replay + sampled decision checks
├─ R6 Predictive (C6)
│  └─ edits to named parts change behavior as described ........... BB6, BB12, edit tests
└─ R7 Epistemic (C7, parked)
   └─ explicit propositions with evidence .......................... BB12 library entries + BB16 plan states (hooks)
```

---

## 5. What our own experiments already taught us (inputs to the design)

| # | Lesson | Evidence | Design consequence |
|---|---|---|---|
| L1 | A testbed must measure generalization, or it measures memory. | The original 180-sequence testbed (review). | Every test reports held-out-combination and OOD splits. |
| L2 | Sparsity with continuous values is nearly free. Binarization is what costs. | T1/T2: 0.29–0.5 nats from the binary snap. | Make *choices* discrete and keep *values* continuous at declared precision (BB5, BB7). Never binarize. |
| L3 | Undeclared magnitudes are a real hidden channel. | E1: 2.3 nats lived in magnitudes. | Precision must be declared and enforced, not hoped for (BB7). |
| L4 | The iid gap is concentrated in arithmetic modes, and their value tables never moved from identity. | E1c: pred 28%, pls2 23%, succ 13% errors vs ~8% for copy modes; every succ/pls2 table exactly identity. | Arithmetic needs a different representation. |
| L5 | Free lookup tables cannot generalize arithmetic to unseen arguments, **and neither can a dense transformer.** | E2redux: tables freed from their regularizer still left iid NLL flat (0.785 vs 0.778, seed 0) or worse (0.970 vs 0.830, seed 1), and the rulebook grew 8× (65k vs 8.4k). Held-out succ/pred/pls2 accuracy was 0.0–0.025 for **every** model: rulenet_g, T2_large, *and dense*. | An unseen table cell gets no gradient, so it cannot be learned. Generalizing needs an operator shared across values (shift-by-n on a cyclic code). That is legible *and* the right inductive bias (BB6). |
| L6 | Weighted-sum gating lets a rule fire on an accidental mix of conditions. | E1c: succ-labeled rules fired on 70% of wrong cpy3 steps vs 0% of right ones. | Coincidence (AND-like) gating: a unit fires only when *all* its conditions hold (BB3). |
| L7 | With dense voting, ablating "bad" rules hurts. Attribution is non-local. | E1d: targeted ablation made succ worse (0.84→0.77). | Fewer, crisper contributors and exact decomposition. Don't rely on whole-rule ablation as the explanation tool. |
| L8 | Mechanistic stories are cheap to make and often wrong. | E1b refuted, then E1c's story refuted by E1d. | Build intervention and replay tooling into the design (BB9, BB11), so every story gets tested. |
| L9 | Structure already beats dense out of distribution. The in-distribution tax comes from specific mechanisms. | rulenet_g vs dense, NLL: shift 1.41 vs 1.84, long 1.40 vs 1.51, held-out cpy3 accuracy 0.96 vs 0.50. But iid 0.78 vs 0.59. | Target the specific mechanisms behind the iid tax (L4/L5). |
| L10 | The iid "oracle" floor is slightly loose. | Dense reaches 0.593 vs oracle 0.617 on the same split. The oracle ignores the rejection-sampling conditioning that removes held-out events from training-distribution sequences. | Treat the dense NLL as the practical iid floor. Fix the oracle if exact floors matter. |

---

## 6. Building blocks

Status: **exists** = established technique; **partial** = pieces exist, our combination or use is new;
**new** = no known equivalent (to our knowledge, *verify*).

| BB | Block | What it does | Status | Nearest prior art *(verify)* | Parallel that suggested it |
|---|---|---|---|---|---|
| 1 | **Explicit sparse feature bus** | Residual state = list of ≤ k (feature id, value) pairs from a dictionary of N. The d-dim form is only an encoding. | new | Sparse autoencoders (post-hoc); Monet; sparse feature circuits | Cortical sparse coding (Olshausen & Field); the *blackboard* architecture (Hearsay-II) |
| 2 | **Product-key unit pools** | 10^4–10^6 units per layer, top-k_u selected in O(√N) | exists | Product-key memory (Lample 2019); PEER "million experts" (He 2024); memory layers at scale (Berges 2024) | Insect mushroom body: sparse expansion + winner-take-all (Dasgupta 2017, "fly hashing") |
| 3 | **Coincidence gating** | A unit fires only when all its conditions co-occur (product / soft-min), not on a weighted sum | partial | Sigma-pi units, gated linear units | Dendritic NMDA spikes: a branch as an AND, the soma as an OR over branches |
| 4 | **Relay attention** | Score = exact sum over (query feature, key feature) pairs + relative-position kernel. Top-k_attn with logged weights. Moves *named* features with renaming. | partial | Sparsemax/entmax, top-k attention, bilinear decompositions | Mail routing with addressed envelopes vs broadcast hormones |
| 5 | **Top-k instead of binarization** | Discrete choice with exact values, so gradients are exact for chosen items | exists | k-sparse AEs, top-k MoE routing | Lateral inhibition / winner-take-all |
| 6 | **Typed operator library** | shift(+n) on cyclic/ordinal codes, copy, compare, count. Integer parameters chosen discretely and logged. Shared across values, hence equivariant. | partial | Neural module networks (Andreas 2016), NALU (Trask 2018); grokking finds rotations for modular arithmetic (Nanda 2023); LLMs add numbers on a helix (Kantamneni & Tegmark 2025) | Grid cells: path integration as a phase shift; convolution = translation equivariance |
| 7 | **Declared-precision magnitudes** | Each value has b declared bits, enforced by noise injection, so channel capacity is bounded and measured | partial | Information bottleneck, quantization-aware training | Spike-rate codes: analog values carried with noise-limited precision |
| 8 | **Logged norm gain** | RMSNorm's scale is one logged scalar per position, so everything after it is linear | new (trivial) | — | Neuromodulators: a global, low-bandwidth gain |
| 9 | **Decision log ("flight recorder")** | The forward pass emits every discrete choice and declared value | partial | Deterministic replay debuggers (rr), distributed tracing (Dapper/OpenTelemetry) | Aviation flight data recorders |
| 10 | **Clonal selection** | Prune/regrow units by fitness = usefulness × legibility. Undescribable, low-value units are recycled. | partial | Our prune+regrow; RigL | Immune clonal selection, negative selection in the thymus, affinity maturation |
| 11 | **Lockstep executor + books-balance audit + hierarchical summaries** | Independent replay. Every logit = Σ contributions exactly. Explanations at a materiality threshold with exact remainder. | new for NNs | Our `execute_hard_g`; translation validation (compilers) | Double-entry bookkeeping (Pacioli); audit sampling and materiality; chip design-for-test (scan chains cost ~5–10% area) |
| 12 | **Library with provenance** | Knowledge entries = (text, feature signature, key, source, date) | partial | RETRO, Atlas, REALM | Case law and citations; hippocampus ↔ cortex (complementary learning systems) |
| 13 | **Precedent entries** | (context signature → continuation) memories that carry idiom and fluency | exists | kNN-LM (Khandelwal 2020) | Judges citing precedent |
| 14 | **Compositional naming** | A feature's name is the program that produced it ("shift+1 of the value copied from the last succ position") | new | Tracr/RASP (names by construction, compiled not learned) | Chemical nomenclature: names from structure |
| 15 | **Bounded style channel + round-trip parser + non-interference test** | Text realization gets ≤ b bits/sentence of free choice. A parser checks that the text expresses exactly the plan. Hidden context is varied to test that output doesn't change. | partial | Plan-then-generate NLG; translation validation; steganography detection | Score vs performer; Levelt's speech production (conceptualizer → formulator → articulator); information-flow control / noninterference in OS security |
| 16 | **Typed plan states** | Per-sentence plan = sparse set of *content-typed* features from BB1's dictionary | partial | Data-to-text content planning (Reiter & Dale pipeline) | An architect's blueprint vs construction |

---

## 7. Design A — Glass Box Transformer (the sparse feature bus)

**Principle:** *all nonlinearity is a logged decision; everything else is exact accounting.*

### 7.1 Anatomy (one layer; repeated L times)
1. **Bus (state)** `S_t = {(f, a_f)}`, with |S_t| ≤ k_tot (≈100–300 at LLM scale), f ∈ dictionary of N (10^6–10^7)
   named features, and a_f at declared precision b (BB7). Hardware form: `h_t = Σ a_f E[f]` with E tied across layers.
   Every linear read `W h_t = Σ a_f (W E[f])` splits exactly into per-feature terms.
2. **Normalize:** `s_t = 1/rms(h_t)` is **logged** (BB8). After that, everything is linear in h_t.
3. **Relay attention** (BB4). Score `σ(t,t') = Σ_{f∈S_t, g∈S_t'} a_f b_g ⟨Q E[f], K E[g]⟩ + κ(t−t')`. It is computed
   densely for speed and decomposed into its k² pair terms on demand. **Top-k_attn** sources (8–32) are kept with
   their logged weights. Values moved = the source's named features through a learned *renaming* (e.g. `cur → prev`).
   So "what moved" is a list of named features.
4. **Unit pool** (BB2, BB3). Product-key query from h_t selects candidate units, and the top-k_u (≈32–64) fire. Unit u
   has a score `z_u = ⟨key_u, h_t⟩ = Σ a_f ⟨key_u, E[f]⟩` (exact per-feature terms) and a coincidence gate over its
   declared conditions, `g_u = Π_i σ(β(c_{u,i} − θ))`. That gate is the AND that removes accidental firing (L6). The
   unit's write is a sparse set of features, each either a free value or a **typed operator** applied to a read
   value (BB6).
5. **Write and re-sparsify:** add the writes to the bus, then keep the top-k_tot by |value|. **Evictions are logged**,
   so forgetting is visible. The evicted mass goes to the remainder ledger.
6. **Readout** (last layer): `logit_v = Σ_f a_f U[f,v]`, exact per-feature contributions. The final softmax is the
   definition of the output distribution, not a hidden step.

**Nonlinearity inventory.** There are five kinds: re-sparsify top-k, unit selection top-k, unit gate value,
attention top-k plus weights, and norm gain. Every one is a logged choice or a logged scalar computed from
exactly-decomposed inputs. Nothing else in the model is nonlinear.

### 7.2 Why it should perform (C1)
- Capacity lives in *N named features × values* and in *10^4–10^6 units per layer*, not in unnamed superposition.
  Very wide product-key pools have reportedly matched or beaten dense FFNs per FLOP (PEER, memory layers *(verify)*).
  Fine-grained MoE scaling laws favor more, smaller experts *(verify)*.
- Information budget check: k·(log₂N + b) = 200·(20+4) ≈ 4.8k bits/position. Whether LLM residuals use more than
  that is **unknown**. This is A's riskiest assumption (§7.5). k can be raised: decomposition stays exact while
  legibility degrades gracefully, and both are measured.
- Training stays smooth: values are continuous, only the choices are discrete, and top-k has exact gradients for
  the chosen items. MoE routing proves this trains at scale. No binary snap, so the main cost we measured (L2) is
  gone.
- Equivariant typed operators should *beat* dense on unseen combinations (L5, L9).
- FLOPs per FFN-replacement are much lower than dense (illustrative: d=4096, 64 single-neuron units ≈ 1M FLOPs plus
  key search ≈ 5M, vs ≈ 268M for a 4d dense FFN). Parameters and memory bandwidth are much higher. Expect the tax
  to show up in **wall-clock time on current hardware**, not in FLOPs. Report both.

### 7.3 How A meets each criterion

| Criterion | Mechanism | Strength |
|---|---|---|
| C1 Performant | Wide-sparse pools, explicit sparse bus, top-k (no binarization), typed operators, product keys | **Open empirical question.** Riskiest assumption §7.5. Evidence for the pools exists; evidence for the bus does not. |
| C2 Faithful | The trace *is* the forward pass (decision log). Interventions are edits to logged variables, re-run through the same code. | **By construction.** The test only checks the implementation. |
| C3 Complete | State is the explicit list (no dense remainder). Magnitudes are noise-bounded to b bits. Norm gain logged. Attention sparse and logged. Evictions logged. | **By construction, up to declared precision.** The magnitude audit measures it. The declared capacity k·b bits/position is a *stated* channel, not a hidden one. |
| C4 Legible | Names compose (BB14). Coincidence gating keeps firing logical. Contributions are concentrated by a penalty and measured as "terms to 90%". Clonal selection removes units that can't be described. Summaries have an exact remainder. | **Statistical, measured, pushed by training pressure.** The weakest link at scale: no guarantee every latent feature has a short name. |
| C5 Checkable | An independent lockstep executor replays the log. Books balance per logit and per layer. Decisions can be sampled and re-derived (e.g. re-run that top-k). | **By construction.** Replay cost ≈ 1 forward pass. |
| C6 Predictive | Typed operators and named units have semantics. Editing a unit's condition or write changes behavior as its name says. | **Measured.** Strong for typed parts, moderate for learned-latent parts. |
| C7 Epistemic (hook) | Named features and units are where propositions and evidence links can live later. | Hook only. |

### 7.4 Main failure modes and their counters
- **Many tiny contributors** (legible in principle, unreadable in practice). Counter: a concentration penalty, plus
  "terms to 90%" reported as a first-class metric.
- **Polysemantic dictionary features** (a name that's wrong half the time). Counter: a held-out-legibility fitness in
  clonal selection, and feature splitting when a feature's description fails on a coherent subset.
- **Magnitude channel saturating** (the model packs content into the b declared bits). Allowed, because it's
  declared. Raise k or lower b if the audit shows heavy use, and report the bits actually used.
- **Top-k attention starving long-context averaging.** Counter: allow k_attn to vary per head, and log the "dropped
  attention mass" in the remainder ledger.

### 7.5 Riskiest assumption, decisive test, kill criterion
- **Assumption:** a residual constrained to be an explicit k-sparse list over N named features, trained from scratch,
  costs ≤ ~10% compute-equivalent, and the cost **falls as N grows** at fixed k.
- **Decisive test:** T-A0 / T-A1 (§11). Measure tax vs N ∈ {1k…64k} at fixed k, and tax vs k, on real text.
- **Kill:** tax does not fall as N grows at fixed k (named capacity can't substitute for superposition), **or** the
  books-balance check can't be made exact (the design leaks).

---

## 8. Design B — Library and Clerk (knowledge outside the weights)

**Principle:** *knowledge lives where you can read it; the model is a clerk who must cite.* This directly targets
Graepel's second gap, "knowledge entangled with manipulation". It is also the most direct route to C7 later.

### 8.1 Anatomy
- **Library:** millions to billions of entries, each `(text, feature signature in A's dictionary, retrieval key,
  provenance: source, date, trust tag)`. Two kinds:
  - **fact entries**: propositions;
  - **precedent entries** (BB13): `(context signature → continuation)` memories that carry idiom and fluency, so
    "we say X because in contexts like these the next word was X".
- **Clerk:** a *small* Design-A model. Retrieval is a logged decision: top-k entries, each score exactly decomposed
  over feature pairs. Retrieved entries enter the bus as named features with a provenance tag. A logit's
  contributions therefore split into **cited entries / clerk units / context**.
- **Core knowledge** that has to stay parametric (grammar, common sense) lives in the clerk's *declared* A-units. It
  is visible, just not cited.

### 8.2 Training that forces honest citation
- **Counterfactual-edit open-book training.** During training, randomly edit or swap library entries and require
  the output to follow the *edited* entry. Memorizing the fact in the clerk's weights then *hurts* the loss, so
  relying on the library is the only winning strategy.
- **Closed-book leakage audit.** Remove or blank the library and measure how much non-core knowledge the clerk still
  recalls. That recall is the hidden channel, so it must be small.
- **Reliance-under-edit test.** Edit an entry: the output follows it ≥ 90% of the time.

### 8.3 How B meets each criterion

| Criterion | Mechanism | Strength |
|---|---|---|
| C1 | Retrieval substitutes for parameters: RETRO ~25× fewer parameters on some benchmarks, Atlas 11B vs PaLM 540B few-shot QA, kNN-LM perplexity gains *(verify all)*. The clerk is small. | **Plausible, with evidence.** Risks: retrieval latency and I/O, and reasoning that needs knowledge woven too finely to retrieve. |
| C2 | Citations are causal: edit training plus reliance tests show outputs actually depend on cited entries, rather than decorating them. | **Measured** (target ≥ 90% reliance). |
| C3 | The hidden channel would be clerk parametric memory, and the leakage audit bounds it. Core knowledge is declared A-units. Retrieval is logged. | **Measured** (leakage target ≤ 10% of non-core facts). |
| C4 | Entries are text, the most legible form there is, with provenance. Clerk reasoning inherits A's legibility. | **Strong for knowledge; A-level for reasoning.** |
| C5 | Re-run with an entry removed or edited, check provenance, replay the clerk with A's executor. | **By construction plus a cheap counterfactual check.** |
| C6 | Edit an entry and predict the change. | **Measured** (reliance-under-edit). |
| C7 | Entries are explicit propositions with provenance: the seed of a belief store. | **Strongest hook of the three designs.** |

### 8.4 Riskiest assumption, decisive test, kill criterion
- **Assumption:** a small clerk with a library matches a large dense model *and* actually relies on the library
  rather than quietly memorizing.
- **Decisive tests:** T-B1 (reliance under edits) and T-B2 (closed-book leakage, with and without edit training).
- **Kill:** high closed-book leakage, **or** reliance-under-edit < 90% despite edit training.

---

## 9. Design C — Planner and Voice (content separated from wording)

**Principle:** *decide what to say in the open; let the wording be free only within a bounded, audited channel.*
Fluency is the most capacity-hungry and least decision-relevant part of language, so give it its own component and
make sure it can't change the content.

### 9.1 Anatomy
- **Planner** (Design A): for each upcoming sentence it emits a **plan**, a sparse set of *content-typed* features
  from A's dictionary with values (BB16), e.g. {predicate: capital_of, arg: France, answer: Paris, stance: assert}.
  This needs no hand-written schema: "content" vs "style" is a type on dictionary features, verified by the
  non-interference test below.
- **Voice:** renders `(plan_i, style code_i ≤ b bits, previously emitted text)` into words. **The voice sees nothing
  else.** It has no access to the hidden context. That is an information-flow rule (BB15): any content in the text
  must have come through the plan or the already-public text.
  - *Strict* variant: the voice is also Design A.
  - *Pragmatic* variant: the voice is a dense model, made safe by the checks below rather than by its internals.
- **Round-trip parser:** an A-style encoder maps the emitted sentence back to content features. Check
  `parse(text) == plan` on content-typed features. This is translation validation: check each instance instead of
  proving the voice correct.
- **Non-interference test:** hold the plan and style code fixed and vary the hidden context. The output distribution
  must not change. Also train adversarial probes to recover non-plan content from the text (a steganography
  check), and estimate how many bits the style code actually carries.

### 9.2 How C meets each criterion

| Criterion | Mechanism | Strength |
|---|---|---|
| C1 | A pragmatic dense voice gives dense-level fluency, and the planner is small. The tax depends on how expressive plans are. | **Open.** The main risk is open-domain nuance that resists a feature-set plan. |
| C2 | The plan determines content causally: round-trip check, plus plan-edit tests (edit the plan and the text changes accordingly). | **Measured** per instance. |
| C3 | Content can flow only through the plan or the ≤ b-bit style code, enforced by the voice's restricted inputs and *tested* by non-interference and steganography probes. | **By construction for flow; measured for leakage.** |
| C4 | A plan is a short set of named content features. The style code is explicitly "how, not what". | **Strong.** |
| C5 | The round-trip parse is a cheap per-sentence check. The planner replays with A's executor. | **By construction.** |
| C6 | Edit the plan and the text changes predictably. | **Measured.** |
| C7 | The plan is an explicit state that an independent evaluator could check, which is Graepel's "System 2" hook. | **Hook.** |

### 9.3 Riskiest assumption, decisive test, kill criterion
- **Assumption:** a sentence's content can be captured as a sparse set of typed features, at small tax and with
  round-trip fidelity.
- **Decisive test:** T-C1.
- **Kill:** round-trip error > 5%, **or** planner tax no lower than token-level Design A's tax (separating the voice
  bought nothing).

---

## 10. The composite: Layered Glass Box, plus Design D (parked)

A, B and C are not rivals. They are layers:
- **A is the substrate** for the clerk, the planner and the parser.
- **B supplies knowledge.**
- **C supplies fluency without content leakage.**

A full trace of one answer reads like this:

> **"Paris"** ← voice realized plan item {capital_of(France) = Paris}. Round-trip ✓, style code used 3 of 16 bits.
> ← planner unit #88 412 "answer-capital-question" fired. Gate conditions {question-about-capital ✓,
> entity=France ✓}. Its score is 84% from 4 named features, and the other 16% is listed.
> ← feature `answer=Paris` came from library entry #4 411 "Paris is the capital of France" (source: encyclopedia,
> 2025-03), retrieval score decomposed. Under an edit to "Lyon", the output becomes "Lyon" ✓.

**Design D — "Open Notebook" (parked, C7).** An explicit belief state: propositions with credences and evidence
links, revised step by step, with an independent evaluator (Graepel's AlphaGo analogy). Library entries, named
features and plans are its natural ingredients. Nothing in A/B/C blocks it.

---

## 11. Test plan (ordered; 4 CPU cores, so these are trend tests at small scale)

Each test states a prediction **before** running, so it can be wrong.

| Order | Test | Question | Setup | Prediction | Kill / redirect if |
|---|---|---|---|---|---|
| 1 | **T-A3 typed operators** | Does an equivariant shift operator fix the arithmetic gap (L4/L5)? | rulenet_g + a cyclic value code with a shift(+n) operator (n chosen discretely, logged). Control: dense with the same value code. Current testbed. | Held-out succ/pred/pls2 accuracy goes from ~0 to ≥ 0.9. iid NLL closes ≥ half the gap to dense. The dense control stays well below. | Held-out arithmetic < 0.5: operator selection is failing; inspect before moving on. |
| 2 | **T-A0 bus-lite on real text** | A's riskiest assumption, cheaply: what does an explicit k-sparse named residual cost on *real* language? | Small dense transformer vs the same with an explicit top-k-over-dictionary residual at every layer (tied dictionary, ids carried). Tiny real corpus (TinyStories if egress allows, else an offline corpus). Sweep k and N. | Tax falls as N grows at fixed k. At k≈32–64 and N≈16k the tax is ≤ 15%. | Tax flat or rising in N: rethink A before building more of it. |
| 3 | **T-A2 coincidence gating** | Does AND-gating remove cross-mode contamination at no cost (L6)? | rulenet_g, weighted-sum vs product gate. | Cross-mode firing ~0, fewer rules per step, iid NLL equal or better. | NLL worse by > 0.05 with no legibility gain. |
| 4 | **T-A1 capacity scaling** | Can named wide-sparse capacity hold *idiosyncratic* knowledge as efficiently as dense superposition? | Testbed v2 = current rules + K random facts (K ∈ {256, 1k, 4k}). Glass box N_u ∈ {1k…64k} at fixed k vs dense at matched params and FLOPs. | Tax falls with N_u. Within 10% of dense on fact recall at matched params. | Tax flat in N_u. |
| 5 | **T-B1 reliance under edits** | Does the clerk actually use the library? | Testbed v2 with facts moved into a library. Train with and without counterfactual edits. | With edit training, reliance ≥ 90%. | < 90%. |
| 6 | **T-B2 leakage audit** | How much knowledge hides in the clerk? | Same, with the library removed. | Leakage ≤ 10% with edit training, much higher without. | Leakage high even with edit training. |
| 7 | **T-C1 planner/voice** | Does content separate from style at small tax? | Testbed v3: sentences generated from frames plus ~8 bits of style variation. | Round-trip error ≤ 5%. Style code uses ≈ 8 bits. Non-interference holds. | Round-trip > 5%, or no tax gain over token-level A. |
| 8 | **T-A4 glass box on real text** | Full Design A vs dense at matched FLOPs: the honest C1 number at small scale, plus C2–C5 metrics | Real corpus, ~1–5M params | Tax ≤ 10–15%, terms-to-90% ≤ 20, completeness gap ≤ 0.01 | Tax > the gap between dense and dense-at-half-compute (worse than halving the model). |

**Why this order.**
- T-A3 is cheap, attacks the one mechanism we *know* is broken, and its operator is reused by everything later.
- T-A0 comes second because it tests the riskiest assumption in the whole program on real data before we invest in
  the rest of A.
- B and C tests need new testbeds, so they come after A's foundations.
- T-A4 integrates everything.

**Shared metrics for every test:** compute-equivalent tax; held-out and OOD accuracy; completeness gap; magnitude
bits used; terms-to-90%; held-out legibility; replay exactness.

---

## Refinement history
- **v1 (2026-10-24).** First full draft from the requirement tree, the lessons L1–L10 and the parallels in
  `THINKING-LOG.md`. Two independent critics (a performance skeptic and an observability skeptic) review it next.
  v2 will record every change they force, and why.
