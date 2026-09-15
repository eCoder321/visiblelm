# Plan v3 — observability without sacrificing accuracy

Written 2026-09-15 by the principal investigator session, after tasks T1 and T2. Status boxes are the checklist;
nothing below is implemented yet. Prior plan: `PLAN.md`. Evidence: `REPORT.md`, `docs/tasks/T1-report.md`,
`docs/tasks/T2-report.md`.

---

## 1. What we actually know (the facts the reasoning has to fit)

All numbers are iid NLL in nats/token on the switching-rules testbed; oracle floor 0.62; dense transformer 0.59.

| model | NLL | rulebook = model | magnitude audit gap | circuit nodes nameable |
|---|---|---|---|---|
| dense + post-hoc SAEs (null) | 0.59 | n/a | +3.2 | 32% |
| structured, continuous activations (`rng_v1`) | 0.66 | 50% | +2.3 | 50% |
| structured, fully binary, medium capacity (`T1_base`, 3 seeds) | 0.94 | 100% | 0.00 | 50% |
| structured, fully binary, 2× capacity (`T2_large`, 3 seeds) | 0.88 | 100% | 0.00 | 58% |
| structured, fully binary, ½ capacity (`T2_small`, 3 seeds) | 1.28 | 100% | 0.00 | — |

Five facts matter for what comes next:

1. **The continuous structured model is nearly free** (0.66 vs 0.59) and its printed program re-executes exactly
   (continuous re-execution matches the network to 1e-8). What it lacks is *binary* readability: reading its features
   as on/off reproduces only 50% of its predictions, because activation magnitudes carry 2.3 nats of information.
2. **Making it binary costs 0.22–0.28 nats** at medium and large capacity, and the cost decelerates with size
   (halving capacity: +0.34; doubling: −0.07). Capacity fixed the specific near-switch blind spot (98% at large) but
   the average is approaching an asymptote well above the dense model.
3. **Attention was already nearly discrete on its own.** Hard attention alone keeps 91% agreement; binarizing
   activations alone drops it to 30%. So the continuous channel that matters is the activation magnitude, not
   routing.
4. **The binary model is badly calibrated on the stochastic part of the task**: it puts 0.04 probability mass on a
   rule switch where the truth is 0.10. Its argmax is right far from switches (97%); it loses nats on *how sure* it is.
5. **Optimization of discrete programs is fragile**: a longer hard phase and a learning-rate restart both made it
   worse (T1). We have never trained a discrete program any way other than straight-through from a soft warm start.

## 2. First-principles analysis: what does observability actually require?

Go back to the goal's four properties and ask what each one *requires*, as opposed to what we *chose*.

**P1 Faithful** (the explanation is the causal path). Requires: the artifact you read is the computation that runs.
Satisfied by *exact re-executability of the printed program*. Does not require binary values. The continuous
structured model already satisfies it (1e-8 match).

**P2 Legible** (states are nameable units). Requires: each unit has a stable, verifiable meaning. Satisfied by
held-out predicate fit and intervention. The continuous model's features are *more* nameable than the null's
(50% vs 32% of circuit nodes). Binarity did not create legibility; the slot structure did.

**P3 Complete** (no hidden channel). This is where the confusion was. What P3 requires is that **every channel that
carries information is declared, bounded, and readable**. A continuous magnitude is a hidden channel only if its
meaning and its information content are unstated. We *chose* binary because it is the simplest way to make the
magnitude channel's capacity zero. But zero is not required; **finite and declared** is. A magnitude that takes one
of L declared levels is a channel of log2(L) bits per feature, fully readable, and a rule over such inputs is still a
finite table (L³ rows for fan-in 3) or, more naturally, a printed weighted threshold ("IF 0.7·b3 + 0.4·val0=5 > 0.6").

**P4 Compositional** (traceable, bounded description per prediction). Requires: the number of nodes and edges a
reader must follow per prediction stays small. Measured by circuit size and description length. Independent of
binarity; the binary model's circuits are actually *smaller* (median 5 vs 12) — that is a real benefit of hard
decisions that graded levels may partly give back, and must be measured.

So the honest restatement of what we learned: **we over-specified the fix.** The goal needs a *finite, declared
alphabet* for every channel and an *exact executor* for the printed program. Binary is the L=2 special case. The
question the next experiments answer is: **what is the smallest alphabet (fewest levels, or least magnitude
precision) at which accuracy is recovered — and does the description length per prediction stay bounded there?**
That is a Pareto frontier between two measurable quantities, and every point on it is a legitimate design.

### 2.1 Why binary costs accuracy: three separable causes

Decomposing "the discrete gap" into causes that have *different fixes*:

* **(a) Capacity.** A binary state has fewer distinguishable configurations; more features are needed to express
  the same function. T2 shows this cause is real but saturating. Fix: size. Diminishing.
* **(b) Optimization.** Gradient descent through a straight-through estimator finds worse programs than may exist.
  Untested: distillation from the soft teacher; annealed relaxations; symbolic repair after snapping. Fix: training
  method. Unknown magnitude — this is the biggest open question.
* **(c) Expressivity of intermediate uncertainty.** A binary slot cannot hold "probably succ, 0.7". On *this*
  testbed the mode is always exactly determined by the last mode token, so (c) should be near zero here. On real
  text (ambiguity resolved later) it will not be. Fix: graded levels. **We cannot measure (c) on the current
  testbed**; a testbed with genuine intermediate uncertainty is required before any conclusion about text.
* **(d) Calibration of the output**, a special case of (c) at the readout: the 0.10 switch probability must be
  expressed as vote weights gated by a "switch is possible here" feature, which requires counting switches and
  distance — reasoning depth the 2-layer rule net may lack. Fix: could be features (a `switch_possible` boolean),
  not magnitudes at all. Must be checked before blaming discreteness.

The first experiment therefore is not a new model; it is **attribution of the lost nats to (a)–(d)**, using the
oracle we already have (per-position true distributions), on the bundles we already have. Everything after depends
on what that says.

### 2.2 Design space between binary and continuous (candidates, ranked)

| option | what changes | keeps exact executor? | readability cost | prior expectation |
|---|---|---|---|---|
| **Q. Quantized levels** — activations in {0, 1/L, …, 1}, L ∈ {2,4,8,16}; STE to nearest level; L-level executor | the alphabet | yes, exactly | rules become weighted thresholds over graded inputs; per-node reading needs a 1-digit calculator | most likely to recover accuracy at small L; the cleanest frontier |
| **D. Distillation** — train soft, then fit the L-level student to the teacher's outputs and states | the training method | yes | none | tests cause (b); may close most of the gap even at L=2 |
| **N. Noise-limited magnitudes** — continuous values + training noise σ; channel capacity ≈ log2(1/σ) | the channel's capacity, not its alphabet | only approximately (executor must reproduce noise-free forward; agreement < 100% by construction) | reader sees continuous numbers | a useful *dial* for the frontier but weaker guarantee than Q; run only if Q fails |
| **W. Weighted-logic reading of the soft model** — no change to the model; define the readable artifact as the printed weighted program and measure how much precision a reader needs | the definition of "read" | yes (already) | reader must do small arithmetic | E1 answers this for free before any training |
| **S. Symbolic repair** — snap, then local search over rule flips scored on training data, no gradients | post-processing | yes | none | cheap add-on to any of the above; tests whether the STE program is a poor local optimum |

Not pursued: making attention soft again (worth ≤9% agreement, cost unknown, and hard attention makes circuits
smaller); scaling the binary model further (T2 shows saturation).

### 2.3 What accuracy is *not* recoverable, so we don't chase it

* The irreducible entropy of the task (seed tokens, switch events) is in the floor; nobody beats it.
* The dense model's 0.59 < 0.62 floor comes from exploiting the rejection-sampled training distribution (held-out
  arguments never appear); a fully general program cannot and should not match that. The realistic target for a
  readable model is **the floor (0.62), not the dense number**. State this in every report.

## 3. Experiments (the checklist)

Order is by information per unit of compute. E0 and E1 need no training and settle which of (a)–(d) we are fighting.

### E0 — Attribute the lost nats  ☑ done 2026-09-15, see `docs/tasks/E0-E1-report.md`
No training. For `ctrl_dense_sae`, `rng_v1` (soft), `T2_large` (binary), on the iid split, using the oracle's
per-position distribution:
* ☐ Split positions into: seed (uniform, unfixable), deterministic rule steps (switch impossible), stochastic rule
  steps (switch possible, P(switch)=0.10), and held-out steps. Report NLL − oracle per bucket and the bucket's share
  of the total gap.
* ☐ Within deterministic steps, split the gap into argmax errors vs. under-confidence on correct argmax.
* ☐ Add `switch_possible`, `n_switches_so_far`, `dist_since_switch≥5` to the legibility predicate vocabulary; report
  whether any feature of `T2_large` tracks `switch_possible` (held-out F1).
* **Decision rule:** if ≥ 50% of the binary model's gap is in stochastic-step calibration and no feature tracks
  `switch_possible`, cause (d) dominates → the first fix is a *feature*, not levels (E2 still runs, but expectations
  change). If the gap is spread over deterministic argmax errors, causes (a)/(b) dominate → E2/E3 are the main line.

### E1 — Post-hoc quantization of the soft model  ☑ done 2026-09-15, see `docs/tasks/E0-E1-report.md`
No training. Take `rng_v1` (and a soft twin of `T2_large`'s config if cheap to train once, 1 seed) and re-execute
with activations rounded to L levels, L ∈ {2, 3, 4, 8, 16, ∞}, hard attention on/off.
* ☐ Report agreement with the soft model and NLL of the quantized program, per L.
* ☐ Report the L at which agreement ≥ 95% and ≥ 99%.
* **Decision rule:** if L ≤ 8 already gives ≥ 95% agreement, the magnitude channel is *coarse* and option Q will
  recover accuracy cheaply → go to E2 with L ∈ {4, 8}. If agreement stays < 90% until L ≥ 16, magnitudes carry
  fine-grained information → E3 (distillation) and E4 (noise dial) become necessary to see whether that information
  is compressible.

### E2 — Quantization-aware training: the frontier  ☐
Training. `T2_large` config; snap to L levels (STE to nearest level, hard attention) for L ∈ {2, 4, 8}; 3 seeds each.
L=2 is `T2_large` itself (already done — reuse).
* ☐ Implement `n_levels` in the snap phase and an L-level executor; verify executor = model to 1e-8 (same test as
  today's binary executor).
* ☐ Report per L: NLL, rulebook agreement (must be 100% by construction; a deviation is a bug), magnitude audit
  (should be exactly 0 at the quantized values), circuit size, circuit-clean fraction, description length.
* ☐ Plot the frontier: NLL vs. bits per feature (log2 L). Mark the floor (0.62) and the soft model (0.66).
* **Gate:** a point with NLL ≤ 0.70 (within ~0.05 of the soft model and ~0.08 of the floor) and median circuit
  size ≤ 8 nodes is a design that answers the user's question on this testbed. If L=4 gets there, the alphabet is
  2 bits/feature — small enough that a person can read a rule's inputs as {off, low, mid, high}.

### E3 — Distillation instead of straight-through  ☐
Training. Same L as the best E2 point plus L=2. Teacher: the soft model (same architecture, no snap). Student:
quantized, trained on the teacher's output distribution (KL) plus a state-matching term on the quantized
activations; then a short fine-tune on the true labels. 3 seeds.
* ☐ Report NLL vs. E2 at the same L. The difference is the size of cause (b).
* **Decision rule:** if distillation at L=2 closes most of the binary gap (≤ 0.70), the whole "discreteness cost" was
  optimization and binary is back on the table. If it helps at L=4/8 but not L=2, both (a)/(b) and the alphabet
  matter, and the frontier from E2 stands with a better training method.

### E4 — Symbolic repair after snapping (cheap add-on)  ☐
No gradient training. On the best E2/E3 model: greedy local search over rule flips / threshold nudges scored by
training NLL under the exact executor. Budget: fixed number of evaluations.
* ☐ Report NLL before/after and the number of edits. A large gain means the STE optimum is poor and post-hoc
  program repair is a legitimate part of the training recipe for readable models.

### E5 — Testbed v2 with genuine intermediate uncertainty  ☐
Design, no training until designed. The current testbed cannot measure cause (c). Add a variant where the state a
position needs is *not* fully determined at that position:
* ☐ Rule tokens are hidden with probability p (the switch happens but the mode token is not emitted); the mode must
  be inferred from the next few tokens' behaviour, so intermediate belief is required for several steps.
* ☐ Exact oracle (a small HMM forward pass) so floors stay exact.
* ☐ Re-run the E2 frontier on it (best two L values + L=2, 3 seeds). **Prediction, stated before running:** the gap
  between L=2 and L≥4 widens on this testbed relative to the current one, because L=2 cannot hold a belief.
  If the prediction fails, cause (c) is smaller than first-principles suggest and binary may be enough even for
  ambiguous inputs — which would be a surprising, valuable result.

### E6 — Human-simulability and the legibility budget  ☐
Measurement, no training. For each frontier point: a *bounded-precision reader* (2 significant digits, the printed
circuit only) re-executes the model's prediction.
* ☐ Report reader agreement with the model vs. L, and description length per prediction (nodes × inputs per node).
* ☐ Define and record the acceptance criterion in advance: reader agreement ≥ 99% and median description per
  prediction ≤ 40 numbers. A frontier point that passes this *and* the E2 gate is the answer on this testbed.

### E7 — Real text (unchanged from PLAN.md Phase 5, now with the frontier)  ☐
* ☐ Small char-level config; run the frontier's best L and L=2; report tax at plateau, description length per
  prediction, intervention-defined legibility. Gate as in `PLAN.md`.

## 4. Predictions, written down now so they can be wrong

1. E0 will show that ≥ 40% of the binary model's gap is calibration on stochastic steps and that no feature tracks
   `switch_possible` (cause (d) is real and fixable with a feature, independent of levels).
2. E1 will show the soft model reaches ≥ 95% agreement by L = 8 and ≥ 99% by L = 16 (the magnitude channel is
   coarse, not fine).
3. E2 will put L = 4 at NLL ≈ 0.70–0.75 and L = 8 at ≈ 0.68, with circuit size between the binary (5) and soft (12).
4. E3 will show distillation helps at every L by 0.03–0.08, i.e. optimization is a real but not dominant cause.
5. E5 will widen the L=2 vs L≥4 gap by at least 0.1 nats.

## 5. Ordering, cost, and gates

| step | compute | wall-clock (4 CPU cores) | blocks |
|---|---|---|---|
| E0 | none (analysis of existing bundles) | < 1 h | everything |
| E1 | none, or one soft `T2_large` twin (1 seed) | < 2 h | E2 |
| E2 | 2 configs × 3 seeds (L=4, L=8) | ~10 h | E3, E4, E6 |
| E3 | 2 configs × 3 seeds + 1 teacher | ~12 h | — |
| E4 | analysis only | ~1 h | — |
| E5 | testbed code + 3 configs × 3 seeds | ~15 h | E7 |
| E6 | analysis only | ~1 h | E7 |

Gates: E0/E1 are read by the PI before E2 is launched (one turn). E2's gate decides whether E3 is worth running.
E5's prediction is committed to this file before its jobs start. Every job: 3 seeds, resumable `run.py`, results
committed per finished seed, one launcher (the PI or the assigned worker, never both).

## 6. Operational lessons carried forward

* Background jobs die with container restarts and nothing inside the container can report it; `run.py` is
  resumable, and "quiet log with no error" is itself the signal to check `uptime`.
* Exactly one party launches jobs per task. Duplicate monitors and duplicate launchers both happened; both cost.
* Every task file states the falsification criterion and the number that decides it before the run.


## 7. E0/E1 actual results (2026-09-15) — what held, what didn't, and the resulting redesign

Full data and reasoning: `docs/tasks/E0-E1-report.md`. Prediction check against §4:

| prediction | outcome |
|---|---|
| 1. ≥40% of gap in stochastic calibration, no feature tracks `switch_possible` | **Wrong on both counts.** Stochastic share is 36.2%. Five features clear the legibility bar for `switch_possible` (best F1 0.827). |
| 2. Soft model reaches ≥95% agreement by L=8, ≥99% by L=16 | **Wrong.** L=8 gives 61.7%, L=16 gives 60.6% — a 49–62% band, non-monotonic, nowhere near either bar. |
| 3–5 (E2/E3/E5 predictions) | Not yet tested; #3 is now suspect given #2's outcome (see below). |

**E0 verdict:** causes (a) capacity / (b) optimization dominate the binary model's gap, not (d) calibration. 58.6% of
the gap is in deterministic rule steps, and 69.4% of *that* is the model's argmax being outright wrong.

**E1 verdict, and the mechanism, which is the important part:** agreement with the soft model never rises above ~62%
at any finite quantization level up to 16, then jumps to 91% (the true ceiling — hard attention alone already caps
it there, not 100%) only once magnitude rounding is removed entirely. This is not "the rule layer needs more
brightness levels" — it is specifically **routing's attention argmax**, computed as a dot product over the code,
that breaks under quantization: telling two candidate source positions apart needs the *difference* between their
scores preserved, which a coarse shared alphabet destroys long before the rule layer would need that precision.

**This changes the plan.** E2 as originally scoped (train at L∈{4,8}, one shared alphabet for routing and rules) is
now a weaker bet: it doesn't separate the two channels E1 just showed need different treatment, and it inherits a
uniform-binning scheme that already looked like a poor fit (non-monotonic in the post-hoc scan). Before committing
training compute to either E2 (redesigned) or E3 (distillation), the next step is E1b below — a cheap, no-training
check of the routing-tie-break mechanism itself, using bundles already on disk.

### E1b — does routing margin predict deterministic argmax errors?  ☐
No training. On `T2_large` (binary, all 3 seeds), for every deterministic rule step: compute the routing head's
attention margin (top attention score minus runner-up, per head, at the position and layer the north-star circuit
trace shows feeds the eventual output feature) and test whether it's smaller on argmax-wrong steps than on
argmax-right steps (e.g. a simple two-sample comparison, held-out F1 of "margin < threshold" as a predictor of
argmax-wrong). Also check the same thing restricted to the `cpy3`-after-switch case T1 originally diagnosed.
* **Confirms the mechanism** if margin is significantly smaller on error steps (and especially on the `cpy3` case).
  → next task gives routing its own finer alphabet in E2 (decoupled from the rule layer's), or moves straight to E3.
* **Refutes it** if margin doesn't predict errors → the mechanism story is wrong or incomplete, and E2/E3 should not
  be redesigned around it without another look — go back to E0's bucket-by-bucket errors for a different lead.

## 8. Standing note on subagent hand-back overhead

During E0/E1, the worker's own turn was repeatedly force-ended by the harness while blocked on a long-running local
script, producing "still in progress, nothing new" hand-backs every few minutes even after being told not to ping
until real numbers existed. Each resume cost real tokens without adding information. Lesson for future tasks: when a
worker reports it's blocked on a specific named background process, the PI should watch that process directly (a
bounded wait on the log/PID) instead of resuming the worker on every forced hand-back — resume only once the PI's own
wait confirms real output exists, or once genuinely new information arrives unprompted.
