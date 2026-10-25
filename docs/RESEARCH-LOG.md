# Research log — interpretable-by-construction language model

A chronological narrative: what we did, what we found, and how each finding changed the next decision. For raw
numbers see `REPORT.md` and `docs/tasks/*-report.md`; for the current plan see `docs/PLAN-v3-observability-frontier.md`.
This file is updated at every phase transition, in order, and nothing here is rewritten after the fact — if a step
turned out to be wrong, that's recorded as its own entry, not edited away.

---

## Origin (before this repo)

A prior chat session asked: if we could redesign a neural network from scratch, how would we build one whose
behavior is legible — where "why did it do that" is answered by the mechanism itself, not a probe fitted afterward?
That session built a sparse-dictionary bottleneck design (`legacy/`) and handed off to this one. First move here was
a supervisor review (`docs/REVIEW-2026-09-12.md`): the handoff's synthetic testbed had only 180 distinct sequences,
and every "held-out" test sequence had actually been seen in training — nothing about generalization had been
measured. That finding threw out the old testbed and the old evidence before building anything new.

## Step 1 — a testbed that can actually test the goal

Built `vlm/testbed.py`: ~19,000 distinct training sequences, mid-sequence rule switches, an exact oracle for every
prediction, and — the part that matters — events deliberately withheld from training (specific rule+argument pairs,
specific switch transitions) so "does it generalize" has a real answer. Also built the measurement harness
(`vlm/metrics.py`): path faithfulness by ablation, held-out legibility, a magnitude-hiding-channel audit, and a
mandatory null model (dense transformer + post-hoc sparse autoencoders) that any new design has to beat, not just
"do okay in absolute terms."

## Step 2 — first redesign: rules over named features (`rulenet` / `rulenet_g`)

Replaced the old dense-vector bottleneck with a design that has no dense vector anywhere: state is a sparse set of
named features, computation is small printable rules ("IF ≤3 features THEN write ≤2"), routing is a sparse lookup
table, output is a separate vote register (so logits are exact per-rule attribution, not a linear-readout fiction).
**Finding**: propositional rules ("IF mode=succ AND cur=7") cannot generalize to a value never seen in that mode —
0% accuracy on held-out arithmetic arguments. **Redesign**: grouped features into value-holding *slots* so a rule can
say "copy their previous-token slot into my slot," one parameter covering every value. This fixed generalization on
copy-type rules (69–100% on held-out arguments) but arithmetic stayed near-zero, which is the correct behavior for a
value nothing in the data ever links across modes — not a bug.

**Finding (the discreteness question)**: even after the slot redesign, reading the model's activations as strictly
on/off reproduced only 50% of its own predictions — continuous *magnitudes* were carrying information the "readable"
description didn't have. Forcing the model to actually run as a discrete program (binary features, hard attention,
trained with straight-through gradients) closed that gap to 100% agreement — the printed rulebook became provably
identical to the model — but cost real accuracy: 0.66 nats (continuous) → 1.17 nats (binary), against a 0.62 floor
and a 0.59 dense-transformer baseline.

## Step 3 — task T1: is the discrete cost fixable with training tricks?

Two guesses, tested over 3 seeds each: more time in the "hard" training phase, and one extra attention head aimed at
the specific weak case found (copying a value from 4 positions back right after a rule switch). **Finding: both made
it worse**, not better — longer hard phase: 1.03 nats; extra head: 1.18 nats, and the extra head hurt its own target
case (91%→71% accuracy on exactly the thing it was meant to fix). This ruled out "it just needs more training time"
and "it's missing one mechanism" as the explanation.

## Step 4 — task T2: is it a capacity limit instead?

Tested a 3-point ladder — half, medium (T1's baseline), and double capacity — 3 seeds each. **Finding: yes, capacity
matters, but with sharply diminishing returns.** Halving capacity cost 0.34 nats; doubling it only recovered 0.07.
The near-switch accuracy specifically jumped from failing on every bucket to clearing 90%+ everywhere at large
capacity — so capacity fixes *specific mechanisms*, not the average uniformly. But doubling the model only closed
19% of the remaining gap to the dense baseline. Scaling the discrete model bigger looked like an asymptote, not a
path back to parity.

## Step 5 — reframing the goal (plan v3)

Went back to first principles on what "observable" actually requires, rather than what had been assumed. The four
goal properties (faithful, legible, complete, compositional) need an *exactly re-executable, declared-alphabet*
program — they do not require the alphabet to be binary. Binary is the L=2 special case of a more general design: a
feature can take a few *declared, discrete* levels instead of a continuous or strictly on/off value, and stay
exactly as readable. This reopened the design space between "fully continuous, 50% readable" and "fully binary,
100% readable but expensive," and set up a frontier to map rather than a single fix to find. Full reasoning:
`docs/PLAN-v3-observability-frontier.md` §2.

## Step 6 — E0: where does the binary model's remaining gap actually come from?

Before training anything new, attributed the lost nats using the oracle already in hand. **Finding: the gap is
concentrated in deterministic-step errors (58.6%), and most of that (69%) is the model picking the wrong answer
outright, not being unsure of the right one.** This ruled out the leading alternate theory (that the model just
can't express *uncertainty* about whether a switch is coming) — a feature tracking "a switch is possible here"
was found to be legible (F1 0.83) on the model's own state, so that information exists internally; the model just
isn't using it correctly to produce the answer.

## Step 7 — E1: how coarse is the "hidden" magnitude channel, really?

Re-executed the trained continuous model at 2, 3, 4, 8, and 16 discrete levels, no training. **Finding: agreement
stays in a 49–62% band at every level up to 16, then jumps to 91% only once rounding is removed entirely** (91%, not
100%, is itself the real ceiling — matching an already-known "hard attention alone" figure). A handful of discrete
levels does not approximate the continuous model well. Along the way, a real bug was found and fixed in the new
quantization code (wrong reference point for non-clamped models, compounded by writing back raw magnitudes instead
of normalized levels) — caught by cross-checking against an already-committed baseline number, which is exactly why
every new metric gets checked against a known figure before being trusted.

**Interpretation, cross-referencing E0 and E1**: the worker's proposed mechanism is that *routing* — deciding which
earlier position to read a value from — is what actually needs fine precision, because telling two candidate
positions apart requires their score *difference* to survive quantization, which a coarse shared alphabet destroys.
The rule layer likely doesn't need nearly as much precision. If true, this reframes the entire discreteness cost as
a routing problem, not a general "not enough levels" problem, and changes what the next experiment should test.

## Step 8 — E1b: the routing-tie-break mechanism, tested directly and refuted

Rather than build a redesigned quantization-aware training run on top of a plausible story, checked it first: do
deterministic-step errors cluster on routing decisions with a narrow margin between the top two candidates?
**Finding: no.** Aggregate wrong-step and right-step margins are statistically indistinguishable (0.0375 vs 0.0380);
a held-out threshold classifier does *worse* than just guessing the majority class. The predicted direction holds on
2 of 3 independently-trained seeds but reverses on the third. Most decisively: on the exact case the mechanism was
built to explain (`cpy3` mode, within 4 positions of a switch), the effect **reverses and gets stronger** — wrong
steps have 2.6x *larger* margins than right steps, the opposite of the prediction. The routing story from step 7 was
wrong. What stands: E0's finding that these errors dominate the gap is untouched — only the proposed explanation for
*why* they happen is dead. This is exactly the kind of guess the project is built to make cheaply and kill quickly,
before it costs a multi-hour training run.

## Step 9 — E1c: two concrete, distinct mechanisms found — arithmetic never learned, and cross-mode vote contamination

Characterized the wrong steps directly instead of guessing a third mechanism. **Mode-wise, the errors concentrate in
the arithmetic-transform modes (pred 28%, pls2 23%, succ 13%) roughly 2-3x more than copy/identity modes (cpy2 8%,
rept 8%, cpy3 14.5%)** — a different pattern from T1's near-switch finding, since this holds even with no switch
involved at all. **Layer-wise, 73% of wrong steps have the correct value completely absent from the state after
layer 0** — most errors originate upstream, not from a later step discarding an already-correct answer. The
rule-table inspection found the specific, decisive fact: **every value-transform table gated on `succ`/`pls2` that
was checked, across all 3 seeds, is still exactly its random-init identity matrix** — the `+1`/`+2 mod 16` arithmetic
these modes need was never learned by any rule using the intended value-map pathway. A runtime firing trace then
showed *why* this doesn't explain everything uniformly: these identity-stuck rules fire on 78.5% of wrong `succ`
steps vs. 8.8% of right ones (a clean causal signal), and — unexpectedly — the same succ-labeled rules also
**cross-fire during entirely different modes** (0% of right `cpy3` steps, 70% of wrong ones) because their firing
threshold is a weighted sum that unrelated boolean co-activations can satisfy alone, injecting a spurious
identity-echo vote that degrades cpy2/cpy3/rept's otherwise-correct mechanism. `pred` and `pls2` share the same
stuck-identity table fact but *not* this firing-rate signature (one fires regardless of correctness, the other
fires *more* on correct steps) — flagged as genuinely unexplained rather than folded into the same story.

**Next hypothesis, stated precisely and falsifiable without training**: ablating the identified identity-stuck rules
from the readout should measurably improve accuracy on succ/cpy2/cpy3/rept specifically, while pred/pls2 should show
little or no improvement — a directly testable prediction, including an explicit prediction of where it will *not*
generalize. Full detail: `docs/tasks/E1c-report.md`.

## Step 10 — E1d: ablating the "bad" rules made things worse (refuted, informatively)

Ran by the PI after the worker hit a rate limit. The first attempt gave a bit-identical null result, which was a red
flag, not a finding. The ablation had missed the second pathway (`Gd` → output groups) by which rules vote. After
fixing it: **removing the identity-stuck rules made `succ` worse in both seeds where any were selected (0.84→0.77;
0.82→0.74).** Copy modes did not recover. So the rules are doing real, specific work, but not the "purely spurious
vote" E1c proposed. With 40+ rules voting per step, a single rule's contribution isn't separable by whole-rule
ablation. The static fact still stands: the arithmetic tables never moved from identity. Proposed cause: the
`l1_table` regularizer pulling them back. Full detail: `docs/tasks/E1d-report.md`.

## Step 11 — E2redux (2 of 3 seeds; seed 2 left unfinished): the regularizer was not the cause

Retrained `T2_large` with `l1_table` = 0, everything else identical.
- **iid NLL did not improve:** 0.785 vs 0.778 (seed 0), 0.970 vs 0.830 (seed 1).
- The rulebook grew 8× (description length 65k vs 8.4k), because the tables were free to move and spent bits doing
  so, but arithmetic still wasn't learned.
- The decisive comparison came from checking the dense baseline on the same metric. **On held-out arithmetic
  (succ/pred/pls2 applied to a value never seen with that mode), every model scores ~0%, the dense transformer
  included.** "Arithmetic not learned" was never a cost of observability. Any lookup representation fails this way:
  a table cell that never appears in training never gets a gradient.
- Generalizing to unseen arguments needs an operator *shared across values* (shift-by-n on a cyclic code). That is
  also the most legible representation there is.
- Two side findings:
  - The structured model already beats dense out of distribution (shift NLL 1.41 vs 1.84; held-out `cpy3` accuracy
    0.96 vs 0.50) while losing in distribution (0.78 vs 0.59).
  - The iid oracle floor is slightly loose. Dense reaches 0.593 vs the oracle's 0.617, because the oracle ignores
    the rejection sampling that removes held-out events from training-distribution sequences.

Seed 2 was not finished. The user redirected the project before it completed (next step), and the two finished
seeds agree.

## Step 12 — two kinds of visibility (Graepel, MIT Tech Review, 2026-10-02)

Read "Don't be fooled — LLMs don't reason". It argues for an explicit, inspectable epistemic state plus an
independent evaluator. Agreed with the user on a distinction:
- **mechanism visibility** (what the model computed: this project);
- **epistemic visibility** (what it believes and why: his).

They are complementary. The second is parked, with one constraint: nothing we build may rule it out.

## Step 13 — pivot: from fixing rulenet to redesigning from the goal backwards

The user asked for a radically different approach: start from "as performant as current models *and* observable",
decompose what must be true, and design candidates that meet every criterion on paper before testing. Output:
- `docs/redesign/DESIGNS.md`: criteria C1–C7, a requirement tree down to building blocks, and three designs with
  criterion-by-criterion tables:
  - A, Glass Box: an explicit sparse named feature bus; every nonlinearity is a logged decision;
  - B, Library & Clerk: knowledge outside the weights, honest citation enforced by counterfactual edits;
  - C, Planner & Voice: content planned in the open, wording confined to a bounded, non-interference-tested channel.
- `docs/redesign/THINKING-LOG.md`: the reasoning, the parallels and the rejected approaches.

The key reframing is that dense models are already *decomposable*. What they lack is a named basis, few active
terms, and a guarantee the named terms are everything. Every design supplies those three.

The lessons from Steps 1–11 became design inputs:
- keep values continuous, never binarize (Steps 3–4);
- declare and enforce magnitude precision (Step 7);
- use AND-gating instead of weighted sums (Step 9);
- use equivariant typed operators instead of free tables (Step 11).

First test planned: T-A3 (typed shift operator on the current testbed). Second: T-A0 (does an explicit sparse
residual cost much on *real* text?), which tests the riskiest assumption in the program.


## Step 14 — the designs meet their critics: observability becomes a curve

Two independent critics reviewed the v1 designs.
- **Observability critic.** v1's "by construction" guarantees were vacuous: a dense model that simply logs its
  numbers passes them.
- **Performance critic.** v1's small, re-cut-every-layer working memory would likely starve the model.

Both point the same way. An explanation is a description *shorter* than the thing it explains that still
reproduces it. So observability is the **explanation frontier** (description length vs fidelity), measured against
a dense model plus the best post-hoc tools. It is not a yes/no property of the substrate.

v2 (`docs/redesign/DESIGNS.md`) therefore:
- lets the computation be as large as performance needs (big persistent named bus);
- hard-quantizes every logged number at inference;
- trains for short descriptions;
- folds the library and planner/voice designs into the core model as views rather than separately taxed stages.

The test plan now starts with three cheap diagnostics on one small dense model trained on real text: the post-hoc
bus tax, the precision tax and the attention tax. Then a from-scratch ladder adds one constraint at a time.


## Step 15 — two more critique rounds: the Ledger Transformer (design v2.2)

**Round 2.** Both critics independently found v2's weak block: re-compressing the model's state into a named list
after every layer. It was costly, lossy, and made names ambiguous, since many lists fit one vector. The fix came
from asking why we decode at all. A name can come from **provenance** instead. If every write into the residual
stream comes from a logged decision (a unit firing, an attention head choosing sources), the stream is *exactly*
the sum of named writes by bookkeeping. Many lists fit one vector, but only one history produced it.

**What that makes the model.** Architecturally, a wide-sparse (PEER/mixture-of-experts-style) transformer whose
every nonlinearity is logged on an 8-bit grid, which is close to designs known to perform. The novelty moves into:
- the training objective: pressure toward short descriptions;
- the audit: a fixed description language run by an independent interpreter, a held-out second language, a dense
  control given the same pressure, and planted smugglers.

**Round 3.** No fatal issues. The major refinements:
- per-unit firing thresholds, so competition between units isn't a hidden coupling;
- normalization as a described node;
- completeness measured causally, with the model as its own decoder;
- a smuggler test scaled down to 0.01 bits/token, with slow, rare-trigger and keyed smugglers;
- a style cap on style→content *flow* only;
- a cheaper, scale-free pressure term;
- a compute-feasible test plan that puts the one real risk first.

**Where this leaves the program.** One empirical question remains, and paper can't answer it. Is there a pressure
level λ with ≤ 10% performance tax at which the model's explanations are ≥ 10× shorter, for equal fidelity, than a
dense model given the same pressure and described by the best post-hoc tools? Next experiment: T-A0 stage 1
(dense → PEER → PEER + three λ values, on real text, CPU scale; a necessary test, not a scale claim).


## Step 16 — the first real-text experiment is built (T-A0 stage 1), waiting for a GPU

The design reduces to one question: does an "explain yourself" pressure buy much shorter explanations at little
performance cost? The code to test it is in `lm/`, and the brief, with predictions written before any run, is
`docs/tasks/TA0-stage1.md`.

**What was built:**
- a pinned 46 MB real-text corpus (prose + Python code);
- a normal transformer and a specialists (PEER-style) transformer, both recording every write into the residual
  stream;
- the pressure term;
- an evaluator that replays the model from a truncated explanation and measures how far its predictions drift.

**Checked on CPU:**
- every job type runs end to end;
- interrupted runs resume;
- at full budget the replay reproduces the model exactly (KL ≈ 1e-8);
- the full-size configuration fits in memory (≤ 4.4 GB).

**Fixes made during testing:**
- a NaN from differentiating a vector norm at zero;
- an out-of-memory error from computing the dense model's pressure on too many positions.

A full-size step takes 5–10 s on these 4 cores, about 15 hours per run, so the 18 runs go to a GPU.


## Step 17 — on GPUs: the self-check catches a compiler bug before any real run

We moved to Modal GPUs (NVIDIA L4) and ran smoke tests first. Training matched CPU. The explanation evaluator's
built-in check did not. At full budget, a replay of the specialists model must reproduce it exactly: KL ≈ 1e-8 on
CPU, but **0.6 on the GPU**.

**Diagnosis:**
- Run step by step, the replay matched the model to 2e-6.
- Compiled as one function, it diverged by up to 2.0 in output scores, starting at the second layer's attention
  output.
- Switching off one GPU compiler optimization, XLA's Triton GEMM fusion, made the compiled version match exactly
  (difference 0.0).

That optimization is now disabled for **every** GPU run, training included. With it off, the check passes on the
GPU (KL ≈ 1e-8), and GPU explanation numbers agree with the CPU smoke runs.

Without the built-in check, every explanation number from the GPU would have been silently wrong. This is lesson
L8 again: build the check into the instrument. All 18 stage-1 trainings were then launched in parallel, one L4
each.


## Step 18 — T-A0 stage 1 results: the substrate helps, the pressure was gamed

18 trainings ran on Modal L4 GPUs (≈ 4.6 GPU-hours). Full detail: `docs/tasks/TA0-stage1-report.md`.

**The good news: the specialists substrate is far more explainable, at no performance cost.** The published PEER
design (many small units, few active) beat a normal transformer per FLOP: 1.32× dense-equivalent compute. Its
explanations are also far more faithful:
- at the most generous budget tested, a replay through its strongest connections reaches KL 0.04;
- the normal model's best is 0.61, at twice the description length;
- the normal model never reaches KL 0.3 anywhere on the grid.
The comparison is still against raw neurons; a transcoder control is owed.

**Two of our additions failed:**
1. **Per-unit sigmoid thresholds**, meant to remove coupling between units, cost ~4× compute (+0.27 bpb). 59% of
   units were never used, and the top-k cap still decided firing 30–40% of the time.
2. **The explanation pressure did not shorten explanations.** Stronger pressure made them *worse*, and the pressured
   models became fragile to any truncation.

**Mechanism, found by testing two hypotheses on the trained checkpoints:**
- *Refuted:* "information moved into spread-out attention". Attention spread was unchanged across pressure levels.
- *Confirmed:* "the pressure was gamed by cancellation". The pressure asked whether 8 writers carry almost all of
  each read's gross contribution. The model complied by making a few contributions huge and opposite in sign:
  - on the output read, gross contribution grew 7× (16.5 → 112.6) while the net stayed ~20;
  - the cancellation ratio rose from 1.2 to 5.4 (7.3 for the pressured dense model).
  Large canceling terms make the computation fragile, so any small upstream error swamps the small difference that
  carries the signal. The observability critic predicted exactly this in round 1.

**Lesson.** A proxy for "explainable" that can be satisfied by rescaling will be gamed. The pressure has to be on the
real target: does a short description reproduce the output?

**Redirect (pre-registered decision rule: prediction 4 failed):** stage 1b, built from these pieces:
- softmax PEER as the base, with its normalizer described explicitly;
- truncation self-distillation: penalize KL between the model and its own top-m truncated replay;
- a cancellation-penalty variant for comparison;
- the fair dense + transcoders control.

---

## What this log is for

Each entry above follows the same shape: a decision, made for a stated reason; what was actually measured; and what
that measurement changed about the next decision. The shape is deliberate — negative results (T1's two failed
fixes, the wrong predictions in step 6/7, the bug in step 7) are kept as entries, not edited out, because a wrong
guess that narrows the space is real progress, and the point of this file is to make that narrowing visible end to
end rather than only showing the steps that worked.
