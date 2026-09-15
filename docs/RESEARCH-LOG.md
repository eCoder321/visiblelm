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

---

## What this log is for

Each entry above follows the same shape: a decision, made for a stated reason; what was actually measured; and what
that measurement changed about the next decision. The shape is deliberate — negative results (T1's two failed
fixes, the wrong predictions in step 6/7, the bug in step 7) are kept as entries, not edited out, because a wrong
guess that narrows the space is real progress, and the point of this file is to make that narrowing visible end to
end rather than only showing the steps that worked.
