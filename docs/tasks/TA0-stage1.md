# T-A0 stage 1 — does explanation pressure buy short explanations cheaply?

**Design under test:** Ledger Transformer, `docs/redesign/DESIGNS.md` v2.2, §7 and §11.
**Code:** `lm/` (see `lm/README.md`).
**Status:** code written and smoke-tested on CPU (2026-10-24); waiting for a GPU.

This brief is written **before any results exist**, so the predictions below can be wrong.

## The question
Is there a strength λ of the description-pressure term at which:
- the specialists model ("PEER": many small units, few active) loses little performance, **and**
- its decisions become much cheaper to explain?

This is the one empirical question the redesign reduces to (§13). Stage 1 is a **necessary** test at small scale. A
pass is not a scale claim. A clear fail is informative.

## Setup
- **Data:** about 46 MB of real text, byte-level: 34 MB prose (books, news, speeches, web text), 11.6 MB Python
  code. It is pinned and checksummed (`lm/data.py`). Validation is scored separately on prose and code.
- **Models:** d=256, 4 layers, 4 heads, context 256, 8,000 steps × 32 × 256 bytes (65M bytes, ~1.4 epochs).

| job | what it is | why it's here |
|---|---|---|
| `dense_d128`, `dense_d192`, `dense_d256` | normal transformers at three widths | the ruler: converts a loss difference into "how much more compute a normal model would need" |
| `peer_softmax` | PEER as published (softmax router) | the strongest no-observability baseline in the specialists family |
| `peer_sig_lam0` | PEER with per-unit thresholds (sigmoid gates), no pressure | **the base**. Isolates the cost of removing the router's coupling. |
| `peer_sig_lam{0.03, 0.1, 0.3}` | the same plus explanation pressure | the λ curve |
| `dense_d256_lam0.1` | normal transformer with the same pressure | a first "pressured control": does pressure alone, without the architecture, buy the same thing? |

- **Seeds:** 2 per job, **paired**: the same seed sees the same batches in every job. Seed noise (~0.02 nats) is
  comparable to the effects we care about, so all tax comparisons are paired differences.
- **Pressure:** for sampled reads, the share of gross contribution mass outside the top-8 writers. The reads are
  unit routing and activation inputs, head value inputs, and the next byte's logit (DESIGNS.md BB10).
  - It is ramped in from 10% to 30% of training.
  - It is sampled on 1/16 of positions for PEER and 1/128 for dense. Dense has ~1,000 neurons per layer as
    writers, and full sampling ran out of memory. The estimate is unbiased, just noisier.

## Metrics
- **Performance:** validation bits per byte. Reported as paired Δ vs `peer_sig_lam0`, and as a dense-equivalent
  compute multiple from the three dense widths.
- **Explanation:** the local frontier (`lm/frontier.py`).
  - **Budget:** keep the n strongest units per layer, the m strongest writer contributions per read, and the m_a
    heaviest attention sources.
  - **Replay:** re-run the model from the input through only those edges.
  - **Score:** KL from the real model's next-byte distribution, against description bits per byte position.
  - **Headline numbers:** *bits needed to reach mean KL ≤ 0.1* and *≤ 0.3 nats*, separately for prose and code.
  - **Correctness check:** at full budget the replay must reproduce the model. On CPU at full size it does, at
    KL ≈ 1e-8 for PEER. Dense keeps at most 256 candidate neurons per layer, so its full-budget KL is ≈ 6e-3, which
    is reported.
- **Health:**
  - dead-unit fraction and usage entropy;
  - cap-binding rate: how often the top-k limit, rather than a unit's own threshold, decides firing. DESIGNS.md
    v2.2 wants this rare. Stage 1 only measures it.
  - loss spikes.

**Converting "10% compute" into loss.** Read it off the dense fit's slope: Δbpb₁₀ ≈ −slope × ln(1/0.9). At this
scale we expect that to be around 0.01 bpb, i.e. at the seed-noise level, which is why seeds are paired.

## Predictions (written before running)
1. **Thresholds are cheap.** `peer_sig_lam0` is within 0.02 bpb of `peer_softmax`.
2. **Specialists vs normal.** PEER (λ=0) has a dense-equivalent compute multiple ≥ 1. It uses far fewer FLOPs per
   byte, so per FLOP it matches or beats dense, as the PEER paper reports.
3. **The tax rises smoothly with λ.**
   - λ=0.03: within 1× the 10%-compute loss.
   - λ=0.1: within 3×.
   - λ=0.3: may exceed the 30% band.
4. **Pressure buys explanation length.** bits@KL≤0.1 falls monotonically with λ. At the largest λ still inside the
   10–30% band, it is **≥ 2× lower** than at λ=0, on both prose and code.
5. **The architecture matters, not just the pressure.** At equal KL, `peer_sig_lam0.1` needs **≥ 3× fewer bits**
   than `dense_d256_lam0.1`. This is the small-scale, necessary version of the design's ≥ 10× target against a
   pressured control.
6. **Health.** Dead-unit fraction < 50% at λ=0 and does not rise by more than 15 points under pressure. Cap binding
   is reported, with no prediction (first measurement).

## Decision rules
- **Proceed to stage 2** (8-bit logged decisions, sparse attention, style cap, at the chosen λ) if predictions 4
  and 5 hold.
- **Redirect, before stage 2,** if 4 fails, i.e. pressure doesn't shorten explanations within the tax band. Then:
  - the pressure *form* is wrong: try KL-greedy edge selection in the evaluator, a different m, pressure on attention
    queries/keys;
  - or the evaluator is too crude, which needs fixing before trusting any frontier.
- **Kill (at this scale) the "architecture adds something" claim** if 5 fails badly: < 1.5× even at the best λ. That
  would mean a normal model given the same pressure explains about as well, so the specialists architecture isn't
  what buys observability. Report it, and re-examine before scaling up.
- **Health failures** (dead fraction > 70%, loss spikes, NaNs) are fixed before any result is interpreted.

## Known limitations (stated up front)
- **Small scale.** Tiny models on a 46 MB corpus use few features and are easy to describe, so a pass here does
  not predict scale. Confirming needs a width trend (stage 3) and, eventually, larger models.
- **Crude edge selection.** Edges are ranked by contribution magnitude in the full run, not chosen to minimize KL.
  Edge weights come from model parameters. This is the *local* frontier, not yet the global 𝓛 interpreter. It
  applies equally to every model.
- **The dense control is not yet the full "dense + transcoders + pressure fine-tuning" control** from DESIGNS.md.
  It is raw neurons with pressure from scratch. Stage 1 uses it only for the necessary-condition test (prediction
  5).
