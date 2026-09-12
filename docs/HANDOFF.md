# HANDOFF — Interpretable-by-construction language model project

**Written:** 2026-09-12. **From:** a claude.ai chat session (compute-limited). **To:** a Claude Code cloud session that will take over experiments.
**Read this file first. Then `interpretable_lm_design.md` (the report). Then the code.**

The person you are working with wants to continue *this* project without going back and forth between chats. Treat this document as the shared memory. Everything below was actually done and measured unless explicitly marked as a plan or a negative result.

---

## 0. How to work with this person (from the chat)

- They asked for **first-principles reasoning on every request** — derive from fundamentals, don't recite industry patterns. When I fell back on transformer conventions they called it out (correctly).
- They want **honest status**, including when something is broken or hasn't run. They caught me twice saying "it's still running in the background" when the process had actually been killed. Never claim a run is alive without checking.
- They asked to **not receive obfuscated blobs** (I once sent code as a base64 self-extracting bundle for phone-pasting convenience; they flagged it). Keep code readable.
- They're often on a **phone**: short status updates, lead with the answer. Long deliverables go in files.
- They asked to be told **what they can do to help** when there are blockers.
- Their standing instruction when I started building: *"test this, improve it till we have a good design which can be used to train at least a small language model — don't stop till you achieve this."* Then: *"put your detailed analysis/steps and findings in an .md file."*

---

## 1. How this project started (conversation arc)

1. They shared a Forbes article (2026-09-09): an Anthropic pretraining researcher (Jacob Coxon) resigned saying labs are racing toward self-improving superintelligence they can't control; Anthropic's alignment science lead (Evan Hubinger) publicly agreed, put >10% on AI killing all humans within a decade, and said Anthropic doesn't yet have a plan to solve alignment for superintelligence. They asked *why* alignment is hard.
2. I gave a first-principles breakdown: (1) values resist specification; (2) optimizing a proxy diverges from the target (Goodhart); (3) we can mostly only check behavior, not internal computation; (4) oversight breaks once the system outpaces the overseer; (5) power-seeking is instrumentally convergent; plus race dynamics.
3. They pushed on (3): *"Do you believe we can only check behavior, not what's happening inside?"* I corrected myself: that's an empirical maturity limit (interpretability is partial), not a principle. Then: *"what do we need to see in order to check behavior?"* Answer: a target spec; coverage of the situations that matter (unobserved, costly-to-comply); a competent judge; and **a warranted basis for generalizing from tested to untested cases — which is a claim about mechanism.** So behavior-checking that generalizes needs mechanistic grounding; the behavior/internals dichotomy collapses.
4. They asked: *"If we redesigned a neural network from scratch, how would we design it to have visibility into how its behavior forms? Restate the goal, propose the ideal design, then test and improve it until it can train a small LM."* → Sections 2–5.
5. After results, they asked for a plan to fix the remaining opaque part (dense computation inside each layer), then asked deep questions about the sparse-feature representation (who defines features, what bounds Ω, can it learn new concepts, is there a better way). → Sections 6–7.
6. Compute became the blocker (single CPU core, background jobs killed between turns). We evaluated GitHub Actions and Claude Code; they chose a **Claude Code cloud session** and asked for this handoff.

---

## 2. The goal (restated, agreed)

Build an LM where "why did it produce this output?" is answered by **the computation itself**, expressed in a fixed human-readable vocabulary, verifiable by intervention, with no hidden channel — so behavior on *untested* inputs can be predicted from mechanism.

Four properties, all required at once:

| | Property | Test |
|---|---|---|
| P1 | **Faithful** — the legible representation *is* the causal path | Ablate the named features → output changes; ablate others → it doesn't |
| P2 | **Legible** — states are expressed in a small fixed set of nameable units | Each unit maps to one identifiable concept (scored vs ground truth where available) |
| P3 | **Complete** — no dense side channel | All information crossing a layer is carried by the legible units, by construction |
| P4 | **Compositional** — outputs decompose into units, units into earlier units | Additive attribution; traceable causal graph input→logit |

---

## 3. Design derivation (first principles)

- **D1** Interface between layers = top-k, non-negative code over a dictionary of unit-norm feature directions. No bypass. (P2, P3)
- **D2** The embedding is also a code — input enters through the same legible channel. (P3)
- **D3** Readout is linear in the final code: `logits = code · W` → exact per-feature attribution. (P4)
- **D4** Tied encoder/decoder: one direction = one meaning. (P2)
- **D5** Faithfulness measured by intervention (ablation), never correlation. (P1)
- **D6** Legibility measured against **ground truth** → primary testbed is a synthetic language with known latent causes; real text is secondary.
- Everything inside a block (attention, MLP, LN) was left standard **on purpose**, to isolate the interface problem first. That is the acknowledged remaining opacity (§6).

Later additions: **v3 code-space residual** (block writes a sparse signed delta in feature space; `code_{l+1} = topk(relu(code_l + Δ))`, persistence by default; `k_write` limits how much a layer may rewrite) and **AuxK** (dead features reconstruct residual error).

---

## 4. Testbeds, metrics, implementation

**Synthetic language:** vocab = 4 mode tokens + 12 content tokens. Sequence = mode token, 2 random content tokens, then 13 tokens by the mode's rule: repeat / +1 / copy-2-back / −1. Deterministic from t≥2. Entropy floor ≈ **0.205 nats/token**. Ground-truth predicates: `mode=m, cur=v, prev=v, prev2=v, next=v, pos=t, mode&cur`.

**Real text:** char-level Tiny Shakespeare (65 chars), T=64, 60k windows, 3000 steps, d=96, 3 layers, dict 384, k=12.

**Metrics** (held-out): interpretability tax (NLL vs dense twin); faithfulness (Δ target log-prob when ablating top-3 attributed features vs 3 random active features vs keeping *only* top-3); completeness (top-3 share of target logit); legibility (best-F1 vs any single ground-truth predicate; % live features with F1>0.8); dead-feature rate; layer invariance (same best label at every layer?); recursive circuit trace (single-node ablation, parents = drop in child activation).

**Stack:** JAX + Optax, CPU. PyTorch could not be installed (CUDA wheels filled disk; pytorch.org index blocked). The code is small and portable.

---

## 5. Everything that was run, with numbers

### v0 dense baseline (synthetic): NLL 0.2058 (floor). Control.

### v1 bottleneck (dict 128, k=8, shared, tied, linear readout)
NLL **0.2053** (no tax). Dead 45/32/24%. Ablate top-3 / random-3 / keep-only-top-3: **−1.73 / −0.22 / −0.035**. Completeness 0.62. Clean features 9% (median F1 0.49). **Discovered exact ground-truth features** (e.g. `mode=repeat & cur=10`, F1=1.00). Per-layer dictionaries ≈ same numbers.
Bugs found: anti-dead loss written on `(code>0)` → zero gradient → silent no-op. Late LR divergence → warmup+cosine+clip.

### v2 AuxK (differentiable dead-feature loss)
| | v1 | v2 k=8 | v2 k=4 |
|---|---|---|---|
| NLL | 0.2053 | 0.2216 | 0.2123 |
| dead (L1/L2) | 32/24% | **4/9%** | 5/34% |
| ablate top/rand/keep | −1.73/−0.22/−0.035 | **−8.45/−1.08/−0.012** | −4.45/−2.27/−0.002 |
| completeness | 0.62 | **1.10** | 1.09 |
| clean final feats | 9% | **19%** | 19% |
**Negative:** k=4 did not improve legibility; random ablations became far more damaging (each feature more load-bearing).

### Circuit trace (v2) — works, and exposed a flaw
Copy-2 mode predicting token 10 at pos 9: `bn2 pos8 f8 [next=10]` ← `bn1 pos8 f8` (same feature carried) and `bn1 pos8 f26` ← `bn0 pos1/3/5 f58 [cur=10]` — attention to same-parity positions, i.e. the period-2 mechanism, readable. Succ/pred modes trace back to **position 0 (mode token)** via attention.
**Negative:** shared dictionary ≠ stable meaning. Features with same label at every layer: v1 23%, v2 **3%**. Directions are reused for different concepts at different depths because each block can rewrite the residual and the bottleneck re-encodes from scratch.

### v3 code-space residual
| | v2 | v3 k_write=8 | v3 k_write=4 |
|---|---|---|---|
| NLL | 0.2216 | **0.2069** | 0.2640 |
| ablate top/rand/keep | −8.45/−1.08/−0.012 | −2.79/−0.70/−0.003 | −3.71/−1.29/−0.116 |
| completeness | 1.10 | 0.86 | 0.95 |
| clean final feats | 19% | 7% | **27%** |
| same label all layers | 3% | **27%** | **56%** |
`k_write` = a clean dial: how much a layer may rewrite, traded against capacity. Final-layer features were polysemantic on the output side ("next=5 or next=7 depending on mode") — the linear readout permits it.

### v4 sparse per-feature readout (each feature votes for ≤ m tokens) — **negative**
Hard top-m mask: stuck at 2.29. Straight-through estimator: stuck at 2.49 (≈ ln 12 = uniform). Gradient about *which* token to predict only enters through ≤ 2·k logit entries. Not retried. Likely fix: anneal m from V downward, or larger m. **Parked.**

### Real text
| model | NLL nats/char | bits/char |
|---|---|---|
| dense (d=96, 3L) | **1.633** | 2.36 |
| v2 bottleneck (dict 384, k=12) | **2.029** (still improving at 3000 steps) | 2.93 |
| v3 code-residual k_write=12 | 2.269 @ step 1000 (run killed) — ahead of v2 at every logged step (2.438 vs 2.534 @500; 2.269 vs 2.340 @1000); dense was 1.866 @1000 | — |

v2 text faithfulness: base −2.23; ablate top-3 **−10.39**; random-3 −3.00; keep-only-top-3 **−1.96** (better than base — other 9 features add noise). Completeness 1.18. Dead 45/6/2/0%. Legibility vs hand predicates weak (2–5% clean) but clean ones are unmistakable: embedding features for `cur=' '` (0.97), `cur='t'`, `cur=','`, `cur='a'`, `cur='\n'`; final layer has *several distinct* "at a word boundary" features the predicate set can't distinguish.

### Summary of what's established
1. Faithfulness by construction — holds on synthetic and text.
2. Zero tax on synthetic; **+0.4 nats/char on text** at this scale (partly convergence, partly capacity). v3 pays less of it.
3. True causal variables get learned as features without supervision.
4. Mechanisms trace as a sparse causal graph.
5. Negatives: sparsity alone ≠ legibility; shared dict ≠ stable meaning; sparse readout didn't train.
6. Every legibility constraint had a gradient-flow failure mode first.

**Caveat on all of the above: single seed per config.** Differences like 7% vs 19% vs 27% clean features are within plausible seed noise. Nothing counts until it holds over ≥3 seeds. This is the first thing to fix (Phase 0).

---

## 6. Conceptual answers given (so you don't contradict them)

- **Who defines features?** The model; dictionary directions are learned. Labels were assigned *afterward* by matching to ground-truth predicates. Format is fixed, vocabulary is discovered.
- **Can it learn concepts it wasn't taught?** Yes — it did (`mode=succ & cur=6`; several unnamed word-boundary features on text). An unnamed feature is observable and intervenable but not yet understood; legibility scores drop, observability doesn't.
- **What bounds Ω?** Dictionary size F (a hyperparameter) and learned directions. Limitations: (1) Ω is capped → make the dictionary growable (dead features are the free pool; AuxK error is the minting signal); (2) **the state is a set, not a structure** — no binding of roles to values; flat features can't say "subject=X, object=Y" without combinatorial blowup → add **provenance pointers** on routed features so the state becomes a sparse graph; (3) continuous strengths could be a side channel → audit; (4) no superposition = no free lunch; that's the tax.
- **Better ways?** Sparse named features are the best-known *format* for "one unit, one meaning, verifiable," but a floor not a ceiling. Directions: definitional hierarchy (atoms defined as rules over atoms — model writes its own glossary); text-as-state (Ω = language; faithful only with no hidden state; per-step computation still opaque); hybrid: model self-labels features, labels verified by intervention.

---

## 7. The plan for the remaining opacity (inside each layer) — agreed with the person

**Principle:** exposing activations inside a dense neuron gives numbers, not meaning. The primitive must be a **rule**: `IF <few named features> THEN <write few named features>`. A layer is a rulebook; what we read is what runs.

A layer must do only two things: (1) compute new features from features at this position; (2) move features between positions.
1. **Compute → conjunctive rules.** Fan-in ≤3, fan-out ≤2, AND via min/product of non-negative strengths, learned bias threshold, signed writes (can erase). Structural constraints, not penalties. Printable: `R17: IF mode=succ AND cur=6 THEN write next=7`.
2. **Route → feature lookup.** Sparse feature-pair compatibility table (match), relabel table (copy f as g), explicit relative-distance features (`dist=1, dist=2, dist=any`). Printable: `H3: WHEN cur=* LOOK FOR dist=2 AND cur=X, COPY cur=X AS prev2=X`. Every routed feature carries **provenance** ("from position s").
3. **Persist** — keep v3 code-space residual.
Depth stays (finite, readable traces).

**Trainability (the v4 lesson):** start dense, anneal to sparse (relaxed Bernoulli / temperature-annealed soft-top-r), snap to hard rulebook at the end, then **verify by ablation that the hard form reproduces the soft one** — the design's own honesty test.

**Improvements added after discussion:** open Ω (growable dictionary); provenance pointers; **intervention-defined legibility** (a feature is "understood" if an auto-generated description predicts, on held-out data, when it fires and what ablating it does — no hand-written predicates); anti-smuggling audits (probe magnitudes; description length per prediction to catch "1000 individually legible rules that collectively aren't"); north-star test moved to Phase 1; ≥3 seeds; plateau/matched-compute comparisons; stronger faithfulness control (ablate top-3 by *activation magnitude*, not attribution — already in `run_job.py`); gradient smoke test for every new constraint; time-box each idea (one attempt + one principled fix); stochastic synthetic rules; small fast text config for iteration.

### Phases (each gated)
- **Phase 0 — Testbed & harness.** Compositional *and* stochastic synthetic rules; 3-seed protocol; gradient smoke tests; magnitude-ablation control; held-out distribution-shift split; fast text config. *Gate:* v3 re-run reproduces prior numbers across seeds.
- **Phase 1 — Rule-bank MLP + open Ω.** Replace MLP with fan-in-3 rules under annealed sparsity; growable dictionary. *Gate:* printed rules match generating rules; rule ablation breaks exactly what it claims; **predict behavior on the shifted split from the rulebook alone, then run and score.**
- **Phase 2 — Lookup routing with provenance.** *Gate:* routing rules read as "look at mode token" / "look 2 back"; state graph exposes binding.
- **Phase 3 — Compose + audits.** *Gate:* tax vs v3 across seeds; % predictions explained by an end-to-end readable chain.
- **Phase 4 — Real text with intervention-defined legibility.** *Gate:* tax at plateau; legibility score; predict-before-run on a held-out text domain.
- **Phase 5 — Scale probe.** 3–5 sizes: does the tax shrink or grow?

---

## 8. Infrastructure lessons (so you don't repeat them)
- The chat sandbox had 1 CPU core, ~3 GB RAM, no GPU; container restarted between turns → background jobs died. Text runs took 45–60 min. **Use the cloud session's compute; keep jobs restartable; write checkpoints/results to disk as you go.**
- `pip install torch` pulled CUDA wheels and filled the disk; JAX CPU wheels were fine.
- Two self-inflicted process bugs: `pgrep -f run4.py` inside a shell whose own command line contained "run4.py" matched itself (waiters hung); `pkill -f run5.py` killed the calling shell. Use PIDs.
- Non-differentiable losses fail silently. **Always smoke-test that a new loss term changes the gradient.**

---

## 9. Code inventory (all plain Python, in `code/`)
- `itlm.py` — model (dense / v1–v3 bottleneck / v4 readout via cfg flags), synthetic data, training loop (AdamW, warmup+cosine, clipping, dictionary renorm), AuxK. Key cfg keys: `bottleneck, share_dict, bottleneck_emb, dict, k, k_aux, code_residual, k_write, sparse_readout, readout_ste, ortho`. `base_cfg(**overrides)`.
- `analyze.py` — `get_codes, dead_features, feature_latent_alignment, attribution, ablation_test, completeness, full_report, predicates, feature_f1, legibility_report, trace_circuit, trace_recursive, print_tree, layer_invariance`.
- `analyze_text.py` — `text_predicates, text_report`.
- `textdata.py` — Tiny Shakespeare loader (expects `shakespeare.txt` in cwd; `run_job.py` downloads it).
- `run_job.py` — **generic multi-seed job runner**: `python run_job.py job.json` or `JOB_JSON=... python run_job.py`. Writes `results/<name>/seed*.json` + `summary.json` (mean/std). Includes magnitude-ablation control. Job spec documented in its docstring.
- `run2.py … run5.py, run_text.py, run_text2.py` — the historical scripts that produced §5 (kept for provenance).
- `results/*.json`, `*.log` — raw outputs behind the numbers in §5 and the report.
- `interpretable_lm_design.md` — the full report (same numbers as §5, more narrative).
- `github_actions/` — a workflow + README for running jobs on GitHub Actions (built, tested locally, **not deployed**; superseded by the cloud session, kept in case useful).

---

## 10. First tasks for the cloud session (in order)
1. `pip install jax jaxlib optax numpy` (or torch if you prefer and have the disk — porting is small). Run `python run_job.py` with `{"name":"smoke","testbed":"synthetic","cfg":{"code_residual":true,"k_write":4},"steps":60,"seeds":[0,1]}` to confirm the harness.
2. **Phase 0a — replicate across seeds:** run v1, v2(k=8), v3(k_write=8), v3(k_write=4) on synthetic with `seeds:[0,1,2,3,4]`, 1500 steps. Report mean±std for NLL, ablation triple, clean-feature %, layer invariance. Decide which §5 conclusions survive.
3. **Phase 0b — finish the text comparison properly:** dense vs v2 vs v3 (k_write=12) to plateau (or matched compute), 3 seeds. Include the magnitude-ablation control.
4. **Phase 0c — build the new testbed:** compositional + stochastic rules with known mechanism, plus a distribution-shift split. Document the generating rules so "printed rule matches generating rule" is checkable.
5. Then Phase 1 (rule-bank MLP with annealed sparsity + growable dictionary). Time-box: one attempt, one principled fix.
6. Keep `interpretable_lm_design.md` as the living report; append a dated changelog section per phase. Report status to the person in short, honest updates: what ran, what the numbers are, what's blocked.

---

## 11. Condensed transcript (user turns and the substance of each reply)

**Full version: `TRANSCRIPT.md` in this package has every user prompt verbatim with a summary of each response. This section is the short index.**

1. **U:** Forbes link; why can't labs align AI? — **A:** news summary + 5-point first-principles breakdown (see §1.2).
2. **U:** Do you believe "we can only check behavior"? What do we need to see to check behavior? — **A:** corrected the claim (maturity limit, not principle); four requirements; generalization needs mechanism, dichotomy collapses.
3. **U:** Redesign a NN from scratch for visibility; restate goal; propose design; test and improve until it trains a small LM; put everything in an .md. — **A:** goal P1–P4; design D1–D6; built JAX implementation; v0/v1 results; faithfulness passes.
4. **U:** Continue. — **A:** v2 AuxK; circuit trace; shared-dict negative result; v3 code-space residual; v3 results; v4 sparse readout attempt; text runs started; report drafted.
5. **U:** No updates, two failures — still progressing? Blockers? Help? — **A:** blocker is single-core compute + per-turn limits; offered "deliver now" vs wait.
6. **U:** Continue. — **A:** text v2 analysis; report delivered with code.
7. **U:** Brief status for someone who hasn't read the report. — **A:** what/where/what's left summary.
8. **U:** It's been hours, still running? Also: plan for the dense function inside each layer, pure redesign, only constraint is observability. — **A:** admitted the run had died (container restart); the rule-based layer plan (§7).
9. **U:** How would you improve the plan? Who defines features / what bounds Ω / can it learn new things / better way? — **A:** §6 and the improvements in §7.
10. **U:** Longer workloads — connect to something? Other adjustments? Update phases? — **A:** options (Claude Code local, cloud VM, GitHub Actions); further adjustments (seeds, convergence confound, magnitude control, smoke tests, time-boxing, stochastic testbed); Phases 0–5.
11. **U:** GitHub Actions (can't use Claude Code, dies with laptop; wants phone). — **A:** built workflow + runner + bundle; phone instructions.
12. **U:** Can't I connect you to GitHub / Claude Code instead? — **A:** no GitHub write connector available; a connected Mac browser exists but needs the laptop; won't handle raw tokens; Claude Code **cloud sessions** run on Anthropic infra with laptop closed (verified in docs) — separate session, needs a handoff. Asked which path.
13. **U:** What is "Run workflow" for? Why was the code hex? — **A:** it launches real training jobs, not tests; the blob was base64 of a compressed tar of five files for one-paste convenience; offered plain files instead.
14. **U:** Transition to the Claude cloud session; dump everything. — **A:** this document + package.

---

*End of handoff.*
