# Tasks E0 + E1 — attribute the lost nats; scan how coarse the magnitude channel is

Issued by the principal investigator session, 2026-09-15. Worker: Sonnet. Read `docs/PLAN-v3-observability-frontier.md`
in full first — this task is exactly its §3 E0 and E1, combined here because neither requires training a new model
(E1 needs at most one small training run, noted below). No jobs need the 4-core budget managed like T1/T2; this is
analysis code plus at most one short run, so you have more headroom than usual, but still confirm nothing else is
running (`ps aux | grep run.py`) before starting anything.

## First: verify the inputs exist
`.pkl` files (trained params) are gitignored and only ever lived on local disk, never committed. Before writing any
analysis code, confirm these are present and check their mtimes look intact (not zero-byte, not from a stale run):
* `results/ctrl_dense_sae/seed0.pkl` (has `bundle['sae']`)
* `results/rng_v1/seed0.pkl` (or find the exact path — check `results/rng_v1/*.pkl`)
* `results/T2_large/seed0.pkl`, `seed1.pkl`, `seed2.pkl`

If any is missing, say so plainly in the report rather than silently substituting something else, and tell me before
improvising a replacement — a missing bundle changes which comparisons E0/E1 can make.

## E0 — attribute the lost nats (no training)
For `ctrl_dense_sae` (opts `{'use_error': 1.0}`), `rng_v1` (soft rulenet_g), `T2_large` (binary, all 3 seeds — report
per-seed and mean), on the **iid** split (`testbed.make_splits`, same seed/params `run.py` uses so it matches the
committed eval numbers):

1. Using `testbed.annotate`'s oracle (`oracle_lp`, `oracle_p`), split every predicted position (t < T-1) into four
   buckets: **seed** (t ≤ 2, uniform, unfixable), **deterministic** (`heldout==0` and the oracle's max probability
   for that step is ≥ 0.99 — i.e. a switch was not possible), **stochastic** (switch was possible, oracle max prob
   < 0.99), **held-out** (`heldout==1`, report separately, don't fold into the other three). For each bucket, report:
   count, share of all positions, mean(model NLL − oracle NLL), and that bucket's share of the *model's total*
   (NLL − oracle) gap (should sum to ~100% across the four buckets).
2. Within the deterministic bucket only, split the gap into **argmax-wrong** (model's argmax ≠ oracle's argmax) vs
   **argmax-right-but-underconfident** (argmax matches, gap is from probability mass, not from picking wrong).
   Report each sub-bucket's share of the deterministic-bucket gap.
3. Add three predicates to `metrics.predicate_matrix` (or a local copy — see the "don't touch shared metrics code"
   rule below) built from `ann['dist_mode']` and a new computed field, "switch was possible at this step": (a)
   `switch_possible` (bool, computable from `testbed.State.switch_possible` logic — reconstruct it from `ann` or add
   a field to `annotate()` if that's cleaner, see note below), (b) `n_switch_far` = `dist_mode>=6` — reuse existing,
   (c) nothing else needed, the existing vocabulary already has `dist_mode`. Fit held-out F1 (same method as
   `metrics.legibility`) for `switch_possible` against every feature of `T2_large`'s final state (seed 0). Report the
   single best-matching feature and its held-out F1.
4. Write up the decision rule from the plan file (§E0) with the actual numbers plugged in and state which branch it
   lands on.

**If you add `switch_possible` to `testbed.annotate`**: do it as a new key, don't rename or remove any existing key
(other code depends on the current keys), and confirm `python vlm/testbed.py` still runs clean after. This is
allowed — flag it in the report as a deviation from "don't touch shared code."

## E1 — post-hoc quantization scan (no training required for `rng_v1`; one optional short run for a `T2_large` soft twin)
1. Implement an L-level version of the discrete executor. In `vlm/rules.py`, generalize `execute_hard_g`'s
   `binarize` step: instead of a single threshold at 0.5, round each activation to the nearest of L evenly-spaced
   levels in [0, clamp] (L ∈ {2, 3, 4, 8, 16}; L=2 must reproduce today's `execute_hard_g` exactly — write a smoke
   test asserting this, using `T2_large` seed 0, before running anything else). Keep the existing function working
   unchanged for existing callers (add a new `n_levels` parameter defaulting to 2, or a new function
   `execute_quantized_g` — your call, whichever is less invasive).
2. On `rng_v1` (soft model, never snapped — `cfg['ste']` is absent/false, so this only tests the "read the trained
   soft model at L levels" question, not "train at L levels"): for each L, run the quantized executor on the iid
   split, report agreement with the soft model's own argmax predictions (`agree_model_rule` equivalent — reuse
   `north_star_any`'s pattern) and the quantized program's NLL. Also report at L=∞ (no rounding) as a sanity ceiling
   — should be ~100% agreement.
3. Optional, only if E0 finishes with time to spare: train one soft (`ste` absent) twin of `T2_large`'s exact config
   (same `n_bool/R/k/n_a/n_u/n_p`, no `clamp`, no `ste_from`/`ste_ramp`, everything else identical), 1 seed, same
   `steps`. Launch it, don't block on it — do the L-scan on `rng_v1` and write up E0 first, then check on this job
   and add its L-scan to the report if it's done; if not, note it as pending and give me a separate short update
   when it finishes rather than holding the whole report for it.
4. Report per L: agreement %, NLL. State the L at which agreement crosses 95% and 99%. Apply the plan's decision
   rule (§E1) and state which branch it lands on.

## Constraints
* **Don't touch shared metrics semantics.** `metrics.py`'s existing functions (used by every prior committed result)
  must keep producing identical numbers for existing configs — if you need a variant, add a new function or an
  optional parameter with the old behavior as default, don't change existing call signatures' defaults.
* This is analysis work; if a script takes more than ~15 minutes to run once, that's a sign you're recomputing
  something already in a committed `seed*.json` (e.g. don't retrain to get numbers `results/T2_large/summary.json`
  already has).
* Standing rules still apply: you're the sole launcher for anything you start in this task; bounded poll loops, not
  recurring monitors; commit and push incrementally (code + the writeup) rather than in one giant commit at the end.

## Report (`docs/tasks/E0-E1-report.md`, commit and push)
1. E0: the four-bucket table, the deterministic-gap sub-split, the `switch_possible` legibility result, the decision
   branch reached.
2. E1: the L-vs-agreement/NLL table for `rng_v1`, the L=2 exactness smoke-test confirmation, the decision branch
   reached, and (if it finished in time) the same table for the `T2_large` soft twin.
3. A short "what this means for E2" paragraph: given the E0 and E1 results, does the plan's §3 E2 (train at L ∈
   {4,8}) still make sense as written, should the L values tested change, or does something upstream of E2 need
   fixing first (e.g. if E0 shows `switch_possible` isn't legible in the current architecture, that's a routing
   issue E2 alone won't fix)? State this plainly — I will read it before deciding what to dispatch next.
4. Any deviations from this task file, and why.

When done: commit, push, then give me the concise numeric summary (the two decision-rule outcomes, the one-paragraph
recommendation for E2, anything blocking or any input file that was missing).
