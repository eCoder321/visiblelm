# T1 report — replicate `rulenet_g` over seeds, test H1 (long hard phase) and H2 (4 heads)

All jobs: `rulenet_g`, data `{"n_train":20000,"n_test":1000}`, seeds `[0,1,2]`, `analyze: true`. Configs in
`vlm/jobs/T1_base.json` / `T1_h1_longhard.json` (copied exactly from `results/rng_snap3_big/summary.json` → `job.cfg`,
plus each hypothesis's changes); `T1_h2_4heads` config is the base plus `heads: 4`.

## 1. `report_table.py ctrl_dense_sae T1_base T1_h1_longhard T1_h2_4heads`

Generalization (NLL; oracle floors: iid 0.62, held-out switch 0.00, held-out argument 0.32)

| name | model | seeds | iid | shift_sw | shift_arg | acc_cpy2 | acc_cpy3 | acc_succ | long |
|---|---|---|---|---|---|---|---|---|---|
| ctrl_dense_sae | dense | 3 | 0.593 ±0.000 | 0.181 ±0.053 | 4.130 ±0.110 | 0.78 ±0.03 | 0.60 ±0.12 | 0.01 ±0.02 | 1.607 ±0.098 |
| T1_base | rulenet_g | 3 | 0.942 ±0.064 | 2.385 ±0.256 | 2.746 ±0.216 | 0.84 ±0.07 | 0.86 ±0.01 | 0.02 ±0.02 | 1.429 ±0.117 |
| T1_h1_longhard | rulenet_g | 3 | 1.034 ±0.056 | 2.845 ±0.319 | 3.251 ±0.083 | 0.66 ±0.11 | 0.82 ±0.09 | 0.00 ±0.01 | 1.670 ±0.090 |
| T1_h2_4heads | rulenet_g | 3 | 1.177 ±0.257 | 2.789 ±0.428 | 3.365 ±0.171 | 0.73 ±0.09 | 0.61 ±0.21 | 0.00 ±0.00 | 1.741 ±0.172 |

Interpretability (completeness gap, magnitude gap, circuit sufficiency vs random, argmax kept, circuit size, clean fraction, description length)

| name | compl | mag | suff | suff_rand | argmax_ok | circ | circ_clean | dl |
|---|---|---|---|---|---|---|---|---|
| ctrl_dense_sae | 0.003 ±0.001 | 3.066 ±0.128 | 0.58 ±0.02 | 0.01 ±0.02 | 0.39 ±0.05 | 4.7 ±0.5 | 0.32 ±0.04 | — |
| T1_base | 0.000 ±0.000 | 0.000 ±0.000 | 0.69 ±0.20 | -0.01 ±0.01 | 0.46 ±0.15 | 14.7 ±2.1 | 0.50 ±0.11 | 5027 ±112 |
| T1_h1_longhard | 0.000 ±0.000 | 0.000 ±0.000 | 0.83 ±0.04 | -0.00 ±0.00 | 0.46 ±0.02 | 10.5 ±0.7 | 0.56 ±0.08 | 5318 ±207 |
| T1_h2_4heads | 0.000 ±0.000 | 0.000 ±0.000 | 0.61 ±0.31 | -0.00 ±0.01 | 0.30 ±0.17 | 10.5 ±2.0 | 0.43 ±0.09 | 5392 ±316 |

## 2. Per-job metric table (mean ± std over 3 seeds)

| metric | T1_base | T1_h1_longhard | T1_h2_4heads |
|---|---|---|---|
| eval.iid.nll | 0.9423 ± 0.0642 | 1.0344 ± 0.0564 | 1.1766 ± 0.2573 |
| eval.shift.nll_heldout_sw | 2.3853 ± 0.2558 | 2.8454 ± 0.3189 | 2.7885 ± 0.4281 |
| eval.shift.acc_heldout_cpy2 | 0.8444 ± 0.0750 | 0.6648 ± 0.1116 | 0.7259 ± 0.0856 |
| eval.shift.acc_heldout_cpy3 | 0.8554 ± 0.0095 | 0.8218 ± 0.0918 | 0.6149 ± 0.2058 |
| analysis.main.north_star_iid.agree_model_rule | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9995 ± 0.0007 |
| analysis.main.faithfulness_iid.sufficiency_retained_frac | 0.6862 ± 0.2023 | 0.8267 ± 0.0360 | 0.6135 ± 0.3058 |
| analysis.main.faithfulness_iid.circuit_clean_frac | 0.4959 ± 0.1070 | 0.5627 ± 0.0826 | 0.4308 ± 0.0850 |
| analysis.main.description_length | 5027 ± 112 | 5318 ± 207 | 5392 ± 316 |

## 3. Near-switch diagnostic (`near_switch_report`, seed 0 of each job)

Accuracy on rule steps by `dist_mode` (oracle `p_mode_far` = 0.10):

| job | dist=1 | dist=2 | dist=3 | dist=4 | dist≥5 (far) | p_mode_far |
|---|---|---|---|---|---|---|
| T1_base | 0.908 | 0.762 | 0.825 | 0.965 | 0.955 | 0.067 |
| T1_h1_longhard | 0.646 | 0.851 | 0.654 | 0.965 | 0.992 | 0.057 |
| T1_h2_4heads | 0.714 | 0.892 | 0.836 | 0.815 | 0.783 | 0.052 |

Accuracy by mode:

| job | rept | succ | pred | pls2 | cpy2 | cpy3 |
|---|---|---|---|---|---|---|
| T1_base | 0.962 | 0.869 | 0.840 | 0.858 | 0.926 | 0.908 |
| T1_h1_longhard | 0.921 | 0.764 | 0.950 | 0.954 | 0.888 | 0.946 |
| T1_h2_4heads | 0.946 | 0.911 | 0.873 | 0.971 | 0.857 | **0.350** |

## 4. Verdicts

**H1 (long hard phase + LR restart) — REFUTED.** Target: iid NLL < 0.85 without changing architecture.
`T1_h1_longhard` iid NLL = **1.034 ± 0.056**, not only above the 0.85 bar but *worse* than the `T1_base` replication
(0.942 ± 0.064). Near-switch accuracy did not improve either — dist=1 dropped from 0.908 (base) to 0.646, dist=3
dropped from 0.825 to 0.654. Doubling the hard phase and restarting the LR schedule bought more `sufficiency_retained_frac`
(0.827 vs 0.686) but did not fix the loss the hypothesis targeted; if anything the extra optimization time re-opened a
generalization gap on `shift.acc_heldout_cpy2` (0.66 vs 0.84).

**H2 (4th head fixes near-switch routing) — REFUTED, and not merely "the wrong trade": the extra head did not even help its target case.**
Target: `heads: 4` should raise near-switch accuracy (dist_mode 1–4) above 90%. On `T1_h2_4heads` seed 0, no
dist_mode bucket reaches 90% (max is 0.892 at dist=2); the mean over dist 1–4 is **0.814**, *lower* than `T1_base`'s
**0.865** over the same buckets. The specific failure case named in the hypothesis — copy-3-back right after a mode
token (dist=1, needing a token 4 positions back) — got *worse* with the 4th head: 0.714 vs 0.908 for the 3-head base.
The `cpy3` mode collapses on this seed (0.350 vs 0.908 for base), and the aggregate `eval.shift.acc_heldout_cpy3`
drops from 0.855 (base) to 0.615 (4 heads). So the extra head bought no targeted improvement to trade against its
cost elsewhere — both the near-switch metric it was meant to fix and the headline iid NLL got worse, and variance
across seeds roughly quadrupled (iid NLL std 0.257 vs 0.064). The routing-gap diagnosis behind H2 may still be right,
but a bare 4th head is not the fix: with the same pruning/regrowth budget it likely gave the search more heads to
misallocate rather than dedicating one to the missing case.

## 5. Surprising results

1. **Both hypotheses moved iid NLL the wrong way.** H1 predicted <0.85, got 1.034; H2 predicted an improvement via
   near-switch accuracy, got 1.177 iid NLL and *lower* near-switch accuracy on average than base. Neither guessed fix
   is the lever on this loss — that should shape what's tried next (e.g. capacity/search-budget interactions, not just
   phase length or head count in isolation), rather than being read as within-noise variation.

## 6. Deviations from the task file

* `vlm/train.py`: added `lr_restart_at_snap` exactly as specified (re-creates the optimizer with a fresh
  warmup+cosine schedule over the remaining steps at snap time); gradient smoke test on a tiny config passed before
  the full runs.
* `vlm/diagnostics.py`: added `near_switch_report(bundle_path)` per the PI's notes, unchanged in semantics.
* `vlm/run.py`: **beyond the task's brief**, added seed-level resumability (skip a seed whose `results/<name>/seedK.json`
  already exists) after a container restart silently killed `T1_base` and `T1_h2_4heads` mid-run with no error logged.
  This only changes run/resume bookkeeping, not training or metric semantics — no seed's config, data, or evaluation
  code path changed. `metrics.py` and `rules.py` were not touched.
* Infrastructure incident (not a code deviation, noted for the record): the execution container restarted at least
  twice during this task, silently killing background jobs. `T1_base` seed 1 was retrained from step 0 after a lost
  `seed1.json`; a brief multi-process race during recovery (duplicate relaunches of the same job) was caught and
  cleaned up (killed extra processes, deleted a corrupted partial checkpoint) before any seed finished, so no result
  in this report was produced by more than one process. All reported numbers come from single, uncontested runs to
  `### done`.
