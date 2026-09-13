# Task T1 — replicate the discrete structured model over seeds; test two hypotheses for its remaining loss

Issued by the principal investigator session, 2026-09-13. Worker: Sonnet.

## Context you need (read these, in order)
1. `PLAN.md` (goal, metrics, gates), `REPORT.md` (results so far), `docs/REVIEW-2026-09-12.md` §3–§5 (why).
2. `vlm/README-worker.md` (how to run jobs; written for this task).
3. `results/rng_snap3_big/log.txt` and `results/rng_snap2/log.txt` (the runs being replicated).

## What is established (single seed, so not yet counted)
The structured rule network `rulenet_g`, trained with the snap phase (`ste_from`, `ste_ramp`, `clamp: 1.0`), is a
discrete program: the printed rulebook agrees with the model on 100% of predictions and the magnitude audit is 0.
Its cost on iid data is 0.98 nats/token (config `rng_snap3_big`) vs 0.59 for the dense model and 0.62 floor.
Diagnosis on `rng_snap2`: accuracy is 97% far from mode switches but 69–82% within 4 positions after a switch, and
the model puts 0.04 probability mass on a switch where the truth is 0.10.

## Hypotheses
* **H1 (optimization, not capacity):** the hard phase is too short. Doubling the hard phase and restarting the
  learning rate at the snap will bring iid NLL below 0.85 without changing the architecture.
* **H2 (routing gap):** right after a mode token, copy-3-back needs the 3rd content token, which is 4 positions back;
  with 3 heads the model cannot afford a head for that case. A 4th head raises near-switch accuracy (dist_mode 1–4)
  above 90%.

## Jobs (all `rulenet_g`, data `{"n_train":20000,"n_test":1000}`, seeds `[0,1,2]`, `analyze: true`)
Base config = `results/rng_snap3_big/summary.json` → `job.cfg` (copy it exactly).
* `T1_base`: base config, 3 seeds.
* `T1_h1_longhard`: base config with `steps: 8000`, `ste_from: 0.35`, `ste_ramp: 0.15`, `prune_at` and
  `regrow_until` scaled so pruning finishes before `ste_from` (use `prune_at: [0.08,0.13,0.18,0.23,0.28]`,
  `regrow_until: 0.35`), and a learning-rate restart at the snap: implement `lr_restart_at_snap: true` in
  `vlm/train.py` (re-create the optimizer with a fresh warmup+cosine schedule over the remaining steps when the
  snap starts). Gradient smoke test first: confirm the loss decreases over 20 steps after the restart on a tiny config.
* `T1_h2_4heads`: base config with `heads: 4`.
Run at most two jobs concurrently (4 CPU cores). Each seed takes ~35–60 min. Use `nohup ... &` and poll the log.

## Report (write to `docs/tasks/T1-report.md`, commit and push; keep it to tables plus at most 10 lines of notes)
1. `python vlm/report_table.py ctrl_dense_sae T1_base T1_h1_longhard T1_h2_4heads` output (both tables).
2. For each job, the mean±std over seeds of: `eval.iid.nll`, `eval.shift.nll_heldout_sw`, `eval.shift.acc_heldout_cpy2`,
   `eval.shift.acc_heldout_cpy3`, `analysis.main.north_star_iid.agree_model_rule`,
   `analysis.main.faithfulness_iid.sufficiency_retained_frac`, `analysis.main.faithfulness_iid.circuit_clean_frac`,
   `analysis.main.description_length`.
3. The near-switch diagnostic for each job, seed 0: accuracy on rule steps by `dist_mode` in {1,2,3,4} and ≥5, per mode,
   and the mean probability mass on mode tokens far from a switch (oracle 0.10). The code for this diagnostic is in
   the PI's notes below; put it in `vlm/diagnostics.py` as `near_switch_report(bundle_path)`.
4. Verdict per hypothesis: confirmed / refuted / inconclusive, with the number that decides it.
5. Anything you changed in `vlm/` beyond `lr_restart_at_snap` and `diagnostics.py`, and why.

## Rules
* Do not change any metric's semantics. If you must touch `vlm/metrics.py` or `vlm/rules.py`, say so in the report.
* Do not reduce seeds or steps to save time. If a job fails, fix and rerun it; report the failure.
* Commit and push to branch `claude/llm-redesign-first-principles-qty0et` after every finished job
  (`git add -A && git commit -m ... && git push origin claude/llm-redesign-first-principles-qty0et`).
  Commit messages end with the attribution lines in `vlm/README-worker.md`.
* Also add the 3-seed control table (from `report_table.py ctrl_dense_sae ctrl_bottleneck_v3 ctrl_named_neurons`)
  to `REPORT.md` as section 3.5 "Controls over 3 seeds", with one sentence: the previous design (bottleneck v3)
  loses to the null model on every interpretability metric on this testbed.

## PI's notes: near-switch diagnostic (adapt into `vlm/diagnostics.py`)
```python
import models, pickle, numpy as np
from metrics import get_states
from testbed import make_splits, annotate, MODE_NAMES, M_MODES
def near_switch_report(bundle_path):
    b = pickle.load(open(bundle_path, 'rb'))
    S = make_splits(n_train=100, n_test=1000); X = S['iid']; A = annotate(X)
    lg, _ = get_states(b, X); pred = lg[:, :-1].argmax(-1); true = A['oracle_p'][:, :-1].argmax(-1)
    rule = A['next'][:, :-1] >= 0; md = A['mode'][:, :-1]; dm = A['dist_mode'][:, :-1]
    per_mode = {MODE_NAMES[m]: float((pred == true)[rule & (md == m)].mean()) for m in range(M_MODES)}
    by_dist = {d: float((pred == true)[rule & (dm == d)].mean()) for d in [1, 2, 3, 4]}
    far = float((pred == true)[rule & (dm >= 5)].mean())
    lp = lg[:, :-1] - np.log(np.exp(lg[:, :-1] - lg[:, :-1].max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg[:, :-1].max(-1, keepdims=True)
    p_mode = float(np.exp(lp[..., :M_MODES]).sum(-1)[rule & (dm >= 5)].mean())
    return dict(per_mode=per_mode, by_dist=by_dist, far=far, p_mode_far=p_mode)
```
Note: `import models` must come before `from models_g import ...` anywhere (circular import).
