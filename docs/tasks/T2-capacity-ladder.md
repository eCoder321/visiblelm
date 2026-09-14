# Task T2 — does the discrete-model gap shrink with more capacity, or does it hit a wall?

Issued by the principal investigator session, 2026-09-14. Worker: Sonnet. Read `docs/tasks/T1-report.md` first — the
question this task answers is a direct consequence of T1's result: neither more training time nor one extra head
closed the gap between the discrete model (0.94 nats) and the null model (0.59 nats), so the next test is whether
the gap is a capacity limit or a fixed cost of discreteness itself.

## Hypothesis
**H3 (capacity, not discreteness itself, is the limiter):** iid NLL falls as the model is given more rule slots,
more scratch features, and more routing-table entries, holding everything else (steps, prune schedule fractions,
snap timing, learning rate) fixed. If true, `T2_large` should beat `T1_base` (0.942 ± 0.064) by a margin larger than
its seed noise, and `T2_small` should be worse than `T1_base` by a comparable margin — a real ladder, not noise.
**Falsified if:** `T2_large` is statistically indistinguishable from `T1_base` (mean within ~1 combined std) while
`T2_small` is worse — that would mean the model already has enough capacity and the remaining gap is a cost of
discreteness itself, not of size.

## Configs
Base = `T1_base`'s exact cfg (in `results/T1_base/seed0.json` → `cfg`; already run, do not rerun — reuse its 3-seed
results as the ladder's midpoint). Build two new configs by scaling capacity, changing nothing else (same `steps`,
`prune_at` fractions, `ste_from`, `ste_ramp`, `lr`, `l1`, `l1_table`, `regrow_*`):

* `T2_small`: `n_bool: 16, R: 128, k: 16, n_a: 2, n_u: 2, n_p: 4` (roughly half of base's capacity on every axis).
* `T2_large`: `n_bool: 64, R: 512, k: 32, n_a: 8, n_u: 8, n_p: 16` (roughly double).

Both: `data {"n_train":20000,"n_test":1000}`, `seeds:[0,1,2]`, `analyze:true`. Run at most two jobs concurrently (you
have 4 cores and no other jobs should be running — confirm with `ps aux | grep run.py` before launching; if you see
processes you didn't start, stop and tell the PI, don't kill them yourself).

**Do not launch more than these two jobs.** If a job's process disappears unexpectedly, per standing policy: report
to the PI and wait, do not relaunch yourself. (`run.py` is resumable — a relaunch skips finished seeds — but launching
is the PI's call only, to avoid the collision that happened in T1.)

## Report (`docs/tasks/T2-report.md`, commit and push)
1. Three-point ladder table: `T2_small`, `T1_base` (cite its existing numbers, don't rerun), `T2_large` — for each,
   mean±std over 3 seeds of `eval.iid.nll`, `analysis.main.description_length`, `analysis.main.north_star_iid.agree_model_rule`,
   `analysis.main.faithfulness_iid.circuit_clean_frac`.
2. Near-switch diagnostic (`vlm/diagnostics.py:near_switch_report`) for each config's seed 0: the `dist_mode` 1–4
   accuracy numbers side by side across the three points, since that was T1's specific failure mode.
3. Verdict on H3 with the numbers that decide it (per the falsification criterion above), stated plainly.
4. If H3 is confirmed (gap shrinks with capacity): note whether `T2_large` closes enough of the gap to null (0.59)
   to be worth scaling further, or whether it's a slow asymptote — eyeball the rate, don't over-interpret 3 points.
5. If H3 is falsified (`T2_large` ≈ `T1_base`): say so plainly as the headline finding — it means discreteness has a
   floor around ~0.9–1.0 nats on this task regardless of size, which redirects the project (see PI's note below).

## PI's note for context (not required reading for the run itself, but relevant to how you frame the verdict)
If H3 is falsified, the natural next move is not "make the discrete model bigger" but "stop making everything
binary" — e.g. let each feature take a few graded levels instead of strictly on/off, and treat the magnitude as an
explicitly audited channel rather than a smuggled one. That is out of scope for this task; just flag clearly if the
result points that direction so the PI can scope it next.
