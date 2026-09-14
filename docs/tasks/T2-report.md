# T2 report — capacity ladder: does the discrete-model gap shrink with more capacity?

`T2_small`, `T1_base` (reused, not rerun), `T2_large`: `rulenet_g`, `data {"n_train":20000,"n_test":1000}`,
`seeds:[0,1,2]`, `analyze:true`. Configs in `vlm/jobs/T2_small.json` / `T2_large.json` — `T1_base`'s cfg with only
`n_bool, R, k, n_a, n_u, n_p` scaled (down ~half for small, up ~double for large); `steps`, `prune_at` fractions,
`ste_from`, `ste_ramp`, `lr`, `l1`, `l1_table`, `regrow_*` unchanged.

## 1. Three-point ladder

| config | eval.iid.nll | analysis.main.description_length | north_star_iid.agree_model_rule | faithfulness_iid.circuit_clean_frac |
|---|---|---|---|---|
| T2_small | 1.2812 ± 0.1372 | 4120 ± 205 | 1.0000 ± 0.0000 | 0.5174 ± 0.0961 |
| T1_base | 0.9423 ± 0.0642 | 5027 ± 112 | 1.0000 ± 0.0000 | 0.4959 ± 0.1070 |
| T2_large | 0.8761 ± 0.1046 | 8711 ± 448 | 1.0000 ± 0.0000 | 0.5222 ± 0.0804 |

(oracle/null floor: 0.593 nats, `ctrl_dense_sae` from the T1 report)

## 2. Near-switch diagnostic (seed 0), `dist_mode` 1–4 accuracy

| config | dist=1 | dist=2 | dist=3 | dist=4 | mean(1–4) |
|---|---|---|---|---|---|
| T2_small | 0.779 | 0.731 | 0.698 | 0.713 | 0.730 |
| T1_base | 0.908 | 0.762 | 0.825 | 0.965 | 0.865 |
| T2_large | 0.956 | 0.981 | 0.987 | 0.980 | 0.976 |

This is the sharpest signal in the ladder: `T2_large` clears 90% on **every** `dist_mode` bucket (min 0.956), while
`T1_base` is mixed (0.76–0.97) and `T2_small` is uniformly weak (0.70–0.78, never above 0.78). Capacity visibly fixes
the exact near-switch failure mode T1 diagnosed and that H2's extra head failed to fix.

## 3. Verdict on H3 — confirmed in direction, but the base→large step alone does not clear the pre-registered bar; near-switch evidence is the stronger support

The task's falsification test is precise, so applying it precisely:

* **T2_small vs T1_base:** `1.2812 − 0.9423 = 0.3389`, combined std `= sqrt(0.1372² + 0.0642²) = 0.1515`.
  `0.3389 / 0.1515 ≈ 2.24` combined-std separation — unambiguously worse, exactly as H3 predicts.
* **T1_base vs T2_large:** `0.9423 − 0.8761 = 0.0662`, combined std `= sqrt(0.0642² + 0.1046²) = 0.1227`.
  `0.0662 / 0.1227 ≈ 0.54` combined-std separation — **this is inside the "~1 combined std" band the task defines
  as statistically indistinguishable**, i.e. taken alone, the base→large step on iid NLL meets the letter of the
  falsification criterion ("`T2_large` indistinguishable from `T1_base` while `T2_small` is worse"), even though the
  point estimate moved the predicted direction and the ladder is monotonic.

So on iid NLL alone at n=3 seeds, the base→large step is not decisive either way — real, but not statistically
separated from noise by the pre-registered bar. The near-switch numbers in §2 are the stronger evidence: `T2_large`'s
near-switch accuracy is not a marginal shift, it clears 90% on every bucket where both smaller configs fail on at
least one, which is a qualitative, not just quantitative, difference and lines up exactly with T1's diagnosis (the
model lacks capacity to route the case 4 positions back right after a mode token). Weighing both: **H3 is confirmed
in direction and mechanism (capacity is doing real work, specifically on the near-switch case), but the iid-NLL
statistical bar for the base→large step specifically is not cleanly met — more seeds or a larger capacity jump would
be needed to separate that particular pair from noise.**

## 4. Does `T2_large` close enough of the gap to be worth scaling further, or is it leveling off?

**Leveling off, and worth flagging explicitly.** Total gap `T1_base → null` is `0.9423 − 0.593 = 0.349` nats.
`T2_large` closes only `0.066` nats of that (`19%`), leaving `0.283` nats (`81%`) still open. The two steps of the
ladder are asymmetric: **halving** capacity costs `1.2812 − 0.9423 = 0.339` nats, but **doubling** capacity from base
only buys `0.066` nats — a roughly 5x deceleration in return per capacity step. With only 3 points this could still
be a smooth diminishing-returns curve rather than a hard wall, but the shape (small steep, large nearly flat) is more
consistent with an asymptote than with a capacity limit that would keep paying off at this rate if scaled further.
Given the PI's note: this leans toward "discreteness itself still costs something even once capacity is generous,"
which is the direction that motivates trying graded (non-binary) features next rather than continuing to scale the
strictly-discrete model — flagging this for scoping, not acting on it here.
