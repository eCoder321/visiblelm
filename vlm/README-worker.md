# Running experiments (for worker sessions)

Environment: 4 CPU cores, no GPU, JAX CPU. `pip install jax jaxlib optax numpy` if imports fail.
Working directory for all commands: `vlm/`.

## Run a job
```
cd vlm
JOB_JSON='{"name":"my_job","data":{"n_train":20000,"n_test":1000},"seeds":[0,1,2],"analyze":true,
           "cfg":{...model config...}}' nohup python3 run.py > /dev/null 2>&1 &
```
Progress: `tail -f ../results/my_job/log.txt`. A job is finished when the log contains `### done`.
Outputs: `results/<name>/seed<k>.json` (eval + full analysis + rulebook text), `seed<k>.pkl` (params, masks, cfg),
`summary.json` (mean/std over seeds). Copy an existing config from `results/<name>/summary.json` → `job.cfg`.
Tables: `python3 report_table.py name1 name2 ...` (run from `vlm/` or repo root).

Timing: a `rulenet_g` seed at 6000 steps takes ~35 min alone, ~60 min when two jobs share the CPU. Never run more
than two jobs at once. Wait with a bounded loop, e.g.
`until grep -q "### done\|Traceback" ../results/my_job/log.txt; do sleep 60; done`.

## Code map
* `testbed.py` — data + exact oracle. `models.py` — dense, dense_sae (null model), bottleneck; registries INIT/APPLY/PRUNE/REG.
* `models_g.py` — the structured rule network (`rulenet_g`): slots, routing tables, rules, snap phase (`ste`).
* `train.py` — training loop: pruning schedule, regrowth, snap phase (`ste_from`, `ste_ramp`). `run.py` — job runner.
* `metrics.py` — the scorecard (completeness gap, magnitude audit, path faithfulness, legibility, rulebook).
* `rules.py` — discrete executor; `north_star_any` compares the printed rulebook with the model and the oracle.
* Import order matters: `import models` before `from models_g import ...`.

## Git
Commit after every finished job and push to `claude/llm-redesign-first-principles-qty0et`. `.pkl` files are ignored.
End every commit message with:
```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01A6K5c33B5s73EZVoUnWG52
```
