# Interpretable-by-construction LM — experiment runner

Two files to create in a **public** repo: `.github/workflows/train.yml` and `bundle.py`.
Then: Actions tab → **train** → *Run workflow*. Paste a job JSON (or leave empty to run every `jobs/*.json`
that has no results yet). Results are committed to `results/<name>/summary.json` and seed files.

Example job:
```json
{"name":"v3_kw4","testbed":"synthetic","cfg":{"code_residual":true,"k_write":4},"steps":1500,"seeds":[0,1,2]}
```
Text example:
```json
{"name":"text_v3","testbed":"text","cfg":{"code_residual":true,"k_write":12},"steps":3000,"seeds":[0]}
```
