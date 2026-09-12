# legacy/ — code and results received from the claude.ai session (2026-09-12)

Kept for provenance. `itlm.py`, `analyze.py`, `analyze_text.py`, `textdata.py`, `run_job.py` and the
`run*.py` scripts produced the numbers in `docs/interpretable_lm_design.md`. `results/` holds the raw JSON
and logs behind those numbers.

One change was made after receipt: the v3 AuxK reconstruction target in `itlm.py` was `upd` (the whole
dense update); it is now `upd - delta @ D` (the part the sparse write missed). See `docs/REVIEW-2026-09-12.md` §6.

New work lives in `vlm/`.
