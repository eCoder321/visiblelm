# Real-text experiments (T-A0)

Code for the first real-text test of the Ledger Transformer design (`docs/redesign/DESIGNS.md` v2.2, §11).
The experiment brief and its pre-registered predictions are in `docs/tasks/TA0-stage1.md`.

| file | what it does |
|---|---|
| `data.py` | builds the pinned byte-level corpus: NLTK prose corpora + tiny-shakespeare + sympy source, about 46 MB |
| `model.py` | transformer with a dense MLP or a PEER-style unit pool. Records the ledger and computes the description-pressure term. |
| `frontier.py` | local explanation frontier: replays the model from a truncated description, KL vs description bits |
| `run.py` | trains + evaluates one job over its seeds. Checkpoints and resumes. `--smoke` = CPU-sized check. |
| `report.py` | summary table: bits per byte, paired tax, dense-equivalent compute, bits needed at a KL target |
| `run_stage1.sh` | runs every stage-1 job in order, then the report |

## On a GPU machine
```bash
git clone <repo> && cd visiblelm && git checkout claude/llm-redesign-first-principles-qty0et
pip install -r lm/requirements-gpu.txt
python -c "import jax; print(jax.devices())"     # should list a CUDA device
python lm/data.py --out data/lm                   # ~20 MB of downloads; sha256s land in data/lm/meta.json
python lm/run.py lm/jobs/ta0s1/peer_sig_lam0.json --smoke   # 1-minute pipeline check
bash lm/run_stage1.sh                              # the full stage: 9 jobs x 2 seeds
```
One modest GPU (L4/T4/A10 class, 16 GB or more) is enough. The heaviest configuration (dense model with the
pressure term) peaked at about 4.4 GB on CPU.

## On Modal (what this project uses)
```bash
pip install 'modal[api-proxy-support]'      # the proxy extra is required to reach Modal from the cloud container
# credentials: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment
modal deploy lm/modal_app.py                 # builds the GPU image (jax[cuda12]) with the current lm/ code
python lm/modal_app.py data                  # builds the corpus on the 'visiblelm-data' volume (checksums match a local build)
python lm/modal_app.py launch                # one L4 call per (job, seed), all in parallel; ids in results/lm/modal_calls.json
python lm/modal_app.py status
python lm/modal_app.py collect && python lm/report.py
```
Re-deploy after changing anything in `lm/`. Calls run on Modal independently of the launching machine. Checkpoints
are committed to the volume at every save, so a retried call resumes where it stopped.

On CPU (this repo's default 4-core container), a full-size step takes about 5–10 s, so one 8,000-step run takes
roughly 15 hours. Use `--smoke` there.
