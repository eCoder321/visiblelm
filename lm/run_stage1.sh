#!/usr/bin/env bash
# T-A0 stage 1: every job, sequentially (one launcher), resumable. Re-running skips finished seeds and resumes
# unfinished ones from their checkpoints.   Usage: bash lm/run_stage1.sh [data_dir] [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-data/lm}; OUT=${2:-results/lm}
[ -f "$DATA/meta.json" ] || python lm/data.py --out "$DATA"
for j in peer_sig_lam0 peer_sig_lam0.03 peer_sig_lam0.1 peer_sig_lam0.3 dense_d256 dense_d192 dense_d128 \
         peer_softmax dense_d256_lam0.1; do
  python lm/run.py "lm/jobs/ta0s1/$j.json" --data "$DATA" --out "$OUT"
done
python lm/report.py --out "$OUT" | tee "$OUT/ta0s1_report.md"
