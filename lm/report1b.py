"""Summarize T-A0 stage 1b: every job's performance and frontier, and each candidate against the k-sweep curve.

    python lm/report1b.py [--out results/lm] [--mode free_strict8]

Per job (mean over seeds): val bpb (prose, code, mean); dead fraction (end, and first logged window);
bits@KL_p90<=0.1 and KL_p90 at 30k / 60k bits for the chosen replay mode, per category, with the bootstrap 90% CI
of the first seed; the noise-vs-strict surrogate gap; selection agreement at the cheapest point reaching
KL_p90 <= 0.3.

Against the k-sweep: the softmax K in {4, 8, 12, 16} points, sorted by bpb, are linearly interpolated in
(bpb, log metric). For each other job whose bpb falls inside the k-sweep's bpb range, it reports
k-sweep(bpb) / job for bits@p90 (above 1 = the job needs fewer bits) and for KL_p90@budget (above 1 = more faithful).
"""
import os, json, glob, math, argparse
import numpy as np

def load(out):
    jobs = {}
    for f in sorted(glob.glob(os.path.join(out, '**', 'ta0s1b_*', 'seed*.json'), recursive=True)):
        r = json.load(open(f)); jobs.setdefault(r['name'][len('ta0s1b_'):], {})[r['seed']] = r
    return jobs

def metrics(seeds, mode):
    rs = list(seeds.values()); out = {'seeds': len(rs)}
    out['bpb'] = float(np.mean([r['val']['all']['bpb'] for r in rs]))
    for c in ['prose', 'code']:
        out[f'bpb_{c}'] = float(np.mean([r['val'][c]['bpb'] for r in rs]))
        fr = [r['frontier2'][c]['modes'][mode] for r in rs]
        vals = [f['bits_at_p90']['0.1'] for f in fr]
        out[f'bits01_{c}'] = None if any(v is None for v in vals) else float(np.exp(np.mean(np.log(vals))))
        for B in ['30000', '60000']:
            v = [f['p90_at_bits'][B] for f in fr]
            out[f'p90@{B}_{c}'] = None if any(x is None for x in v) else float(np.mean(v))
        out[f'ci_bits01_{c}'] = fr[0]['boot'].get('bits_at_p90_0.1')
        out[f'ci_p90@30000_{c}'] = fr[0]['boot'].get('p90_at_30000')
        n = [r['frontier2'][c]['modes'].get('free_noise8', {}).get('p90_at_bits', {}).get('30000') for r in rs]
        s = [f['p90_at_bits']['30000'] for f in fr]
        out[f'noise_vs_strict@30k_{c}'] = None if None in n + s else float(np.mean(n) / max(np.mean(s), 1e-9))
        if 'points' in fr[0]:
            pts = [q for q in fr[0]['points'] if q['kl_p90'] <= 0.3 and 'sel_agree' in q]
            out[f'sel_agree_{c}'] = min(pts, key=lambda q: q['bits'])['sel_agree'] if pts else None
        else:  # compacted on Modal (lm/modal_app.py _compact)
            out[f'sel_agree_{c}'] = fr[0].get('sel_agree_at_p90_0.3')
    dead = [r['stability'].get('dead_frac_mean') for r in rs if r.get('stability')]
    out['dead_end'] = float(np.mean(dead)) if dead else None
    first = [next((h['dead_frac_window'] for h in r['history'] if h.get('dead_frac_window') is not None), None) for r in rs]
    out['dead_first_window'] = float(np.mean([x for x in first if x is not None])) if any(x is not None for x in first) else None
    return out

def sweep_value(sweep, bpb, key):
    """Interpolate the k-sweep's metric at a given bpb (log space); None outside its bpb range or if undefined."""
    pts = sorted([(m['bpb'], m[key]) for m in sweep if m.get(key)], key=lambda t: t[0])
    if len(pts) < 2 or not (pts[0][0] <= bpb <= pts[-1][0]): return None
    for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
        if b0 <= bpb <= b1:
            f = (bpb - b0) / (b1 - b0 + 1e-12)
            return float(np.exp(np.log(v0) + f * (np.log(v1) - np.log(v0))))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/lm'); ap.add_argument('--mode', default='free_strict8')
    a = ap.parse_args(); jobs = load(a.out)
    if not jobs: print('no stage-1b results'); return
    M = {j: metrics(s, a.mode) for j, s in jobs.items()}
    fmt = lambda v: '-' if v is None else (f'{v/1000:.1f}k' if isinstance(v, float) and v > 1000 else
                                           (f'{v:.3f}' if isinstance(v, float) else str(v)))
    cols = ['seeds', 'bpb', 'bpb_prose', 'bpb_code', 'bits01_prose', 'bits01_code', 'p90@30000_prose', 'p90@30000_code',
            'p90@60000_prose', 'p90@60000_code', 'noise_vs_strict@30k_prose', 'dead_first_window', 'dead_end']
    print(f'replay mode: {a.mode}\n')
    print('| job | ' + ' | '.join(cols) + ' |'); print('|---' * (len(cols) + 1) + '|')
    for j in sorted(M): print(f'| {j} | ' + ' | '.join(fmt(M[j].get(c)) for c in cols) + ' |')
    print('\nagainst the softmax k-sweep at equal bpb, PAIRED BY SEED (seed-to-seed bpb differences of 0.04-0.06 make'
          ' unpaired comparisons misleading); above 1 = better than just choosing fewer units:')
    for j in sorted(M):
        if j.startswith('softmax_k'): continue
        for s, r in sorted(jobs[j].items()):
            sweep = [metrics({s: jobs[k][s]}, a.mode) for k in jobs if k.startswith('softmax_k') and s in jobs[k]]
            mj = metrics({s: r}, a.mode); row = []
            for c in ['prose', 'code']:
                for key in [f'bits01_{c}', f'p90@30000_{c}', f'p90@60000_{c}']:
                    sv = sweep_value(sweep, mj['bpb'], key); v = mj.get(key)
                    row.append(f'{key}: ' + ('outside sweep range' if sv is None else ('-' if v is None else f'{sv / v:.2f}x')))
            print(f'  {j} seed{s} (bpb {mj["bpb"]:.4f}; sweep bpb ' +
                  ', '.join(f'{x["bpb"]:.3f}' for x in sorted(sweep, key=lambda x: x["bpb"])) + '): ' + '; '.join(row))
    json.dump(M, open(os.path.join(a.out, 'ta0s1b_summary.json'), 'w'), indent=1)

if __name__ == '__main__':
    main()
