"""Summarize T-A0 results: performance tax and explanation frontier per job.

    python lm/report.py [--out results/lm] [--prefix ta0s1_] [--base ta0s1_peer_sig_lam0] [--kl 0.1 0.3]

Per job (mean over seeds; paired deltas use matching seeds):
  bpb            validation bits per byte (prose, code, mean)
  d_bpb          paired difference vs the base job (the tax in loss units)
  dense_mult     dense-equivalent compute multiple: training compute a dense model needs to reach this loss, divided
                 by this job's training compute (fit of bpb vs log compute over the dense_d* jobs; > 1 = better per FLOP)
  bits@KL        local-frontier description length (bits per byte position) needed to reach mean KL <= target
                 (step function on the Pareto set; '>' means not reached within the evaluated grid)
  dead, cap      PEER dead-unit fraction and cap-binding rate
"""
import os, json, glob, math, argparse
import numpy as np

def load(out, prefix):
    jobs = {}
    for f in sorted(glob.glob(os.path.join(out, '**', prefix + '*', 'seed*.json'), recursive=True)):
        r = json.load(open(f)); jobs.setdefault(r['name'], {})[r['seed']] = r
    return jobs

def bits_at(pareto, kl):
    for q in sorted(pareto, key=lambda q: q['bits']):
        if q['kl_mean'] <= kl: return q['bits']
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/lm'); ap.add_argument('--prefix', default='ta0s1_')
    ap.add_argument('--base', default='ta0s1_peer_sig_lam0'); ap.add_argument('--kl', nargs='+', type=float, default=[0.1, 0.3])
    a = ap.parse_args()
    jobs = load(a.out, a.prefix)
    if not jobs: print('no results'); return
    comp = lambda r: 3 * r['flops_per_token'] * r['train_tokens']
    dense = [(math.log(comp(r)), r['val']['all']['bpb']) for n, s in jobs.items() if n.startswith(a.prefix + 'dense_d')
             and 'lam' not in n for r in s.values()]
    fit = np.polyfit(*zip(*dense), 1) if len({x for x, _ in dense}) >= 2 else None
    base = jobs.get(a.base, {})
    rows = []
    for name, seeds in jobs.items():
        rs = list(seeds.values())
        row = {'job': name[len(a.prefix):], 'seeds': len(rs), 'params_M': rs[0]['n_params'] / 1e6,
               'MFLOP/tok': rs[0]['flops_per_token'] / 1e6}
        for c in ['prose', 'code', 'all']: row[f'bpb_{c}'] = np.mean([r['val'][c]['bpb'] for r in rs])
        paired = [r['val']['all']['bpb'] - base[s]['val']['all']['bpb'] for s, r in seeds.items() if s in base]
        row['d_bpb'] = np.mean(paired) if paired and name != a.base else None
        if fit is not None:
            ce = [math.exp((r['val']['all']['bpb'] - fit[1]) / fit[0]) / comp(r) for r in rs]
            row['dense_mult'] = float(np.mean(ce))
        for kl in a.kl:
            for c in ['prose', 'code']:
                b = [bits_at(r['frontier'][c]['pareto'], kl) for r in rs]
                row[f'bits@{kl}_{c}'] = None if any(x is None for x in b) else float(np.mean(b))
        row['full_kl'] = max(r['frontier'][c]['full_budget_kl'] for r in rs for c in ['prose', 'code'])
        st = [r['stability'] for r in rs if r.get('stability')]
        if st:
            row['dead'] = np.mean([s['dead_frac_mean'] for s in st])
            caps = [l['cap_binding'] for s in st for l in s['layers'] if l['cap_binding'] is not None]
            row['cap'] = np.mean(caps) if caps else None
        rows.append(row)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: '-' if v is None else (f'{v:.4f}' if isinstance(v, float) and abs(v) < 10 else
                                           (f'{v:.0f}' if isinstance(v, float) else str(v)))
    print('| ' + ' | '.join(cols) + ' |'); print('|' + '---|' * len(cols))
    for r in sorted(rows, key=lambda r: r['job']):
        print('| ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |')
    if fit is not None: print(f'\ndense fit: bpb = {fit[1]:.4f} + {fit[0]:.4f} * ln(train FLOPs), from {len(dense)} runs')

if __name__ == '__main__':
    main()
