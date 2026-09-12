"""
Generic experiment runner.  Usage:
    python run_job.py path/to/job.json            # or
    JOB_JSON='{"name":...}' python run_job.py

Job spec (all optional except name):
{
  "name": "v3_kw4_3seeds",
  "testbed": "synthetic" | "text",
  "cfg": {"code_residual": true, "k_write": 4},     # overrides on base_cfg / text base cfg
  "steps": 1500, "lr": 3e-3, "bs": 64, "aux_coef": 1.0,
  "seeds": [0, 1, 2],
  "n_train": 20000, "T": 16                           # text: n_train windows, T context
}
Writes results/<name>/seed<k>.json + summary.json (mean/std over seeds) + log.txt
"""
import os, sys, json, time, urllib.request
import numpy as np, jax, jax.numpy as jnp
from itlm import *
from analyze import *

def load_job():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return json.load(open(sys.argv[1]))
    return json.loads(os.environ['JOB_JSON'])

def ensure_text():
    if not os.path.exists('shakespeare.txt'):
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'shakespeare.txt')

def synthetic_job(job, seed, out):
    rng = np.random.default_rng(seed)
    T = job.get('T', 16)
    Xtr, mtr = gen_synthetic(job.get('n_train', 20000), T, rng); Xte, mte = gen_synthetic(2000, T, rng)
    cfg = base_cfg(T=T, **job.get('cfg', {}))
    p, h = train(cfg, Xtr, job.get('steps', 1500), lr=job.get('lr', 3e-3), bs=job.get('bs', 64),
                 aux_coef=job.get('aux_coef', 1.0), seed=seed, eval_data=Xte, log_every=250)
    r = full_report(p, cfg, Xte, mte, job['name'])
    r['history'] = h; r['seed'] = seed; r['cfg'] = cfg
    if cfg['bottleneck']:
        _, codes = get_codes(p, cfg, Xte)
        r['legibility_final'] = legibility_report(codes[-1], Xte, mte)
        r['layer_invariance'] = layer_invariance(codes, Xte, mte)
        # stronger faithfulness control: ablate top-3 by ACTIVATION magnitude (not attribution)
        r['ablation_magnitude_control'] = ablation_by_magnitude(p, cfg, Xte, codes[-1])
    return r

def ablation_by_magnitude(p, cfg, X, code_last, n_abl=3, n_examples=300, seed=0):
    rng = np.random.default_rng(seed); B, T, F = code_last.shape; L = cfg['layers']
    f = jax.jit(lambda x, fm: forward(p, cfg, x, feat_masks=fm))
    base, mag = [], []
    for i in rng.integers(0, B, n_examples):
        x = X[i:i+1]; t = rng.integers(3, T - 1); tgt = x[0, t + 1]; c = code_last[i, t]
        active = np.where(c > 0)[0]; top = active[np.argsort(-c[active])[:n_abl]]
        def lp(kill):
            fms = [np.ones((1, T, F), np.float32) for _ in range(L + 1)]
            if kill is not None: fms[-1][0, t, kill] = 0.0
            lg = np.asarray(f(jnp.asarray(x), [jnp.asarray(m) for m in fms]))[0, t]
            return lg[tgt] - np.logaddexp.reduce(lg)
        base.append(lp(None)); mag.append(lp(top))
    return {'base': float(np.mean(base)), 'ablate_top_by_magnitude': float(np.mean(mag))}

def text_job(job, seed, out):
    ensure_text()
    from textdata import get_text_data
    from analyze_text import text_predicates
    T = job.get('T', 64)
    Xtr, Xte, V, chars = get_text_data(T=T, n_train=job.get('n_train', 60000), n_test=1000, seed=seed)
    cfg = dict(vocab=V, d=96, layers=3, heads=3, ff=256, T=T, dict=384, k=12, bottleneck=True, share_dict=True,
               bottleneck_emb=True, k_aux=16, code_residual=False, k_write=12)
    cfg.update(job.get('cfg', {}))
    p, h = train(cfg, Xtr, job.get('steps', 3000), lr=job.get('lr', 2e-3), bs=job.get('bs', 32),
                 aux_coef=job.get('aux_coef', 1.0), seed=seed, eval_data=Xte, log_every=250)
    r = {'name': job['name'], 'seed': seed, 'cfg': cfg, 'history': h, 'eval_nll': h[-1][2], 'bits_per_char': h[-1][2] / np.log(2)}
    if cfg['bottleneck']:
        _, codes = get_codes(p, cfg, Xte, bs=100)
        r['dead_frac'] = dead_features(codes)
        P = text_predicates(Xte, chars)
        for i, c in enumerate(codes):
            best, names, live = feature_f1(c, P, min_pos=1)
            r[f'bn{i}_legibility'] = {'n_live': int(live.sum()), 'median_bestF1': float(np.median(best[live])),
                                      'frac_F1>0.8': float((best[live] > 0.8).mean()),
                                      'top': [(int(f), names[f], round(float(best[f]), 2)) for f in np.argsort(-best)[:15]]}
        W = attribution(p, cfg, codes[-1])
        r['ablation'] = ablation_test(p, cfg, Xte, codes[-1], W, n_examples=200)
        r['ablation_magnitude_control'] = ablation_by_magnitude(p, cfg, Xte, codes[-1], n_examples=200)
        r['completeness_top3_median'] = completeness(codes[-1], W, Xte)
    return r

def summarize(rs):
    def walk(prefix, objs):
        out = {}
        keys = set().union(*[o.keys() for o in objs if isinstance(o, dict)])
        for k in keys:
            vals = [o.get(k) for o in objs if isinstance(o, dict)]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                out[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
            elif all(isinstance(v, dict) for v in vals):
                out[k] = walk(prefix + k + '.', vals)
        return out
    return walk('', rs)

if __name__ == '__main__':
    job = load_job()
    out = os.path.join('results', job['name']); os.makedirs(out, exist_ok=True)
    rs = []
    for seed in job.get('seeds', [0]):
        t0 = time.time(); print(f"### {job['name']} seed {seed}", flush=True)
        r = (text_job if job.get('testbed', 'synthetic') == 'text' else synthetic_job)(job, seed, out)
        r['wall_seconds'] = time.time() - t0
        json.dump(r, open(os.path.join(out, f'seed{seed}.json'), 'w'), indent=1, default=float)
        rs.append(r)
    s = summarize(rs); s['job'] = job
    json.dump(s, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in s.items() if k in ('eval_nll', 'ablation', 'ablation_magnitude_control',
          'completeness_top3_median', 'layer_invariance', 'legibility_final')}, indent=1, default=float))
