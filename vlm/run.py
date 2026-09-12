"""Job runner: trains a config over seeds, evaluates on all splits, saves params + results.

    python run.py job.json          or      JOB_JSON='{...}' python run.py

Job spec: {"name": str, "cfg": {model cfg incl. model/steps/lr/bs}, "seeds": [..], "data": {"n_train":..,"n_test":..,"T":..},
           "sae": {"F":.., "k":..}   # optional: for a dense model, also fit post-hoc SAEs (the null model)
           "analyze": true           # optional: run metrics.full_analysis after training (default true)}
Writes results/<name>/seed<k>.json, seed<k>.pkl (params, masks, cfg), summary.json (mean/std over seeds).
"""
import os, sys, json, time, pickle
import numpy as np, jax
from testbed import make_splits, annotate
from train import train, evaluate, fit_saes

def load_job():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]): return json.load(open(sys.argv[1]))
    return json.loads(os.environ['JOB_JSON'])

def get_data(spec, cache={}):
    key = json.dumps(spec, sort_keys=True)
    if key not in cache:
        S = make_splits(**spec); cache[key] = (S, {k: annotate(S[k]) for k in ['iid', 'shift', 'long']})
    return cache[key]

def run_seed(job, seed, out, log):
    S, ann = get_data(job.get('data', {}))
    cfg = dict(job['cfg']); cfg.setdefault('T_max', 32)
    t0 = time.time()
    p, masks, hist = train(cfg, S['train'], seed=seed, evals={'iid': S['iid']}, log=log)
    r = {'name': job['name'], 'seed': seed, 'cfg': cfg, 'history': hist, 'train_seconds': time.time() - t0}
    r['eval'] = {k: evaluate(p, masks, cfg, S[k], ann=ann[k]) for k in ['iid', 'shift', 'long']}
    log(f"  eval: " + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r['eval'].items()}))
    bundle = {'params': jax.tree.map(np.asarray, p), 'masks': jax.tree.map(np.asarray, masks), 'cfg': cfg}
    if cfg['model'] == 'dense' and job.get('sae'):
        cfg_s = {**cfg, 'model': 'dense_sae', **job['sae']}
        ps = fit_saes(cfg_s, p, S['train'], seed=seed, log=log)
        r['eval_sae'] = {ue: {k: evaluate(ps, {}, cfg_s, S[k], opts={'use_error': ue}, ann=ann[k]) for k in ['iid', 'shift']}
                         for ue in [1.0, 0.0]}
        log(f"  sae completeness gap (iid): {r['eval_sae'][0.0]['iid']['nll'] - r['eval_sae'][1.0]['iid']['nll']:.4f}")
        bundle['sae'] = {'params': jax.tree.map(np.asarray, ps), 'cfg': cfg_s}
    pickle.dump(bundle, open(os.path.join(out, f'seed{seed}.pkl'), 'wb'))
    if job.get('analyze', True):
        from metrics import full_analysis
        r['analysis'] = full_analysis(bundle, S, ann, log=log)
    return r

def summarize(rs):
    def walk(objs):
        out = {}
        keys = set().union(*[o.keys() for o in objs if isinstance(o, dict)])
        for k in keys:
            vals = [o.get(k) for o in objs if isinstance(o, dict) and k in o]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                out[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
            elif all(isinstance(v, dict) for v in vals):
                out[k] = walk(vals)
        return out
    return walk(rs)

if __name__ == '__main__':
    job = load_job()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', job['name']); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'a')
    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    rs = []
    for seed in job.get('seeds', [0]):
        log(f"### {job['name']} seed {seed}  {time.strftime('%H:%M:%S')}")
        r = run_seed(job, seed, out, log)
        json.dump(r, open(os.path.join(out, f'seed{seed}.json'), 'w'), indent=1, default=float); rs.append(r)
    s = {'job': job, 'n_seeds': len(rs), 'summary': summarize([{k: v for k, v in r.items() if k in ('eval', 'eval_sae', 'analysis')} for r in rs])}
    json.dump(s, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=float)
    log("### done " + json.dumps(s['summary'].get('eval', {}), default=float)[:2000])
