"""Train + evaluate one T-A0 job over its seeds; resumable from checkpoints.

    python lm/run.py lm/jobs/ta0s1/peer_sig_l0.json [--data data/lm] [--out results/lm] [--smoke]

Job spec: {"name": str, "seeds": [..], "cfg": {model + training keys, see DEFAULTS}}.
Writes <out>/<name>/seed<k>.json (history, val loss by category, local frontier, stability) and checkpoints
<out>/<name>/ckpt_seed<k>.pkl (params, optimizer state, balance bias, data-rng state) every ckpt_every steps.
--seeds overrides the job's seed list (to run seeds in parallel). Data order depends only on the seed, so runs with the same seed see the same batches (paired comparisons).
--smoke shrinks everything to a CPU-sized check of the whole pipeline.
"""
import os, sys, json, time, math, pickle, argparse
import numpy as np, jax, jax.numpy as jnp, optax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Corpus
from model import init_params, init_balance, forward, n_params, flops_per_token
import frontier

DEFAULTS = dict(d=256, layers=4, heads=4, mlp='peer', ff_mult=4, n_keys=128, d_key=128, r_heads=4, topk=16,
                gate='sigmoid', balance_eta=1e-3,
                T=256, bs=32, steps=8000, lr=2e-3, warmup=200, wd=0.1, clip=1.0, log_every=200, ckpt_every=1000,
                lam=0.0, press_m=8, press_frac=1 / 16, press_slots=16, press_start=0.1, press_ramp=0.2,
                press_w={'unit': 1.0, 'value': 1.0, 'logit': 1.0},
                n_eval=256, frontier_windows=4, frontier_T=128, frontier_n_max=256)
SMOKE = dict(d=64, layers=2, heads=2, n_keys=16, d_key=32, r_heads=2, topk=4, T=64, bs=8, steps=30, log_every=10,
             ckpt_every=20, n_eval=8, frontier_windows=1, frontier_T=32, frontier_n_max=32, press_frac=1 / 4,
             press_slots=4, press_start=0.0, press_ramp=0.1)

def lam_at(cfg, i):
    s, r = cfg['press_start'] * cfg['steps'], max(cfg['press_ramp'] * cfg['steps'], 1)
    return cfg['lam'] * float(np.clip((i - s) / r, 0.0, 1.0))

def make_step(cfg, opt):
    B, T = cfg['bs'], cfg['T']
    P = max(1, int(B * T * cfg['press_frac']))
    n_nodes = cfg['r_heads'] * cfg['topk'] if cfg['mlp'] == 'peer' else cfg['ff_mult'] * cfg['d']
    w = cfg['press_w']

    def loss_fn(p, xb, bias, key, lam):
        x, y = xb[:, :-1], xb[:, 1:]
        press = None
        if cfg['lam'] > 0:
            k1, k2 = jax.random.split(key)
            pos = jax.random.choice(k1, B * T, (P,), replace=False)
            press = {'pos': pos, 'slots': jax.random.randint(k2, (P, cfg['press_slots']), 0, n_nodes),
                     'y': y.reshape(-1)[pos], 'm': cfg['press_m']}
        logits, aux = forward(p, cfg, x, bias=bias, press=press)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), y[..., None], -1).mean()
        pen = jnp.zeros(()); parts = {}
        if press is not None:
            parts = aux['press']
            pen = sum(w[k] * parts[k] for k in parts) / sum(w[k] for k in parts)
        return ce + lam * pen, (ce, pen, parts, aux['usage'])

    @jax.jit
    def step(p, os_, bias, xb, key, lam):
        (_, (ce, pen, parts, usage)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, xb, bias, key, lam)
        upd, os_ = opt.update(g, os_, p); p = optax.apply_updates(p, upd)
        if bias is not None:
            bias = [bb - cfg['balance_eta'] * jnp.sign(u - u.mean(-1, keepdims=True)) for bb, u in zip(bias, usage)]
        return p, os_, bias, ce, pen, parts
    return step

@jax.jit
def _nll_rows(p_cfg_logits, y):
    return -jnp.take_along_axis(jax.nn.log_softmax(p_cfg_logits, -1), y[..., None], -1)[..., 0]

def eval_loss(p, cfg, X, bias=None, bs=16):
    f = jax.jit(lambda p, x: forward(p, cfg, x)[0])
    tot = []
    for i in range(0, len(X), bs):
        xb = jnp.asarray(X[i:i + bs]); tot.append(np.asarray(_nll_rows(f(p, xb[:, :-1]), xb[:, 1:])))
    nats = float(np.concatenate(tot).mean())
    return {'nats': nats, 'bpb': nats / math.log(2)}

def stability(p, cfg, X, bs=16):
    """PEER only: dead-unit fraction, normalized usage entropy, cap-binding rate (sigmoid gate: the lowest-ranked
    selected unit of a retrieval head still has gate > 0.5, so the top-k cap may be cutting units off)."""
    if cfg['mlp'] != 'peer': return {}
    f = jax.jit(lambda p, x: forward(p, cfg, x, record=True)[1]['record'])
    N = cfg['n_keys'] ** 2; K = cfg['topk']
    use = [np.zeros(N) for _ in range(cfg['layers'])]; bind = [[] for _ in range(cfg['layers'])]
    for i in range(0, len(X), bs):
        rec = f(p, jnp.asarray(X[i:i + bs, :-1]))
        for l, r in enumerate(rec['layers']):
            np.add.at(use[l], np.asarray(r['unit']).ravel(), 1)
            g = np.asarray(r['gate']).reshape(r['gate'].shape[:2] + (cfg['r_heads'], K))
            sc = np.asarray(r['score']).reshape(g.shape)
            low = np.take_along_axis(g, sc.argmin(-1)[..., None], -1)[..., 0]
            bind[l].append((low > 0.5).mean())
    out = []
    for l in range(cfg['layers']):
        q = use[l] / use[l].sum(); nz = q[q > 0]
        out.append({'dead_frac': float((use[l] == 0).mean()), 'usage_entropy': float(-(nz * np.log(nz)).sum() / np.log(N)),
                    'cap_binding': float(np.mean(bind[l])) if cfg.get('gate', 'sigmoid') == 'sigmoid' else None})
    return {'layers': out, 'dead_frac_mean': float(np.mean([o['dead_frac'] for o in out]))}

def save(path, obj):
    tmp = path + '.tmp'; pickle.dump(obj, open(tmp, 'wb')); os.replace(tmp, path)

def run_seed(job, seed, corpus, outdir, log):
    cfg = job['cfg']; T = cfg['T']
    ckpt = os.path.join(outdir, f'ckpt_seed{seed}.pkl')
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg['lr'], min(cfg['warmup'], cfg['steps'] // 10), cfg['steps'],
                                              cfg['lr'] * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(cfg['clip']),
                      optax.adamw(sched, weight_decay=cfg['wd'], mask=lambda p: jax.tree.map(lambda a: a.ndim >= 2, p)))
    step = make_step(cfg, opt)
    if os.path.exists(ckpt):
        st = pickle.load(open(ckpt, 'rb')); p, os_, bias, i0, hist = st['p'], st['os'], st['bias'], st['step'], st['hist']
        rng = np.random.default_rng(); rng.bit_generator.state = st['rng']; t_prev = st['seconds']
        log(f'  resumed seed {seed} at step {i0}')
    else:
        p = init_params(cfg, jax.random.PRNGKey(seed)); os_ = opt.init(p); bias = init_balance(cfg)
        i0, hist, rng, t_prev = 0, [], np.random.default_rng(1000 + seed), 0.0
    key = jax.random.PRNGKey(10_000 + seed)
    Xval = {c: corpus.eval_windows('val', c, cfg['n_eval'] // 2, T, seed=0) for c in ['prose', 'code']}
    t0 = time.time()
    for i in range(i0, cfg['steps']):
        xb = jnp.asarray(corpus.batch(rng, cfg['bs'], T))
        p, os_, bias, ce, pen, parts = step(p, os_, bias, xb, jax.random.fold_in(key, i), jnp.float32(lam_at(cfg, i)))
        if i % cfg['log_every'] == 0 or i == cfg['steps'] - 1:
            ce = float(ce)
            if not math.isfinite(ce): raise RuntimeError(f'non-finite loss at step {i}')
            hv = eval_loss(p, cfg, np.concatenate([Xval['prose'][:16], Xval['code'][:16]]))
            hist.append({'step': i, 'train_nats': ce, 'val_bpb_quick': hv['bpb'], 'lam': lam_at(cfg, i),
                         'press': float(pen), **{f'press_{k}': float(v) for k, v in parts.items()}})
            log(f'  step {i:6d} train {ce:.4f} nats  val~ {hv["bpb"]:.4f} bpb  press {float(pen):.3f} '
                f'lam {lam_at(cfg, i):.3g}  ({t_prev + time.time() - t0:.0f}s)')
        if (i + 1) % cfg['ckpt_every'] == 0 and i + 1 < cfg['steps']:
            save(ckpt, {'p': p, 'os': os_, 'bias': bias, 'step': i + 1, 'hist': hist, 'rng': rng.bit_generator.state,
                        'seconds': t_prev + time.time() - t0})
    secs = t_prev + time.time() - t0
    save(ckpt, {'p': p, 'os': os_, 'bias': bias, 'step': cfg['steps'], 'hist': hist, 'rng': rng.bit_generator.state,
                'seconds': secs})
    r = {'name': job['name'], 'seed': seed, 'cfg': cfg, 'history': hist, 'train_seconds': secs,
         'n_params': n_params(p), 'flops_per_token': flops_per_token(cfg, T),
         'train_tokens': cfg['steps'] * cfg['bs'] * T}
    r['val'] = {c: eval_loss(p, cfg, X) for c, X in Xval.items()}
    r['val']['all'] = {k: (r['val']['prose'][k] + r['val']['code'][k]) / 2 for k in ['nats', 'bpb']}
    log(f"  val bpb: prose {r['val']['prose']['bpb']:.4f} code {r['val']['code']['bpb']:.4f}")
    r['stability'] = stability(p, cfg, np.concatenate([Xval['prose'][:32], Xval['code'][:32]]))
    r['frontier'] = {}
    for c in ['prose', 'code']:
        Xf = corpus.eval_windows('val', c, cfg['frontier_windows'], cfg['frontier_T'], seed=1)
        with jax.default_matmul_precision('highest'):  # GPUs default to TF32; the replay check needs full fp32
            fr = frontier.evaluate(p, cfg, Xf, n_max=cfg['frontier_n_max'])
        r['frontier'][c] = fr
        log(f"  frontier[{c}] full-budget KL {fr['full_budget_kl']:.2e}  KL@bits " +
            ' '.join(f"{k}:{v:.3f}" for k, v in fr['kl_at_bits'].items() if v is not None))
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('job'); ap.add_argument('--data', default='data/lm'); ap.add_argument('--out', default='results/lm')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--seeds', type=int, nargs='+')
    a = ap.parse_args()
    job = json.load(open(a.job))
    if a.seeds is not None: job['seeds'] = a.seeds
    job['cfg'] = {**DEFAULTS, **job['cfg'], **(SMOKE if a.smoke else {})}
    if a.smoke: job['name'] += '_smoke'
    outdir = os.path.join(a.out, job['name']); os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, 'log_seed' + '_'.join(map(str, job.get('seeds', [0]))) + '.txt'), 'a')
    def log(s): print(s, flush=True); logf.write(s + '\n'); logf.flush()
    log(f"== {job['name']} on {jax.devices()} cfg {json.dumps(job['cfg'], sort_keys=True)}")
    corpus = Corpus(a.data)
    for seed in job.get('seeds', [0]):
        res = os.path.join(outdir, f'seed{seed}.json')
        if os.path.exists(res): log(f'  seed {seed} done, skipping'); continue
        r = run_seed(job, seed, corpus, outdir, log)
        json.dump(r, open(res, 'w'))

if __name__ == '__main__':
    main()
