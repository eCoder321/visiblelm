"""Mechanism check for T-A0 stage 1: where did the description pressure push the information?

Hypothesis (from the frontier breakdown): pressure on per-position writer concentration (unit, value and logit reads)
moved information into the one unpressured channel, attention spread over source positions. If so, attention rows
get more diffuse (higher entropy, more mass outside the top-k sources) as lambda grows, while the pressured
quantities themselves fall. Measured on fixed validation windows, at every position (not sampled), from the final
checkpoints.

    used by lm/modal_app.py (function analyze); returns a dict per (job, seed)
"""
import os, sys, pickle, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Corpus
from model import forward

def analyze_ckpt(ckpt, cfg, X, m=8):
    p = pickle.load(open(ckpt, 'rb'))['p']
    x = jnp.asarray(X[:, :-1]); B, T = x.shape
    with jax.default_matmul_precision('highest'):
        _, aux = forward(p, cfg, x, record=True)
        A = [np.asarray(l['A']) for l in aux['record']['layers']]               # (B,H,T,T) per layer
        n_nodes = cfg['r_heads'] * cfg['topk'] if cfg['mlp'] == 'peer' else cfg['ff_mult'] * cfg['d']
        press = {'pos': jnp.arange(B * T), 'slots': jnp.tile(jnp.arange(min(n_nodes, 64))[None], (B * T, 1)),
                 'y': jnp.asarray(X[:, 1:]).reshape(-1), 'm': m}
        _, aux_p = forward(p, cfg, x, press=press)
    out = {'press_all_positions': {k: float(v) for k, v in aux_p['press'].items()}, 'attention': []}
    for a in A:
        rows = a.transpose(0, 2, 1, 3).reshape(-1, a.shape[1], T)                # (B*T, H, T)
        t = np.tile(np.arange(T), B)
        ent = -(rows * np.log(np.clip(rows, 1e-12, None))).sum(-1)              # (B*T, H)
        srt = -np.sort(-rows, -1)
        out['attention'].append({
            'entropy_nats': float(ent[t >= 15].mean()),                       # positions with >= 16 sources
            'outside_top4': float((1 - srt[..., :4].sum(-1))[t >= 15].mean()),
            'outside_top16': float((1 - srt[..., :16].sum(-1))[t >= 15].mean())})
    return out

def run(results_root, data_dir, jobs, seeds=(0, 1), cfg_loader=None):
    corpus = Corpus(data_dir)
    X = np.concatenate([corpus.eval_windows('val', c, 2, 128, seed=1) for c in ['prose', 'code']])
    res = {}
    for j in jobs:
        for s in seeds:
            d = os.path.join(results_root, f'seed{s}', f'ta0s1_{j}')
            ck = os.path.join(d, f'ckpt_seed{s}.pkl')
            if not os.path.exists(ck): continue
            import json
            cfg = json.load(open(os.path.join(d, f'seed{s}.json')))['cfg']
            res[f'{j}/seed{s}'] = analyze_ckpt(ck, cfg, X)
    return res

def cancellation(ckpt, cfg, X):
    """Cancellation ratio sum|c| / |sum c| of per-writer contributions, for the next byte's (centered) logit and for
    unit/neuron reads, plus the share of a read's net value carried by its top-8 writers (net, signed)."""
    import frontier
    p = pickle.load(open(ckpt, 'rb'))['p']
    x = jnp.asarray(X[:, :-1]); y = jnp.asarray(X[:, 1:])
    with jax.default_matmul_precision('highest'):
        _, aux = forward(p, cfg, x, record=True); rec = aux['record']
        WV = [rec['emb'][:, :, None, :]]; unit_ratio = []
        for b, r in zip(p['blocks'], rec['layers']):
            WV.append(r['heads'])
            Wc = jnp.concatenate(WV, 2) * b['ln2']
            if cfg['mlp'] == 'peer':
                from model import routing_vectors
                unit = r['unit']; slots = jnp.broadcast_to(jnp.arange(unit.shape[-1]), unit.shape)
                reads = routing_vectors(b, cfg, slots, unit); wv = b['v'][unit]; coef = r['coef']
            else:
                k = 64; imp = jnp.abs(r['coef']) * jnp.linalg.norm(b['w_out'], axis=-1); idx = jax.lax.top_k(imp, k)[1]
                reads = b['w_in'].T[idx]; wv = b['w_out'][idx]; coef = jnp.take_along_axis(r['coef'], idx, -1)
                full = b['w_out'][None, None] * r['coef'][..., None]
            C = jnp.einsum('btwd,btnd->btnw', Wc, reads) * r['s2'][:, :, None, None]
            ratio = jnp.abs(C).sum(-1) / (jnp.abs(C.sum(-1)) + 1e-6)
            w = jnp.abs(coef) * jnp.linalg.norm(wv, axis=-1)
            unit_ratio.append(float(jnp.median(ratio)))
            WV.append(coef[..., None] * wv if cfg['mlp'] == 'peer' else full)
        Wc = jnp.concatenate(WV, 2) * p['ln_f']
        U = p['unemb'] - p['unemb'].mean(1, keepdims=True)
        c = jnp.einsum('btwd,dbt->btw', Wc, U[:, y]) * rec['sf'][..., None]
        net = c.sum(-1); gross = jnp.abs(c).sum(-1)
        top8 = jnp.take_along_axis(c, jax.lax.top_k(jnp.abs(c), 8)[1], -1).sum(-1)
    return {'logit_cancel_median': float(jnp.median(gross / (jnp.abs(net) + 1e-6))),
            'logit_gross_median': float(jnp.median(gross)), 'logit_net_median': float(jnp.median(jnp.abs(net))),
            'logit_top8_net_share_median': float(jnp.median(top8 / (net + 1e-9))),
            'unit_read_cancel_median_per_layer': unit_ratio}

def run_cancel(results_root, data_dir, jobs, seeds=(0, 1)):
    import json
    corpus = Corpus(data_dir)
    X = np.concatenate([corpus.eval_windows('val', c, 1, 128, seed=1) for c in ['prose', 'code']])
    res = {}
    for j in jobs:
        for s in seeds:
            d = os.path.join(results_root, f'seed{s}', f'ta0s1_{j}'); ck = os.path.join(d, f'ckpt_seed{s}.pkl')
            if os.path.exists(ck):
                res[f'{j}/seed{s}'] = cancellation(ck, json.load(open(os.path.join(d, f'seed{s}.json')))['cfg'], X)
    return res
