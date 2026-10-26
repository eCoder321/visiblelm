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

def gate_diag(ckpt, cfg, X):
    """Why did sigmoid gates cost ~4x compute? Per layer: gate saturation and sum per retrieval head, unit-pool
    output size relative to the residual, and the gradient each parameter group gets (|grad| / |param|)."""
    import optax
    st = pickle.load(open(ckpt, 'rb')); p, bias = st['p'], st['bias']
    x = jnp.asarray(X[:, :-1]); y = jnp.asarray(X[:, 1:])
    R, K = cfg['r_heads'], cfg['topk']
    with jax.default_matmul_precision('highest'):
        _, aux = forward(p, cfg, x, bias=bias, record=True); rec = aux['record']
        def ce(p):
            lg, _ = forward(p, cfg, x, bias=bias); return -jnp.take_along_axis(jax.nn.log_softmax(lg, -1), y[..., None], -1).mean()
        g = jax.grad(ce)(p)
    out, h = [], rec['emb']
    for l, (b, r, gb) in enumerate(zip(p['blocks'], rec['layers'], g['blocks'])):
        h = h + r['heads'].sum(2)
        w = r['coef'][..., None] * b['v'][r['unit']]                       # (B,T,RK,d)
        mlp_out = w.sum(2); ratio = jnp.linalg.norm(mlp_out, axis=-1) / jnp.linalg.norm(h, axis=-1)
        gate = r['gate'].reshape(r['gate'].shape[:2] + (R, K))
        share = gate / (gate.sum(-1, keepdims=True) + 1e-9)          # each unit's share of its head's total gate
        ent = -(share * jnp.log(share + 1e-12)).sum(-1) / jnp.log(K)    # 1 = all K units equal, 0 = one unit
        d = {'gate_sum_per_head': float(gate.sum(-1).mean()), 'gate_max_mean': float(gate.max(-1).mean()),
             'top_share_mean': float(share.max(-1).mean()), 'share_entropy_norm': float(ent.mean()),
             'coef_top4_share': float((jax.lax.top_k(jnp.abs(r['coef']), 4)[0].sum(-1) / (jnp.abs(r['coef']).sum(-1) + 1e-9)).mean()),
             'mlp_out_over_resid': float(jnp.median(ratio))}
        if 'gate_raw' in r:
            raw = r['gate_raw']; d.update(raw_gt_0p9=float((raw > 0.9).mean()), raw_lt_0p1=float((raw < 0.1).mean()))
        rel = lambda k: float(jnp.linalg.norm(gb[k]) / (jnp.linalg.norm(b[k]) + 1e-12))
        d['grad_rel'] = {k: rel(k) for k in ['keys1', 'keys2', 'wqk', 'u', 'v', 'theta'] if k in gb}
        out.append(d); h = h + mlp_out
    return out

def run_gate_diag(results_root, data_dir, jobs, seeds=(0,)):
    import json
    corpus = Corpus(data_dir)
    X = np.concatenate([corpus.eval_windows('val', c, 4, 128, seed=1) for c in ['prose', 'code']])
    res = {}
    for j in jobs:
        for s in seeds:
            d = os.path.join(results_root, f'seed{s}', f'ta0s1_{j}'); ck = os.path.join(d, f'ckpt_seed{s}.pkl')
            if os.path.exists(ck):
                res[f'{j}/seed{s}'] = gate_diag(ck, json.load(open(os.path.join(d, f'seed{s}.json')))['cfg'], X)
    return res

def rescore(results_root, data_dir, jobs, qbits=8, seeds=(0, 1)):
    """Re-evaluate saved checkpoints: validation and train-window loss with and without the routing bias, stability
    and the local frontier with the bias, exact and with a quantized replay (edge values on a qbits grid)."""
    import json, frontier, run
    corpus = Corpus(data_dir); res = {}
    Xtr = corpus.batch(np.random.default_rng(123), 64, 256)
    for j in jobs:
        for s in seeds:
            d = os.path.join(results_root, f'seed{s}', f'ta0s1_{j}'); ck = os.path.join(d, f'ckpt_seed{s}.pkl')
            if not os.path.exists(ck): continue
            cfg = json.load(open(os.path.join(d, f'seed{s}.json')))['cfg']
            st = pickle.load(open(ck, 'rb')); p, bias = st['p'], st['bias']
            Xval = {c: corpus.eval_windows('val', c, cfg['n_eval'] // 2, cfg['T'], seed=0) for c in ['prose', 'code']}
            out = {'val_bias': {c: run.eval_loss(p, cfg, X, bias) for c, X in Xval.items()},
                   'val_nobias': {c: run.eval_loss(p, cfg, X, None) for c, X in Xval.items()},
                   'train_bias': run.eval_loss(p, cfg, Xtr, bias), 'train_nobias': run.eval_loss(p, cfg, Xtr, None),
                   'stability_bias': run.stability(p, cfg, np.concatenate([Xval['prose'][:32], Xval['code'][:32]]), bias),
                   'frontier': {}}
            for c in ['prose', 'code']:
                Xf = corpus.eval_windows('val', c, cfg['frontier_windows'], cfg['frontier_T'], seed=1)
                for qb in [None, qbits]:
                    with jax.default_matmul_precision('highest'):
                        fr = frontier.evaluate(p, cfg, Xf, n_max=cfg['frontier_n_max'], qbits=qb, bias=bias)
                    fr.pop('points'); out['frontier'][f'{c}_q{qb}'] = fr
            res[f'{j}/seed{s}'] = out
    return res
