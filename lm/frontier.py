"""Local explanation frontier, stage 1 (DESIGNS.md v2.2 §1 C4, §11 T-A0).

A *local description* of a forward pass, at budget (m, n, m_a), keeps:
  - per layer and position, the n strongest nodes (units / neurons, ranked by |coefficient| * |write vector|);
  - for every kept read (a node's routing and activation inputs, a head's value input at each source position, and
    the logit vector), only its m largest writer contributions;
  - for every head and query position, only its m_a heaviest source positions, with their logged weights.
Norm gains and attention weights are logged values: the description states them, and they are not recomputed.
The description is executed from the input bytes: every kept read is recomputed from the *replayed* writers through
the kept edges only; dropped edges and nodes contribute nothing. Fidelity is KL(model || replay) on the full next-byte
distribution, per position. Description length counts, per position, every kept node id, edge pointer and b-bit value
(unamortized). The frontier is fidelity against description length over a grid of budgets. Unit selection uses the
load-balancing bias the model was trained with (bias=...), since that is part of its deployed routing.

Stage-1 simplifications (applied identically to every model, so comparisons stay fair):
  - edges are ranked by contribution magnitude in the full run, not by KL-greedy selection or patching (which can
    only improve every model's frontier);
  - edge weights come from the model's parameters (the description points at them), not from a separate b-bit table,
    so this is the local frontier, not yet the global 𝓛 interpreter;
  - dense models keep at most n_max neurons per layer and position as candidate nodes.
At full budget the replay must reproduce the model (KL ~ 0); evaluate() reports this as a correctness check.
"""
import math, numpy as np, jax, jax.numpy as jnp
from model import forward, routing_vectors

def rank_desc(x, axis=-1):
    return jnp.argsort(jnp.argsort(-x, axis=axis), axis=axis)

def setup(p, cfg, x, n_max=256, bias=None):
    """Full run on x (B,T) plus everything the replay needs: writer vectors, node read/write vectors, and the ranks
    that define which edges and nodes a budget keeps."""
    logits, aux = forward(p, cfg, x, bias=bias, record=True)
    rec = aux['record']; B, T = x.shape
    S = {'logits': logits, 'emb': rec['emb'], 'layers': [], 'sf': rec['sf']}
    WV = [rec['emb'][:, :, None, :]]
    for l, (b, r) in enumerate(zip(p['blocks'], rec['layers'])):
        L = {'s1': r['s1'], 's2': r['s2'], 'A': r['A']}
        Wc = jnp.concatenate(WV, 2) * b['ln1']
        nv = jnp.linalg.norm(jnp.einsum('btwd,dhe->bthwe', Wc, b['wv']), axis=-1) * r['s1'][:, :, None, None]
        L['rank_v'] = rank_desc(nv)                                  # (B,T,H,W): value reads over writers
        L['rank_a'] = rank_desc(r['A'])                              # (B,H,T,T): sources per query
        WV.append(r['heads'])
        if cfg['mlp'] == 'peer':
            unit = r['unit']; slots = jnp.broadcast_to(jnp.arange(unit.shape[-1]), unit.shape)
            reads = jnp.stack([routing_vectors(b, cfg, slots, unit), b['u'][unit]], 3)   # (B,T,N,2,d)
            wvec = b['v'][unit]; coef = r['coef']; L.update(unit=unit, theta=b['theta'][unit], bias=None)
        else:
            k = min(n_max, r['coef'].shape[-1])
            imp = jnp.abs(r['coef']) * jnp.linalg.norm(b['w_out'], axis=-1)
            idx = jax.lax.top_k(imp, k)[1]                            # (B,T,N) candidate neurons
            reads = b['w_in'].T[idx][:, :, :, None, :]; wvec = b['w_out'][idx]
            coef = jnp.take_along_axis(r['coef'], idx, -1); L.update(bias=b['b_in'][idx])
        L['reads'], L['wvec'] = reads, wvec
        L['rank_n'] = rank_desc(jnp.abs(coef) * jnp.linalg.norm(wvec, axis=-1))
        Wc = jnp.concatenate(WV, 2) * b['ln2']
        C = jnp.einsum('btwd,btnrd->btnrw', Wc, reads) * r['s2'][:, :, None, None, None]
        L['rank_r'] = rank_desc(jnp.abs(C))                          # (B,T,N,reads,W)
        WV.append(coef[..., None] * wvec)
        S['layers'].append(L)
    Wc = jnp.concatenate(WV, 2) * p['ln_f']
    U = p['unemb'] - p['unemb'].mean(1, keepdims=True)
    S['rank_o'] = rank_desc(jnp.linalg.norm(jnp.einsum('btwd,dv->btwv', Wc, U), axis=-1) * rec['sf'][:, :, None])
    S['n_nodes'] = int(S['layers'][0]['rank_n'].shape[-1])
    return S

def quantize(c, axes, bits):
    """Round contributions to a signed b-bit grid whose step is set by the largest kept term over `axes` (one read).
    A description states each edge value at b bits relative to its read's largest term, so canceling giants cannot
    carry a small net signal through it (critique round 4)."""
    if bits is None: return c
    step = jnp.abs(c).max(axis=axes, keepdims=True) / (2 ** (bits - 1)) + 1e-30
    return jnp.round(c / step) * step

def make_replay(p, cfg, qbits=None):
    """qbits=None: edge values at the model's float precision. qbits=b: every kept edge value, kept attention weight
    and logit contribution is rounded to b bits relative to its read's largest kept term."""
    gate_kind = cfg.get('gate', 'sigmoid')
    q = lambda c, axes: quantize(c, axes, qbits)
    @jax.jit
    def replay(S, m, n, ma):
        WV = [S['emb'][:, :, None, :]]
        for b, L in zip(p['blocks'], S['layers']):
            Wc = jnp.concatenate(WV, 2) * b['ln1']
            Cv = jnp.einsum('btwd,dhe->bthwe', Wc, b['wv']) * L['s1'][:, :, None, None, None]
            v = q(Cv * (L['rank_v'] < m)[..., None], (3, 4)).sum(3)            # (B,T,H,e)
            A = q(L['A'] * (L['rank_a'] < ma), (-1,))
            heads = jnp.einsum('bthe,hed->bthd', jnp.einsum('bhts,bshe->bthe', A, v), b['wo'])
            WV.append(heads)
            Wc = jnp.concatenate(WV, 2) * b['ln2']
            C = jnp.einsum('btwd,btnrd->btnrw', Wc, L['reads']) * L['s2'][:, :, None, None, None]
            pre = q(C * (L['rank_r'] < m), (-1,)).sum(-1)                         # (B,T,N,reads)
            keep = L['rank_n'] < n
            if cfg['mlp'] == 'peer':
                route, act = pre[..., 0], pre[..., 1]
                R, K = cfg['r_heads'], cfg['topk']; sh = route.shape
                if gate_kind == 'softmax':                          # normalizer over the kept units of each head
                    z = jnp.where(keep, route, -1e30).reshape(sh[:-1] + (R, K))
                    g = jax.nn.softmax(z, -1).reshape(sh)
                else:
                    g = jax.nn.sigmoid(route - L['theta'])
                    if gate_kind == 'sigmoid_scaled': g = g / K
                    if gate_kind == 'sigmoid_norm':
                        gk = (g * keep).reshape(sh[:-1] + (R, K))
                        g = (gk / (gk.sum(-1, keepdims=True) + 1e-9)).reshape(sh)
                coef = g * jax.nn.gelu(act) * keep
            else:
                coef = jax.nn.gelu(pre[..., 0] + L['bias']) * keep
            WV.append(coef[..., None] * L['wvec'])
        Wc = jnp.concatenate(WV, 2) * p['ln_f'] * (S['rank_o'] < m)[..., None]
        if qbits is None: return jnp.einsum('btwd,dv->btv', Wc, p['unemb']) * S['sf'][..., None]
        Co = jnp.einsum('btwd,dv->btwv', Wc, p['unemb']) * S['sf'][..., None, None]
        return q(Co, (2, 3)).sum(2)
    return replay

def description_bits(cfg, T, n_nodes, m, n, ma, b=8):
    """Description length per position (bits), unamortized, for budget (m, n, m_a). Returns (T,) array."""
    H, L = cfg['heads'], cfg['layers']
    n_ids = cfg['n_keys'] ** 2 if cfg['mlp'] == 'peer' else cfg.get('ff_mult', 4) * cfg['d']
    reads = 2 if cfg['mlp'] == 'peer' else 1
    t = np.arange(T) + 1.0
    lg = lambda z: np.log2(np.maximum(z, 1))
    bits = np.zeros(T)
    for l in range(L):
        Wa = 1 + l * (H + n_nodes); Wm = Wa + H
        bits += 2 * b                                                          # two logged norm gains
        bits += H * np.minimum(ma, t) * (lg(t) + b)                           # kept sources: position + weight
        bits += H * min(m, Wa) * (lg(Wa) + b)                                 # value-read edges at this position
        bits += min(n, n_nodes) * (lg(n_ids) + b + reads * min(m, Wm) * (lg(Wm) + b))
    Wf = 1 + L * (H + n_nodes)
    return bits + min(m, Wf) * (lg(Wf) + b)

def kl_rows(lf, lr):
    pf = jax.nn.log_softmax(lf, -1); pr = jax.nn.log_softmax(lr, -1)
    return (jnp.exp(pf) * (pf - pr)).sum(-1)

def evaluate(p, cfg, X, grid=None, n_max=256, bits=8, budgets_bits=(256, 512, 1024, 2048, 4096, 8192, 16384),
             qbits=None, bias=None):
    """X: (B, T+1) eval windows. Returns the frontier: every grid point, the Pareto set, KL at bit budgets, and the
    full-budget correctness check."""
    x, y = jnp.asarray(X[:, :-1]), X[:, 1:]
    S = setup(p, cfg, x, n_max, bias); exact = make_replay(p, cfg); replay = make_replay(p, cfg, qbits)
    T = x.shape[1]; N = S['n_nodes']; Wf = 1 + cfg['layers'] * (cfg['heads'] + N)
    nll = lambda lg: float(-jnp.take_along_axis(jax.nn.log_softmax(lg, -1), jnp.asarray(y)[..., None], -1).mean())
    full = exact(S, jnp.int32(Wf), jnp.int32(N), jnp.int32(T))
    out = {'n_nodes': N, 'nll_model': nll(S['logits']), 'full_budget_kl': float(kl_rows(S['logits'], full).mean()),
           'qbits': qbits, 'points': []}
    if qbits is not None:  # full budget, values on the declared grid: what precision alone costs
        out['full_budget_kl_quantized'] = float(kl_rows(S['logits'], replay(S, jnp.int32(Wf), jnp.int32(N), jnp.int32(T))).mean())
    grid = grid or {'m': [1, 2, 4, 8, 16, 32], 'n': [2, 4, 8, 16, 32, 64, 128, 256], 'ma': [1, 2, 4, 8, 16]}
    for m in grid['m']:
        for n in grid['n']:
            if n > N: continue
            for ma in grid['ma']:
                lr = replay(S, jnp.int32(m), jnp.int32(n), jnp.int32(ma))
                kl = np.asarray(kl_rows(S['logits'], lr)).ravel()
                dl = description_bits(cfg, T, N, m, n, ma, bits)
                out['points'].append({'m': m, 'n': n, 'ma': ma, 'bits': float(dl.mean()), 'kl_mean': float(kl.mean()),
                                      'kl_median': float(np.median(kl)), 'kl_p90': float(np.quantile(kl, 0.9)),
                                      'nll': nll(lr)})
    pts = sorted(out['points'], key=lambda q: q['bits'])
    pareto, best = [], float('inf')
    for q in pts:
        if q['kl_mean'] < best: pareto.append(q); best = q['kl_mean']
    out['pareto'] = pareto
    out['kl_at_bits'] = {str(B): min([q['kl_mean'] for q in pts if q['bits'] <= B], default=None) for B in budgets_bits}
    out['p90_at_bits'] = {str(B): min([q['kl_p90'] for q in pts if q['bits'] <= B], default=None) for B in budgets_bits}
    return out
