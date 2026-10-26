"""Stage-1b local frontier and faithful-replay core (docs/redesign/FAITHFUL-REPLAY-PRESSURE.md v3.1). PEER models only.

A local description at budget (m, n, m_a) is a sparse, quantized re-execution of the forward pass from the input
bytes. Every read of the residual stream sums only its m largest writer contributions:
  - the query, key and value input of every attention head;
  - every retrieval head's routing query;
  - every kept unit's activation input;
  - the logit vector.
Each kept contribution is rounded on a b-bit grid whose step comes from a power-of-two scale at or above the read's
largest kept term. Every term is rounded, including the largest; the scale costs 5 bits.

  sel='free'   everything else is derived from the replay itself: norm gains from the replayed residual; attention
               weights from the replayed q, k (top m_a sources per row, weights quantized); unit selection by product
               keys on the replayed routing query (with the trained routing bias); then the top n units per position by
               replayed write size, with the gate renormalized over the kept units.
  sel='given'  the full run's norm gains, attention weights and unit selection are taken as stated facts (an upper
               bound, close to stage-1 semantics).
Edges are ranked by the replay's own contribution magnitudes, so the description is self-consistent.
Description length counts, per position and unamortized: every kept edge (pointer + b bits), every read's scale
(5 bits), every kept unit id and every kept attention source (position + weight). Derived quantities cost nothing.

The same replay() is used as the training pressure (run.py), with quant='noise' (uniform +-1/2-step noise, step from
the read's continuous largest term times r ~ log-uniform[1,2), so the gradient sees the cost of large terms) or
quant='ste' (strict rounding with a straight-through gradient).
"""
import math, numpy as np, jax, jax.numpy as jnp
from model import forward, gain, rope, select_from_query, peer_gate

SCALE_BITS = 5

def rank_desc(x, axis=-1):
    return jnp.argsort(jnp.argsort(-x, axis=axis), axis=axis)

def _quant(c, axes, quant, key):
    """quant: None | ('strict', b) | ('ste', b) | ('noise', b)."""
    if quant is None: return c
    kind, b = quant
    mx = jnp.abs(c).max(axis=axes, keepdims=True)
    if kind == 'noise':
        r = jnp.exp(jax.random.uniform(key, mx.shape) * math.log(2.0))
        step = mx * r / 2 ** (b - 1) + 1e-30
        return c + (jax.random.uniform(jax.random.fold_in(key, 1), c.shape) - 0.5) * step
    scale = jax.lax.stop_gradient(2.0 ** jnp.ceil(jnp.log2(mx + 1e-30)))
    step = scale / 2 ** (b - 1)
    q = jnp.round(c / step) * step
    return q if kind == 'strict' else c + jax.lax.stop_gradient(q - c)

class _Keys:
    def __init__(self, key): self.key, self.i = key, 0
    def __call__(self):
        self.i += 1
        return None if self.key is None else jax.random.fold_in(self.key, self.i)

def _read_vec(C, m, quant, keys):
    """C: (..., W, e) per-writer vector contributions of one read -> (..., e), keeping the m largest by norm."""
    keep = rank_desc(jnp.sqrt((C * C).sum(-1) + 1e-30)) < m
    return _quant(C * keep[..., None], (-2, -1), quant, keys()).sum(-2)

def _read_scalar(C, m, quant, keys):
    """C: (..., W) per-writer scalar contributions -> (...,), keeping the m largest by magnitude."""
    keep = rank_desc(jnp.abs(C)) < m
    return _quant(C * keep, (-1,), quant, keys()).sum(-1)

def replay(p, cfg, x, rec, m, n, ma, sel='free', quant=None, key=None, bias=None, want=()):
    """Sparse quantized re-execution of the model on x (B,T) from a description at budget (m, n, m_a).
    rec: the full run's record (forward(..., record=True)), used for sel='given' and for diagnostics.
    want: extra outputs: 'select' (replayed unit ids per layer), 'qr' (routing queries, for the ranking hinge).
    Returns (logits (B,T,V), aux)."""
    keys = _Keys(key); B, T = x.shape; H = cfg['heads']; R, K = cfg['r_heads'], cfg['topk']
    WV = [p['emb'][x][:, :, None, :]]
    aux = {'select': [], 'qr': [], 'step_qr': []}
    causal = jnp.tril(jnp.ones((T, T), bool))
    for l, b in enumerate(p['blocks']):
        L = rec['layers'][l]
        Wc = jnp.concatenate(WV, 2)
        s1 = gain(Wc.sum(2))[..., None] if sel == 'free' else L['s1'][..., None, None]
        X = Wc * b['ln1'] * s1                                                    # (B,T,W,d)
        v = _read_vec(jnp.einsum('btwd,dhe->bthwe', X, b['wv']), m, quant, keys)  # (B,T,H,e)
        if sel == 'free':
            q = _read_vec(jnp.einsum('btwd,dhe->bthwe', X, b['wq']), m, quant, keys)
            k = _read_vec(jnp.einsum('btwd,dhe->bthwe', X, b['wk']), m, quant, keys)
            e = q.shape[-1]
            att = jnp.einsum('bhte,bhse->bhts', rope(q.transpose(0, 2, 1, 3)), rope(k.transpose(0, 2, 1, 3))) / math.sqrt(e)
            A = jax.nn.softmax(jnp.where(causal, att, -1e30), -1)
        else:
            A = L['A']
        A = _quant(A * (rank_desc(A) < ma), (-1,), quant, keys())
        heads = jnp.einsum('bthe,hed->bthd', jnp.einsum('bhts,bshe->bthe', A, v), b['wo'])
        WV.append(heads)
        Wc = jnp.concatenate(WV, 2)
        s2 = gain(Wc.sum(2))[..., None] if sel == 'free' else L['s2'][..., None, None]
        X = Wc * b['ln2'] * s2
        Cr = jnp.einsum('btwd,drk->btrwk', X, b['wqk'])                          # (B,T,R,W,dk)
        qr = _read_vec(Cr, m, quant, keys)                                       # (B,T,R,dk)
        if 'qr' in want:
            aux['qr'].append(qr)
            aux['step_qr'].append(2.0 ** jnp.ceil(jnp.log2(jnp.abs(Cr).max((-2, -1)) + 1e-30)) / 2 ** 7)
        if sel == 'free':
            unit, score, _, _ = select_from_query(b, qr, cfg, None if bias is None else bias[l])
        else:
            unit = L['unit'].reshape(B, T, R, K); nk = cfg['n_keys']; dk = cfg['d_key']
            score = (jnp.einsum('btrk,btrjk->btrj', qr[..., :dk // 2], b['keys1'][unit // nk]) +
                     jnp.einsum('btrk,btrjk->btrj', qr[..., dk // 2:], b['keys2'][unit % nk]))
        if 'select' in want: aux['select'].append(unit)
        theta = b['theta'][unit]
        act = jax.nn.gelu(_read_scalar(jnp.einsum('btwd,btrjd->btrjw', X, b['u'][unit]), m, quant, keys))
        V = b['v'][unit]                                                          # (B,T,R,K,d)
        g0, _ = peer_gate(score, theta, cfg)
        imp = (jnp.abs(g0 * act) * jnp.sqrt((V * V).sum(-1) + 1e-30)).reshape(B, T, R * K)
        keep = (rank_desc(jax.lax.stop_gradient(imp)) < n).reshape(B, T, R, K)
        g, _ = peer_gate(score, theta, cfg, keep=keep)
        coef = (g * act * keep).reshape(B, T, R * K)
        WV.append(coef[..., None] * V.reshape(B, T, R * K, -1))
    Wc = jnp.concatenate(WV, 2)
    sf = gain(Wc.sum(2))[..., None] if sel == 'free' else rec['sf'][..., None, None]
    Co = jnp.einsum('btwd,dv->btwv', Wc * p['ln_f'] * sf, p['unemb'])
    return _read_vec(Co, m, quant, keys), aux

def description_bits(cfg, T, m, n, ma, sel='free', b=8):
    """Bits per position (T,), unamortized."""
    H, L, R, K = cfg['heads'], cfg['layers'], cfg['r_heads'], cfg['topk']
    RK = R * K; ids = math.log2(cfg['n_keys'] ** 2)
    t = np.arange(T) + 1.0
    edges = lambda W: min(m, W) * (math.log2(W) + b) + SCALE_BITS
    bits = np.zeros(T)
    for l in range(L):
        Wa = 1 + l * (H + RK); Wm = Wa + H
        reads_attn = 3 if sel == 'free' else 1                      # q, k, v (free) or v only (given)
        bits += H * (reads_attn * edges(Wa) + np.minimum(ma, t) * (np.log2(t) + b) + SCALE_BITS)
        bits += R * edges(Wm) + min(n, RK) * (ids + edges(Wm))
        if sel == 'given': bits += 2 * b                              # stated norm gains
    Wf = 1 + L * (H + RK)
    return bits + edges(Wf) + (b if sel == 'given' else 0)

# ------------------------------------------------------------------------------------------------ fidelity metrics
def byte_class(y):
    """0 letter, 1 digit, 2 whitespace, 3 other."""
    y = np.asarray(y)
    letter = ((y >= 65) & (y <= 90)) | ((y >= 97) & (y <= 122))
    digit = (y >= 48) & (y <= 57); space = (y == 32) | (y == 10) | (y == 9) | (y == 13)
    return np.where(letter, 0, np.where(digit, 1, np.where(space, 2, 3)))

class NgramNull:
    """Fixed order-2 byte n-gram null for the explained fraction: P = .6 P3 + .3 P2 + .1 P1, add-0.01 smoothing."""
    def __init__(self, train_bytes=None, path=None):
        if path is not None and train_bytes is None:
            z = np.load(path); self.c3, self.c2, self.c1 = z['c3'], z['c2'], z['c1']; return
        d = np.asarray(train_bytes, np.int64)
        self.c1 = np.bincount(d, minlength=256).astype(np.float64)
        self.c2 = np.bincount(d[:-1] * 256 + d[1:], minlength=256 ** 2).reshape(256, 256).astype(np.float32)
        self.c3 = np.bincount(d[:-2] * 65536 + d[1:-1] * 256 + d[2:], minlength=256 ** 3).reshape(256, 256, 256).astype(np.float32)
        if path is not None: np.savez_compressed(path, c3=self.c3, c2=self.c2, c1=self.c1)
    def nll(self, X):
        """X: (B, T+1) windows -> (B, T) null NLL of X[:, 1:] (the first position backs off to order 1)."""
        X = np.asarray(X, np.int64); prev2 = np.concatenate([X[:, :1], X[:, :-2]], 1); prev1 = X[:, :-1]; y = X[:, 1:]
        p1 = (self.c1[y] + 0.01) / (self.c1.sum() + 2.56)
        p2 = (self.c2[prev1, y] + 0.01) / (self.c2[prev1].sum(-1) + 2.56)
        p3 = (self.c3[prev2, prev1, y] + 0.01) / (self.c3[prev2, prev1].sum(-1) + 2.56)
        return -np.log(0.6 * p3 + 0.3 * p2 + 0.1 * p1)

def _interp_bits(pts, kl_key, target):
    """Interpolated bits (log-linear) on the Pareto set where kl_key first reaches target; None if never reached."""
    pts = sorted(pts, key=lambda q: q['bits']); par, best = [], float('inf')
    for q in pts:
        if q[kl_key] < best: par.append(q); best = q[kl_key]
    for a, c in zip(par, par[1:]):
        if a[kl_key] > target >= c[kl_key]:
            f = (a[kl_key] - target) / (a[kl_key] - c[kl_key] + 1e-12)
            return float(np.exp(np.log(a['bits']) + f * (np.log(c['bits']) - np.log(a['bits']))))
    return float(par[0]['bits']) if par and par[0][kl_key] <= target else None

def _kl_at_bits(pts, kl_key, B):
    """Lowest kl_key among points with bits <= B, linearly interpolated in log-bits between Pareto points."""
    pts = sorted(pts, key=lambda q: q['bits']); par, best = [], float('inf')
    for q in pts:
        if q[kl_key] < best: par.append(q); best = q[kl_key]
    below = [q for q in par if q['bits'] <= B]
    if not below: return None
    a = below[-1]; above = [q for q in par if q['bits'] > B]
    if not above: return float(a[kl_key])
    c = above[0]; f = (np.log(B) - np.log(a['bits'])) / (np.log(c['bits']) - np.log(a['bits']))
    return float(a[kl_key] + f * (c[kl_key] - a[kl_key]))

GRID = {'m': [2, 3, 4, 6, 8, 12, 16, 24, 32], 'n': [4, 6, 8, 12, 16, 24, 32, 48, 64], 'ma': [2, 4, 8, 16]}

def evaluate(p, cfg, X, cat, null, bias=None, modes=(('free', ('strict', 8)), ('given', ('strict', 8)),
             ('free', ('noise', 8)), ('free', ('strict', 6))), grid=None, budgets=(30000, 60000), n_boot=200, seed=0):
    """X: (Nw, T+1) windows of one category. Returns per mode: the grid points (bits, KL mean/median/p90, EF overall
    and per byte class), interpolated bits@KL_p90 in {0.1, 0.3}, KL_p90 and EF at fixed bit budgets, bootstrap CIs over
    windows, and (free mode) selection agreement. Also the full-budget correctness check (exact, given and free)."""
    grid = grid or GRID
    x = jnp.asarray(X[:, :-1]); y = np.asarray(X[:, 1:]); T = x.shape[1]
    with jax.default_matmul_precision('highest'):
        logits, aux = forward(p, cfg, x, bias=bias, record=True)
    rec = jax.tree.map(jax.lax.stop_gradient, aux['record'])
    lp_m = jax.nn.log_softmax(logits, -1)
    nll_m = np.asarray(-jnp.take_along_axis(lp_m, jnp.asarray(y)[..., None], -1)[..., 0])
    nll_0 = null.nll(X); cls = byte_class(y)
    RK = cfg['r_heads'] * cfg['topk']; Wf = 1 + cfg['layers'] * (cfg['heads'] + RK)
    fns = {}
    def get_fn(sel, quant):
        k = (sel, quant)
        if k not in fns:
            fns[k] = jax.jit(lambda p, m, n, ma, key: replay(p, cfg, x, rec, m, n, ma, sel, quant, key, bias, ('select',)))
        return fns[k]
    def stats(lr):
        lr = jax.nn.log_softmax(lr, -1)
        kl = np.asarray((jnp.exp(lp_m) * (lp_m - lr)).sum(-1))
        nll_r = np.asarray(-jnp.take_along_axis(lr, jnp.asarray(y)[..., None], -1)[..., 0])
        return kl, nll_r
    out = {'cat': cat, 'nll_model': float(nll_m.mean()), 'nll_null': float(nll_0.mean()), 'modes': {}}
    key = jax.random.PRNGKey(seed)
    with jax.default_matmul_precision('highest'):
        for sel in ['given', 'free']:
            lr, _ = get_fn(sel, None)(p, jnp.int32(Wf), jnp.int32(RK), jnp.int32(T), key)
            out[f'full_budget_kl_{sel}_exact'] = float(stats(lr)[0].mean())
        for sel, quant in modes:
            name = f"{sel}_{quant[0]}{quant[1]}"; fn = get_fn(sel, quant); pts = []; per_window = []
            for m in grid['m']:
                for n in [v for v in grid['n'] if v <= RK]:
                    for ma in grid['ma']:
                        lr, ra = fn(p, jnp.int32(m), jnp.int32(n), jnp.int32(ma), key)
                        kl, nll_r = stats(lr)
                        bits = float(description_bits(cfg, T, m, n, ma, sel).mean())
                        num = nll_r - nll_m; den = nll_0 - nll_m
                        q = {'m': m, 'n': n, 'ma': ma, 'bits': bits, 'kl_mean': float(kl.mean()),
                             'kl_median': float(np.median(kl)), 'kl_p90': float(np.quantile(kl, 0.9)),
                             'ef': float(1 - num.sum() / den.sum()),
                             'ef_by_class': {str(c): float(1 - num[cls == c].sum() / den[cls == c].sum())
                                             for c in range(4) if (cls == c).any()}}
                        if sel == 'free':
                            q['sel_agree'] = _selection_agreement(rec, ra['select'], n, cfg)
                        pts.append(q); per_window.append((kl, num, den))
            res = {'points': pts, 'bits_at_p90': {str(t): _interp_bits(pts, 'kl_p90', t) for t in (0.1, 0.3)},
                   'p90_at_bits': {str(B): _kl_at_bits(pts, 'kl_p90', B) for B in budgets},
                   'ef_at_bits': {str(B): _ef_at_bits(pts, B) for B in budgets}}
            res['boot'] = _bootstrap(pts, per_window, budgets, n_boot, seed)
            out['modes'][name] = res
    return out

def _ef_at_bits(pts, B):
    ok = [q for q in pts if q['bits'] <= B]
    return max((q['ef'] for q in ok), default=None)

def _selection_agreement(rec, sel_rep, n, cfg):
    """Mass-weighted and unweighted agreement between the replay's selected units and the full run's top-n units."""
    out = []
    for L, ur in zip(rec['layers'], sel_rep):
        uf = L['unit']; w = jnp.abs(L['coef'])                                  # (B,T,RK)
        top = rank_desc(w) < n
        ur = ur.reshape(ur.shape[0], ur.shape[1], -1)
        present = (uf[..., :, None] == ur[..., None, :]).any(-1)
        out.append((float((w * top * present).sum() / ((w * top).sum() + 1e-9)),
                    float((top * present).sum() / (top.sum() + 1e-9))))
    return {'mass': float(np.mean([a for a, _ in out])), 'unweighted': float(np.mean([b for _, b in out]))}

def _bootstrap(pts, per_window, budgets, n_boot, seed):
    """Percentile CIs (5, 95) over windows for bits@p90 in {0.1, 0.3} and p90@budgets."""
    rng = np.random.default_rng(seed); Nw = per_window[0][0].shape[0]; draws = {}
    for _ in range(n_boot):
        idx = rng.integers(0, Nw, Nw)
        bp = [{'bits': q['bits'], 'kl_p90': float(np.quantile(kl[idx], 0.9))} for q, (kl, _, _) in zip(pts, per_window)]
        for t in (0.1, 0.3):
            v = _interp_bits(bp, 'kl_p90', t); draws.setdefault(f'bits_at_p90_{t}', []).append(np.nan if v is None else v)
        for B in budgets:
            v = _kl_at_bits(bp, 'kl_p90', B); draws.setdefault(f'p90_at_{B}', []).append(np.nan if v is None else v)
    ci = lambda a: [float(np.nanpercentile(a, 5)), float(np.nanpercentile(a, 95)), float(np.mean(np.isnan(a)))]
    return {k: ci(np.array(v)) for k, v in draws.items()}
