"""
Models, all with one interface:

    params, masks = init(cfg, key)
    logits, states = apply(params, masks, cfg, x, feat_masks=None, opts=None)

`states` is the list of legible states (one (B,T,F) non-negative sparse code per state boundary). `feat_masks`
is an optional list of the same length with (B,T,F) or (F,) multiplicative masks used for ablation.
`masks` are structural sparsity masks (rulenet only; empty dict otherwise). `opts` are eval-time switches.

Model families
  dense      standard pre-LN transformer. Legible states: none (states = residual streams, dense).
  dense_sae  the null model: a trained dense transformer with post-hoc top-k sparse autoencoders at every
             residual boundary. States = SAE codes. opts['use_error']=1 keeps the SAE error term (the hidden channel);
             0 forces the model to run through the codes only.
  bottleneck the previous session's design (v1/v3 port): sparse code over a tied dictionary at every boundary,
             standard dense block inside; code-space residual with k_write sparse signed writes (v3) when
             cfg['code_residual']. cfg['identity_dict'] makes the dictionary the identity (F = d): "named neurons".
  rulenet    the new design. State is only ever a sparse non-negative code over F named features; there is no
             dense vector anywhere. Input: token v -> feature v (identity). Each layer = ROUTE (per head, a sparse
             feature-compatibility table A[f,g] + relative-distance bias selects a source position; a sparse
             relabel table P[g,f'] copies its features) then COMPUTE (R rules; rule r reads <= fan_in features
             through W1[:, r], thresholds, and writes <= fan_out features through W2[r, :]). Both are sparse
             signed writes into a persistent state: code <- topk(relu(code + delta)). Output: fixed readout —
             the last V features are "next=v" and logits are their activations (times a per-token scale, plus bias).
             Sparsity is structural (masks) and reached by progressive magnitude pruning during training.
"""
import jax, jax.numpy as jnp, numpy as np
from testbed import VOCAB

# ----------------------------------------------------------------------------- shared pieces
def ln(x, g, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * g

def topk_mask(z, k):
    thr = jnp.sort(z, axis=-1)[..., -k][..., None]
    return jnp.where((z >= thr) & (z > 0), z, 0.0)

def attn_dense(x, b, H, T):
    d = x.shape[-1]; hd = d // H
    q = (x @ b['wq']).reshape(*x.shape[:-1], H, hd); k_ = (x @ b['wk']).reshape(*x.shape[:-1], H, hd)
    v = (x @ b['wv']).reshape(*x.shape[:-1], H, hd)
    a = jnp.einsum('bthd,bshd->bhts', q, k_) / jnp.sqrt(hd)
    a = jnp.where(jnp.tril(jnp.ones((T, T), bool))[None, None], a, -1e9)
    a = jax.nn.softmax(a, -1)
    return jnp.einsum('bhts,bshd->bthd', a, v).reshape(*x.shape[:-1], d) @ b['wo']

def _block_init(k, d, ff, s=0.02):
    kk = jax.random.split(k, 6)
    return {'ln1': jnp.ones(d), 'ln2': jnp.ones(d),
            'wq': jax.random.normal(kk[0], (d, d)) * s, 'wk': jax.random.normal(kk[1], (d, d)) * s,
            'wv': jax.random.normal(kk[2], (d, d)) * s, 'wo': jax.random.normal(kk[3], (d, d)) * s,
            'w1': jax.random.normal(kk[4], (d, ff)) * s, 'b1': jnp.zeros(ff),
            'w2': jax.random.normal(kk[5], (ff, d)) * s, 'b2': jnp.zeros(d)}

def _mask_apply(c, fm):
    """fm: None, a multiplicative mask, or a dict with optional 'mask' and 'patch' (patch replaces active values)."""
    if fm is None: return c
    if isinstance(fm, dict):
        if 'mask' in fm: c = c * fm['mask']
        if 'patch' in fm: c = jnp.where(c > 0, fm['patch'], 0.0)
        return c
    return c * fm

# ----------------------------------------------------------------------------- dense
def init_dense(cfg, key):
    d, L = cfg['d'], cfg['layers']; ks = jax.random.split(key, L + 3)
    p = {'emb': jax.random.normal(ks[0], (VOCAB, d)) * 0.02, 'pos': jax.random.normal(ks[1], (cfg['T_max'], d)) * 0.02,
         'out': jax.random.normal(ks[2], (d, VOCAB)) * 0.02, 'ln_f': jnp.ones(d),
         'blocks': [_block_init(ks[3 + l], d, cfg['ff']) for l in range(L)]}
    return p, {}

def apply_dense(p, masks, cfg, x, feat_masks=None, opts=None):
    B, T = x.shape; H = cfg['heads']
    h = p['emb'][x] + p['pos'][:T][None]
    res = [h]
    for b in p['blocks']:
        h = h + attn_dense(ln(h, b['ln1']), b, H, T)
        h = h + (jax.nn.gelu(ln(h, b['ln2']) @ b['w1'] + b['b1']) @ b['w2'] + b['b2'])
        res.append(h)
    return ln(h, p['ln_f']) @ p['out'], res

# ----------------------------------------------------------------------------- dense + post-hoc SAE (null model)
def init_sae(key, d, F):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (F, d)); W = W / jnp.linalg.norm(W, axis=1, keepdims=True)
    return {'enc': W.T * 0.5, 'b_enc': jnp.zeros(F), 'dec': W, 'b_dec': jnp.zeros(d)}

def sae_encode(s, h, k):
    return topk_mask(jax.nn.relu((h - s['b_dec']) @ s['enc'] + s['b_enc']), k)

def sae_decode(s, c):
    return c @ s['dec'] + s['b_dec']

def apply_dense_sae(p, masks, cfg, x, feat_masks=None, opts=None):
    """p = {'dense': dense params, 'saes': [sae per residual boundary]}."""
    opts = opts or {}; use_err = opts.get('use_error', 1.0)
    B, T = x.shape; H = cfg['heads']; k = cfg['k']; pd = p['dense']
    h = pd['emb'][x] + pd['pos'][:T][None]
    states = []
    def through(h, i):
        s = p['saes'][i]; c = sae_encode(s, h, k)
        c = _mask_apply(c, None if feat_masks is None else feat_masks[i])
        c_clean = sae_encode(s, h, k)
        err = h - sae_decode(s, c_clean)
        states.append(c)
        return sae_decode(s, c) + use_err * err
    h = through(h, 0)
    for l, b in enumerate(pd['blocks']):
        h = h + attn_dense(ln(h, b['ln1']), b, H, T)
        h = h + (jax.nn.gelu(ln(h, b['ln2']) @ b['w1'] + b['b1']) @ b['w2'] + b['b2'])
        h = through(h, l + 1)
    return ln(h, pd['ln_f']) @ pd['out'], states

# ----------------------------------------------------------------------------- bottleneck (v1 / v3 port)
def init_bottleneck(cfg, key):
    d, L, F = cfg['d'], cfg['layers'], cfg['F']; ks = jax.random.split(key, L + 4)
    p = {'emb': jax.random.normal(ks[0], (VOCAB, d)) * 0.02, 'pos': jax.random.normal(ks[1], (cfg['T_max'], d)) * 0.02,
         'out': jax.random.normal(ks[2], (d, VOCAB)) * 0.02, 'emb_b': jnp.zeros(F),
         'blocks': [_block_init(ks[4 + l], d, cfg['ff']) for l in range(L)]}
    if not cfg.get('identity_dict', False):
        D = jax.random.normal(ks[3], (F, d)); p['dict'] = D / jnp.linalg.norm(D, axis=1, keepdims=True)
    for b in p['blocks']: b['enc_b'] = jnp.zeros(F)
    return p, {}

def _dict(p, cfg):
    return jnp.eye(cfg['F']) if cfg.get('identity_dict', False) else p['dict']

def apply_bottleneck(p, masks, cfg, x, feat_masks=None, opts=None):
    B, T = x.shape; H, k, kw = cfg['heads'], cfg['k'], cfg['k_write']; D = _dict(p, cfg)
    h = p['emb'][x] + p['pos'][:T][None]
    code = topk_mask(jax.nn.relu(h @ D.T + p['emb_b']), k)
    code = _mask_apply(code, None if feat_masks is None else feat_masks[0])
    h = code @ D; states = [code]; aux = []
    for l, b in enumerate(p['blocks']):
        a = attn_dense(ln(h, b['ln1']), b, H, T)
        m = jax.nn.gelu(ln(h + a, b['ln2']) @ b['w1'] + b['b1']) @ b['w2'] + b['b2']
        upd = a + m
        if cfg.get('code_residual', True):
            dz = upd @ D.T + b['enc_b']
            thr = jnp.sort(jnp.abs(dz), axis=-1)[..., -kw][..., None]
            delta = jnp.where(jnp.abs(dz) >= thr, dz, 0.0)
            z = code + delta
            aux.append((jax.nn.relu(z), upd - delta @ D))
            code = topk_mask(jax.nn.relu(z), k)
        else:
            hn = h + upd; z = hn @ D.T + b['enc_b']
            code = topk_mask(jax.nn.relu(z), k); aux.append((jax.nn.relu(z), hn - code @ D))
        code = _mask_apply(code, None if feat_masks is None else feat_masks[l + 1])
        h = code @ D; states.append(code)
    logits = h @ p['out']
    if opts is not None and opts.get('return_aux'): return logits, states, aux
    return logits, states

def bottleneck_aux_loss(p, cfg, aux, states):
    """AuxK: features unused in the batch reconstruct the missed part of the update (keeps features alive)."""
    D = _dict(p, cfg); tot = 0.0
    for (z, err), c in zip(aux, states[1:]):
        used = (c > 0).any(axis=(0, 1))
        z_dead = jnp.where(used[None, None, :], 0.0, z)
        recon = topk_mask(z_dead, cfg['k_aux']) @ D
        tot = tot + jnp.mean((jax.lax.stop_gradient(err) - recon) ** 2)
    return tot

# ----------------------------------------------------------------------------- rulenet (the new design)
# State at a position = (code: F non-negative hidden features, at most k active; out: V signed output votes).
# Features 0..V-1 are the input features ("cur=v", written by the identity embedding). The output register is not
# subject to top-k and is never read by anything: it only accumulates votes, so logits[v] = out[v]*scale+bias is
# exact per-rule attribution. Everything a layer does is a sparse signed write into (code, out).
def init_rulenet(cfg, key):
    F, R, H, L, Tm = cfg['F'], cfg['R'], cfg['heads'], cfg['layers'], cfg['T_max']
    assert F >= VOCAB, "need at least V input features"
    ks = jax.random.split(key, L + 1)
    p = {'out_scale': jnp.full(VOCAB, cfg.get('out_scale_init', 1.0)), 'out_bias': jnp.zeros(VOCAB), 'layers': []}
    m = {'layers': []}
    if cfg.get('emb', 'identity') == 'learned':
        # token -> non-negative sparse feature set; identity on the first V features plus small random extras
        E = jnp.eye(VOCAB, F) + jnp.abs(jax.random.normal(ks[-1], (VOCAB, F))) * 0.1
        p['emb'] = E; m['emb'] = jnp.ones((VOCAB, F))
    for l in range(L):
        kk = jax.random.split(ks[l], 5)
        lp = {'A': jax.random.normal(kk[0], (H, F, F)) * cfg.get('a_init', 0.3),   # query-key compatibility
              'u': jax.random.normal(kk[1], (H, F)) * cfg.get('a_init', 0.3),      # key-only: "look for g"
              'P': jax.random.normal(kk[2], (H, F, F + VOCAB)) * 0.1,               # relabel/copy table
              'rel': jnp.zeros((H, Tm + 1)),                                        # relative distance bias
              'W1': jax.random.normal(kk[3], (F, R)) * 0.3, 'b1': jnp.zeros(R),     # rule conditions
              'W2': jax.random.normal(kk[4], (R, F + VOCAB)) * 0.1}                 # rule writes
        p['layers'].append(lp)
        m['layers'].append({'A': jnp.ones((H, F, F)), 'u': jnp.ones((H, F)), 'P': jnp.ones((H, F, F + VOCAB)),
                            'W1': jnp.ones((F, R)), 'W2': jnp.ones((R, F + VOCAB))})
    return p, m

def _act(z, cfg):
    z = jax.nn.relu(z)
    c = cfg.get('clamp', 0.0)
    return jnp.minimum(z, c) if c > 0 else z

def route(code, lp, lm, cfg, T):
    """Attention as lookup in feature space. Returns (delta (B,T,F+V), attn (B,H,T,T+1)); last column = null source."""
    A = lp['A'] * lm['A']; u = lp['u'] * lm['u']; P = lp['P'] * lm['P']
    S = jnp.einsum('btf,hfg,bsg->bhts', code, A, code) + jnp.einsum('hg,bsg->bhs', u, code)[:, :, None, :]
    dist = jnp.arange(T)[:, None] - jnp.arange(T)[None, :]                    # t - s
    rel = jnp.take(lp['rel'], jnp.clip(dist, 0, lp['rel'].shape[1] - 1), axis=1)   # (H,T,T)
    S = S + rel[None]
    S = jnp.where((dist > 0)[None, None], S, -1e9)                            # strictly causal, no self
    S = jnp.concatenate([S, jnp.zeros(S.shape[:-1] + (1,))], -1)              # null source, score 0
    a = jax.nn.softmax(S, -1)
    src = jnp.concatenate([code, jnp.zeros(code.shape[:1] + (1, code.shape[-1]))], 1)
    copied = jnp.einsum('bhts,bsg->bhtg', a, src)
    return jnp.einsum('bhtg,hgf->btf', copied, P), a

def compute(code, lp, lm, cfg):
    hid = _act(code @ (lp['W1'] * lm['W1']) + lp['b1'], cfg)
    return hid @ (lp['W2'] * lm['W2']), hid

def apply_rulenet(p, masks, cfg, x, feat_masks=None, opts=None):
    opts = opts or {}
    B, T = x.shape; F, k = cfg['F'], cfg['k']
    if 'emb' in p:
        code = jax.nn.relu(p['emb'] * masks['emb'])[x]                       # token v -> its (sparse) feature set
    else:
        code = jax.nn.one_hot(x, F)                                           # token v -> feature v ("cur=v")
    out = jnp.zeros((B, T, VOCAB))
    code = _mask_apply(code, None if feat_masks is None else feat_masks[0])
    states, attns, hids, writes = [code], [], [], []
    i = 1
    for lp, lm in zip(p['layers'], masks['layers']):
        delta, a = route(code, lp, lm, cfg, T); attns.append(a); writes.append(delta)
        code = topk_mask(_act(code + delta[..., :F], cfg), k); out = out + delta[..., F:]
        code = _mask_apply(code, None if feat_masks is None else feat_masks[i]); states.append(code); i += 1
        delta, hid = compute(code, lp, lm, cfg); hids.append(hid); writes.append(delta)
        code = topk_mask(_act(code + delta[..., :F], cfg), k); out = out + delta[..., F:]
        code = _mask_apply(code, None if feat_masks is None else feat_masks[i]); states.append(code); i += 1
    logits = out * p['out_scale'] + p['out_bias']
    if opts.get('return_internals'): return logits, states, {'attn': attns, 'hid': hids, 'writes': writes, 'out': out}
    return logits, states

# ----------------------------------------------------------------------------- registry
INIT = {'dense': init_dense, 'bottleneck': init_bottleneck, 'rulenet': init_rulenet}
APPLY = {'dense': apply_dense, 'dense_sae': apply_dense_sae, 'bottleneck': apply_bottleneck, 'rulenet': apply_rulenet}
PRUNE = {}
REG = {}

def n_states(cfg):
    return {'dense': cfg['layers'] + 1, 'dense_sae': cfg['layers'] + 1, 'bottleneck': cfg['layers'] + 1,
            'rulenet': 2 * cfg['layers'] + 1, 'rulenet_g': 2 * cfg['layers'] + 1}[cfg['model']]

def state_dim(cfg):
    return cfg['d'] if cfg['model'] == 'dense' else cfg['F']

# ----------------------------------------------------------------------------- pruning (rulenet)
# Each structure is viewed as (slices, entries): W1 per rule (column), W2 per rule (row), A/u/P per head, emb per token.
_VIEWS = {  # name -> (to 2D view, from 2D view) given cfg
    'W1': (lambda w: w.T, lambda w2, shape: w2.T),
    'W2': (lambda w: w, lambda w2, shape: w2),
    'A': (lambda w: w.reshape(w.shape[0], -1), lambda w2, shape: w2.reshape(shape)),
    'u': (lambda w: w, lambda w2, shape: w2),
    'P': (lambda w: w.reshape(w.shape[0], -1), lambda w2, shape: w2.reshape(shape)),
    'emb': (lambda w: w, lambda w2, shape: w2),
}

def _targets(cfg):
    F = cfg['F']
    return {'W1': (F, cfg['fan_in']), 'W2': (F + VOCAB, cfg['fan_out']), 'A': (F * F, cfg['n_a']),
            'u': (F, cfg['n_u']), 'P': (F * (F + VOCAB), cfg['n_p']), 'emb': (F, cfg.get('emb_fan', 3))}

def _select(w, m, g, n_keep, n_grow):
    """w, m, g: (slices, entries). Keep exactly the n_keep - n_grow largest |w| among currently unmasked entries,
    then grow the n_grow largest |g| among the remaining entries (weights reset to 0). Returns (mask, w)."""
    n_keep = max(int(n_keep), 1); n_grow = min(max(1, int(round(n_grow))), n_keep - 1) if (g is not None and n_grow > 0) else 0
    E = w.shape[1]; n_keep = min(n_keep, E)
    score = jnp.abs(w) * m + (m - 1) * 1e9
    idx = jax.lax.top_k(score, n_keep - n_grow)[1]
    keep = jnp.zeros_like(w, dtype=bool).at[jnp.arange(w.shape[0])[:, None], idx].set(True) & (m > 0)
    if n_grow > 0:
        gs = jnp.where(keep, -1e9, jnp.abs(g))
        gidx = jax.lax.top_k(gs, n_grow)[1]
        grow = jnp.zeros_like(w, dtype=bool).at[jnp.arange(w.shape[0])[:, None], gidx].set(True) & ~keep
        w = jnp.where(grow, 0.0, w)
        keep = keep | grow
    return keep.astype(w.dtype), w

def prune_masks(p, masks, cfg, frac, grads=None, regrow_frac=0.0):
    """Prune every structure to a fraction `frac` of the way (log-linear) from dense to its target size, optionally
    swapping `regrow_frac` of the kept entries for the highest-gradient masked-out entries (RigL-style regrowth).
    frac=1 -> target sizes. Returns (masks, params) — params change only where entries were regrown."""
    tg = _targets(cfg); groups = cfg.get('prune_groups', ['W1', 'W2', 'A', 'u', 'P', 'emb'])
    def keep_n(dense_n, target_n):
        return int(round(np.exp(np.log(dense_n) + frac * (np.log(target_n) - np.log(dense_n)))))
    def do(name, w, m, g):
        if name not in groups: return m, w
        to2, from2 = _VIEWS[name]; dn, tn = tg[name]; n = keep_n(dn, tn)
        m2, w2 = _select(to2(w), to2(m), None if g is None else to2(g), n, regrow_frac * n)
        return from2(m2, w.shape), from2(w2, w.shape)
    p = dict(p); new = {'layers': []}; p['layers'] = list(p['layers'])
    if 'emb' in masks:
        new['emb'], p['emb'] = do('emb', p['emb'], masks['emb'], None if grads is None else grads['emb'])
    for l, (lp, lm) in enumerate(zip(p['layers'], masks['layers'])):
        lp = dict(lp); nm = {}
        for name in ['W1', 'W2', 'A', 'u', 'P']:
            nm[name], lp[name] = do(name, lp[name], lm[name], None if grads is None else grads['layers'][l][name])
        new['layers'].append(nm); p['layers'][l] = lp
    return new, p

def mask_stats(masks):
    out = {'emb': int(masks['emb'].sum())} if 'emb' in masks else {}
    for l, lm in enumerate(masks.get('layers', [])):
        out[f'L{l}'] = {k: int(v.sum()) for k, v in lm.items()}
    return out

PRUNE['rulenet'] = lambda p, masks, cfg, frac, grads=None, regrow_frac=0.0: prune_masks(p, masks, cfg, frac, grads, regrow_frac)
def _reg_rulenet(p, masks, cfg):
    reg = 0.0
    for lp, lm in zip(p['layers'], masks['layers']):
        reg = reg + sum(jnp.abs(lp[n] * lm[n]).sum() for n in ['A', 'u', 'P', 'W1', 'W2'])
    if 'emb' in p: reg = reg + jnp.abs(p['emb'] * masks['emb']).sum()
    return reg
REG['rulenet'] = _reg_rulenet
from models_g import init_rulenet_g, apply_rulenet_g, prune_rulenet_g, reg_rulenet_g
INIT['rulenet_g'] = init_rulenet_g; APPLY['rulenet_g'] = apply_rulenet_g; PRUNE['rulenet_g'] = prune_rulenet_g; REG['rulenet_g'] = reg_rulenet_g
