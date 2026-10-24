"""Byte-level transformer for the real-text experiments, with two MLP kinds and ledger instrumentation.

mlp='dense': a standard GELU MLP. mlp='peer': a PEER-style pool of n_keys**2 single-neuron units (He 2024; product
keys, Lample et al. 2019). Each of r_heads retrieval heads picks its top-k units exactly for the factorized score
s = q1.k1[a] + q2.k2[b]; unit u then has coefficient gate(s) * gelu(u_u . x) and writes coefficient * v_u.
  gate='softmax'  PEER's router: softmax over the selected units (couples them)
  gate='sigmoid'  sigma(s - theta_u): a per-unit threshold, no coupling (DESIGNS.md v2.2, BB2)
Selection may add a load-balancing bias per half-key (aux-loss-free, updated outside the gradient); the bias only
affects which units are selected, never their coefficients.

Ledger view (DESIGNS.md §7): every write into the residual stream is the token embedding, one attention head's
output, or one unit's (neuron's) write. Norm gains are scalars, so every read of the residual -- a unit's routing or
activation input, a head's value input, a logit -- is an exact sum of per-writer contributions given the gains.
  forward(..., press=...)  also returns the description-pressure terms on sampled positions (BB10): for each sampled
                           read, the share of its gross contribution mass that comes from writers outside its top-m.
  forward(..., record=True) also returns what frontier.py needs to replay the model from a truncated description.
"""
import math, jax, jax.numpy as jnp

def init_params(cfg, key):
    d, L, H, V = cfg['d'], cfg['layers'], cfg['heads'], cfg.get('vocab', 256)
    e = d // H
    ks = iter(jax.random.split(key, 8 + 12 * L))
    nrm = lambda shape, scale: scale * jax.random.normal(next(ks), shape)
    out_scale = 1 / math.sqrt(2 * L)
    p = {'emb': nrm((V, d), 1.0), 'ln_f': jnp.ones(d), 'unemb': nrm((d, V), d ** -0.5), 'blocks': []}
    for _ in range(L):
        b = {'ln1': jnp.ones(d), 'ln2': jnp.ones(d), 'wq': nrm((d, H, e), d ** -0.5), 'wk': nrm((d, H, e), d ** -0.5),
             'wv': nrm((d, H, e), d ** -0.5), 'wo': nrm((H, e, d), d ** -0.5 * out_scale)}
        if cfg['mlp'] == 'dense':
            f = cfg.get('ff_mult', 4) * d
            b.update(w_in=nrm((d, f), d ** -0.5), b_in=jnp.zeros(f), w_out=nrm((f, d), f ** -0.5 * out_scale))
        else:
            n, dk, R = cfg['n_keys'], cfg['d_key'], cfg['r_heads']
            b.update(wqk=nrm((d, R, dk), d ** -0.5), keys1=nrm((n, dk // 2), (dk // 2) ** -0.5),
                     keys2=nrm((n, dk // 2), (dk // 2) ** -0.5), u=nrm((n * n, d), d ** -0.5),
                     v=nrm((n * n, d), (R * cfg['topk']) ** -0.5 * out_scale), theta=jnp.zeros(n * n))
        p['blocks'].append(b)
    return p

def init_balance(cfg):
    if cfg['mlp'] != 'peer': return None
    return [jnp.zeros((2, cfg['n_keys'])) for _ in range(cfg['layers'])]

def gain(h, eps=1e-6):
    """RMSNorm's scalar gain s (B,T,1): normalized input = s * gamma * h."""
    return jax.lax.rsqrt(jnp.mean(h * h, -1, keepdims=True) + eps)

def rope(x, base=10000.0):
    T, e = x.shape[-2], x.shape[-1]; half = e // 2
    ang = jnp.arange(T)[:, None] * base ** (-jnp.arange(half) / half)[None]
    c, s = jnp.cos(ang), jnp.sin(ang); x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)

def attention(b, x):
    """x: normalized input (B,T,d). Returns per-head writes (B,T,H,d) and attention weights (B,H,T,T)."""
    e = b['wq'].shape[-1]; T = x.shape[1]
    q = rope(jnp.einsum('btd,dhe->bhte', x, b['wq'])); k = rope(jnp.einsum('btd,dhe->bhte', x, b['wk']))
    v = jnp.einsum('btd,dhe->bhte', x, b['wv'])
    att = jnp.einsum('bhte,bhse->bhts', q, k) / math.sqrt(e)
    att = jnp.where(jnp.tril(jnp.ones((T, T), bool)), att, -1e30)
    A = jax.nn.softmax(att, -1)
    heads = jnp.einsum('bhte,hed->bthd', jnp.einsum('bhts,bhse->bhte', A, v), b['wo'])
    return heads, A

def peer_select(b, x, cfg, bias=None):
    """Exact top-k per retrieval head for s = q1.k1[a] + q2.k2[b]. Returns unit ids, true scores (B,T,R,K), half-key
    ids, and the routing read vectors' factors."""
    n, dk, K = cfg['n_keys'], cfg['d_key'], cfg['topk']
    q = jnp.einsum('btd,drk->btrk', x, b['wqk'])
    s1 = jnp.einsum('btrk,nk->btrn', q[..., :dk // 2], b['keys1'])
    s2 = jnp.einsum('btrk,nk->btrn', q[..., dk // 2:], b['keys2'])
    z1, z2 = (s1, s2) if bias is None else (s1 + jax.lax.stop_gradient(bias[0]), s2 + jax.lax.stop_gradient(bias[1]))
    v1, i1 = jax.lax.top_k(z1, K); v2, i2 = jax.lax.top_k(z2, K)
    cand = (v1[..., :, None] + v2[..., None, :]).reshape(v1.shape[:-1] + (K * K,))
    _, ci = jax.lax.top_k(cand, K)
    a = jnp.take_along_axis(i1, ci // K, -1); c = jnp.take_along_axis(i2, ci % K, -1)
    score = jnp.take_along_axis(s1, a, -1) + jnp.take_along_axis(s2, c, -1)
    return a * n + c, score, a, c

def peer_gate(score, theta, cfg):
    if cfg.get('gate', 'sigmoid') == 'softmax': return jax.nn.softmax(score, -1)
    return jax.nn.sigmoid(score - theta)

def mlp(b, x, cfg, bias=None):
    """Returns (out (B,T,d), node dict). Nodes are the units (peer: R*K slots) or neurons (dense)."""
    B, T, _ = x.shape
    if cfg['mlp'] == 'dense':
        pre = x @ b['w_in'] + b['b_in']; act = jax.nn.gelu(pre)
        return act @ b['w_out'], {'pre': pre, 'coef': act}
    unit, score, a, c = peer_select(b, x, cfg, bias)
    g = peer_gate(score, b['theta'][unit], cfg)
    pre = jnp.einsum('btd,btrkd->btrk', x, b['u'][unit])
    coef = g * jax.nn.gelu(pre)
    out = jnp.einsum('btrk,btrkd->btd', coef, b['v'][unit])
    flat = lambda z: z.reshape(B, T, -1)
    return out, {'unit': flat(unit), 'score': flat(score), 'gate': flat(g), 'pre': flat(pre), 'coef': flat(coef),
                 'a': flat(a), 'c': flat(c)}

def routing_vectors(b, cfg, unit_slots, unit):
    """Read vector of each selected unit's routing score: s = x . r with r = Wq_r[:, :dk/2] k1[a] + Wq_r[:, dk/2:] k2[b].
    unit_slots: slot index (..., ) in [0, R*K) -> retrieval head = slot // K; unit: unit id (...,)."""
    n, dk, K = cfg['n_keys'], cfg['d_key'], cfg['topk']
    rv1 = jnp.einsum('drk,nk->rnd', b['wqk'][..., :dk // 2], b['keys1'])   # (R, n, d): per head, per half-key
    rv2 = jnp.einsum('drk,nk->rnd', b['wqk'][..., dk // 2:], b['keys2'])
    r = unit_slots // K
    return rv1[r, unit // n] + rv2[r, unit % n]

def safe_norm(z, axis=-1):
    """Euclidean norm with a finite gradient at zero (zero writes are common: gelu underflows, unused units)."""
    return jnp.sqrt((z * z).sum(axis) + 1e-12)

def outside_topm(c_abs, m):
    """Share of gross mass outside the m largest entries, along the last axis (scale-free; DESIGNS.md BB10)."""
    m = min(m, c_abs.shape[-1])
    top = jax.lax.top_k(c_abs, m)[0].sum(-1)
    return 1.0 - top / (c_abs.sum(-1) + 1e-9)

def forward(p, cfg, x, bias=None, press=None, record=False):
    """x: (B,T) int bytes -> logits (B,T,V), aux. press: {'pos': (P,) flat positions, 'slots': (P,S) node indices,
    'y': (P,) next bytes at those positions, 'm': top-m} -> aux['press'] = per-read-kind mean outside-top-m share."""
    B, T = x.shape
    h = p['emb'][x]
    aux = {'usage': []}
    rec = {'emb': h, 'layers': []} if record else None
    if press is not None:
        pos, m = press['pos'], press['m']
        at = lambda z: z.reshape((B * T,) + z.shape[2:])[pos]
        Wsp = [at(h)[:, None, :]]                               # writer vectors at the sampled positions, (P, w, d) each
        pen = {'value': [], 'unit': []}
    for l, b in enumerate(p['blocks']):
        s1 = gain(h); heads, A = attention(b, h * s1 * b['ln1'])
        if press is not None:  # value reads: head h at a sampled position as a source, over that position's writers
            Wc = jnp.concatenate(Wsp, 1) * b['ln1']
            Cv = jnp.einsum('pwd,dhe->phwe', Wc, b['wv']) * at(s1)[:, :, None, None]
            pen['value'].append(outside_topm(safe_norm(Cv), m).mean())
            Wsp.append(at(heads))
        h = h + heads.sum(2)
        s2 = gain(h); out, node = mlp(b, h * s2 * b['ln2'], cfg, None if bias is None else bias[l])
        if cfg['mlp'] == 'peer':
            aux['usage'].append(jnp.stack([jnp.zeros(cfg['n_keys']).at[node['a'].ravel()].add(1.0),
                                           jnp.zeros(cfg['n_keys']).at[node['c'].ravel()].add(1.0)]))
        if press is not None:  # unit/neuron reads at the sampled positions, over that position's writers
            Wc = jnp.concatenate(Wsp, 1) * b['ln2']; sl = press['slots']; P = sl.shape[0]
            coef = at(node['coef'])                                       # (P, nodes)
            if cfg['mlp'] == 'dense':
                reads = [b['w_in'].T[sl]]                                 # (P,S,d)
                wnorm = safe_norm(b['w_out'])[sl]
                writes = coef[:, :, None] * b['w_out'][None]               # (P,f,d)
            else:
                unit = at(node['unit'])                                   # (P,RK)
                us = jnp.take_along_axis(unit, sl, 1)
                reads = [routing_vectors(b, cfg, sl, us), b['u'][us]]
                wnorm = safe_norm(b['v'][us])
                writes = coef[:, :, None] * b['v'][unit]                  # (P,RK,d)
            wgt = jax.lax.stop_gradient(jnp.abs(jnp.take_along_axis(coef, sl, 1)) * wnorm)
            sp = at(s2)[:, :, None]
            share = sum(outside_topm(jnp.abs(jnp.einsum('pwd,psd->psw', Wc, r) * sp), m) for r in reads) / len(reads)
            pen['unit'].append((share * wgt).sum() / (wgt.sum() + 1e-9))
            Wsp.append(writes)
        h = h + out
        if record:
            rec['layers'].append({'s1': s1[..., 0], 'A': A, 'heads': heads, 's2': s2[..., 0], **node})
    sf = gain(h); logits = (h * sf * p['ln_f']) @ p['unemb']
    if press is not None:  # the next byte's logit, centered over the vocabulary
        Wc = jnp.concatenate(Wsp, 1) * p['ln_f']
        U = p['unemb'] - p['unemb'].mean(1, keepdims=True)
        c = jnp.einsum('pwd,dp->pw', Wc, U[:, press['y']]) * at(sf)
        aux['press'] = {'value': jnp.mean(jnp.stack(pen['value'])), 'unit': jnp.mean(jnp.stack(pen['unit'])),
                        'logit': outside_topm(jnp.abs(c), m).mean()}
    if record:
        rec['sf'] = sf[..., 0]; rec['logits'] = logits; aux['record'] = rec
    return logits, aux

def n_params(p):
    return int(sum(x.size for x in jax.tree.leaves(p)))

def flops_per_token(cfg, T):
    """Approximate forward FLOPs per token (multiply-add = 2), excluding the pressure term."""
    d, L, H, V = cfg['d'], cfg['layers'], cfg['heads'], cfg.get('vocab', 256)
    attn = 2 * 4 * d * d + 2 * 2 * (T / 2) * d
    if cfg['mlp'] == 'dense':
        ff = 2 * 2 * d * cfg.get('ff_mult', 4) * d
    else:
        R, K, n, dk = cfg['r_heads'], cfg['topk'], cfg['n_keys'], cfg['d_key']
        ff = 2 * d * R * dk + 2 * R * n * dk + 2 * 2 * R * K * d
    return L * (attn + ff) + 2 * d * V
