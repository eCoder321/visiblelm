"""
rulenet_g: the rule network with a *structured* state (slots), so that mechanisms can be first-order.

Why (see results of rn_diag_rulesonly): a propositional rule "IF mode=succ AND cur=7 THEN next=8" is a table entry
and cannot apply to a value it never saw. To generalize across values the state needs variables. Here features
are organized into GROUPS: a value group is a slot that holds one content value (size C) or one mode (size M);
a boolean group is a single feature. Operations are expressed at group level:
  * ROUTE (per head): LOOK FOR a position holding any value in group g (u[g]); MATCH my group g1 against their
    group g2 by value equality (A[g1,g2]); COPY their group k into my group g value-for-value (P[k,g], identity
    within the group) or, when g is a boolean, as "they have something in k".
  * RULE r: IF (<= fan_in individual features, thresholded) THEN write <= fan_out individual features (W2) and/or
    ONE group map: dst <- T_r(src), where T_r is a value table initialised to the identity (a copy) and
    regularised towards it; deviations from identity are the learned value transforms (e.g. succ).
  * Persistence, output register and readout as in rulenet.
Everything is a sparse signed write; the printed program is the computation.

Group layout (in feature order):
  tok_mode (M) | tok_content (C) | val slots x n_val (C each) | mode slots x n_mode (M each) | bools x n_bool (1 each)
  and, in the output register: out_mode (M) | out_content (C).
"""
import jax, jax.numpy as jnp, numpy as np
from testbed import VOCAB, M_MODES, C_TOK
from models import topk_mask, _mask_apply, _act

def layout(cfg):
    """Returns dict with F, groups (list of (name, start, size, kind)), and index tensors."""
    n_val, n_mode, n_bool = cfg.get('n_val', 4), cfg.get('n_mode', 1), cfg.get('n_bool', 16)
    groups = [('tok_mode', 0, M_MODES, 'mode'), ('tok_content', M_MODES, C_TOK, 'content')]
    pos = VOCAB
    for i in range(n_val): groups.append((f'val{i}', pos, C_TOK, 'content')); pos += C_TOK
    for i in range(n_mode): groups.append((f'mode{i}', pos, M_MODES, 'mode')); pos += M_MODES
    for i in range(n_bool): groups.append((f'b{i}', pos, 1, 'bool')); pos += 1
    F = pos
    groups.append(('out_mode', F, M_MODES, 'mode')); groups.append(('out_content', F + M_MODES, C_TOK, 'content'))
    G = len(groups); Cmax = C_TOK
    # scatter matrix feature -> (group, value index), over F+V features
    Mv = np.zeros((F + VOCAB, G, Cmax), np.float32)
    for gi, (name, start, size, kind) in enumerate(groups):
        for v in range(size): Mv[start + v, gi, v] = 1.0
    kinds = [k for _, _, _, k in groups]
    same = np.array([[kinds[a] == kinds[b] and kinds[a] != 'bool' for b in range(G)] for a in range(G)], np.float32)
    to_bool = np.array([[kinds[b] == 'bool' for b in range(G)] for a in range(G)], np.float32)
    is_state = np.array([start < F for _, start, _, _ in groups], np.float32)      # groups that live in the state (not out)
    return dict(F=F, G=G, Cmax=Cmax, groups=groups, Mv=jnp.asarray(Mv), same=jnp.asarray(same),
                to_bool=jnp.asarray(to_bool), is_state=jnp.asarray(is_state), names=[g[0] for g in groups])

def init_rulenet_g(cfg, key):
    L_ = layout(cfg); F, G, Cm = L_['F'], L_['G'], L_['Cmax']; cfg['F'] = F
    R, H, L, Tm = cfg['R'], cfg['heads'], cfg['layers'], cfg['T_max']
    ks = jax.random.split(key, L + 2)
    p = {'out_scale': jnp.full(VOCAB, cfg.get('out_scale_init', 1.0)), 'out_bias': jnp.zeros(VOCAB), 'layers': []}
    m = {'layers': []}
    E = jnp.eye(VOCAB, F) + jnp.abs(jax.random.normal(ks[-1], (VOCAB, F))) * 0.05
    p['emb'] = E; m['emb'] = jnp.ones((VOCAB, F))
    for l in range(L):
        kk = jax.random.split(ks[l], 6)
        lp = {'u': jax.random.normal(kk[0], (H, G)) * 0.3, 'A': jax.random.normal(kk[1], (H, G, G)) * 0.3,
              'P': jax.random.normal(kk[2], (H, G, G)) * 0.1, 'rel': jnp.zeros((H, Tm + 1)),
              'W1': jax.random.normal(kk[3], (F, R)) * 0.3, 'b1': jnp.zeros(R),
              'W2': jax.random.normal(kk[4], (R, F + VOCAB)) * 0.1,
              'Gr': jax.random.normal(kk[5], (R, G, G)) * 0.1,
              'T': jnp.broadcast_to(jnp.eye(Cm), (R, Cm, Cm)).copy()}
        p['layers'].append(lp)
        m['layers'].append({'u': jnp.ones((H, G)), 'A': jnp.ones((H, G, G)), 'P': jnp.ones((H, G, G)),
                            'W1': jnp.ones((F, R)), 'W2': jnp.ones((R, F + VOCAB)), 'Gr': jnp.ones((R, G, G))})
    return p, m

def _grouped(code, L_):
    """(B,T,F) -> (B,T,G,Cmax) value-aligned view, and (B,T,G) group sums."""
    Mv = L_['Mv'][:code.shape[-1]]
    cg = jnp.einsum('btf,fgv->btgv', code, Mv)
    return cg, cg.sum(-1)

def _ungroup(dg, L_, n):
    """(B,T,G,Cmax) -> (B,T,n) features."""
    return jnp.einsum('btgv,fgv->btf', dg, L_['Mv'][:n])

def route_g(code, lp, lm, cfg, L_, T):
    H = cfg['heads']; F = L_['F']
    cg, gs = _grouped(code, L_)
    u = lp['u'] * lm['u']; A = lp['A'] * lm['A'] * L_['same']; P = lp['P'] * lm['P']
    S = jnp.einsum('btgv,hgk,bskv->bhts', cg, A, cg) + jnp.einsum('hg,bsg->bhs', u, gs)[:, :, None, :]
    dist = jnp.arange(T)[:, None] - jnp.arange(T)[None, :]
    rel = jnp.take(lp['rel'], jnp.clip(dist, 0, lp['rel'].shape[1] - 1), axis=1)
    S = jnp.where((dist > 0)[None, None], S + rel[None], -1e9)
    S = jnp.concatenate([S, jnp.zeros(S.shape[:-1] + (1,))], -1)
    a = jax.nn.softmax(S, -1)
    cg_src = jnp.concatenate([cg, jnp.zeros(cg.shape[:1] + (1,) + cg.shape[2:])], 1)
    copied = jnp.einsum('bhts,bskv->bhtkv', a, cg_src)                        # (B,H,T,G,Cmax)
    # value-preserving copies between same-kind groups; summed copies into booleans
    d_same = jnp.einsum('bhtkv,hkg->btgv', copied, P * L_['same'])
    d_bool = jnp.einsum('bhtk,hkg->btg', copied.sum(-1), P * L_['to_bool'])
    dg = d_same.at[..., 0].add(d_bool)                                        # booleans live at value index 0
    return _ungroup(dg, L_, F + VOCAB), a

def compute_g(code, lp, lm, cfg, L_):
    F = L_['F']
    hid = _act(code @ (lp['W1'] * lm['W1']) + lp['b1'], cfg)                   # (B,T,R)
    delta = hid @ (lp['W2'] * lm['W2'])
    cg, gs = _grouped(code, L_)
    Gr = lp['Gr'] * lm['Gr']
    mapped = jnp.einsum('btkv,rvw->btrkw', cg, lp['T'])                       # each rule's table applied to every group
    d_same = jnp.einsum('btr,btrkw,rkg->btgw', hid, mapped, Gr * L_['same'])
    d_bool = jnp.einsum('btr,btk,rkg->btg', hid, gs, Gr * L_['to_bool'])
    dg = d_same.at[..., 0].add(d_bool)
    return delta + _ungroup(dg, L_, F + VOCAB), hid

def apply_rulenet_g(p, masks, cfg, x, feat_masks=None, opts=None):
    opts = opts or {}; L_ = layout(cfg)
    B, T = x.shape; F, k = L_['F'], cfg['k']
    code = jax.nn.relu(p['emb'] * masks['emb'])[x]
    out = jnp.zeros((B, T, VOCAB))
    code = _mask_apply(code, None if feat_masks is None else feat_masks[0])
    states, attns, hids = [code], [], []
    i = 1
    for lp, lm in zip(p['layers'], masks['layers']):
        delta, a = route_g(code, lp, lm, cfg, L_, T); attns.append(a)
        code = topk_mask(_act(code + delta[..., :F], cfg), k); out = out + delta[..., F:]
        code = _mask_apply(code, None if feat_masks is None else feat_masks[i]); states.append(code); i += 1
        delta, hid = compute_g(code, lp, lm, cfg, L_); hids.append(hid)
        code = topk_mask(_act(code + delta[..., :F], cfg), k); out = out + delta[..., F:]
        code = _mask_apply(code, None if feat_masks is None else feat_masks[i]); states.append(code); i += 1
    logits = out * p['out_scale'] + p['out_bias']
    if opts.get('return_internals'): return logits, states, {'attn': attns, 'hid': hids, 'out': out}
    return logits, states

def reg_rulenet_g(p, masks, cfg):
    """L1 on all structural weights, and on each rule table's deviation from the identity."""
    reg = jnp.abs(p['emb'] * masks['emb']).sum()
    for lp, lm in zip(p['layers'], masks['layers']):
        reg = reg + sum(jnp.abs(lp[n] * lm[n]).sum() for n in ['u', 'A', 'P', 'W1', 'W2', 'Gr'])
        reg = reg + cfg.get('l1_table', 1.0) * jnp.abs(lp['T'] - jnp.eye(lp['T'].shape[-1])).sum()
    return reg

# ----------------------------------------------------------------------------- pruning
from models import _select
def prune_rulenet_g(p, masks, cfg, frac, grads=None, regrow_frac=0.0):
    L_ = layout(cfg); F, G = L_['F'], L_['G']
    tg = {'u': (G, cfg['n_u']), 'A': (G * G, cfg['n_a']), 'P': (G * G, cfg['n_p']),
          'W1': (F, cfg['fan_in']), 'W2': (F + VOCAB, cfg['fan_out']), 'Gr': (G * G, cfg.get('n_gr', 1)), 'emb': (F, cfg.get('emb_fan', 3))}
    views = {'W1': (lambda w: w.T, lambda w2, s: w2.T), 'W2': (lambda w: w, lambda w2, s: w2), 'u': (lambda w: w, lambda w2, s: w2),
             'A': (lambda w: w.reshape(w.shape[0], -1), lambda w2, s: w2.reshape(s)),
             'P': (lambda w: w.reshape(w.shape[0], -1), lambda w2, s: w2.reshape(s)),
             'Gr': (lambda w: w.reshape(w.shape[0], -1), lambda w2, s: w2.reshape(s)), 'emb': (lambda w: w, lambda w2, s: w2)}
    def keep_n(dn, tn): return int(round(np.exp(np.log(dn) + frac * (np.log(tn) - np.log(dn)))))
    def do(name, w, m, g):
        to2, from2 = views[name]; dn, tn = tg[name]; n = keep_n(dn, tn)
        m2, w2 = _select(to2(w), to2(m), None if g is None else to2(g), n, regrow_frac * n)
        return from2(m2, w.shape), from2(w2, w.shape)
    p = dict(p); p['layers'] = list(p['layers']); new = {'layers': []}
    new['emb'], p['emb'] = do('emb', p['emb'], masks['emb'], None if grads is None else grads['emb'])
    for l, (lp, lm) in enumerate(zip(p['layers'], masks['layers'])):
        lp = dict(lp); nm = {}
        for name in ['u', 'A', 'P', 'W1', 'W2', 'Gr']:
            nm[name], lp[name] = do(name, lp[name], lm[name], None if grads is None else grads['layers'][l][name])
        new['layers'].append(nm); p['layers'][l] = lp
    return new, p

# ----------------------------------------------------------------------------- printing
def rulebook_g(bundle, labels=None, top_rel=3):
    from metrics import token_name
    p, m, cfg = bundle['params'], bundle['masks'], bundle['cfg']; L_ = layout(cfg)
    F, G = L_['F'], L_['G']; names = L_['names']
    fn = []
    for (name, start, size, kind) in L_['groups']:
        for v in range(size):
            fn.append(f"{name}={token_name(v) if kind == 'mode' else v}" if kind != 'bool' else name)
    def lab(i):
        s = fn[i]
        if labels:
            for key in sorted(labels):
                if key.startswith('state') and i < len(labels[key]) and labels[key][i][0] and labels[key][i][1] > 0.8:
                    return f"{s}[{labels[key][i][0]}]"
        return s
    lines = []; dl = 0
    E = np.maximum(np.asarray(p['emb'] * m['emb']), 0); lines.append("=== embedding: token -> features")
    for v in range(VOCAB):
        nz = [(f, E[v, f]) for f in np.where(E[v] > 0)[0]]; dl += len(nz)
        lines.append(f"  {token_name(v):>5s} -> " + ', '.join(f"{lab(f)}:{w:.2f}" for f, w in nz))
    for l, (lp, lm) in enumerate(zip(p['layers'], m['layers'])):
        lines.append(f"=== layer {l} : ROUTE")
        for h in range(cfg['heads']):
            u = np.asarray(lp['u'][h] * lm['u'][h]); A = np.asarray(lp['A'][h] * lm['A'][h] * L_['same']); P = np.asarray(lp['P'][h] * lm['P'][h] * (L_['same'] + L_['to_bool']))
            rel = np.asarray(lp['rel'][h][1:cfg['T_max'] + 1]); top = np.argsort(-rel)[:top_rel]
            us = [f"{names[g]}:{u[g]:+.2f}" for g in np.argsort(-np.abs(u)) if u[g] != 0]
            As = [f"(my {names[a]} == their {names[b]}):{A[a, b]:+.2f}" for a, b in zip(*np.where(A != 0))]
            Ps = [f"their {names[a]} -> my {names[b]}:{P[a, b]:+.2f}" for a, b in zip(*np.where(P != 0))]
            dl += len(us) + len(As) + len(Ps)
            lines.append(f"  head {h}: prefer distance {[(int(d + 1), round(float(rel[d]), 2)) for d in top]}")
            lines.append(f"    LOOK FOR: {us}"); lines.append(f"    MATCH: {As}"); lines.append(f"    COPY: {Ps}")
        lines.append(f"=== layer {l} : RULES")
        W1 = np.asarray(lp['W1'] * lm['W1']); W2 = np.asarray(lp['W2'] * lm['W2']); Gr = np.asarray(lp['Gr'] * lm['Gr'] * (L_['same'] + L_['to_bool'])[None])
        for r in range(cfg['R']):
            ins = [(f, W1[f, r]) for f in np.where(W1[:, r] != 0)[0]]
            outs = [(g, W2[r, g]) for g in np.where(W2[r] != 0)[0]]
            maps = [(a, b, Gr[r, a, b]) for a, b in zip(*np.where(Gr[r] != 0))]
            if not ins or (not outs and not maps): continue
            dl += len(ins) + len(outs) + 1
            cond = ' + '.join(f"{w:+.2f}*{lab(f)}" for f, w in ins)
            eff = [f"{fn[g]} {w:+.2f}" for g, w in outs]
            for a, b, w in maps:
                Tt = np.asarray(lp['T'][r]); dev = np.abs(Tt - np.eye(Tt.shape[0])); nz = int((dev > 0.05).sum()); dl += 1 + nz
                if nz == 0: eff.append(f"{names[b]} <- {names[a]} (copy) x{w:+.2f}")
                else:
                    ent = [f"{v}->{np.argmax(Tt[v])}" for v in range(Tt.shape[0]) if np.argmax(Tt[v]) != v and Tt[v].max() > 0.3]
                    eff.append(f"{names[b]} <- table({names[a]}) x{w:+.2f} [{nz} non-identity entries: {', '.join(ent[:20])}]")
            rl = f"rule[{labels['rules'+str(l)][r][0]}:{labels['rules'+str(l)][r][1]:.2f}]" if labels and f'rules{l}' in labels and labels[f'rules{l}'][r][0] else 'rule'
            lines.append(f"  L{l}R{r:<3d} {rl}: IF {cond} {float(lp['b1'][r]):+.2f} > 0 THEN " + '; '.join(eff))
    lines.append(f"=== readout: logits[v] = out[v]*scale[v] + bias[v]; description length = {dl}")
    return '\n'.join(lines), dl
