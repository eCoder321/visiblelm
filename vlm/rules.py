"""
The north-star test for rulenet: execute the *printed* rulebook as a discrete program and compare it with the
neural model and with the truth.

The discrete reading drops everything a person cannot read off the rulebook:
  * a feature is either present (1) or absent (0)  -> no magnitudes;
  * a head attends to exactly one source position, the best-scoring one, or to nothing  -> no soft mixtures;
  * a rule fires iff its printed condition holds on the present features, and then writes its printed votes.
If the discrete program agrees with the model, the rulebook is the mechanism. If it agrees with the oracle on
held-out events, the mechanism generalizes.
"""
import numpy as np
from testbed import VOCAB

def execute_hard(bundle, X, binarize=True, hard_attn=True, keep_topk=True):
    p, m, cfg = bundle['params'], bundle['masks'], bundle['cfg']
    F, k, H, Tm = cfg['F'], cfg['k'], cfg['heads'], cfg['T_max']
    B, T = X.shape
    clampv = cfg.get('clamp', 0.0)
    def act(z):
        z = np.maximum(z, 0.0)
        if clampv > 0: z = np.minimum(z, clampv)
        return z
    def topk(z):
        if not keep_topk: return z
        thr = np.sort(z, -1)[..., -k][..., None]
        return np.where((z >= thr) & (z > 0), z, 0.0)
    def binz(z):
        return (z > 0).astype(np.float64) if binarize else z
    if 'emb' in p:
        E = np.maximum(np.asarray(p['emb']) * np.asarray(m['emb']), 0); code = binz(E[X])
    else:
        code = np.eye(F)[X]
    out = np.zeros((B, T, VOCAB))
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    for lp, lm in zip(p['layers'], m['layers']):
        A = np.asarray(lp['A'] * lm['A']); u = np.asarray(lp['u'] * lm['u']); P = np.asarray(lp['P'] * lm['P'])
        rel = np.asarray(lp['rel'])
        S = np.einsum('btf,hfg,bsg->bhts', code, A, code) + np.einsum('hg,bsg->bhs', u, code)[:, :, None, :]
        S = S + np.take(rel, np.clip(dist, 0, Tm), axis=1)[None]
        S = np.where((dist > 0)[None, None], S, -1e9)
        S = np.concatenate([S, np.zeros(S.shape[:-1] + (1,))], -1)
        if hard_attn:
            a = np.zeros_like(S); idx = S.argmax(-1)
            np.put_along_axis(a, idx[..., None], 1.0, -1)
        else:
            e = np.exp(S - S.max(-1, keepdims=True)); a = e / e.sum(-1, keepdims=True)
        src = np.concatenate([code, np.zeros((B, 1, F))], 1)
        delta = np.einsum('bhts,bsg,hgf->btf', a, src, P)
        code = binz(topk(act(code + delta[..., :F]))); out += delta[..., F:]
        W1 = np.asarray(lp['W1'] * lm['W1']); W2 = np.asarray(lp['W2'] * lm['W2']); b1 = np.asarray(lp['b1'])
        hid = act(code @ W1 + b1)
        if binarize: hid = (hid > 0).astype(np.float64)
        delta = hid @ W2
        code = binz(topk(act(code + delta[..., :F]))); out += delta[..., F:]
    return out * np.asarray(p['out_scale']) + np.asarray(p['out_bias'])

def north_star(bundle, X, ann, soft_logits):
    """Agreement of the discrete rulebook with the model and with the oracle, on rule steps and on held-out steps."""
    hard = execute_hard(bundle, X)
    hp = hard[:, :-1].argmax(-1); sp = soft_logits[:, :-1].argmax(-1); op = ann['oracle_p'][:, :-1].argmax(-1)
    rule = ann['next'][:, :-1] >= 0; held = ann['heldout'][:, :-1].astype(bool)
    r = {'agree_model_rule': float((hp == sp)[rule].mean()), 'agree_oracle_rule': float((hp == op)[rule].mean()),
         'model_oracle_rule': float((sp == op)[rule].mean())}
    if held.any():
        r.update({'agree_model_heldout': float((hp == sp)[held].mean()), 'agree_oracle_heldout': float((hp == op)[held].mean()),
                  'model_oracle_heldout': float((sp == op)[held].mean())})
    return r

# ----------------------------------------------------------------------------- structured model (rulenet_g)
def execute_hard_g(bundle, X, binarize=True, hard_attn=True, keep_topk=True):
    from models_g import layout
    p, m, cfg = bundle['params'], bundle['masks'], bundle['cfg']; L_ = layout(cfg)
    F, G, Cm, k, Tm = L_['F'], L_['G'], L_['Cmax'], cfg['k'], cfg['T_max']
    Mv = np.asarray(L_['Mv']); same = np.asarray(L_['same']); to_bool = np.asarray(L_['to_bool'])
    B, T = X.shape; clampv = cfg.get('clamp', 0.0); ste = cfg.get('ste', False); thr0 = cfg.get('ste_thr', 0.5) if ste else 0.0
    if ste: binarize = True; hard_attn = True                                  # a snapped model IS the discrete program
    def act(z):
        z = np.maximum(z, 0.0); return np.minimum(z, clampv) if clampv > 0 else z
    def topk(z):
        if not keep_topk: return z
        thr = np.sort(z, -1)[..., -k][..., None]; return np.where((z >= thr) & (z > 0), z, 0.0)
    def binz(z): return (z > thr0).astype(np.float64) if binarize else z
    def step_state(z): return binz(topk(act(z)))                              # same order as the model: act, top-k on soft values, threshold
    def grouped(code): cg = np.einsum('btf,fgv->btgv', code, Mv[:code.shape[-1]]); return cg, cg.sum(-1)
    def ungroup(dg, n): return np.einsum('btgv,fgv->btf', dg, Mv[:n])
    code = binz(act(np.asarray(p['emb'] * m['emb'])[X])); out = np.zeros((B, T, VOCAB))
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    for lp, lm in zip(p['layers'], m['layers']):
        cg, gs = grouped(code)
        u = np.asarray(lp['u'] * lm['u']); A = np.asarray(lp['A'] * lm['A']) * same; P = np.asarray(lp['P'] * lm['P'])
        S = np.einsum('btgv,hgk,bskv->bhts', cg, A, cg) + np.einsum('hg,bsg->bhs', u, gs)[:, :, None, :]
        S = np.where((dist > 0)[None, None], S + np.take(np.asarray(lp['rel']), np.clip(dist, 0, Tm), axis=1)[None], -1e9)
        S = np.concatenate([S, np.zeros(S.shape[:-1] + (1,))], -1)
        if hard_attn:
            a = np.zeros_like(S); np.put_along_axis(a, S.argmax(-1)[..., None], 1.0, -1)
        else:
            e = np.exp(S - S.max(-1, keepdims=True)); a = e / e.sum(-1, keepdims=True)
        cg_src = np.concatenate([cg, np.zeros((B, 1, G, Cm))], 1)
        copied = np.einsum('bhts,bskv->bhtkv', a, cg_src)
        dg = np.einsum('bhtkv,hkg->btgv', copied, P * same)
        dg[..., 0] += np.einsum('bhtk,hkg->btg', copied.sum(-1), P * to_bool)
        delta = ungroup(dg, F + VOCAB)
        code = step_state(code + delta[..., :F]); out += delta[..., F:]
        cg, gs = grouped(code)
        W1 = np.asarray(lp['W1'] * lm['W1']); W2 = np.asarray(lp['W2'] * lm['W2']); b1 = np.asarray(lp['b1'])
        hid = binz(act(code @ W1 + b1))
        delta = hid @ W2
        Gs = np.asarray(lp['Gs'] * lm['Gs']); Gd = np.asarray(lp['Gd'] * lm['Gd']); Tt = np.asarray(lp['T'])
        src = np.einsum('btkv,rk->btrv', cg, Gs); mapped = np.einsum('btrv,rvw->btrw', src, Tt) * hid[..., None]
        kind_ok = np.einsum('rk,kg->rg', Gs != 0, same) > 0; bool_ok = np.einsum('rk,kg->rg', Gs != 0, to_bool) > 0
        dg = np.einsum('btrw,rg->btgw', mapped, np.where(kind_ok, Gd, 0.0))
        dg[..., 0] += np.einsum('btr,rg->btg', mapped.sum(-1), np.where(bool_ok, Gd, 0.0))
        delta = delta + ungroup(dg, F + VOCAB)
        code = step_state(code + delta[..., :F]); out += delta[..., F:]
    return out * np.asarray(p['out_scale']) + np.asarray(p['out_bias'])

def _quantize(z, n_levels, clamp_hint=0.0):
    """Round z (>=0) to the nearest of n_levels evenly-spaced values in [0, clampv]. clampv is clamp_hint if that is
    > 0 (a trained model's own cfg['clamp']), else the batch's own max (for a model that was never clamped, e.g. a
    soft model with no `clamp` in its cfg) -- so every quantization level is used, whatever the model's natural
    activation scale is. n_levels=None means no rounding at all (the L=infinity sanity ceiling).
    Boundaries between levels are placed at the midpoints and compared with a strict '>' (as the existing binarize
    threshold does), so n_levels=2 with clamp_hint=1.0 exactly reproduces `binz`'s `z > 0.5` for a clamp=1.0, ste
    model: level 0 = 0, level 1 = clampv, boundary at clampv/2."""
    if n_levels is None:
        return z
    clampv = clamp_hint if clamp_hint > 0 else float(z.max()) if z.size else 1.0
    if clampv <= 0: clampv = 1.0
    if n_levels <= 1:
        return np.zeros_like(z)
    step = clampv / (n_levels - 1)
    boundaries = step * (np.arange(n_levels - 1) + 0.5)
    idx = (z[..., None] > boundaries).sum(-1)
    return idx.astype(z.dtype) * step

def execute_quantized_g(bundle, X, n_levels=2, hard_attn=True, keep_topk=True):
    """Generalizes execute_hard_g's binarize step to n_levels evenly-spaced activation levels (n_levels=2 with a
    clamp=1.0, ste model is byte-for-byte the same as execute_hard_g(binarize=True); n_levels=None reproduces the
    continuous re-execution with no rounding at all, a sanity ceiling)."""
    from models_g import layout
    p, m, cfg = bundle['params'], bundle['masks'], bundle['cfg']; L_ = layout(cfg)
    F, G, Cm, k, Tm = L_['F'], L_['G'], L_['Cmax'], cfg['k'], cfg['T_max']
    Mv = np.asarray(L_['Mv']); same = np.asarray(L_['same']); to_bool = np.asarray(L_['to_bool'])
    B, T = X.shape
    clampv = cfg.get('clamp', 0.0); ste = cfg.get('ste', False)
    thr0 = cfg.get('ste_thr', 0.5) if ste else 0.0
    def act(z):
        z = np.maximum(z, 0.0); return np.minimum(z, clampv) if clampv > 0 else z
    def topk(z):
        if not keep_topk: return z
        thr = np.sort(z, -1)[..., -k][..., None]; return np.where((z >= thr) & (z > 0), z, 0.0)
    def qz(z):
        if ste and n_levels == 2:                                         # exact backward-compat path: z > ste_thr
            return (z > thr0).astype(np.float64) if n_levels is not None else z
        return _quantize(z, n_levels, clamp_hint=clampv)
    def step_state(z): return qz(topk(act(z)))
    def grouped(code): cg = np.einsum('btf,fgv->btgv', code, Mv[:code.shape[-1]]); return cg, cg.sum(-1)
    def ungroup(dg, n): return np.einsum('btgv,fgv->btf', dg, Mv[:n])
    code = qz(act(np.asarray(p['emb'] * m['emb'])[X])); out = np.zeros((B, T, VOCAB))
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]
    for lp, lm in zip(p['layers'], m['layers']):
        cg, gs = grouped(code)
        u = np.asarray(lp['u'] * lm['u']); A = np.asarray(lp['A'] * lm['A']) * same; P = np.asarray(lp['P'] * lm['P'])
        S = np.einsum('btgv,hgk,bskv->bhts', cg, A, cg) + np.einsum('hg,bsg->bhs', u, gs)[:, :, None, :]
        S = np.where((dist > 0)[None, None], S + np.take(np.asarray(lp['rel']), np.clip(dist, 0, Tm), axis=1)[None], -1e9)
        S = np.concatenate([S, np.zeros(S.shape[:-1] + (1,))], -1)
        if hard_attn:
            a = np.zeros_like(S); np.put_along_axis(a, S.argmax(-1)[..., None], 1.0, -1)
        else:
            e = np.exp(S - S.max(-1, keepdims=True)); a = e / e.sum(-1, keepdims=True)
        cg_src = np.concatenate([cg, np.zeros((B, 1, G, Cm))], 1)
        copied = np.einsum('bhts,bskv->bhtkv', a, cg_src)
        dg = np.einsum('bhtkv,hkg->btgv', copied, P * same)
        dg[..., 0] += np.einsum('bhtk,hkg->btg', copied.sum(-1), P * to_bool)
        delta = ungroup(dg, F + VOCAB)
        code = step_state(code + delta[..., :F]); out += delta[..., F:]
        cg, gs = grouped(code)
        W1 = np.asarray(lp['W1'] * lm['W1']); W2 = np.asarray(lp['W2'] * lm['W2']); b1 = np.asarray(lp['b1'])
        hid = qz(act(code @ W1 + b1))
        delta = hid @ W2
        Gs = np.asarray(lp['Gs'] * lm['Gs']); Gd = np.asarray(lp['Gd'] * lm['Gd']); Tt = np.asarray(lp['T'])
        src = np.einsum('btkv,rk->btrv', cg, Gs); mapped = np.einsum('btrv,rvw->btrw', src, Tt) * hid[..., None]
        kind_ok = np.einsum('rk,kg->rg', Gs != 0, same) > 0; bool_ok = np.einsum('rk,kg->rg', Gs != 0, to_bool) > 0
        dg = np.einsum('btrw,rg->btgw', mapped, np.where(kind_ok, Gd, 0.0))
        dg[..., 0] += np.einsum('btr,rg->btg', mapped.sum(-1), np.where(bool_ok, Gd, 0.0))
        delta = delta + ungroup(dg, F + VOCAB)
        code = step_state(code + delta[..., :F]); out += delta[..., F:]
    return out * np.asarray(p['out_scale']) + np.asarray(p['out_bias'])

def north_star_any(bundle, X, ann, soft_logits, **kw):
    ex = execute_hard_g if bundle['cfg']['model'] == 'rulenet_g' else execute_hard
    hard = ex(bundle, X, **kw)
    hp = hard[:, :-1].argmax(-1); sp = soft_logits[:, :-1].argmax(-1); op = ann['oracle_p'][:, :-1].argmax(-1)
    rule = ann['next'][:, :-1] >= 0; held = ann['heldout'][:, :-1].astype(bool)
    r = {'agree_model_rule': float((hp == sp)[rule].mean()), 'agree_oracle_rule': float((hp == op)[rule].mean()),
         'model_oracle_rule': float((sp == op)[rule].mean())}
    if held.any():
        r.update({'agree_model_heldout': float((hp == sp)[held].mean()), 'agree_oracle_heldout': float((hp == op)[held].mean()),
                  'model_oracle_heldout': float((sp == op)[held].mean())})
    return r
