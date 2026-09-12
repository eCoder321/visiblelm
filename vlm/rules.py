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
