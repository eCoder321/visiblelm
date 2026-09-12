from itlm import *

def get_codes(p, cfg, X, bs=500):
    outs = []
    f = jax.jit(lambda x: forward(p, cfg, x, return_all=True))
    for i in range(0, len(X), bs):
        lg, codes, attns = f(jnp.asarray(X[i:i+bs]))
        outs.append((np.asarray(lg), [None if c is None else np.asarray(c) for c in codes]))
    logits = np.concatenate([o[0] for o in outs])
    nb = len(outs[0][1])
    codes = [None if outs[0][1][j] is None else np.concatenate([o[1][j] for o in outs]) for j in range(nb)]
    return logits, codes

def dead_features(codes):
    out = []
    for c in codes:
        if c is None: out.append(None); continue
        out.append(float(((c > 0).sum(axis=(0, 1)) == 0).mean()))
    return out

def feature_latent_alignment(code, latent, n_latent):
    """code: (B,T,F) codes; latent: (B,) ground-truth latent per sequence (mode).
    For each feature: P(mode | feature fires) -> selectivity = max_mode P(mode|fire).
    Chance = 1/n_latent. Returns per-feature selectivity and preferred latent."""
    B, T, F = code.shape
    fires = (code[:, 2:, :] > 0)                    # ignore positions 0,1 (mode tok / random tok)
    fires = fires.reshape(B, -1, F).any(axis=1)      # (B,F): does feature fire anywhere in seq
    sel, pref, cnt = np.zeros(F), np.zeros(F, int), np.zeros(F, int)
    for f in range(F):
        m = fires[:, f]
        cnt[f] = m.sum()
        if m.sum() == 0: sel[f] = np.nan; continue
        pm = np.bincount(latent[m], minlength=n_latent) / m.sum()
        sel[f] = pm.max(); pref[f] = pm.argmax()
    return sel, pref, cnt

def attribution(p, cfg, code_last):
    """logits = code_last @ (D @ out). Return per-feature contribution matrix W (F,V)."""
    if cfg.get('sparse_readout', 0) > 0:
        R = np.asarray(p['readout']); m = cfg['sparse_readout']
        thr = np.sort(np.abs(R), axis=-1)[:, -m][:, None]
        return np.where(np.abs(R) >= thr, R, 0.0)
    D = np.asarray(p['dict'] if cfg['share_dict'] else p['blocks'][-1]['dict'])
    W = D @ np.asarray(p['out'])                     # (F,V)
    return W

def ablation_test(p, cfg, X, code_last, W, n_abl=3, n_examples=400, seed=0):
    """For each (seq, position): rank features by contribution to the target logit.
    Ablate top-n vs random-n active features in the FINAL bottleneck and measure
    change in target log-prob. Faithful attribution => top-n ablation hurts much more."""
    rng = np.random.default_rng(seed)
    B, T, F = code_last.shape
    L = cfg['layers']
    f = jax.jit(lambda x, fm: forward(p, cfg, x, feat_masks=fm))
    base_lp, top_lp, rnd_lp, suff_lp = [], [], [], []
    idx = rng.integers(0, B, n_examples)
    for i in idx:
        x = X[i:i+1]
        # choose a position t>=3 to explain prediction of x[t+1]
        t = rng.integers(3, T - 1); tgt = x[0, t + 1]
        c = code_last[i, t]                          # (F,)
        contrib = c * W[:, tgt]                      # per-feature contribution to target logit
        active = np.where(c > 0)[0]
        top = active[np.argsort(-contrib[active])[:n_abl]]
        rnd = rng.choice(active, size=min(n_abl, len(active)), replace=False)
        def masks_for(feats_to_kill=None, keep_only=None):
            fms = [np.ones((1, T, F), np.float32) for _ in range(L + 1)]
            if feats_to_kill is not None:
                fms[-1][0, t, feats_to_kill] = 0.0
            if keep_only is not None:
                m = np.zeros(F, np.float32); m[keep_only] = 1.0; fms[-1][0, t] = m
            return [jnp.asarray(m_) for m_ in fms]
        def lp(fm):
            lg = np.asarray(f(jnp.asarray(x), fm))[0, t]
            return lg[tgt] - np.logaddexp.reduce(lg)
        base_lp.append(lp(masks_for()))
        top_lp.append(lp(masks_for(feats_to_kill=top)))
        rnd_lp.append(lp(masks_for(feats_to_kill=rnd)))
        suff_lp.append(lp(masks_for(keep_only=top)))
    base_lp, top_lp, rnd_lp, suff_lp = map(np.array, (base_lp, top_lp, rnd_lp, suff_lp))
    return dict(base=float(base_lp.mean()),
                ablate_top=float(top_lp.mean()), ablate_random=float(rnd_lp.mean()),
                keep_only_top=float(suff_lp.mean()),
                # fraction of examples where top-ablation flips argmax
                )

def completeness(code_last, W, X, top_n=3):
    """Fraction of the target-logit (minus mean logit) explained by top_n contributing features."""
    B, T, F = code_last.shape
    fr = []
    for i in range(min(B, 500)):
        for t in range(3, T - 1):
            c = code_last[i, t]; tgt = X[i, t + 1]
            contrib = c * (W[:, tgt] - W.mean(1))
            tot = contrib.sum()
            if abs(tot) < 1e-6: continue
            top = np.sort(contrib)[::-1][:top_n].sum()
            fr.append(top / tot)
    fr = np.array(fr)
    return float(np.median(fr))

def full_report(p, cfg, X, modes, name):
    logits, codes = get_codes(p, cfg, X)
    rep = {'name': name, 'eval_nll': eval_loss(p, cfg, X)}
    if not cfg['bottleneck']:
        return rep
    rep['dead_frac_per_bottleneck'] = dead_features(codes)
    rep['mean_active_k'] = [None if c is None else float((c > 0).sum(-1).mean()) for c in codes]
    sel, pref, cnt = feature_latent_alignment(codes[-1], modes, M_MODES)
    live = ~np.isnan(sel)
    rep['final_layer_feature_mode_selectivity'] = {
        'chance': 1 / M_MODES,
        'mean_selectivity_live': float(np.nanmean(sel)),
        'frac_live_features_selective>0.9': float((sel[live] > 0.9).mean()),
        'n_live': int(live.sum()),
    }
    W = attribution(p, cfg, codes[-1])
    rep['ablation'] = ablation_test(p, cfg, X, codes[-1], W)
    rep['completeness_top3_median'] = completeness(codes[-1], W, X)
    return rep

if __name__ == "__main__":
    import sys
    print(json.dumps(full_report(*sys.argv[1:]), indent=1))

# ---------------------------------------------------------------------------
# Better legibility metric: for each feature, best F1 against any single
# ground-truth predicate  (mode==m) | (prev_tok==v) | (prev2_tok==v) | (pos==t)
# A monosemantic feature has one predicate with high F1.
# ---------------------------------------------------------------------------
def predicates(X, modes):
    B, T = X.shape
    preds = {}
    pos = np.broadcast_to(np.arange(T)[None], (B, T))
    md = np.broadcast_to(modes[:, None], (B, T))
    prev = np.concatenate([np.full((B, 1), -1), X[:, :-1]], 1)
    prev2 = np.concatenate([np.full((B, 2), -1), X[:, :-2]], 1)
    for m in range(M_MODES): preds[f'mode={m}'] = (md == m)
    for v in range(M_MODES): preds[f'cur=MODE{v}'] = (X == v)
    for v in range(M_MODES, VOCAB):
        preds[f'cur={v}'] = (X == v); preds[f'prev={v}'] = (prev == v); preds[f'prev2={v}'] = (prev2 == v)
    nxt = np.concatenate([X[:, 1:], np.full((B, 1), -1)], 1)
    for v in range(M_MODES, VOCAB): preds[f'next={v}'] = (nxt == v)
    for t in range(T): preds[f'pos={t}'] = (pos == t)
    # conjunctions mode x cur token (the true "next token" cause for repeat/succ/pred modes)
    for m in range(M_MODES):
        for v in range(M_MODES, VOCAB):
            preds[f'mode={m}&cur={v}'] = (md == m) & (X == v)
    return preds

def feature_f1(code, preds, min_pos=2):
    B, T, F = code.shape
    fires = (code > 0)[:, min_pos:].reshape(-1, F)
    P = {k: v[:, min_pos:].reshape(-1) for k, v in preds.items()}
    names = list(P.keys()); PM = np.stack([P[n] for n in names], 1).astype(np.float32)  # (N, npred)
    fr = fires.astype(np.float32)
    tp = fr.T @ PM                          # (F, npred)
    fp = fr.sum(0)[:, None] - tp
    fn = PM.sum(0)[None] - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    best = f1.max(1); best_name = [names[i] for i in f1.argmax(1)]
    live = fr.sum(0) > 0
    return best, best_name, live

def legibility_report(code, X, modes):
    best, names, live = feature_f1(code, predicates(X, modes))
    return {'n_live': int(live.sum()),
            'median_bestF1_live': float(np.median(best[live])),
            'frac_live_F1>0.8': float((best[live] > 0.8).mean()),
            'frac_live_F1>0.5': float((best[live] > 0.5).mean()),
            'examples': [(int(f), names[f], round(float(best[f]), 2)) for f in np.argsort(-best)[:12]]}

# ---------------------------------------------------------------------------
# Circuit tracing: for one prediction, which earlier (layer, position, feature)
# nodes causally drive the top final feature?  Measured by ablation (exact
# causal effect), tractable because only k features are active per node.
# ---------------------------------------------------------------------------
def trace_circuit(p, cfg, x, t, codes_x, top_final=1, top_parents=4):
    """x: (1,T); t: position. codes_x: list of (1,T,F) codes for x.
    Returns nested dict of causal parents."""
    L = cfg['layers']; F = cfg['dict']; T = x.shape[1]
    f = jax.jit(lambda xx, fm: forward(p, cfg, xx, feat_masks=fm, return_all=True))
    tgt = x[0, t + 1]
    W = attribution(p, cfg, None)
    c_last = codes_x[-1][0, t]
    contrib = c_last * W[:, tgt]
    finals = np.argsort(-contrib)[:top_final]
    def masks():
        return [np.ones((1, T, F), np.float32) for _ in range(L + 1)]
    out = {'target_tok': int(tgt), 'final_features': []}
    for ff in finals:
        node = {'feat': int(ff), 'contrib_to_target_logit': float(contrib[ff]), 'parents': []}
        # candidate parents: all active features at bottleneck L-1 (previous layer) at positions <= t
        c_prev = codes_x[-2][0]
        effects = []
        for s in range(t + 1):
            for g in np.where(c_prev[s] > 0)[0]:
                fm = masks(); fm[-2][0, s, g] = 0.0
                _, cc, _ = f(jnp.asarray(x), [jnp.asarray(m) for m in fm])
                new = float(np.asarray(cc[-1])[0, t, ff])
                effects.append((float(c_last[ff] - new), int(s), int(g)))
        effects.sort(reverse=True)
        node['parents'] = [{'layer': L - 1, 'pos': s, 'feat': g, 'effect_on_final_feat': round(e, 3)}
                           for e, s, g in effects[:top_parents]]
        out['final_features'].append(node)
    return out

def trace_recursive(p, cfg, x, t, codes_x, top_parents=3, depth=None):
    """Recursive causal trace: final top feature -> parents at layer L-1 -> parents at L-2 ... -> embedding features.
    Effect = drop in the child feature's activation when the parent is ablated (exact, single-parent ablation)."""
    L = cfg['layers']; F = cfg['dict']; T = x.shape[1]
    f = jax.jit(lambda xx, fm: forward(p, cfg, xx, feat_masks=fm, return_all=True))
    tgt = x[0, t + 1]
    W = attribution(p, cfg, None)
    c_last = codes_x[-1][0, t]; contrib = c_last * W[:, tgt]
    ff = int(np.argmax(contrib))
    def masks(): return [np.ones((1, T, F), np.float32) for _ in range(L + 1)]
    def parents(bidx, pos, feat):
        # candidates: active features at bottleneck bidx-1, positions <= pos
        c_prev = codes_x[bidx - 1][0]
        base = float(codes_x[bidx][0, pos, feat])
        effs = []
        for s in range(pos + 1):
            for g in np.where(c_prev[s] > 0)[0]:
                fm = masks(); fm[bidx - 1][0, s, int(g)] = 0.0
                _, cc, _ = f(jnp.asarray(x), [jnp.asarray(m) for m in fm])
                effs.append((base - float(np.asarray(cc[bidx])[0, pos, feat]), int(s), int(g)))
        effs.sort(reverse=True)
        return [(round(e, 2), s, g) for e, s, g in effs[:top_parents] if e > 0.05 * max(base, 1e-6)]
    def rec(bidx, pos, feat):
        node = {'bn': bidx, 'pos': pos, 'feat': feat, 'act': round(float(codes_x[bidx][0, pos, feat]), 2)}
        if bidx > 0:
            node['parents'] = [dict(effect=e, **rec(bidx - 1, s, g)) for e, s, g in parents(bidx, pos, feat)]
        return node
    return {'target': int(tgt), 'logit_contrib': round(float(contrib[ff]), 2), 'tree': rec(L, t, ff)}

def print_tree(node, names=None, ind=0):
    lab = f"bn{node['bn']} pos{node['pos']} f{node['feat']} act={node['act']}"
    if names is not None: lab += f"  [{names[node['feat']]}]"
    if 'effect' in node: lab = f"(effect {node['effect']}) " + lab
    print("  " * ind + lab)
    for c in node.get('parents', []): print_tree(c, names, ind + 1)

def layer_invariance(codes, X, modes):
    """Does a shared-dictionary feature mean the same thing at every layer?
    Label each live feature per bottleneck (best-F1 predicate, all positions) and
    measure agreement of labels across bottlenecks."""
    P = predicates(X, modes)
    labs = []
    for c in codes:
        b, n, live = feature_f1(c, P, min_pos=0); labs.append((n, b, live))
    F = codes[0].shape[-1]
    agree, tot = 0, 0
    for f in range(F):
        ls = [(n[f], b[f]) for n, b, live in labs if live[f]]
        if len(ls) < 2: continue
        tot += 1
        if len(set(l for l, _ in ls)) == 1: agree += 1
    return {'frac_features_same_label_all_layers': agree / max(tot, 1), 'n_multi_layer_live': tot,
            'per_bottleneck_frac_F1>0.8': [float((b[live] > 0.8).mean()) for n, b, live in labs],
            'per_bottleneck_median_F1': [float(np.median(b[live])) for n, b, live in labs]}
