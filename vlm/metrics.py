"""
Metrics that operationalize the four goal properties on a trained model (any family).

  completeness gap   NLL(run only through the legible states) - NLL(normal). Zero by construction for models whose
                     only channel is the states; the SAE error term for the null model.  [P3]
  magnitude audit    NLL when every active feature's activation is replaced by that feature's mean activation.
                     The increase is the information carried by continuous magnitudes rather than by *which*
                     features are on.  [P3, hidden channel inside the legible states]
  path faithfulness  For a prediction, the circuit = every (state, position, feature) node whose single ablation
                     changes the target log-prob by > tau. Necessity: ablate the circuit. Sufficiency: ablate every
                     active node NOT in the circuit and see what the prediction retains, versus a random set of the
                     same size. Model-agnostic, all states, all positions — not just the last layer.  [P1, P4]
  legibility         For every feature (and every rule, for rulenet), fit the best predicate over ground-truth
                     variables on half of the iid split and score its F1 on the other half. Clean = held-out F1 > 0.8.
                     Reported as counts and as the fraction of activation mass / of circuit nodes that is clean.  [P2]
  rulebook           rulenet only: the printed rules and routing tables plus their total description length.
"""
import jax, jax.numpy as jnp, numpy as np, itertools, json
from functools import partial
from models import APPLY, n_states
from testbed import VOCAB, M_MODES, C_TOK, MODE_NAMES, annotate

def _cfg_t(cfg):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()))

@partial(jax.jit, static_argnums=(2, 5))
def _fwd(p, masks, cfg_t, x, feat_masks, opts_t):
    cfg = dict(cfg_t)
    return APPLY[cfg['model']](p, masks, cfg, x, feat_masks=feat_masks, opts=dict(opts_t))

def forward(bundle, x, feat_masks=None, opts=None, which='main'):
    """Returns numpy (logits, states). which='sae' uses the null-model params in the bundle."""
    if which == 'sae':
        p, masks, cfg = bundle['sae']['params'], {}, bundle['sae']['cfg']
    else:
        p, masks, cfg = bundle['params'], bundle['masks'], bundle['cfg']
    lg, st = _fwd(p, masks, _cfg_t(cfg), jnp.asarray(x), feat_masks, tuple(sorted((opts or {}).items())))
    return np.asarray(lg), [np.asarray(s) for s in st]

def get_states(bundle, X, which='main', bs=500):
    L, S = [], None
    for i in range(0, len(X), bs):
        lg, st = forward(bundle, X[i:i + bs], which=which); L.append(lg)
        S = st if S is None else [np.concatenate([a, b]) for a, b in zip(S, st)]
    return np.concatenate(L), S

def target_logprob(logits, X):
    lp = logits[:, :-1] - np.log(np.exp(logits[:, :-1] - logits[:, :-1].max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits[:, :-1].max(-1, keepdims=True)
    return np.take_along_axis(lp, X[:, 1:, None], -1)[..., 0]

def nll_of(bundle, X, which='main', feat_masks=None, opts=None):
    lg, _ = forward(bundle, X, feat_masks=feat_masks, opts=opts, which=which)
    return float(-target_logprob(lg, X).mean())

# ----------------------------------------------------------------------------- names
def feature_names(cfg):
    F = cfg['F']
    tok = [f"cur={MODE_NAMES[v]}" if v < M_MODES else f"cur={v - M_MODES}" for v in range(VOCAB)]
    if cfg['model'] == 'rulenet':
        return tok + [f"h{i}" for i in range(VOCAB, F)]
    return [f"f{i}" for i in range(F)]

def token_name(v):
    return MODE_NAMES[v] if v < M_MODES else str(v - M_MODES)

# ----------------------------------------------------------------------------- P3: completeness + magnitude audit
def completeness_gap(bundle, X):
    if 'sae' in bundle:
        return nll_of(bundle, X, which='sae', opts={'use_error': 0.0}) - nll_of(bundle, X, which='sae', opts={'use_error': 1.0})
    return 0.0

def magnitude_audit(bundle, Xfit, X, which='main'):
    """Replace each active activation by the feature's mean active value (from Xfit). Returns (nll_base, nll_binarized)."""
    _, S = get_states(bundle, Xfit, which=which)
    means = []
    for s in S:
        cnt = (s > 0).sum((0, 1)); tot = s.sum((0, 1)); means.append(np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0).astype(np.float32))
    B, T = X.shape
    fm = [{'patch': jnp.asarray(np.broadcast_to(m, (B, T, len(m))).copy())} for m in means]
    opts = {'use_error': 1.0} if which == 'sae' else None
    return nll_of(bundle, X, which=which, opts=opts), nll_of(bundle, X, which=which, feat_masks=fm, opts=opts)

# ----------------------------------------------------------------------------- P1/P4: path faithfulness
def path_faithfulness(bundle, X, ann, n_examples=50, tau=0.1, seed=0, which='main', opts=None, log=None):
    rng = np.random.default_rng(seed)
    B, T = X.shape
    _, S = get_states(bundle, X, which=which)
    nS = len(S); F = S[0].shape[-1]
    rows = []
    rule_steps = np.argwhere(ann['next'][:, :-1] >= 0)
    picks = rule_steps[rng.choice(len(rule_steps), n_examples, replace=False)]
    for (i, t) in picks:
        x = X[i:i + 1]; tgt = X[i, t + 1]
        def lp_of(fm_list, n):
            lg, _ = forward(bundle, np.repeat(x, n, 0), feat_masks=fm_list, opts=opts, which=which)
            lg = lg[:, t]; return lg[:, tgt] - (np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) + lg.max(-1))
        base = lp_of(None, 1)[0]
        # candidate nodes: active (state, pos<=t, feature)
        nodes = [(si, s, f) for si in range(nS) for s in range(t + 1) for f in np.where(S[si][i, s] > 0)[0]]
        N = len(nodes)
        fm = [np.ones((N, T, F), np.float32) for _ in range(nS)]
        for j, (si, s, f) in enumerate(nodes): fm[si][j, s, f] = 0.0
        lps = np.concatenate([lp_of([jnp.asarray(m[a:a + 512]) for m in fm], min(512, N - a)) for a in range(0, N, 512)])
        eff = base - lps
        circ = [nodes[j] for j in np.where(eff > tau)[0]]
        # necessity: ablate the circuit; sufficiency: ablate everything else; control: random set of same size
        def masks_for(kill):
            m = [np.ones((1, T, F), np.float32) for _ in range(nS)]
            for (si, s, f) in kill: m[si][0, s, f] = 0.0
            return [jnp.asarray(a) for a in m]
        others = [n for n in nodes if n not in set(circ)]
        rnd_keep = [nodes[j] for j in rng.choice(N, min(len(circ), N), replace=False)]
        rnd_others = [n for n in nodes if n not in set(rnd_keep)]
        lp_nec = lp_of(masks_for(circ), 1)[0] if circ else base
        lp_suf = lp_of(masks_for(others), 1)[0]
        lp_rnd = lp_of(masks_for(rnd_others), 1)[0]
        lp_none = lp_of(masks_for(nodes), 1)[0]
        rows.append(dict(i=int(i), t=int(t), base=float(base), necessity=float(lp_nec), sufficiency=float(lp_suf),
                         random_same_size=float(lp_rnd), all_ablated=float(lp_none), n_nodes=N, circuit_size=len(circ),
                         circuit=[(int(a), int(b), int(c)) for a, b, c in circ],
                         effects={f"{a},{b},{c}": float(e) for (a, b, c), e in zip(nodes, eff) if e > tau}))
    r = np.array([[q['base'], q['necessity'], q['sufficiency'], q['random_same_size'], q['all_ablated'], q['circuit_size'], q['n_nodes']] for q in rows])
    def frac(v):  # fraction of the (base - all_ablated) gap retained
        return float(np.mean((v - r[:, 4]) / np.maximum(r[:, 0] - r[:, 4], 1e-3)))
    summ = dict(base=float(r[:, 0].mean()), necessity=float(r[:, 1].mean()), sufficiency=float(r[:, 2].mean()),
                random_same_size=float(r[:, 3].mean()), all_ablated=float(r[:, 4].mean()),
                sufficiency_retained_frac=frac(r[:, 2]), random_retained_frac=frac(r[:, 3]),
                sufficiency_argmax_ok=float(np.mean(r[:, 2] > np.log(0.5))),
                circuit_size_median=float(np.median(r[:, 5])), circuit_size_mean=float(r[:, 5].mean()), n_nodes_mean=float(r[:, 6].mean()),
                tau=tau, n_examples=len(rows))
    return summ, rows

# ----------------------------------------------------------------------------- P2: legibility
def predicate_matrix(ann, T_use):
    """Boolean predicates over ground-truth variables at every (seq, position<T_use). Returns (N, npred) and names."""
    def flat(k): return ann[k][:, :T_use].reshape(-1)
    v = {k: flat(k) for k in ['mode', 'cur', 'prev', 'prev2', 'prev3', 'next', 'arg', 'is_mode', 'dist_mode']}
    P, names = [], []
    def add(name, arr): P.append(arr); names.append(name)
    for m in range(M_MODES): add(f"mode={MODE_NAMES[m]}", v['mode'] == m)
    for k in ['cur', 'prev', 'prev2', 'prev3', 'next', 'arg']:
        for c in range(C_TOK): add(f"{k}={c}", v[k] == c)
    add("is_mode_tok", v['is_mode'] == 1)
    for d in range(1, 6): add(f"dist_mode={d}", v['dist_mode'] == d)
    add("dist_mode>=6", v['dist_mode'] >= 6)
    for m in range(M_MODES):
        for k in ['cur', 'prev', 'prev2', 'prev3', 'next', 'arg']:
            for c in range(C_TOK): add(f"mode={MODE_NAMES[m]}&{k}={c}", (v['mode'] == m) & (v[k] == c))
        add(f"mode={MODE_NAMES[m]}&is_mode_tok", (v['mode'] == m) & (v['is_mode'] == 1))
    # cur x prev (for copy rules) and next x is_mode
    for a in range(C_TOK):
        for b in range(C_TOK): add(f"cur={a}&prev={b}", (v['cur'] == a) & (v['prev'] == b))
    return np.stack(P, 1).astype(np.float32), names

def fit_predicates(act, PM):
    """act: (N, F) activations; PM: (N, npred) bool. Returns best F1 per feature and index of best predicate."""
    fr = (act > 0).astype(np.float32)
    tp = fr.T @ PM; fp = fr.sum(0)[:, None] - tp; fn = PM.sum(0)[None] - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return f1

def legibility(bundle, X, ann, which='main', states=None, extra=None, log=None):
    """Held-out legibility per state (and for `extra` named activation sets, e.g. rulenet rules)."""
    T_use = X.shape[1] - 1
    PM, names = predicate_matrix(ann, T_use)
    if states is None: _, states = get_states(bundle, X, which=which)
    N = len(X) * T_use; half = np.arange(N) % 2 == 0          # interleaved halves
    out = {}; labels = {}
    def analyze(name, act):
        A = act[:, :T_use].reshape(N, -1)
        f1_a = fit_predicates(A[half], PM[half]); best = f1_a.argmax(1)
        f1_b = fit_predicates(A[~half], PM[~half])
        held = f1_b[np.arange(len(best)), best]
        live = A.sum(0) > 0; mass = A.sum(0)
        clean = live & (held > 0.8)
        out[name] = dict(n_live=int(live.sum()), n_clean=int(clean.sum()), n_half_clean=int((live & (held > 0.5)).sum()),
                         median_heldF1_live=float(np.median(held[live])) if live.any() else 0.0,
                         clean_mass_frac=float(mass[clean].sum() / max(mass.sum(), 1e-9)),
                         clean_event_frac=float((A[:, clean] > 0).sum() / max((A > 0).sum(), 1)))
        labels[name] = [(names[b], float(h)) if l else (None, 0.0) for b, h, l in zip(best, held, live)]
    for i, s in enumerate(states): analyze(f"state{i}", s)
    for k, v in (extra or {}).items(): analyze(k, v)
    return out, labels

# ----------------------------------------------------------------------------- rulenet: rulebook
def rulebook(bundle, labels=None, top_rel=3):
    p, m, cfg = bundle['params'], bundle['masks'], bundle['cfg']
    F = cfg['F']; fn = feature_names(cfg); outn = [f"next={token_name(v)}" for v in range(VOCAB)]
    allw = fn + outn
    def lab(i, state_labels=None):
        s = fn[i]
        if labels and i >= VOCAB:
            for key in sorted(labels):
                if key.startswith('state') and labels[key][i][0] and labels[key][i][1] > 0.8:
                    return f"{s}[{labels[key][i][0]}]"
        return s
    lines = []; dl = 0
    if 'emb' in p:
        E = np.maximum(p['emb'] * m['emb'], 0); lines.append("=== embedding: token -> features")
        for v in range(VOCAB):
            nz = [(f, E[v, f]) for f in np.where(E[v] > 0)[0]]; dl += len(nz)
            lines.append(f"  {token_name(v):>5s} -> " + ', '.join(f"{lab(f)}:{w:.2f}" for f, w in nz))
    for l, (lp, lm) in enumerate(zip(p['layers'], m['layers'])):
        lines.append(f"=== layer {l} : ROUTE")
        for h in range(cfg['heads']):
            A = lp['A'][h] * lm['A'][h]; u = lp['u'][h] * lm['u'][h]; P = lp['P'][h] * lm['P'][h]
            rel = lp['rel'][h][1:cfg['T_max'] + 1]
            top = np.argsort(-rel)[:top_rel]
            us = [f"{lab(g)}:{u[g]:+.2f}" for g in np.argsort(-np.abs(u)) if u[g] != 0]
            As = [f"({lab(f)},{lab(g)}):{A[f, g]:+.2f}" for f, g in zip(*np.where(A != 0))]
            Ps = [f"{lab(g)}->{allw[f]}:{P[g, f]:+.2f}" for g, f in zip(*np.where(P != 0))]
            dl += len(us) + len(As) + len(Ps)
            lines.append(f"  head {h}: prefer distance {[(int(d + 1), round(float(rel[d]), 2)) for d in top]}")
            lines.append(f"    LOOK FOR: {us}")
            lines.append(f"    MATCH (mine,theirs): {As}")
            lines.append(f"    COPY: {Ps}")
        lines.append(f"=== layer {l} : RULES")
        W1 = lp['W1'] * lm['W1']; W2 = lp['W2'] * lm['W2']
        for r in range(cfg['R']):
            ins = [(f, W1[f, r]) for f in np.where(W1[:, r] != 0)[0]]; outs = [(g, W2[r, g]) for g in np.where(W2[r] != 0)[0]]
            if not ins or not outs: continue
            dl += len(ins) + len(outs) + 1
            cond = ' + '.join(f"{w:+.2f}*{lab(f)}" for f, w in ins)
            rl = f"rule[{labels['rules'+str(l)][r][0]}:{labels['rules'+str(l)][r][1]:.2f}]" if labels and f'rules{l}' in labels and labels[f'rules{l}'][r][0] else 'rule'
            lines.append(f"  L{l}R{r:<3d} {rl}: IF {cond} {lp['b1'][r]:+.2f} > 0 THEN " + ', '.join(f"{allw[g]} {w:+.2f}" for g, w in outs))
    lines.append(f"=== readout: logits[v] = next=v * scale[v] + bias[v]; description length (nonzero parameters) = {dl}")
    return '\n'.join(lines), dl

# ----------------------------------------------------------------------------- everything
def full_analysis(bundle, S, ann, n_faith=40, log=print):
    cfg = bundle['cfg']; out = {}
    Xi, Xs = S['iid'][:1000], S['shift'][:1000]
    ai = {k: v[:1000] for k, v in ann['iid'].items()}; as_ = {k: v[:1000] for k, v in ann['shift'].items()}
    targets = [('main', None)] if cfg['model'] != 'dense' else []
    if 'sae' in bundle: targets.append(('sae', {'use_error': 1.0}))
    for which, opts in targets:
        r = {}
        r['completeness_gap'] = completeness_gap(bundle, Xi) if which == 'sae' else 0.0
        b, m = magnitude_audit(bundle, S['train'][:1000], Xi, which=which); r['magnitude_audit'] = {'nll': b, 'nll_binarized': m, 'gap': m - b}
        log(f"  [{which}] completeness gap {r['completeness_gap']:.4f}   magnitude gap {m - b:.4f} ({b:.4f} -> {m:.4f})")
        extra = None
        if cfg['model'] == 'rulenet' and which == 'main':
            _, st, internals = _fwd(bundle['params'], bundle['masks'], _cfg_t(cfg), jnp.asarray(Xi), None, (('return_internals', True),))
            extra = {f"rules{l}": np.asarray(h) for l, h in enumerate(internals['hid'])}
        leg, labels = legibility(bundle, Xi, ai, which=which, extra=extra)
        r['legibility'] = leg
        log(f"  [{which}] legibility: " + json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in leg.items()}))
        for split, X, A in [('iid', Xi, ai), ('shift', Xs, as_)]:
            summ, rows = path_faithfulness(bundle, X, A, n_examples=n_faith, which=which, opts=opts)
            # fraction of circuit nodes that are clean features
            clean_nodes = tot_nodes = 0
            for q in rows:
                for (si, s, f) in q['circuit']:
                    tot_nodes += 1; clean_nodes += int(labels[f"state{si}"][f][1] > 0.8)
            summ['circuit_clean_frac'] = clean_nodes / max(tot_nodes, 1)
            r[f'faithfulness_{split}'] = summ
            log(f"  [{which}] faithfulness {split}: " + json.dumps({k: round(v, 3) for k, v in summ.items()}))
        if cfg['model'] == 'rulenet' and which == 'main':
            text, dl = rulebook(bundle, labels); r['rulebook'] = text; r['description_length'] = dl
            log(f"  rulebook description length {dl}")
        out[which] = r
    return out
