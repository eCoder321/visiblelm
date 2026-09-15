"""
Task E1d, run directly by the PI (worker hit a hard rate/spend limit before starting).
Ablation test of E1c's hypothesis: identity-stuck, cross-firing rules on succ/pred/pls2 are causally responsible
for those modes' errors AND for degrading cpy2/cpy3/rept via spurious cross-mode votes.
No training. T2_large, all 3 seeds, iid split matching E0/E1/E1b/E1c.
"""
import models, pickle, numpy as np, json
from models_g import layout, apply_rulenet_g
from testbed import make_splits, annotate, MODE_NAMES, M_MODES

rng_master = np.random.default_rng(0)
S = make_splits(n_train=20000, n_test=1000)
Xte = S['iid']
A = annotate(Xte)
det_mask = (A['heldout'] == 0) & (A['oracle_p'].max(-1) >= 0.99)
det_mask[:, -1] = False  # last col filler

results = {}
for seed in [0, 1, 2]:
    b = pickle.load(open(f'../results/T2_large/seed{seed}.pkl', 'rb'))
    p, m, cfg = b['params'], b['masks'], b['cfg']
    L_ = layout(cfg)
    F = L_['F']

    def forward(feat_masks=None):
        lg, states, internals = apply_rulenet_g(p, m, cfg, Xte, feat_masks=feat_masks, opts={'return_internals': True})
        return np.asarray(lg), internals

    lg_base, internals_base = forward()
    pred_base = lg_base[:, :-1].argmax(-1)
    true = A['oracle_p'][:, :-1].argmax(-1)
    wrong_base = (pred_base != true) & det_mask[:, :-1]
    right_base = (pred_base == true) & det_mask[:, :-1]

    # --- identify candidate rules per layer: cond weights tok_mode in {succ=1,pred=2,pls2=3} > 0.15,
    #     reads a state content-kind group via Gs and routes (via T) into a content-kind DESTINATION that
    #     includes out_content/out_mode (the output-vote pathway), and T is (near-)identity.
    out_group_idx = [gi for gi, (name, start, size, kind) in enumerate(L_['groups']) if name in ('out_mode', 'out_content')]
    candidates = []  # (layer, rule, mode_idx)
    for l, (lp, lm) in enumerate(zip(p['layers'], m['layers'])):
        W1 = np.asarray(lp['W1']) * np.asarray(lm['W1'])
        Gs = np.asarray(lp['Gs']) * np.asarray(lm['Gs'])
        Gd = np.asarray(lp['Gd']) * np.asarray(lm['Gd'])
        T = np.asarray(lp['T'])
        src_groups = [gi for gi, (name, start, size, kind) in enumerate(L_['groups']) if kind == 'content' and start < F]
        for r in range(cfg['R']):
            for mode_idx in [1, 2, 3]:
                if W1[mode_idx, r] > 0.15:
                    src_content = any(abs(Gs[r, g]) > 1e-6 for g in src_groups)
                    dst_out = any(abs(Gd[r, g]) > 1e-6 for g in out_group_idx)
                    if src_content and dst_out:
                        Tr = T[r]
                        is_identity = np.mean(np.argmax(Tr, axis=1) == np.arange(Tr.shape[0])) > 0.9
                        if is_identity:
                            candidates.append((l, r, mode_idx))
    candidates = sorted(set(candidates))

    # --- firing-rate discrimination: which candidates fire much more on wrong than right steps, in >=1 mode
    hid = [np.asarray(h) for h in internals_base['hid']]  # per layer, (B,T,R), pre-threshold->post-STE 0/1 already (snapped)
    modes_bT = np.broadcast_to(A['mode'][:, :-1], wrong_base.shape)
    selected = []
    per_rule_disc = {}
    for (l, r, mode_idx) in candidates:
        fires = hid[l][:, :-1, r] > 0.5
        best_disc, best_mode = 0.0, None
        for mi in range(M_MODES):
            mmask = modes_bT == mi
            w = wrong_base & mmask; rr = right_base & mmask
            if w.sum() < 10 or rr.sum() < 10: continue
            fw = fires[w].mean(); fr = fires[rr].mean()
            disc = fw - fr
            if abs(disc) > abs(best_disc): best_disc, best_mode = disc, mi
        per_rule_disc[(l, r)] = (best_disc, best_mode, MODE_NAMES[mode_idx])
        if best_disc > 0.15:  # fires notably more on wrong steps in some mode
            selected.append((l, r))
    selected = sorted(set(selected))

    # --- ablation: zero these rules' OUTPUT-REGISTER votes only, via BOTH pathways that can write there:
    #     (1) the direct W2[r, F:] columns, (2) the group-map's Gd[r, out_mode/out_content] entries.
    #     State writes (W2[r,:F] and Gd[r, state-groups]) are left completely untouched.
    def make_ablated_params(rule_list):
        p2 = dict(p); p2['layers'] = [dict(lp) for lp in p['layers']]
        for (l, r) in rule_list:
            W2 = np.array(p2['layers'][l]['W2']); W2[r, F:] = 0.0; p2['layers'][l]['W2'] = W2
            Gd = np.array(p2['layers'][l]['Gd'])
            for g in out_group_idx: Gd[r, g] = 0.0
            p2['layers'][l]['Gd'] = Gd
        return p2

    p_ablated = make_ablated_params(selected)
    lg_abl, _ = apply_rulenet_g(p_ablated, m, cfg, Xte, opts={})
    lg_abl = np.asarray(lg_abl)
    pred_abl = lg_abl[:, :-1].argmax(-1)

    # random control: same count, drawn from all rules that fire at all on deterministic steps, excluding selected
    n_sel = len(selected)
    all_rules = [(l, r) for l in range(cfg['layers']) for r in range(cfg['R']) if (l, r) not in selected]
    rng = np.random.default_rng(100 + seed)
    rand_rules = [all_rules[i] for i in rng.choice(len(all_rules), size=min(n_sel, len(all_rules)), replace=False)] if n_sel > 0 else []
    p_rand = make_ablated_params(rand_rules)
    lg_rand, _ = apply_rulenet_g(p_rand, m, cfg, Xte, opts={})
    lg_rand = np.asarray(lg_rand)
    pred_rand = lg_rand[:, :-1].argmax(-1)

    def acc_by_mode(pred):
        out = {}
        for mi in range(M_MODES):
            mmask = det_mask[:, :-1] & (modes_bT == mi)
            if mmask.sum() == 0: out[MODE_NAMES[mi]] = None; continue
            out[MODE_NAMES[mi]] = float((pred == true)[mmask].mean())
        return out

    def nll_of(lg):
        lp = lg[:, :-1] - np.log(np.exp(lg[:, :-1] - lg[:, :-1].max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg[:, :-1].max(-1, keepdims=True)
        return float(-np.take_along_axis(lp, Xte[:, 1:, None], -1)[..., 0].mean())

    results[seed] = dict(
        n_candidates=len(candidates), n_selected=len(selected), selected=[list(x) for x in selected],
        acc_before=acc_by_mode(pred_base), acc_after_targeted=acc_by_mode(pred_abl), acc_after_random=acc_by_mode(pred_rand),
        nll_before=nll_of(lg_base), nll_after_targeted=nll_of(lg_abl), nll_after_random=nll_of(lg_rand),
        rule_discrimination={f"L{l}R{r}": v for (l, r), v in per_rule_disc.items()},
    )
    print(f"seed {seed}: {len(candidates)} candidates, {len(selected)} selected -> {selected}", flush=True)
    print(f"  acc before: {results[seed]['acc_before']}", flush=True)
    print(f"  acc after targeted: {results[seed]['acc_after_targeted']}", flush=True)
    print(f"  acc after random: {results[seed]['acc_after_random']}", flush=True)
    print(f"  nll before/targeted/random: {results[seed]['nll_before']:.4f} / {results[seed]['nll_after_targeted']:.4f} / {results[seed]['nll_after_random']:.4f}", flush=True)

json.dump(results, open('../results/E1d_ablation.json', 'w'), indent=1)
print("### E1D DONE")
