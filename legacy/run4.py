from analyze import *
import pickle
rng = np.random.default_rng(0)
Xtr, mtr = gen_synthetic(20000, 16, rng); Xte, mte = gen_synthetic(2000, 16, rng)
variants = [("v3_coderes_k8_d128", base_cfg(code_residual=True), 1.0),
            ("v3_coderes_k8_d128_kw4", base_cfg(code_residual=True, k_write=4), 1.0)]
for name, cfg, aux in variants:
    print("==", name, flush=True)
    p, h = train(cfg, Xtr, 1500, lr=3e-3, aux_coef=aux, eval_data=Xte, log_every=750)
    r = full_report(p, cfg, Xte, mte, name)
    _, codes = get_codes(p, cfg, Xte)
    r['legibility_final'] = legibility_report(codes[-1], Xte, mte)
    r['layer_invariance'] = layer_invariance(codes, Xte, mte)
    summ = {k: r[k] for k in ['name','eval_nll','dead_frac_per_bottleneck','ablation','completeness_top3_median','layer_invariance']}
    summ['leg_final'] = {k: v for k, v in r['legibility_final'].items() if k != 'examples'}
    print(json.dumps(summ, indent=1), flush=True)
    pickle.dump((cfg, jax.tree.map(np.asarray, p)), open(f'ckpt_{name}.pkl', 'wb'))
    json.dump(r, open(f'report_{name}.json','w'), indent=1)
