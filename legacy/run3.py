from analyze import *
import pickle, sys
rng = np.random.default_rng(0)
Xtr, mtr = gen_synthetic(20000, 16, rng); Xte, mte = gen_synthetic(2000, 16, rng)
variants = [("v1_k8_d128", base_cfg(), 0.0),
            ("v2_k8_d128_auxk", base_cfg(), 1.0),
            ("v2_k4_d128_auxk", base_cfg(k=4), 1.0),
            ("v2_k4_d256_auxk", base_cfg(k=4, dict=256), 1.0),
            ("v2_k2_d256_auxk", base_cfg(k=2, dict=256, k_aux=4), 1.0)]
reports = []
for name, cfg, aux in variants:
    print("==", name, flush=True)
    p, h = train(cfg, Xtr, 1500, lr=3e-3, aux_coef=aux, eval_data=Xte, log_every=750)
    r = full_report(p, cfg, Xte, mte, name)
    _, codes = get_codes(p, cfg, Xte)
    r['legibility_final'] = legibility_report(codes[-1], Xte, mte)
    r['legibility_mid'] = legibility_report(codes[1], Xte, mte)
    reports.append(r)
    summ = {k: r[k] for k in ['name','eval_nll','dead_frac_per_bottleneck','ablation','completeness_top3_median']}
    summ['leg_final'] = {k: v for k, v in r['legibility_final'].items() if k != 'examples'}
    summ['leg_mid'] = {k: v for k, v in r['legibility_mid'].items() if k != 'examples'}
    print(json.dumps(summ, indent=1), flush=True)
    pickle.dump((cfg, jax.tree.map(np.asarray, p)), open(f'ckpt_{name}.pkl', 'wb'))
json.dump(reports, open('reports_run3.json', 'w'), indent=1)
