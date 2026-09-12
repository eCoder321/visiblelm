from analyze import *
import pickle
rng = np.random.default_rng(0)
Xtr, mtr = gen_synthetic(20000, 16, rng); Xte, mte = gen_synthetic(2000, 16, rng)
reports = []
for name, cfg, aux in [("dense", base_cfg(bottleneck=False), 0.0),
                       ("v1_shared_k8", base_cfg(), 0.0),
                       ("v1_shared_k8_aux", base_cfg(), 3.0),
                       ("v1_perlayer_k8", base_cfg(share_dict=False), 0.0)]:
    print("==", name, flush=True)
    p, h = train(cfg, Xtr, 1500, lr=3e-3, aux_coef=aux, eval_data=Xte, log_every=500)
    r = full_report(p, cfg, Xte, mte, name); reports.append(r)
    print(json.dumps(r, indent=1), flush=True)
    pickle.dump((cfg, jax.tree.map(np.asarray, p)), open(f'ckpt_{name}.pkl', 'wb'))
json.dump(reports, open('reports_run2.json', 'w'), indent=1)
