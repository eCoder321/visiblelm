from analyze import *
from textdata import get_text_data
import pickle
Xtr, Xte, V, chars = get_text_data(T=64, n_train=60000, n_test=1000)
print("vocab", V, "train windows", Xtr.shape, flush=True)
def tcfg(**kw):
    c = dict(vocab=V, d=96, layers=3, heads=3, ff=256, T=64, dict=384, k=12,
             bottleneck=True, share_dict=True, bottleneck_emb=True, k_aux=16)
    c.update(kw); return c
variants = [("text_dense", tcfg(bottleneck=False), 0.0),
            ("text_v2_k12_d384", tcfg(), 1.0)]
for name, cfg, aux in variants:
    print("==", name, flush=True)
    p, h = train(cfg, Xtr, 3000, lr=2e-3, bs=32, aux_coef=aux, eval_data=Xte, log_every=500)
    pickle.dump((cfg, jax.tree.map(np.asarray, p), h), open(f'ckpt_{name}.pkl', 'wb'))
    print(name, "final eval nll", h[-1][2], "bits/char", h[-1][2]/np.log(2), flush=True)
