from analyze import *
from textdata import get_text_data
import pickle
Xtr, Xte, V, chars = get_text_data(T=64, n_train=60000, n_test=1000)
cfg = dict(vocab=V, d=96, layers=3, heads=3, ff=256, T=64, dict=384, k=12, bottleneck=True, share_dict=True,
           bottleneck_emb=True, k_aux=16, code_residual=True, k_write=12)
name = "text_v3_coderes_kw12"
print("==", name, flush=True)
p, h = train(cfg, Xtr, 3000, lr=2e-3, bs=32, aux_coef=1.0, eval_data=Xte, log_every=500)
pickle.dump((cfg, jax.tree.map(np.asarray, p), h), open(f'ckpt_{name}.pkl', 'wb'))
print(name, "final eval nll", h[-1][2], "bits/char", h[-1][2]/np.log(2), flush=True)
