from analyze import *
from textdata import get_text_data
import pickle, sys

def text_predicates(X, chars):
    B, T = X.shape
    prev = np.concatenate([np.full((B, 1), -1), X[:, :-1]], 1)
    nxt = np.concatenate([X[:, 1:], np.full((B, 1), -1)], 1)
    P = {}
    for v, ch in enumerate(chars):
        P[f'cur={ch!r}'] = (X == v); P[f'next={ch!r}'] = (nxt == v); P[f'prev={ch!r}'] = (prev == v)
    vow = np.array([c.lower() in 'aeiou' for c in chars]); up = np.array([c.isupper() for c in chars])
    sp = np.array([c in ' \n' for c in chars]); pun = np.array([c in ".,;:!?'" for c in chars])
    P['cur=vowel'] = vow[X]; P['cur=upper'] = up[X]; P['cur=space/newline'] = sp[X]; P['cur=punct'] = pun[X]
    P['next=vowel'] = vow[np.clip(nxt, 0, None)] & (nxt >= 0); P['next=space/newline'] = sp[np.clip(nxt, 0, None)] & (nxt >= 0)
    P['prev=space/newline'] = sp[np.clip(prev, 0, None)] & (prev >= 0)
    return P

def text_report(name):
    cfg, p, h = pickle.load(open(f'ckpt_{name}.pkl', 'rb')); p = jax.tree.map(jnp.asarray, p)
    Xtr, Xte, V, chars = get_text_data(T=64, n_train=100, n_test=1000)
    rep = {'name': name, 'eval_nll': eval_loss(p, cfg, Xte), 'bits_per_char': eval_loss(p, cfg, Xte) / np.log(2)}
    if not cfg['bottleneck']:
        return rep
    logits, codes = get_codes(p, cfg, Xte, bs=100)
    rep['dead_frac'] = dead_features(codes)
    P = text_predicates(Xte, chars)
    for i, c in enumerate(codes):
        best, names, live = feature_f1(c, P, min_pos=1)
        rep[f'bn{i}_legibility'] = {'n_live': int(live.sum()), 'median_bestF1': float(np.median(best[live])),
                                    'frac_F1>0.8': float((best[live] > 0.8).mean()), 'frac_F1>0.5': float((best[live] > 0.5).mean()),
                                    'top': [(int(f), names[f], round(float(best[f]), 2)) for f in np.argsort(-best)[:10]]}
    W = attribution(p, cfg, codes[-1])
    rep['ablation'] = ablation_test(p, cfg, Xte, codes[-1], W, n_examples=200)
    rep['completeness_top3_median'] = completeness(codes[-1], W, Xte)
    return rep

if __name__ == '__main__':
    for n in sys.argv[1:]:
        r = text_report(n); json.dump(r, open(f'report_{n}.json', 'w'), indent=1)
        print(json.dumps({k: (v if not isinstance(v, dict) or 'top' not in v else {kk: vv for kk, vv in v.items() if kk != 'top'}) for k, v in r.items()}, indent=1))
