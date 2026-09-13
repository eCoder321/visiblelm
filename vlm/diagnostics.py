"""Diagnostics that go beyond the main metric suite in metrics.py.

near_switch_report: accuracy on rule steps broken down by distance-to-last-mode-token, per mode, plus the mean
probability mass the model puts on mode tokens far from a switch (oracle 0.10). Used to check whether the residual
loss after the snap phase is concentrated near mode switches (see docs/tasks/T1-replicate-and-close-discrete-gap.md).
"""
import models, pickle, numpy as np
from metrics import get_states
from testbed import make_splits, annotate, MODE_NAMES, M_MODES

def near_switch_report(bundle_path):
    b = pickle.load(open(bundle_path, 'rb'))
    S = make_splits(n_train=100, n_test=1000); X = S['iid']; A = annotate(X)
    lg, _ = get_states(b, X); pred = lg[:, :-1].argmax(-1); true = A['oracle_p'][:, :-1].argmax(-1)
    rule = A['next'][:, :-1] >= 0; md = A['mode'][:, :-1]; dm = A['dist_mode'][:, :-1]
    per_mode = {MODE_NAMES[m]: float((pred == true)[rule & (md == m)].mean()) for m in range(M_MODES)}
    by_dist = {d: float((pred == true)[rule & (dm == d)].mean()) for d in [1, 2, 3, 4]}
    far = float((pred == true)[rule & (dm >= 5)].mean())
    lp = lg[:, :-1] - np.log(np.exp(lg[:, :-1] - lg[:, :-1].max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg[:, :-1].max(-1, keepdims=True)
    p_mode = float(np.exp(lp[..., :M_MODES]).sum(-1)[rule & (dm >= 5)].mean())
    return dict(per_mode=per_mode, by_dist=by_dist, far=far, p_mode_far=p_mode)

if __name__ == '__main__':
    import sys, json
    print(json.dumps(near_switch_report(sys.argv[1]), indent=1))
