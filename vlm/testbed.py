"""
Generalizing synthetic language with a known, compositional mechanism and an exact oracle.

Why this exists (see docs/REVIEW-2026-09-12.md §3): the previous testbed had 180 distinct sequences and every
"held-out" sequence appeared in training, so nothing measured generalization. This one is designed so that
(a) the space of sequences is combinatorially large (fresh test sequences are essentially never seen),
(b) the mechanism is compositional (a rule is applied to arguments; the rule can change mid-sequence),
(c) specific (rule, argument) events are held out of training entirely, so behaviour on them can only come
    from having learned the rule, and
(d) the true conditional distribution is computable, so every NLL has an exact floor.

Vocabulary: M mode tokens (ids 0..M-1) and C content tokens (ids M..M+C-1).
Process:
  x[0]      = random mode token
  x[1..3]   = random content tokens (the seed)
  x[t], t>=4: with prob P_SWITCH (only if t >= SWITCH_MIN_POS, fewer than MAX_SWITCHES so far, and at least
              SWITCH_GAP positions since the last mode token) a uniformly random mode token; otherwise the
              current mode's rule applied to the content history (mode tokens are not content and are skipped).
Rules (c = content history):
  repeat: c[-1]      succ: c[-1]+1      pred: c[-1]-1      plus2: c[-1]+2      copy2: c[-2]      copy3: c[-3]
Held-out events: for every mode m one argument value HELD[m]; a training sequence never contains a prediction
step whose (mode, argument) is held out. Test sequences may. "Argument" = the content token the rule reads.
A second family of held-out events: ordered mode-switch pairs (a -> b) that never occur in training.
"""
import numpy as np

M_MODES, C_TOK = 6, 16
VOCAB = M_MODES + C_TOK
MODE_NAMES = ['rept', 'succ', 'pred', 'pls2', 'cpy2', 'cpy3']
P_SWITCH, SWITCH_MIN_POS, MAX_SWITCHES, SWITCH_GAP = 0.1, 6, 2, 5
# held-out argument value per mode (content index 0..C-1) and held-out ordered switch pairs
HELD_ARG = {0: 3, 1: 7, 2: 11, 3: 14, 4: 5, 5: 9}
HELD_SWITCH = {(1, 4), (4, 2), (3, 5), (5, 0)}

def rule_apply(mode, c):
    """c: list of content indices (0..C-1), most recent last. Returns next content index."""
    if mode == 0: return c[-1]
    if mode == 1: return (c[-1] + 1) % C_TOK
    if mode == 2: return (c[-1] - 1) % C_TOK
    if mode == 3: return (c[-1] + 2) % C_TOK
    if mode == 4: return c[-2]
    if mode == 5: return c[-3]

def rule_arg(mode, c):
    return c[-1] if mode in (0, 1, 2, 3) else (c[-2] if mode == 4 else c[-3])

class State:
    """Exact state of the process after observing a prefix; drives both generation and the oracle."""
    __slots__ = ('mode', 'c', 'n_switch', 'last_switch', 'prev_mode')
    def __init__(self):
        self.mode = None; self.c = []; self.n_switch = 0; self.last_switch = 0; self.prev_mode = None
    def observe(self, t, tok):
        if tok < M_MODES:
            if t > 0:
                self.n_switch += 1; self.last_switch = t; self.prev_mode = self.mode
            self.mode = tok
        else:
            self.c.append(tok - M_MODES)
    def switch_possible(self, t):
        """Can position t (the token about to be generated) be a mode token?"""
        return t >= SWITCH_MIN_POS and self.n_switch < MAX_SWITCHES and (t - self.last_switch) >= SWITCH_GAP
    def next_dist(self, t):
        """Exact P(x[t] | prefix) as a (VOCAB,) array. t >= 1."""
        p = np.zeros(VOCAB)
        if t <= 3:
            p[M_MODES:] = 1.0 / C_TOK; return p
        ps = P_SWITCH if self.switch_possible(t) else 0.0
        p[:M_MODES] = ps / M_MODES
        p[M_MODES + rule_apply(self.mode, self.c)] += 1.0 - ps
        return p
    def event(self, t):
        """(mode, argument) for the rule step producing x[t], plus the switch pair if a switch just happened."""
        if t <= 3: return None, None
        return (self.mode, rule_arg(self.mode, self.c)), ((self.prev_mode, self.mode) if self.last_switch == t - 1 and t - 1 > 0 else None)

def _gen_one(T, rng, allow_heldout):
    while True:
        x = np.zeros(T, np.int32); st = State(); ok = True
        x[0] = rng.integers(M_MODES); st.observe(0, x[0])
        for t in range(1, T):
            if t <= 3:
                x[t] = M_MODES + rng.integers(C_TOK)
            else:
                if not allow_heldout:
                    (m, a), sw = st.event(t)
                    if HELD_ARG[m] == a or (sw is not None and sw in HELD_SWITCH): ok = False; break
                if st.switch_possible(t) and rng.random() < P_SWITCH:
                    x[t] = rng.integers(M_MODES)
                else:
                    x[t] = M_MODES + rule_apply(st.mode, st.c)
            st.observe(t, x[t])
        if ok: return x

def generate(n, T, seed, allow_heldout=False, require_heldout=False):
    """Returns X (n,T) int32. allow_heldout=False -> training distribution (held-out events rejected).
    require_heldout=True -> only sequences containing at least one held-out event (the shift split)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x = _gen_one(T, rng, allow_heldout or require_heldout)
        if require_heldout and not annotate(x[None])['heldout'].any(): continue
        out.append(x)
    return np.stack(out)

def annotate(X):
    """Per-position ground truth for X (B,T). All arrays are (B,T); entry t describes the step that PREDICTS x[t+1]
    (i.e. the state after observing x[0..t]). Last position is filler.
      mode: effective mode; cur/prev/prev2/prev3: content history (or -1); next: oracle argmax content token
      (content index) or -1; arg: the argument the rule reads; is_mode: x[t] is a mode token; dist_mode: t - last
      mode-token position; heldout: the step (mode,arg) or the switch pair is held out; heldout_arg / heldout_sw
      split the two; oracle_lp: log P_true(x[t+1] | x[..t]); oracle_p: full (B,T,VOCAB) true distribution."""
    B, T = X.shape
    keys = ['mode', 'cur', 'prev', 'prev2', 'prev3', 'next', 'arg', 'is_mode', 'dist_mode', 'heldout', 'heldout_arg', 'heldout_sw']
    A = {k: np.full((B, T), -1, np.int64) for k in keys}
    lp = np.zeros((B, T)); P = np.zeros((B, T, VOCAB))
    for b in range(B):
        st = State(); last_mode_pos = 0
        for t in range(T):
            st.observe(t, X[b, t])
            if X[b, t] < M_MODES: last_mode_pos = t
            A['mode'][b, t] = st.mode; A['is_mode'][b, t] = int(X[b, t] < M_MODES); A['dist_mode'][b, t] = t - last_mode_pos
            c = st.c
            for i, k in enumerate(['cur', 'prev', 'prev2', 'prev3']):
                A[k][b, t] = c[-1 - i] if len(c) > i else -1
            if t + 1 < T:
                p = st.next_dist(t + 1); P[b, t] = p; lp[b, t] = np.log(p[X[b, t + 1]])
                if t + 1 > 3:
                    A['next'][b, t] = rule_apply(st.mode, c); A['arg'][b, t] = rule_arg(st.mode, c)
                    ha = int(HELD_ARG[st.mode] == A['arg'][b, t])
                    # switch pair is held out if x[t] is a mode token and (prev_mode -> mode) is in HELD_SWITCH
                    hs = int(X[b, t] < M_MODES and t > 0 and (st.prev_mode, st.mode) in HELD_SWITCH)
                    A['heldout_arg'][b, t] = ha; A['heldout_sw'][b, t] = hs; A['heldout'][b, t] = int(ha or hs)
            if t + 1 >= T or t + 1 <= 3:
                A['heldout'][b, t] = A['heldout_arg'][b, t] = A['heldout_sw'][b, t] = 0
    A['oracle_lp'] = lp; A['oracle_p'] = P
    return A

def make_splits(T=24, n_train=40000, n_test=3000, seed=0, T_long=32):
    """train (no held-out events), iid test (same distribution), shift test (contains held-out events),
    long test (same process, longer sequences, held-out events allowed)."""
    return dict(train=generate(n_train, T, seed, allow_heldout=False),
                iid=generate(n_test, T, seed + 1, allow_heldout=False),
                shift=generate(n_test, T, seed + 2, require_heldout=True),
                long=generate(n_test, T_long, seed + 3, allow_heldout=True))

def oracle_nll(X, mask=None):
    """Mean true NLL per predicted token (positions 0..T-2), optionally restricted to a (B,T) mask."""
    A = annotate(X); lp = -A['oracle_lp'][:, :-1]
    if mask is None: return float(lp.mean())
    m = mask[:, :-1].astype(bool); return float(lp[m].mean())

def describe(x):
    return ' '.join(MODE_NAMES[t][:4].upper() if t < M_MODES else str(t - M_MODES) for t in x)

if __name__ == '__main__':
    import time; t0 = time.time()
    S = make_splits(n_train=20000, n_test=2000)
    print('gen %.1fs' % (time.time() - t0))
    tr = {tuple(r) for r in S['train']}
    for k in ['iid', 'shift', 'long']:
        u = {tuple(r) for r in S[k]}; print(f"{k}: unique {len(u)}/{len(S[k])}, seen in train {len(u & tr)}")
    print('unique train', len(tr))
    for k, X in S.items():
        A = annotate(X)
        print(f"{k:6s} oracle nll/token {oracle_nll(X):.4f}  heldout steps {A['heldout'][:, :-1].mean():.3f}  "
              f"(arg {A['heldout_arg'][:, :-1].mean():.3f}, sw {A['heldout_sw'][:, :-1].mean():.3f})")
    for i in range(3): print(describe(S['shift'][i]))
