"""Training loop shared by all models, progressive pruning for rulenet, SAE fitting for the null model,
and split evaluation against the oracle."""
import time, jax, jax.numpy as jnp, numpy as np, optax
from functools import partial
from models import INIT, APPLY, bottleneck_aux_loss, prune_masks, mask_stats, init_sae, sae_encode, sae_decode, topk_mask
from testbed import annotate, VOCAB

def nll_from_logits(logits, x, mask=None):
    lp = jax.nn.log_softmax(logits[:, :-1], -1)
    tok = -jnp.take_along_axis(lp, x[:, 1:, None], -1)[..., 0]
    if mask is None: return tok.mean()
    m = mask[:, :-1]; return (tok * m).sum() / jnp.maximum(m.sum(), 1)

def make_loss(cfg):
    def loss(p, masks, x):
        if cfg['model'] == 'bottleneck':
            logits, states, aux = APPLY['bottleneck'](p, masks, cfg, x, opts={'return_aux': True})
            nll = nll_from_logits(logits, x)
            return nll + cfg.get('aux_coef', 1.0) * bottleneck_aux_loss(p, cfg, aux, states), nll
        logits, states = APPLY[cfg['model']](p, masks, cfg, x)
        nll = nll_from_logits(logits, x)
        reg = 0.0
        if cfg['model'] == 'rulenet' and cfg.get('l1', 0) > 0:
            for lp, lm in zip(p['layers'], masks['layers']):
                reg = reg + sum(jnp.abs(lp[n] * lm[n]).sum() for n in ['A', 'u', 'P', 'W1', 'W2'])
            if 'emb' in p: reg = reg + jnp.abs(p['emb'] * masks['emb']).sum()
            reg = cfg['l1'] * reg
        return nll + reg, nll
    return loss

def train(cfg, Xtr, seed=0, evals=None, log=print):
    """cfg keys: model, steps, lr, bs, plus model-specific. rulenet: prune_at (list of step fractions), the last
    reaching the target. Returns params, masks, history."""
    key = jax.random.PRNGKey(seed)
    p, masks = INIT[cfg['model']](cfg, key)
    steps, lr = cfg['steps'], cfg['lr']
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(100, steps // 10), steps, lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=cfg.get('wd', 0.01)))
    loss = make_loss(cfg)
    @jax.jit
    def step(p, masks, os, x):
        (l, nll), g = jax.value_and_grad(loss, has_aux=True)(p, masks, x)
        upd, os = opt.update(g, os, p); p = optax.apply_updates(p, upd)
        if cfg['model'] == 'bottleneck' and not cfg.get('identity_dict', False):
            p['dict'] = p['dict'] / jnp.linalg.norm(p['dict'], axis=1, keepdims=True)
        return p, os, nll
    os = opt.init(p)
    rng = np.random.default_rng(seed); hist = []; t0 = time.time()
    prune_at = sorted(cfg.get('prune_at', [])) if cfg['model'] == 'rulenet' else []
    prune_steps = [int(f * steps) for f in prune_at]
    regrow_every = cfg.get('regrow_every', 0); regrow_until = int(cfg.get('regrow_until', 0.8) * steps)
    ones = jax.tree.map(jnp.ones_like, masks)
    grad_fn = jax.jit(lambda p, x: jax.grad(lambda p_, x_: loss(p_, ones, x_)[0])(p, x))
    cur_frac = 0.0
    for i in range(steps):
        is_prune = i in prune_steps
        is_regrow = regrow_every and prune_steps and i > prune_steps[0] and i <= regrow_until and i % regrow_every == 0 and not is_prune
        if is_prune or is_regrow:
            if is_prune: cur_frac = (prune_steps.index(i) + 1) / len(prune_steps)
            rf = cfg.get('regrow_frac', 0.0)
            g = grad_fn(p, jnp.asarray(Xtr[rng.integers(0, len(Xtr), 256)])) if rf > 0 else None
            masks, p = prune_masks(p, masks, cfg, cur_frac, grads=g, regrow_frac=rf)
            if is_prune: log(f"  step {i}: pruned to frac {cur_frac:.2f} -> {mask_stats(masks)}")
        idx = rng.integers(0, len(Xtr), cfg['bs'])
        p, os, nll = step(p, masks, os, jnp.asarray(Xtr[idx]))
        if i % cfg.get('log_every', 250) == 0 or i == steps - 1:
            ev = {k: evaluate(p, masks, cfg, X)['nll'] for k, X in (evals or {}).items()}
            hist.append({'step': i, 'train_nll': float(nll), **ev})
            log(f"  step {i:5d} train {float(nll):.4f} " + ' '.join(f"{k} {v:.4f}" for k, v in ev.items()) + f"  ({time.time()-t0:.0f}s)")
    return p, masks, hist

@partial(jax.jit, static_argnums=(2, 4))
def _logits(p, masks, cfg_t, x, opts_t):
    cfg = dict(cfg_t); return APPLY[cfg['model']](p, masks, cfg, x, opts=dict(opts_t))[0]

def _cfg_t(cfg):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()))

def logits_of(p, masks, cfg, X, opts=None, bs=500):
    out = []
    for i in range(0, len(X), bs):
        out.append(np.asarray(_logits(p, masks, _cfg_t(cfg), jnp.asarray(X[i:i + bs]), tuple(sorted((opts or {}).items())))))
    return np.concatenate(out)

def evaluate(p, masks, cfg, X, opts=None, ann=None):
    """NLL per predicted token overall and on held-out-event steps; oracle floors for both."""
    lg = logits_of(p, masks, cfg, X, opts)
    lp = jax.nn.log_softmax(jnp.asarray(lg[:, :-1]), -1)
    tok = -np.asarray(jnp.take_along_axis(lp, jnp.asarray(X[:, 1:, None]), -1)[..., 0])
    r = {'nll': float(tok.mean())}
    if ann is not None:
        r['oracle'] = float(-ann['oracle_lp'][:, :-1].mean())
        for key in ['heldout', 'heldout_arg', 'heldout_sw']:
            m = ann[key][:, :-1].astype(bool)
            if m.sum() > 0:
                r[f'nll_{key}'] = float(tok[m].mean()); r[f'oracle_{key}'] = float(-ann['oracle_lp'][:, :-1][m].mean())
        # argmax agreement with the oracle on rule steps (excludes seed positions)
        rule = ann['next'][:, :-1] >= 0
        pred = lg[:, :-1].argmax(-1); true = ann['oracle_p'][:, :-1].argmax(-1)
        r['acc_rule'] = float((pred == true)[rule].mean())
        m = ann['heldout'][:, :-1].astype(bool)
        if m.sum() > 0: r['acc_heldout'] = float((pred == true)[m].mean())
    return r

# ----------------------------------------------------------------------------- SAE fitting (null model)
def fit_saes(cfg, p_dense, Xtr, seed=0, steps=3000, lr=3e-3, bs=256, log=print):
    """Fit one top-k SAE per residual boundary of a trained dense model; returns the dense_sae params."""
    F, k = cfg['F'], cfg['k']; d = cfg['d']
    res_fn = jax.jit(lambda x: APPLY['dense'](p_dense, {}, cfg, x)[1])
    saes = []
    for i in range(cfg['layers'] + 1):
        key = jax.random.PRNGKey(seed + 100 * i); s = init_sae(key, d, F)
        opt = optax.adam(lr); os = opt.init(s)
        def loss(s, h):
            c = sae_encode(s, h, k); rec = sae_decode(s, c)
            mse = jnp.mean((h - rec) ** 2)
            # AuxK: dead features reconstruct the residual error
            used = (c > 0).any(axis=(0, 1))
            z = jax.nn.relu((h - s['b_dec']) @ s['enc'] + s['b_enc'])
            zd = jnp.where(used[None, None], 0.0, z); rec_d = topk_mask(zd, 2 * k) @ s['dec']
            return mse + 0.1 * jnp.mean((jax.lax.stop_gradient(h - rec) - rec_d) ** 2), mse
        @jax.jit
        def step(s, os, h):
            (l, mse), g = jax.value_and_grad(loss, has_aux=True)(s, h)
            u, os = opt.update(g, os, s); s = optax.apply_updates(s, u)
            s['dec'] = s['dec'] / jnp.linalg.norm(s['dec'], axis=1, keepdims=True)
            return s, os, mse
        rng = np.random.default_rng(seed)
        for j in range(steps):
            idx = rng.integers(0, len(Xtr), bs); h = res_fn(jnp.asarray(Xtr[idx]))[i]
            s, os, mse = step(s, os, h)
        h = res_fn(jnp.asarray(Xtr[:512]))[i]
        fvu = float(jnp.mean((h - sae_decode(s, sae_encode(s, h, k))) ** 2) / jnp.var(h))
        log(f"  SAE {i}: final mse {float(mse):.4f}, FVU {fvu:.3f}")
        saes.append(s)
    return {'dense': p_dense, 'saes': saes}
