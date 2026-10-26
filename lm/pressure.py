"""Faithful-replay pressure (docs/redesign/FAITHFUL-REPLAY-PRESSURE.md v3.1): the training-time version of
frontier2's evaluation.

On one window, run the model's own sparse quantized replay at a budget sampled per application, and penalize:
  KL(stop-grad(model) || replay)
plus a ranking hinge. The hinge asks that, under the replayed routing query, the units the full run selected outscore
the best unit it did not select, by a margin measured in quantization steps of that query read (scale-free: inflating
scores does not help). sel='given' replays with the full run's selection (teacher-forced); sel='free' derives it.
"""
import jax, jax.numpy as jnp
from model import forward, select_from_query
import frontier2

def ranking_hinge(p, cfg, rec, aux, bias):
    R, K, nk, dk = cfg['r_heads'], cfg['topk'], cfg['n_keys'], cfg['d_key']; out = []
    for l, b in enumerate(p['blocks']):
        qr, stepq = aux['qr'][l], aux['step_qr'][l]                              # (B,T,R,dk), (B,T,R)
        uf = rec['layers'][l]['unit']; B, T = uf.shape[:2]; uf = uf.reshape(B, T, R, K)
        a, c = uf // nk, uf % nk
        bb = jnp.zeros((2, nk)) if bias is None else jax.lax.stop_gradient(bias[l])
        k1, k2 = b['keys1'][a], b['keys2'][c]                                     # (B,T,R,K,dk/2)
        s_sel = (jnp.einsum('btrk,btrjk->btrj', qr[..., :dk // 2], k1) + jnp.einsum('btrk,btrjk->btrj', qr[..., dk // 2:], k2)
                 + bb[0][a] + bb[1][c])
        cu, cs, ca, cc = select_from_query(b, qr, cfg, None if bias is None else bias[l], K=2 * K)
        cz = cs + bb[0][ca] + bb[1][cc]
        in_set = (cu[..., :, None] == uf[..., None, :]).any(-1)
        best_other = jnp.where(in_set, -1e30, cz).max(-1)                         # (B,T,R)
        knorm = jnp.sqrt((k1 * k1).sum(-1) + (k2 * k2).sum(-1)).mean(-1)
        margin = (s_sel.min(-1) - best_other) / (stepq * knorm + 1e-9)
        # smoothly capped at 4: a violation measured in quantization steps can be huge when a small budget changes the
        # replayed query a lot, and an uncapped hinge then spikes the gradient (seen in the smoke test: 20.8)
        out.append((4.0 * jnp.tanh(jax.nn.relu(1.0 - margin) / 4.0)).mean())
    return jnp.mean(jnp.stack(out))

def replay_loss(p, cfg, xw, bias, key, sel, quant, hinge_w):
    """xw: (1, T) one window. Returns (kl, hinge)."""
    k_b, k_q = jax.random.split(key)
    kb = jax.random.split(k_b, 3)
    pick = lambda k, ch: jnp.asarray(ch, jnp.int32)[jax.random.randint(k, (), 0, len(ch))]
    m, n, ma = pick(kb[0], cfg['press_m_choices']), pick(kb[1], cfg['press_n_choices']), pick(kb[2], cfg['press_ma_choices'])
    logits, aux = forward(jax.lax.stop_gradient(p), cfg, xw, bias=bias, record=True)
    rec = jax.tree.map(jax.lax.stop_gradient, aux['record']); lp = jax.lax.stop_gradient(jax.nn.log_softmax(logits, -1))
    lr, ra = frontier2.replay(p, cfg, xw, rec, m, n, ma, sel, (quant, 8), k_q, bias, ('qr',) if hinge_w > 0 else ())
    kl = (jnp.exp(lp) * (lp - jax.nn.log_softmax(lr, -1))).sum(-1).mean()
    hinge = ranking_hinge(p, cfg, rec, ra, bias) if hinge_w > 0 else jnp.zeros(())
    return kl, hinge
