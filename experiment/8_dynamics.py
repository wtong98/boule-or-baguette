"""Toy model of gradient dynamics"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp

import sys
sys.path.append('../')
from common import *
from train import *
from model.transformer import *
from task.graph import *


def extract_summary(state, train_task):
    n_hop = train_task.samp_dist[1]

    emb = np.array(state.params['Embed_freeze']['embedding'])
    A = np.array(state.params['Dense_0']['kernel'])
    W = np.array(state.params['Dense_1']['kernel'])

    readout = emb @ W
    att = emb @ t(A) @ t(emb)

    a_L = np.diag(att)
    tok_signs = np.sign(a_L)

    readout_unsigned = tok_signs[:,None] * readout
    att_unsigned = tok_signs[None,:] * att

    w_no = readout_unsigned[5:-n_hop,1]
    w_yes = readout_unsigned[5:-n_hop,2]

    mask = np.ones(readout.shape)
    mask = np.triu(mask, k=-2 * n_hop - 2) * np.tril(mask, k=2 * n_hop - 2)
    mask_u = np.triu(mask, k=-1)
    mask_l = np.tril(mask, k=-3)

    w_t = np.mean(w_no + w_yes)
    w_L = np.mean(np.diag(readout_unsigned, k=-2)[5:-n_hop])

    w_g = (mask_u * readout_unsigned).sum(axis=0) / (np.sum(mask_u, axis=0) + 1e-8) 
    w_g = np.mean(w_g[5:-n_hop])

    w_p = (mask_l * readout_unsigned).sum(axis=0) / (np.sum(mask_l, axis=0) + 1e-8)
    w_p = np.mean(w_p[5:-n_hop])

    mask = np.ones(att.shape)
    mask = np.triu(mask, k=-2 * n_hop) * np.tril(mask, k=2 * n_hop)
    mask_u = np.triu(mask, k=1)
    mask_l = np.tril(mask, k=-1)

    a_L = np.mean(np.diag(att_unsigned)[5:-n_hop])

    a_g = (mask_l * att_unsigned).sum(axis=1) / (np.sum(mask_l, axis=1) + 1e-8)
    a_g = np.mean(a_g[5:-n_hop])

    a_p = (mask_u * att_unsigned).sum(axis=1) / (np.sum(mask_u, axis=1) + 1e-8)
    a_p = np.mean(a_p[5:-n_hop])

    xs, _ = next(train_task)
    logits = state.apply_fn({'params': state.params}, xs)
    probs = jax.nn.softmax(logits)
    probs = probs[:,:-1]
    xs_next = xs[:,1:]

    nt_idx = xs_next > 0
    nt_idx = nt_idx.at[:,:2].set(False)
    nt_probs = probs[nt_idx]

    out_prob = nt_probs[np.arange(len(nt_probs)), xs_next[nt_idx]]
    pred = xs_next[nt_idx]

    term_idx = (pred == 1) | (pred == 2)
    ds_t = np.mean(out_prob[term_idx])
    ds_nt = np.mean(out_prob[~term_idx])

    return np.array([w_t, w_g, w_p, w_L, a_g, a_p, a_L, ds_t, ds_nt])


depth = 10
n_vocab = 2 * depth + 1 + StarfishTask.offset
n_hidden = 512
batch_size = 512

n_layers = 1

cot = True
ttr = False
nouveau = True
n_hop = 3
test_n_hop = 5

train_task = StarfishTask(depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)
test_task = StarfishTask(depth=depth, samp_dist=test_n_hop, batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)

# print(next(train_task)[0][-12:])

# <codecell>
config = TrConfig(n_vocab=n_vocab, 
                  pos_emb=False,
                  n_out=n_vocab if cot else 1,
                  n_hidden=n_hidden, 
                  return_format=None if cot else True)

# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_iters=1,
                    loss='ce_mask' if cot else 'bce',
                    test_every=1000,
                    train_iters=10_000,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot] if cot else None,
                    summary_fn=lambda state: extract_summary(state, train_task),
                    print_fn=print_gen if cot else None,
                    lr=1,
                    optim=optax.sgd,
                    )

emb = np.array(state.params['Embed_freeze']['embedding'])
A = np.array(state.params['Dense_0']['kernel'])
W = np.array(state.params['Dense_1']['kernel'])

# <codecell>
xs = jnp.array([[12, 20, 18, 16]])
X = emb[xs]

att = X @ t(A) @ t(X)
logits = np.tril(att) @ X @ W
plt.plot(logits[0, -1], '--o')

np.round(np.tril(att), decimals=2)

# <codecell>
xs = jnp.array([[9, 10, 0, 0, 0]])
gen1(state, xs, beta=1)

# <codecell>
idx = 100
summs = np.array(hist['summary'])[:idx]

ts = np.arange(len(summs))
wt, wg, wp, wl, ag, ap, al, ds_t, ds_nt = summs.T

plt.plot(ts, wt, label='wt')
plt.plot(ts, wg, label='wg')
plt.plot(ts, wp, label='wp')
plt.plot(ts, wl, label='wl')

plt.plot(ts, ag, label='ag')
plt.plot(ts, ap, label='ap')
plt.plot(ts, al, label='al')

plt.legend()
# plt.savefig('fig/true_dynamics.png')

# <codecell>
plt.plot(ds_t, label='term')
plt.plot(ds_nt, label='nt')
plt.legend()

# <codecell>
# NON-TERM ESTIMATE
next_tok_logit = ag * wg + al * wl + 2 * ap * wp
# next_tok_logit = 1 * ag * wg + al * wl + 2 * ap * wp

term_logit = (ag + al + 2 * ap) * wt / 2
# term_logit = (1 * ag + al + 2 * ap) * wt / 2

other_logit = 0
# other_logit = 1 * ag * (wg + wl) + al * (wg + wp) + 2 * ap * (wl + wg)

prob = np.exp(next_tok_logit) / (np.exp(next_tok_logit) + np.exp(term_logit) + np.exp(other_logit))

plt.plot(prob)
plt.plot(ds_nt)

# plt.plot(next_tok_logit)
# plt.plot(term_logit)
# plt.plot(other_logit)

# <codecell>
# TERM ESTIMATE
next_tok_logit = 2 * al * wl + 2 * ap * wp
# next_tok_logit = 2 * al * wl + 3 * ap * wp
term_logit = (al + 2 * ap) * wt
# term_logit = (2 * al + 3 * ap) * wt / 2
other_logit = 0
# other_logit = 2 * al * (wg + wp) + 3 * ap * (wl + wg)

prob = np.exp(term_logit) / (np.exp(next_tok_logit) + np.exp(term_logit) + np.exp(other_logit))

plt.plot(prob)
plt.plot(ds_t)

# plt.plot(next_tok_logit)
# plt.plot(term_logit)
# plt.plot(other_logit)


# <codecell>
emb = np.array(state.params['Embed_freeze']['embedding'])
A = np.array(state.params['Dense_0']['kernel'])
W = np.array(state.params['Dense_1']['kernel'])

readout = emb @ W
att = emb @ t(A) @ t(emb)

a_L = np.diag(att)
tok_signs = np.sign(a_L)

readout_unsigned = tok_signs[:,None] * readout
att_unsigned = tok_signs[None,:] * att

w_no = readout_unsigned[5:-n_hop,1]
w_yes = readout_unsigned[5:-n_hop,2]

mask = np.ones(readout.shape)
mask = np.triu(mask, k=-2 * n_hop - 2) * np.tril(mask, k=2 * n_hop - 2)
mask_u = np.triu(mask, k=-1)
mask_l = np.tril(mask, k=-3)

w_t = np.mean(w_no + w_yes)
w_L = np.mean(np.diag(readout_unsigned, k=-2)[5:-n_hop])

w_g = (mask_u * readout_unsigned).sum(axis=0) / (np.sum(mask_u, axis=0) + 1e-8) 
w_g = np.mean(w_g[5:-n_hop])

w_p = (mask_l * readout_unsigned).sum(axis=0) / (np.sum(mask_l, axis=0) + 1e-8)
w_p = np.mean(w_p[5:-n_hop])

mask = np.ones(att.shape)
mask = np.triu(mask, k=-2 * n_hop) * np.tril(mask, k=2 * n_hop)
mask_u = np.triu(mask, k=1)
mask_l = np.tril(mask, k=-1)

a_L = np.mean(np.diag(att_unsigned)[5:-n_hop])

a_g = (mask_l * att_unsigned).sum(axis=1) / (np.sum(mask_l, axis=1) + 1e-8)
a_g = np.mean(a_g[5:-n_hop])

a_p = (mask_u * att_unsigned).sum(axis=1) / (np.sum(mask_u, axis=1) + 1e-8)
a_p = np.mean(a_p[5:-n_hop])


# <codecell>
readout = emb @ W
thr = max(np.max(readout), -np.min(readout))

plt.imshow(readout_unsigned, cmap='BrBG', vmin=-thr, vmax=thr)
plt.colorbar()
plt.title('readout')

xs = np.linspace(0, 21)
plt.plot(xs, xs + 2, color='red', alpha=0.3)

plt.savefig('fig/lin_readout.png')

# <codecell>
att = emb @ t(A) @ t(emb)
thr = max(np.max(att), -np.min(att))

plt.imshow(att_unsigned, cmap='BrBG', vmin=-thr, vmax=thr)
plt.colorbar()

xs = np.linspace(0, 23)
plt.plot(xs, xs, color='red', alpha=0.3)

plt.title('att')
plt.savefig('fig/lin_att.png')

# <codecell>
xs, ys = next(train_task)
logits = state.apply_fn({'params': state.params}, xs)
probs = jax.nn.softmax(logits)
probs = probs[:,:-1]
xs_next = xs[:,1:]

nt_idx = xs_next > 0
nt_idx = nt_idx.at[:,:2].set(False)
nt_probs = probs[nt_idx]

out_prob = nt_probs[np.arange(len(nt_probs)), xs_next[nt_idx]]
pred = xs_next[nt_idx]

term_idx = (pred == 1) | (pred == 2)
print(np.mean(out_prob[term_idx]))
print(np.mean(out_prob[~term_idx]))

# <codecell>
### TRACING DYNAMICS
# def grad_old(t, xs):
#     pt = 0.5
#     eta = 0.1

#     wt, wg, wp, wl, ag, ap, al = xs

#     ds = 1 / (1 + np.exp(-al))
#     df = ds

#     wt_n = pt * ds * (al + ap) - (1 - pt) * df * (al + ap + ag)
#     wg_n = - (1 - pt) * df * (al  + ag)
#     wp_n = - (1 - pt) * df * (al + ap) - pt * df * (al + ap)
#     wl_n = (1 - pt) * (al * ds - ap * df - ag * df) - pt * df * (al + ap + ag)

#     ag_n = - (1 - pt) * df * (wt + wl + wg)
#     ap_n = - (1 - pt) * df * (wt + wl + wg) + pt * (wt * ds - wl * df - wp * df)
#     al_n = (1 - pt) * (wl * ds - wt * df - wp * df - wg * df) + pt * (wt * ds - wl * df - wp * df - wg * df)

#     diff = np.array([wt_n, wg_n, wp_n, wl_n, ag_n, ap_n, al_n])
#     return eta * diff

# def grad(t, xs):
#     eta = 0.1
#     k = 3
#     wt, wg, wp, wl, ag, ap, al = xs

#     # logit_nn = ag * wg + al * wl + (k / 2) * ap * wp
#     # logit_tn = (ag + al + (k/2) * ap) * wt
#     # p_nn = np.exp(logit_nn) / (np.exp(logit_nn) + np.exp(logit_tn) + 1e-8)
#     p_nn = 1 / (1 + np.exp(-al * wl))

#     # logit_nt = 2 * al * wl + k * ap * wp
#     # logit_tt = (2 * al + k * ap) * wt
#     # p_tt = np.exp(logit_tt) / (np.exp(logit_nt) + np.exp(logit_tt) + 1e-8)
#     p_tt = 1 / (1 + np.exp(-2 * al * wt))

#     dn_s = 1 - p_nn
#     dn_f = -(p_tt - 1)
#     dt_s = 1 - p_tt
#     dt_f = -(p_nn - 1)

#     print('DN_S', dn_s)
#     print('DN_F', dn_f)
#     print('DT_S', dt_s)
#     print('DT_f', dt_f)

#     at = (1 / k) * al + (k - 1) / k * ap

#     wl_ = dn_s * al - dn_f * (ag + (k/2) * ap) - dt_f * at
#     wg_ = (1 / (2 * k)) * dn_s * ag - dn_f * (ag + (k/2) * ap + al) - dt_f * at

#     wp_ = (1/2) * dn_s * ap - dn_f * (ag + (k/2) * ap + al) - dt_f * at
#     wt_ = dt_s * at - dn_f * (ag + (k/2) * ap + al)

#     al_ = dn_s * wl - dn_f * (2 * k * wg + k * wp) - dt_f * wt \
#             + (2 / k) * (dt_s * wt - dn_f * wl - dn_f * (2 * k * wg + k * wp))
#     ag_ = 1 / (2 * k) * (dn_s * wg - dn_f * ((2*k - 1) * wg + k * wp + wl) - dt_f * wt)

#     ap_ = (1 / 2) * (dn_s * wp - dn_f * (2 * k * wg + (k - 1) * wp + wl) - dt_f * wt \
#                      + (2 / k) * (dt_s * wt - dn_f * (2 * k * wg + k * wp + wl)))
    
#     al_, ag_, ap_ = k * al_, k * ag_, k * ap_

#     diff = np.array([wt_, wg_, wp_, wl_, ag_, ap_, al_])
#     return eta * diff


def grad(t, xs):
    eta = 0.1
    k = 3
    wt, wg, wp, wl, ag, ap, al = xs

    # logit_nn = ag * wg + al * wl + (k / 2) * ap * wp
    # logit_tn = (ag + al + (k/2) * ap) * wt
    # p_nn = np.exp(logit_nn) / (np.exp(logit_nn) + np.exp(logit_tn) + 1e-8)
    p_nn = 1 / (1 + np.exp(-al * wl))

    # logit_nt = 2 * al * wl + k * ap * wp
    # logit_tt = (2 * al + k * ap) * wt
    # p_tt = np.exp(logit_tt) / (np.exp(logit_nt) + np.exp(logit_tt) + 1e-8)
    p_tt = 1 / (1 + np.exp(-2 * al * wt))


    dn_s = 1 - p_nn
    dn_f = -(p_tt - 1)
    dt_s = 1 - p_tt
    dt_f = -(p_nn - 1)

    print('DN_S', dn_s)
    print('DN_F', dn_f)
    print('DT_S', dt_s)
    print('DT_f', dt_f)

    at = (1 / k) * al + (k - 1) / k * ap

    # wl_ = dn_s * al - dn_f * (ag + (k/2) * ap) - dt_f * at
    wl_ = dn_s * al

    # wg_ = (1 / (2 * k)) * dn_s * ag - dn_f * (ag + (k/2) * ap + al) - dt_f * at
    wg_ =  -dn_f * al

    # wp_ = (1/2) * dn_s * ap - dn_f * (ag + (k/2) * ap + al) - dt_f * at
    wp_ = -dn_f * (ag + al)

    # wt_ = dt_s * at - dn_f * (ag + (k/2) * ap + al)
    wt_ = dt_s * al - dn_f * (al + ag)

    # al_ = dn_s * wl - dn_f * (2 * k * wg + k * wp) - dt_f * wt \
    #         + (2 / k) * (dt_s * wt - dn_f * wl - dn_f * (2 * k * wg + k * wp))
    al_ = dn_s * wl

    # ag_ = 1 / (2 * k) * (dn_s * wg - dn_f * ((2*k - 1) * wg + k * wp + wl) - dt_f * wt)
    ag_ = dn_s * wg

    # ap_ = (1 / 2) * (dn_s * wp - dn_f * (2 * k * wg + (k - 1) * wp + wl) - dt_f * wt \
    #                  + (2 / k) * (dt_s * wt - dn_f * (2 * k * wg + k * wp + wl)))
    ap_ = dn_s * wp
    
    al_, ag_, ap_ = k * al_, k * ag_, k * ap_

    diff = np.array([wt_, wg_, wp_, wl_, ag_, ap_, al_])
    return eta * diff


xs0 = 0.01 * np.array([0, -1, -1, 1, 0, -1, 1])
# xs0 = np.random.randn((7)) / 100
out = solve_ivp(grad, (0, 1000), xs0)
out

# <codecell>
ts = out.t
wt, wg, wp, wl, ag, ap, al = out.y

plt.plot(ts, wt, '--o', label='wt', alpha=0.7)
plt.plot(ts, wg, '--o', label='wg', alpha=0.7)
plt.plot(ts, wp, '--o', label='wp', alpha=0.7)
plt.plot(ts, wl, '--o', label='wl', alpha=0.7)

plt.plot(ts, ag, '--o', label='ag', alpha=0.7)
plt.plot(ts, ap, '--o', label='ap', alpha=0.7)
plt.plot(ts, al, '--o', label='al', alpha=0.7)


plt.legend()
plt.savefig('fig/est_dynamics.png')
# plt.yscale('log')