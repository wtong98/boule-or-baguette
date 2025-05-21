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
    ds = np.mean(nt_probs[np.arange(len(nt_probs)), xs_next[nt_idx]])

    return np.array([w_t, w_g, w_p, w_L, a_g, a_p, a_L, ds])


depth = 30
n_vocab = 2 * depth + 1 + StarfishTask.offset
n_hidden = 512
batch_size = 128

n_layers = 1

cot = True
ttr = False
nouveau = True
n_hop = 3
test_n_hop = 7

train_task = StarfishTask(depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)
test_task = StarfishTask(depth=depth, samp_dist=test_n_hop, batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)

config = TrConfig(n_vocab=n_vocab, 
                  pos_emb=False,
                  n_out=n_vocab if cot else 1,
                  n_hidden=n_hidden, 
                  return_final_logits_only=False if cot else True)

# <codecell>
gamma = 1

state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_iters=1,
                    loss='ce_mask' if cot else 'bce',
                    test_every=25,
                    train_iters=2_000,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot] if cot else None,
                    summary_fn=lambda state: extract_summary(state, train_task),
                    print_fn=print_gen if cot else None,
                    lr=1 * gamma,
                    optim=optax.sgd,
                    gamma=gamma
                    )

emb = np.array(state.params['Embed_freeze']['embedding'])
A = np.array(state.params['Dense_0']['kernel'])
W = np.array(state.params['Dense_1']['kernel'])

# <codecell>
idx = 100
summs = np.array(hist['summary'])[:idx]

ts = np.arange(len(summs))
wt, wg, wp, wl, ag, ap, al, ds = summs.T

plt.plot(ts, wt, label='wt')
plt.plot(ts, wg, label='wg')
plt.plot(ts, wp, label='wp')
plt.plot(ts, wl, label='wl')

plt.plot(ts, ag, label='ag')
plt.plot(ts, ap, label='ap')
plt.plot(ts, al, label='al')

plt.legend()

# <codecell>
plt.plot(ds)

# <codecell>
n_hop = 20

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

# term_idx = (xs_next == 1) | (xs_next == 2)
# probs[term_idx][:,1:3]

nt_idx = xs_next > 0
# nt_idx = (nt_idx.astype(int) - term_idx.astype(int)).astype(bool)
nt_idx = nt_idx.at[:,:2].set(False)
nt_probs = probs[nt_idx]
np.mean(nt_probs[np.arange(len(nt_probs)), xs_next[nt_idx]])

# <codecell>


np.sum(nt_idx, axis=1)



# <codecell>
### TRACING DYNAMICS
def grad(t, xs):
    pt = 0.5
    ds = 0.2
    df = 0.1
    eta = 0.1

    wt, wg, wp, wl, ag, ap, al = xs

    wt_n = pt * ds * (al + ap) - (1 - pt) * df * (al + ap + ag)
    wg_n = - (1 - pt) * df * (al  + ag)
    wp_n = - (1 - pt) * df * (al + ap) - pt * df * (al + ap)
    wl_n = (1 - pt) * (al * ds - ap * df - ag * df) - pt * df * (al + ap + ag)

    ag_n = - (1 - pt) * df * (wt + wl + wg)
    ap_n = - (1 - pt) * df * (wt + wl + wg) + pt * (wt * ds - wl * df - wp * df)
    al_n = (1 - pt) * (wl * ds - wt * df - wp * df - wg * df) + pt * (wt * ds - wl * df - wp * df - wg * df)

    diff = np.array([wt_n, wg_n, wp_n, wl_n, ag_n, ap_n, al_n])
    return eta * diff

xs0 = 0.01 * np.array([1, -1, -1, 1, -1, -1, 1])
out = solve_ivp(grad, (0, 1000), xs0)
out

# <codecell>
ts = out.t
wt, wg, wp, wl, ag, ap, al = out.y

plt.plot(ts, wt, '--o', label='wt')
plt.plot(ts, wg, '--o', label='wg')
plt.plot(ts, wp, '--o', label='wp')
plt.plot(ts, wl, '--o', label='wl')

plt.plot(ts, ag, '--o', label='ag')
plt.plot(ts, ap, '--o', label='ap')
plt.plot(ts, al, '--o', label='al')


plt.legend()
# plt.yscale('log')