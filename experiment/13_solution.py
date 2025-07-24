"""Normative solution"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from scipy.stats import norm
import seaborn as sns

import sys
sys.path.append('../')
from common import *
from train import *
from model.transformer import *
from task.graph import *

def relu(x):
    return x * (x > 0)


depth = 10
n_hidden = 512
batch_size = 128

cot = False
ttr = True
nouveau = True
force_bin_label = True
n_arms = 10
n_hop = 10
test_n_hop = 7

n_vocab = n_arms * depth + 1 + StarfishTask.offset

train_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)
test_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(test_n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)

# %%
emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)
a = np.ones((n_hidden, 1))
# a[(n_hidden//2):] = -1

z = np.random.randn(n_vocab, n_hidden)
for i in range(n_vocab):
    z[i] = z[i % n_arms]

W = emb.T @ z

# <codecell>
# xs_toks = np.array([[10, 30]])
train_task.batch_size = 1024
xs_toks, ys = next(train_task)

xs = emb[xs_toks].sum(axis=1)
out = relu(xs @ W) @ a

pos = out[ys == 1]
neg = out[ys == 0]

plt.hist(pos, bins=15, alpha=0.8)
plt.hist(neg, bins=15, alpha=0.8)

pos_mean = np.mean(pos)
pos_var = np.var(pos)
neg_mean = np.mean(neg)
neg_var = np.var(neg)

print(pos_var)
print(neg_var)

diff = pos_mean * pos_var - neg_mean * neg_var
det = np.sqrt(pos_var * neg_var * ((pos_mean - neg_mean)**2 + 2 * (pos_var - neg_var) * np.log(np.sqrt(pos_var / neg_var))))
denom = pos_var - neg_var

c = (diff - det) / denom
plt.axvline(c, color='red')

acc = (np.mean(pos > c) + np.mean(neg < c)) / 2
acc


# <codecell>
def comp_acc(n_arms, depth, n_hop, n_hidden):
    n_vocab = n_arms * depth + 1 + StarfishTask.offset
    emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)
    a = np.ones((n_hidden, 1))

    z = np.random.randn(n_vocab, n_hidden)
    for i in range(n_vocab):
        z[i] = z[i % n_arms]

    W = emb.T @ z

    task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)

    xs_toks, ys = next(task)

    xs = emb[xs_toks].sum(axis=1)
    out = relu(xs @ W) @ a

    pos = out[ys == 1]
    neg = out[ys == 0]

    pos_mean = np.mean(pos)
    pos_var = np.var(pos)
    neg_mean = np.mean(neg)
    neg_var = np.var(neg)

    diff = pos_mean * pos_var - neg_mean * neg_var
    det = np.sqrt(pos_var * neg_var * ((pos_mean - neg_mean)**2 + 2 * (pos_var - neg_var) * np.log(np.sqrt(pos_var / neg_var))))
    denom = pos_var - neg_var

    c = (diff - det) / denom

    acc = (np.mean(pos > c) + np.mean(neg < c)) / 2
    return acc

# <codecell>
ax1 = (2**np.linspace(3, 9, num=10)).astype(int) * 2
ax2 = (2**np.linspace(3, 9, num=10)).astype(int) * 2

with jax.disable_jit():
    Z = []
    for h, d in tqdm(itertools.product(ax1, ax2)):
        val = np.mean([comp_acc(10, d, int(0.5 * d), h) for _ in range(5)])
        Z.append(val)

    Z = np.array(Z).reshape(10, 10)

# comp_acc(500, 10, 5, 1024)

# %%
plt.imshow(Z)
plt.colorbar()

xs = np.linspace(0, 4)
plt.plot(xs, 2 + 1.5 * xs, color='red')

# <codecell>
### AR MODEL
depth = 10
n_hidden = 500
batch_size = 128

cot = True
ttr = True
nouveau = True
force_bin_label = True
n_arms = 1000
n_hop = 5
test_n_hop = 7

n_vocab = n_arms * depth + 1 + StarfishTask.offset

task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)
# next(task)

# %%
emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)
emb[0] = 0

a = np.ones((n_hidden, 1))

z = np.random.randn(n_vocab, n_hidden)
for i in range(n_vocab):
    z[i] = z[i % n_arms]

W = emb.T @ z

task.batch_size = 1024

xs_toks, ys = next(task)
lens = (xs_toks != 0).sum(axis=1, keepdims=True)
xs = emb[xs_toks].sum(axis=1) / lens

out = relu(xs @ W) @ a

pos = out[ys == 1].flatten()
neg = out[ys == 0].flatten()

# rand_idxs1 = np.random.choice(len(pos), size=len(pos), replace=False)
# rand_idxs2 = np.random.choice(len(pos), size=len(pos), replace=False)
# print(np.mean(pos[rand_idxs1] > neg[rand_idxs2]))
print(np.mean(pos > neg))

plt.hist(pos, bins=15, alpha=0.8)
plt.hist(neg, bins=15, alpha=0.8)

# pos_mean = np.mean(pos)
# pos_var = np.var(pos)
# neg_mean = np.mean(neg)
# neg_var = np.var(neg)

# print(pos_var)
# print(neg_var)

# diff = pos_mean * pos_var - neg_mean * neg_var
# det = np.sqrt(pos_var * neg_var * ((pos_mean - neg_mean)**2 + 2 * (pos_var - neg_var) * np.log(np.sqrt(pos_var / neg_var))))
# denom = pos_var - neg_var

# c = (diff - det) / denom
# plt.axvline(c, color='red')

# acc = (np.mean(pos > c) + np.mean(neg < c)) / 2
# acc


# <codecell>
def comp_acc(n_arms, depth, n_hop, n_hidden):
    n_vocab = n_arms * depth + 1 + StarfishTask.offset
    emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)
    emb[0] = 0
    a = np.ones((n_hidden, 1))

    z = np.random.randn(n_vocab, n_hidden)
    for i in range(n_vocab):
        z[i] = z[i % n_arms]

    W = emb.T @ z

    task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=1024, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)

    xs_toks, ys = next(task)
    lens = (xs_toks != 0).sum(axis=1, keepdims=True)
    xs = emb[xs_toks].sum(axis=1) / lens

    out = relu(xs @ W) @ a

    pos = out[ys == 1]
    neg = out[ys == 0]

    return np.mean(pos > neg)

    # pos_mean = np.mean(pos)
    # pos_var = np.var(pos)
    # neg_mean = np.mean(neg)
    # neg_var = np.var(neg)

    # diff = pos_mean * pos_var - neg_mean * neg_var
    # det = np.sqrt(pos_var * neg_var * ((pos_mean - neg_mean)**2 + 2 * (pos_var - neg_var) * np.log(np.sqrt(pos_var / neg_var))))
    # denom = pos_var - neg_var

    # c = (diff - det) / denom

    # acc = (np.mean(pos > c) + np.mean(neg < c)) / 2
    # return acc

# <codecell>
ax1 = (2**np.linspace(3, 9, num=10)).astype(int) * 2
ax2 = (2**np.linspace(3, 9, num=10)).astype(int) * 2

with jax.disable_jit():
    Z = []
    for h, b in tqdm(itertools.product(ax1, ax2)):
        # val = np.mean([comp_acc(10, d, int(0.5 * d), h) for _ in range(5)])
        val = np.mean([comp_acc(b, 10, 5, h) for _ in range(5)])
        Z.append(val)

    Z = np.array(Z).reshape(10, 10)

# comp_acc(500, 10, 5, 1024)

# %%
plt.imshow(Z)
plt.colorbar()

xs = np.linspace(0, 10)
plt.plot(xs, 3 + 0.5 * xs, color='red')
# plt.plot(xs, 3 + 2 * xs, color='red')

# <codecell>
### Measuring distributions
res = []
res_neg = []

n_arms = 20
n_hidden = 100
depth = 10
L = depth // 2
n_hop = depth // 2

n_vocab = n_arms * depth + 1 + StarfishTask.offset

task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1, n_hop), batch_size=1024, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)

# task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1, n_hop), batch_size=1024, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)
# xs, ys = next(task)
# print(xs[:3])

log = []
for _ in tqdm(range(1000)):
    emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)
    z = np.random.randn(n_vocab, n_hidden)
    for i in range(n_vocab):
        z[i] = z[i % n_arms]

    # batch, _ = next(task)
    # lens = (batch != 0).sum(axis=1, keepdims=True)
    # toks = batch[0]
    # lens = lens[0]
    # toks_neg = batch[-1]
    # lens_neg = lens[-1]

    # toks = np.append(np.arange(L - 1) * n_arms, 0)
    # toks_neg = np.append(np.arange(L - 1) * n_arms, 1)
    # lens = L
    # lens_neg = L

    L1, L2 = np.random.randint(3, depth, size=2)
    toks = np.append(np.arange(L1 - 1) * n_arms, 0)
    toks_neg = np.append(np.arange(L2 - 1) * n_arms, 1)
    lens = L1
    lens_neg = L2

    # toks = [1]; L = 1

    W = emb.T @ z
    xs = emb[toks].sum(axis=0, keepdims=True) / lens
    xs_neg = emb[toks_neg].sum(axis=0, keepdims=True) / lens_neg

    # val = xs @ W
    # res.append(val[0])

    val = relu(xs @ W)
    val_neg = relu(xs_neg @ W)
    res.append(np.sum(val).item())
    res_neg.append(np.sum(val_neg).item())

# # <codecell>
# log = np.array(log).squeeze()
# c = np.cov(log.T)

# print(np.sum(c) - np.sum(np.diag(c)))

# # plt.plot(np.cumsum(c[0,1:]))
# plt.plot(c[0])


# %%
def red_fac(L, is_same):
    # b = np.ceil(L / n_arms)
    # return (2 * L - (n_arms * b)) * (b - 1)
    if is_same:
        return L**2 - L - 2
    else:
        return L**2 - 3 * L

def mu(L, is_same):
    rf = red_fac(L, is_same)
    var = ((L + 2 * is_same) / L**2) * (1 + (n_vocab + 1) / n_hidden) + (1 + 1 / n_hidden) * rf / L**2
    loc = (n_hidden / 2) * np.sqrt(2 * var / np.pi)
    return loc

def sig2(L, is_same):
    rf = red_fac(L, is_same)
    var = ((L + 2 * is_same) / L**2) * (1 + (n_vocab + 1) / n_hidden) + (1 + 1 / n_hidden) * rf / L**2
    sig2 = (n_hidden / 2) * ((1 - (1 / np.pi)) * var + 1) # NOTE: variance slightly off, corrected from o3
    return sig2

plt.hist(res, density=True, bins=20)
plt.hist(res_neg, density=True, bins=20, alpha=0.5)

low = np.min(res_neg)
hi = np.max(res)
xs = np.linspace(low, hi)

all_mus = [mu(l, True) for l in range(3, depth)]
loc = np.mean(all_mus)

all_vars = [sig2(l, True) for l in range(3, depth)]
scale = np.sqrt(np.mean(all_vars) + np.var(all_mus))

all_mus_neg = [mu(l, False) for l in range(3, depth)]
loc_neg = np.mean(all_mus_neg)

all_vars_neg = [sig2(l, False) for l in range(3, depth)]
scale_neg = np.sqrt(np.mean(all_vars_neg) + np.var(all_mus_neg))

plt.plot(xs, norm.pdf(xs, loc=loc, scale=scale))
plt.plot(xs, norm.pdf(xs, loc=loc_neg, scale=scale_neg))


print('pred loc', loc)
print('true loc', np.mean(res))
print('pred scl', scale)
print('true scl', np.std(res))

res = np.array(res)
res_neg = np.array(res_neg)
i1, i2 = np.random.choice(len(res), size=(2, len(res)))
print('raw', np.mean(res[i1] > res_neg[i2]))

pos_mean = np.mean(res)
pos_std = np.std(res)
plt.plot(xs, norm.pdf(xs, loc=pos_mean, scale=pos_std), color='pink')

neg_mean = np.mean(res_neg)
neg_std = np.std(res_neg)
plt.plot(xs, norm.pdf(xs, loc=neg_mean, scale=neg_std), color='pink')

# s1 = norm.rvs(size=1000, loc=pos_mean, scale=pos_std)
# s2 = norm.rvs(size=1000, loc=neg_mean, scale=neg_std)
s1 = norm.rvs(size=1000, loc=loc, scale=scale)
s2 = norm.rvs(size=1000, loc=loc_neg, scale=scale_neg)

print('est', np.mean(s1 > s2))


# <codecell>
red_fac(L, True) - red_fac(L, False)

# <codecell>
n_arms = 10000
n_hidden = 100
depth = 10
n_hop = depth // 2

n_vocab = n_arms * depth + 1 + StarfishTask.offset

task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1, n_hop), batch_size=1024, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)

xs, ys = next(task)
lens = np.sum(xs != 0, axis=1)

plt.hist(lens)

# <codecell>
def comp_acc(n_arms, depth, n_hop, n_hidden):
    # res = []
    # res_neg = []

    # n_arms = 20
    # n_hidden = 100
    # depth = 10

    n_vocab = n_arms * depth + 1 + StarfishTask.offset

    # for _ in range(1000):
    #     emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)
    #     z = np.random.randn(n_vocab, n_hidden)
    #     for i in range(n_vocab):
    #         z[i] = z[i % n_arms]

    #     L1, L2 = np.random.randint(3, depth, size=2)
    #     toks = np.append(np.arange(L1 - 1) * n_arms, 0)
    #     toks_neg = np.append(np.arange(L2 - 1) * n_arms, 1)
    #     lens = L1
    #     lens_neg = L2

    #     W = emb.T @ z
    #     xs = emb[toks].sum(axis=0, keepdims=True) / lens
    #     xs_neg = emb[toks_neg].sum(axis=0, keepdims=True) / lens_neg

    #     val = relu(xs @ W)
    #     val_neg = relu(xs_neg @ W)
    #     res.append(np.sum(val).item())
    #     res_neg.append(np.sum(val_neg).item())


    def red_fac(L, is_same):
        if is_same:
            return L**2 - L - 2
        else:
            return L**2 - 3 * L

    def mu(L, is_same):
        rf = red_fac(L, is_same)
        var = ((L + 2 * is_same) / L**2) * (1 + (n_vocab + 1) / n_hidden) + (1 + 1 / n_hidden) * rf / L**2
        loc = (n_hidden / 2) * np.sqrt(2 * var / np.pi)
        return loc

    def sig2(L, is_same):
        rf = red_fac(L, is_same)
        var = ((L + 2 * is_same) / L**2) * (1 + (n_vocab + 1) / n_hidden) + (1 + 1 / n_hidden) * rf / L**2
        sig2 = (n_hidden / 2) * ((1 - (1 / np.pi)) * var + 1) # NOTE: variance slightly off, corrected from o3
        return sig2

    all_mus = [mu(l, True) for l in range(3, depth)]
    loc = np.mean(all_mus)

    all_vars = [sig2(l, True) for l in range(3, depth)]
    scale = np.sqrt(np.mean(all_vars) + np.var(all_mus))

    all_mus_neg = [mu(l, False) for l in range(3, depth)]
    loc_neg = np.mean(all_mus_neg)

    all_vars_neg = [sig2(l, False) for l in range(3, depth)]
    scale_neg = np.sqrt(np.mean(all_vars_neg) + np.var(all_mus_neg))

    # res = np.array(res)
    # res_neg = np.array(res_neg)
    # i1, i2 = np.random.choice(len(res), size=(2, len(res)))

    # pos_mean = np.mean(res)
    # pos_std = np.std(res)

    # neg_mean = np.mean(res_neg)
    # neg_std = np.std(res_neg)

    s1 = norm.rvs(size=1000, loc=loc, scale=scale)
    s2 = norm.rvs(size=1000, loc=loc_neg, scale=scale_neg)

    return np.mean(s1 > s2)

comp_acc(20, 10, 0, 100)

# <codecell>
res = 40

ax1 = (2**np.linspace(3, 9, num=res)).astype(int) * 2
ax2 = (2**np.linspace(3, 9, num=res)).astype(int) * 2

with jax.disable_jit():
    Z = []
    for h, b in tqdm(itertools.product(ax1, ax2)):
        # val = np.mean([comp_acc(10, d, int(0.5 * d), h) for _ in range(5)])
        val = np.mean([comp_acc(b, 10, 5, h) for _ in range(5)])
        Z.append(val)

    Z = np.array(Z).reshape(res, res)

# comp_acc(500, 10, 5, 1024)

# %%
ax = plt.gca()
idxs = list(range(len(ax1)))[::4]

plt.imshow(Z.T, vmin=0.5, vmax=0.95, origin='lower')
ax.set_xticks(idxs)
ax.set_xticklabels(ax1[idxs])
ax.set_yticks(idxs)
ax.set_yticklabels(ax1[idxs])
plt.colorbar()

xs = np.linspace(0, 35)
plt.plot(xs, 3 + 1 * xs, color='red')
plt.plot(xs, -20 + 2 * xs, color='red')
plt.ylim(0, 39)
# plt.plot(xs, 5 + 1 * xs, color='red')
# plt.plot(xs, 3 + 2 * xs, color='red')

plt.savefig('solution_debug.png')

# <codecell>
n_arms = 10
n_depth = 10
n_hop = 5
eps = 1e-8

task = StarfishTask(depth=n_depth, n_arms=n_arms, samp_dist=(1, n_hop), batch_size=64)
next(task)

# <codecell>
# TODO: how does competition look in the AR case?
z = np.random.randn(n_arms * n_depth) * eps

log = []
for _ in tqdm(range(1000)):
    xs, ys = next(task)
    batch_upd = np.zeros(z.shape)

    for (s1, s2), y in zip(xs, ys):
        if z[s1] + z[s2] > 0:
            if y == 1:
                batch_upd[s1] += 1
                batch_upd[s2] += 1
            else:
                batch_upd[s1] -= 1
                batch_upd[s2] -= 1
    
    z += batch_upd
    log.append(np.copy(z))

log = np.array(log)
# <codecell>
# plt.plot(np.sign(log[4]))
plt.plot(log[-1])
plt.plot(log[2])

# <codecell>
for i in range(n_arms):
    print(f'{i}: {np.mean(z[i::n_arms] > 0)}')

