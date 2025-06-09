"""Max margin heuristics"""

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

n_depth = 10
n_dist = 1

v = 2 * n_depth + 4

M1 = np.zeros((v**2, v**2))
d = 2 * n_dist - 1

idx_mask = np.arange(-2*n_dist, 2*n_dist+1)
# idx_mask[(idx_mask < -1) & (idx_mask % 2 == 1)] = 0
print(idx_mask)

for beta in range(2 * n_dist, v - 2 * n_dist):
    bb = v * beta + beta
    # plt.axvline(v * beta)
    # M1[bb, (v*beta):((beta+1)*v)] = -1 / d
    M1[bb, idx_mask + bb] = -1 / d
    M1[bb, v * beta + beta - 2] = 1

    for alpha in range(beta - 2 * n_dist, beta):
        if alpha < 0:
            continue

        ba = v * beta + alpha
        # M1[ba, (v*alpha):((alpha+1)*v)] = -1 / d
        M1[ba, idx_mask + v * alpha + beta] = -1 / d
        M1[ba, v*alpha + beta - 2] = 1

    for par in range(beta + 2, beta + 2*n_dist + 1, 2):
        if par >= v:
            continue

        bp = v * beta + par
        # M1[bp, (v*par):((par+1)*v)] = -1 / d
        M1[bp, idx_mask + v*par + beta] = -1 / d
        M1[bp, v*par + beta - 2] = 1

plt.imshow(M1, cmap='BrBG', vmin=-0.1, vmax=0.1)

print(M1[ba])

idx = np.argmax(M1[ba])
print(idx // v)
print(idx % v)

# <codecell>
plt.imshow(M1[10 * v + 8].reshape(v, v))

# %%
M2 = np.zeros((v**2, v**2))
d = 2 * n_dist - 1

for beta in range(2 * n_dist, v - 2 * n_dist):
    # br = np.arange(v)
    br = idx_mask + beta
    bsb = v * beta + beta - 2

    M2[bsb, v*br + beta] = -1 / d
    M2[bsb, v*beta + beta] = 1

    for gamma in range(beta+1, beta + 2*n_dist + 1):
        if gamma >= v:
            continue

        bsg = v * beta + gamma - 2
        M2[bsg, v*br + beta] = -1 / d
        M2[bsg, v*gamma + beta] = 1

    for s in range(beta - 2*n_dist, beta, 2):
        if s < 0:
            continue

        bs2 = v * beta + s - 2
        M2[bs2, v*br + beta] = -1 / d
        M2[bs2, v*s + beta] = 1

plt.imshow(M2, cmap='BrBG', vmin=-0.1, vmax=0.1)

# <codecell>
plt.imshow(M2[v * 10 + 6].reshape(v, v))


# <codecell>
M = M1 @ M2
# M = M2 @ M1

val, vec = np.linalg.eig(M)
idxs = np.argsort(-val)

i = 0
ve = vec[:,idxs[i]]
print(val[idxs[i]])

ve = np.real(ve)

xs = np.arange(0, 16)
plt.plot(xs, xs)
plt.imshow(ve.reshape(v, v), cmap='BrBG', vmin=-0.5, vmax=0.5)
plt.colorbar()

# <codecell>
plt.plot(val[idxs])

# <codecell>
o = M2 @ ve[:,None]
plt.imshow(o.reshape(v, v))
plt.colorbar()


# <codecell>
val[idxs]
ve = np.sum(vec[:,idxs[:80]], axis=-1)
ve = np.real(ve)

xs = np.arange(0, 16)
plt.plot(xs, xs)
plt.imshow(ve.reshape(v, v), cmap='BrBG', vmin=-1, vmax=1)
plt.colorbar()


# <codecell>
M = np.zeros((2 * v**2, 2 * v**2))
M[:v**2,v**2:] = M2
M[v**2:, :v**2] = M1

val, vec = np.linalg.eig(M)
idxs = np.argsort(-val)

i = 9
ve = vec[:,idxs[i]]
print(val[idxs[i]])

ve = ve[:v**2]
ve = np.real(ve)

xs = np.arange(0, 16)
plt.plot(xs, xs)
plt.imshow(ve.reshape(v, v), cmap='BrBG', vmin=-0.5, vmax=0.5)
plt.colorbar()

# %%
val[idxs]
ve = np.sum(vec[:,idxs[:20]], axis=-1)
ve = np.real(ve[:v**2])

xs = np.arange(0, 16)
plt.plot(xs, xs)
plt.imshow(ve.reshape(v, v), cmap='BrBG', vmin=-1, vmax=1)
plt.colorbar()
