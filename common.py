"""Common utilities"""

from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.special import logsumexp
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')

def set_theme():
    sns.set_theme(style='ticks', font_scale=1.25, rc={
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.figsize': (4, 3)
    })


def new_seed():
    return np.random.randint(0, np.iinfo(np.int32).max)


def t(xs):
    return np.swapaxes(xs, -2, -1)


class Finite:
    def __init__(self, task, data_size, seed=None) -> None:
        self.task = task
        self.data_size = data_size
        self.batch_size = self.task.batch_size
        self.task.batch_size = data_size   # dirty trick (we're all adults here)

        self.data = next(self.task)
        del self.task                      # task is consumed

        self.rng = np.random.default_rng(seed)
    
    def __next__(self):
        idxs = self.rng.choice(self.data_size, self.batch_size, replace=True)
        return self.data[0][idxs], self.data[1][idxs]

    def __iter__(self):
        return self


def split_cases(all_cases, run_split):
    run_idx = sys.argv[1]
    try:
        run_idx = int(run_idx) % run_split
    except ValueError:
        print(f'warn: unable to parse index {run_idx}, setting run_idx=0')
        run_idx = 0

    print('RUN IDX', run_idx)
    all_cases = np.array_split(all_cases, run_split)[run_idx]
    return list(all_cases)


def summon_dir(path: str, clear_if_exists=False):
    new_dir = Path(path)
    if new_dir.exists() and clear_if_exists:
        shutil.rmtree(new_dir)

    new_dir.mkdir(parents=True)
    return new_dir


def collate_dfs(df_dir, show_progress=False, concat=True):
    pkl_path = Path(df_dir)
    dfs = []

    it = pkl_path.iterdir()
    if show_progress:
        it = tqdm(list(it))
    
    for f in it:
        if f.suffix == '.pkl':
            try:
                df = pd.read_pickle(f)
                dfs.append(df)
            except:
                print(f'warn: fail to read {f}')

    if concat:
        dfs = pd.concat(dfs)

    return dfs


def uninterleave(interl_xs):
    xs = interl_xs[:,0::2]
    ys = interl_xs[:,1::2,[0]]
    xs, x_q = xs[:,:-1], xs[:,[-1]]
    return xs, ys, x_q


def unpack(pack_xs):
    xs = pack_xs[:,:-1,:-1]
    ys = pack_xs[:,:-1,[-1]]
    x_q = pack_xs[:,[-1],:-1]
    return xs, ys, x_q


def estimate_dmmse(task, xs, ys, x_q, sig2=0.05):
    '''
    xs: N x P x D
    ys: N x P x 1
    x_q: N x 1 x D
    ws: F x D
    '''
    ws = task.ws
    
    weights = np.exp(-(1 / (2 * sig2)) * np.sum((ys - xs @ ws.T)**2, axis=1))  # N x F
    probs = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-32)
    w_dmmse = np.expand_dims(probs, axis=-1) * ws  # N x F x D
    w_dmmse = np.sum(w_dmmse, axis=1, keepdims=True)  # N x 1 x D
    return (x_q @ t(w_dmmse)).squeeze()


def estimate_ridge(task, xs, ys, x_q, sig2=0.05):
    n_dims = xs.shape[-1]
    w_ridge = np.linalg.pinv(t(xs) @ xs + n_dims * sig2 * np.identity(n_dims)) @ t(xs) @ ys
    return (x_q @ w_ridge).squeeze()

# NOTE: remove once added to optax upstream
def scale_by_sign():
    def init_fn(params):  # no access to init_empty_state
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree.map(lambda g: jnp.sign(g), updates)
        return updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)


def sign_sgd(learning_rate):
    return optax.chain(
        scale_by_sign(),
        optax.scale_by_learning_rate(learning_rate)
    )


def log_gen_like_same(z1, z2, d, sig2):
    t1 = -d * (np.log(2 * np.pi) - np.log(d))
    t2 = (-d/2) * np.log(sig2)
    t3 = (-d / 2) * np.log(2 + sig2)

    a1 = (1 + sig2) / (2 + sig2)
    a2 = (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)
    a3 = (2 / (2 + sig2))
    a4 = np.dot(z1, z2)

    t4 = - d/(2 * sig2)
    t5 = a1 * a2 - a3 * a4

    t_exp = t4 * t5
    log_sol = t1 + t2 + t3 + t_exp
    return log_sol


def log_gen_like_diff(z1, z2, d, sig2):
    t1 = -d * (np.log(2 * np.pi) - np.log(d))
    t2 = -d * np.log(1 + sig2)

    a1 = 1 / (1 + sig2)
    a2 = (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)

    t4 = - d/2
    t5 = a1 * a2

    t_exp = t4 * t5
    log_sol = t1 + t2 + t_exp
    return log_sol


def log_mem_like_same(z1, z2, d, sig2, ss):
    L = len(ss)
    t1 = d * (np.log(d) -  np.log(2 * np.pi * sig2))

    a1 = (-d / (2 * sig2))
    a2 = (1/2) * (np.linalg.norm(z1)**2 + np.linalg.norm(z2)**2)
    a3 = np.dot(z1, z2)

    p1 = t1 + a1 * (a2 - a3)

    d1 = -d / sig2
    d2 = ss - (z1 + z2) / 2
    d_exp = d1 * np.linalg.norm(d2, axis=1)**2

    log_sol = p1 + logsumexp(d_exp) - np.log(L)
    return log_sol


def log_mem_like_diff(z1, z2, d, sig2, ss):
    L = len(ss)
    t1 = d * (np.log(d) -  np.log(2 * np.pi * sig2))

    a1 = -d / (2 * sig2)
    e1s = np.reshape(a1 * np.linalg.norm(ss - z1, axis=1)**2, (-1, 1))
    e2s = np.reshape(a1 * np.linalg.norm(ss - z2, axis=1)**2, (1, -1))

    prods = e1s + e2s
    np.fill_diagonal(prods, -np.inf)

    t2 = -np.log(L * (L - 1))
    log_sol = t1 + t2 + logsumexp(prods)
    return log_sol