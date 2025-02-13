"""Graph connectedness algorithms"""

# <codecell>
import functools
import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.append('../')
from common import new_seed

def fast_binary_pow(a, n):
    for _ in range(n):
        a = _bin_path(a)
    return a

@jax.jit
def _bin_path(a):
    c = a @ a + a
    c = (c > 0).astype(int)
    return c

def samp_children(key, nodes, samp_dist):
    n_min = nodes * 2**samp_dist
    n_max = nodes * 2**samp_dist + 2**samp_dist
    children = jax.random.randint(key, minval=n_min, maxval=n_max, shape=len(n_min))
    return children


@functools.partial(jax.jit, static_argnums=(3,))
def samp_nodes_on_branch(key, depth, samp_dist, batch_size):
    source = key

    key, source = jax.random.split(source)
    upr = 2**(depth - samp_dist)
    nodes = jax.random.randint(key, minval=1, maxval=upr, shape=batch_size)

    key, source = jax.random.split(source)
    children = samp_children(key, nodes, samp_dist)

    xs = jnp.stack((nodes, children), axis=1)
    ys = jnp.ones(batch_size)
    return xs, ys


@functools.partial(jax.jit, static_argnums=(1, 2, 3,))
def samp_nodes_off_branch(key, depth, samp_dist, batch_size):
    source = key

    key, source = jax.random.split(source)
    l = depth - samp_dist
    probs = 2**jnp.arange(l)
    probs = probs / np.sum(probs)
    layers = jax.random.choice(key, l, p=probs, shape=(batch_size,))

    key, source = jax.random.split(source)
    l_low = 2**layers
    l_high = 2**(layers + 1)
    true, shadow = jax.random.randint(key, minval=l_low, maxval=l_high, shape=(2, batch_size))

    ys = (true == shadow).astype(int)

    key, source = jax.random.split(source)
    children = samp_children(key, shadow, samp_dist)
    xs = jnp.stack((true, children), axis=1)
    
    return xs, ys


class GraphTiTask:
    def __init__(self, n_nodes, n_dims=None, p_connect=None, prob_thresh=(0.1, 0.7), samp_adj=True, seed=None, batch_size=128) -> None:
        assert batch_size % 2 == 0, 'batch size must be even'

        self.n_nodes = n_nodes
        self.n_dims = n_dims
        self.p_connect = p_connect
        self.prob_thresh = prob_thresh
        self.samp_adj = samp_adj
        self.batch_size = batch_size

        if self.p_connect is None:
            self.p_connect = 1 / n_nodes

        self.rng = np.random.default_rng(seed)

        do_cont = False
        while not do_cont:
            self.adj = self.rng.binomial(n=1, p=self.p_connect, size=self.n_nodes**2).reshape(self.n_nodes, self.n_nodes)
            self.cnx = fast_binary_pow(self.adj, n_nodes)

            p = np.mean(self.cnx)
            if prob_thresh[0] < p < prob_thresh[1]:
                do_cont = True
            else:
                print(f'warn: rejecting graph with p={p:.2f}')

        ## Enforced binary tree
        # self.adj = np.zeros((self.n_nodes, self.n_nodes))
        # idx = 2
        # while idx < self.n_nodes:
        #     row = idx // 2 - 1
        #     self.adj[row,idx-1] += 1
        #     self.adj[row,idx] += 1
        #     idx += 2

        # self.cnx = fast_binary_pow(self.adj, n_nodes)
        
        self.pat = 2 * self.cnx - self.adj
        idx = np.arange(self.n_nodes**2)
        neg_sel = self.pat.ravel() == 0
        pos_sel_adj = self.pat.ravel() == 1
        pos_sel_nadj = self.pat.ravel() == 2

        self.pos_idx_adj = idx[pos_sel_adj]
        self.pos_idx_nadj = idx[pos_sel_nadj]
        self.neg_idx = idx[neg_sel]

        if n_dims is not None:
            self.emb_mat = self.rng.standard_normal((self.n_nodes, self.n_dims)) / np.sqrt(self.n_dims)

        self.rng = np.random.default_rng(None)

    def __next__(self):
        pos_idx = self.pos_idx_adj if self.samp_adj else self.pos_idx_nadj
        xs_pos = self.rng.choice(pos_idx, replace=True, size=self.batch_size // 2)
        xs_neg = self.rng.choice(self.neg_idx, replace=True, size=self.batch_size // 2)

        xs = np.concat((xs_pos, xs_neg))
        ys = np.zeros(self.batch_size)
        ys[:self.batch_size//2] = 1

        shuffle_idx = self.rng.choice(self.batch_size, size=self.batch_size, replace=False)
        xs = xs[shuffle_idx]
        ys = ys[shuffle_idx]

        xs = np.stack((xs // self.n_nodes, xs % self.n_nodes), axis=1)

        if self.n_dims is not None:
            xs = self.emb_mat[xs]

        return xs, ys

    def __iter__(self):
        return self


class BinaryTreeTiTask:
    def __init__(self, rev, depth, samp_dist=1, on_branch=True, fill_gaps=True, batch_size=128) -> None:
        self.rev = rev
        self.depth = depth
        self.samp_dist = samp_dist
        self.on_branch = on_branch
        self.fill_gaps = fill_gaps
        self.batch_size = batch_size

        self.seed = new_seed()
        self.source = jax.random.key(self.seed)

    
    def __next__(self):
        key, self.source = jax.random.split(self.source)

        if self.on_branch:
            xs, ys = samp_nodes_on_branch(key, self.depth, self.samp_dist, self.batch_size)
        else:
            xs, ys = samp_nodes_off_branch(key, self.depth, self.samp_dist, self.batch_size)

            if self.fill_gaps:
                while np.sum(ys == 0) < self.batch_size:
                    print('warn: insufficient examples, resampling')
                    xs_add, ys_add = samp_nodes_off_branch(key, self.depth, self.samp_dist, self.batch_size)
                    xs = np.concatenate((xs, xs_add))
                    ys = np.concatenate((ys, ys_add))
                
                idxs = np.argsort(ys)
                xs = xs[idxs[:self.batch_size]]
                ys = ys[idxs[:self.batch_size]]

        if self.rev:
            xs = xs[:,::-1]
            ys = np.zeros(ys.shape)
        
        return xs, ys


    def __iter__(self):
        return self


class Chain:
    def __init__(self, tasks) -> None:
        self.tasks = tasks
        self.batch_size = self.tasks[0].batch_size
    

    def __next__(self):
        exs = [next(task) for task in self.tasks]
        xs, ys = zip(*exs)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)

        idx = np.random.choice(len(xs), size=len(xs), replace=False)
        print(idx)
        idx = idx[:self.batch_size]

        return xs[idx], ys[idx]

    
    def __iter__(self):
        return self


# task = Chain([
#     BinaryTreeTiTask(rev=True, depth=5, samp_dist=1, batch_size=10, on_branch=True),
#     BinaryTreeTiTask(rev=False, depth=5, samp_dist=1, batch_size=10, on_branch=True)
# ])

# xs, ys = next(task)

# print(xs)
# print(ys)

# %%
