"""Graph connectedness algorithms"""

# <codecell>
from collections.abc import Iterable
import functools

import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.append('../')
from common import new_seed
from train import extract_pred

def fast_binary_pow(a, n):
    for _ in range(n):
        a = _bin_path(a)
    return a

@jax.jit
def _bin_path(a):
    c = a @ a + a
    c = (c > 0).astype(int)
    return c


@functools.partial(jax.jit, static_argnums=(2, 3))
def samp_nodes_on_branch(key, depth, samp_dist, batch_size):
    source = key

    key, source = jax.random.split(source)
    samp_dist = _split_samp_dist(key, samp_dist, batch_size)

    key, source = jax.random.split(source)
    upr = 2**(depth - samp_dist)
    nodes = jax.random.randint(key, minval=1, maxval=upr, shape=batch_size)

    key, source = jax.random.split(source)
    children = _samp_children(key, nodes, samp_dist)

    xs = jnp.stack((nodes, children), axis=1)
    ys = jnp.ones(batch_size)
    return xs, ys


@functools.partial(jax.jit, static_argnums=(2, 3))
def samp_nodes_off_branch(key, depth, samp_dist, batch_size):
    source = key

    key, source = jax.random.split(source)
    samp_dist = _split_samp_dist(key, samp_dist, batch_size)

    key, source = jax.random.split(source)
    upr = 2**(depth - samp_dist)
    nodes = jax.random.randint(key, minval=1, maxval=upr, shape=batch_size)
    layers = jnp.log2(nodes).astype(int)

    key, source = jax.random.split(source)
    l_low = 2**layers
    l_high = 2**(layers + 1)
    true, shadow = jax.random.randint(key, minval=l_low, maxval=l_high, shape=(2, batch_size))

    ys = (true == shadow).astype(int)

    key, source = jax.random.split(source)
    children = _samp_children(key, shadow, samp_dist)
    xs = jnp.stack((true, children), axis=1)
    
    return xs, ys


def _split_samp_dist(key, samp_dist, batch_size):
    if isinstance(samp_dist, Iterable):
        s_low, s_high = samp_dist
        samp_dist = jax.random.randint(key, minval=s_low, maxval=s_high+1, shape=batch_size)

    return samp_dist


def _samp_children(key, nodes, samp_dist):
    n_min = nodes * 2**samp_dist
    n_max = nodes * 2**samp_dist + 2**samp_dist
    children = jax.random.randint(key, minval=n_min, maxval=n_max, shape=len(n_min))
    return children


def _pred_frac_on(depth):
    return 1.3 * np.exp(-0.45 * (depth - 1.5))


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
    pad_idx = 0
    no_idx = 1
    yes_idx = 2
    sep_idx = 3
    offset = 3

    def __init__(self, depth, order=None, samp_dist=1, 
                 on_branch=True, 
                 shuffle=False, 
                 cot=False, trace_to_start=True, unwrap=False, 
                 rl_prompt=False, n_thought=None,
                 use_sep=True, repeat_first=True,
                 apply_offset=True,
                 batch_size=128) -> None:

        if not on_branch:
            if isinstance(samp_dist, Iterable):
                dist = samp_dist[1]
            else:
                dist = samp_dist

            assert depth - dist > 1, 'impossible to generate off-branch examples'

        self.depth = depth
        self.order = order
        self.samp_dist = samp_dist
        self.on_branch = on_branch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cot = cot
        self.trace_to_start = trace_to_start
        self.rl_prompt = rl_prompt
        self.n_thought = n_thought
        self.unwrap = unwrap
        self.use_sep = use_sep
        self.repeat_first = repeat_first
        self.apply_offset = apply_offset

        self.pred_frac_on = _pred_frac_on(self.depth)
        self.off_branch_eff_batch_size = int(1.1 * self.batch_size / self.pred_frac_on)
        self.seed = new_seed()
        self.source = jax.random.key(self.seed)

        if self.n_thought is None:
            self.n_thought = 2 * depth

    
    def __next__(self):
        key, self.source = jax.random.split(self.source)

        if self.on_branch:
            xs, ys = samp_nodes_on_branch(key, self.depth, self.samp_dist, self.batch_size)
        else:
            xs, ys = samp_nodes_off_branch(key, self.depth, self.samp_dist, self.off_branch_eff_batch_size)
            keep_idx = (ys == 0)
            while np.sum(keep_idx) < self.batch_size:
                # print('warn: insufficient examples, resampling')
                xs_add, ys_add = samp_nodes_off_branch(key, self.depth, self.samp_dist, self.off_branch_eff_batch_size)
                xs = jnp.concatenate((xs, xs_add))
                ys = jnp.concatenate((ys, ys_add))
                keep_idx = (ys == 0)

            xs = xs[keep_idx][:self.batch_size]
            ys = ys[keep_idx][:self.batch_size]

        if self.order == 'rev':
            xs = xs[:,::-1]
            ys = np.zeros(ys.shape)
        elif self.order == 'split':
            split = self.batch_size // 2
            xs = xs.at[:split].set(xs[:split,::-1])
            ys = ys.at[:split].set(0)
        
        
        if self.cot:
            if self.apply_offset:
                xs = xs + BinaryTreeTiTask.offset
            
            xs = _add_chain(xs, self.depth, self.batch_size, self.use_sep, self.repeat_first, trace_to_start=self.trace_to_start)

            if self.unwrap:
                xs, ys = _unwrap(xs, self.depth, self.use_sep)
                keep_idx = ys != 0
                xs = xs[keep_idx]
                ys = ys[keep_idx]
            else:
                ys = xs[:,1:]
                ys = ys.at[:,:2].set(0)   # mask prompt
                xs = xs[:,:-1]
        
        elif self.rl_prompt:
            if self.apply_offset:
                xs = xs + BinaryTreeTiTask.offset
            
            thought_toks = np.zeros((self.batch_size, self.n_thought))
            xs = jnp.concatenate((
                xs,
                BinaryTreeTiTask.sep_idx * np.ones((self.batch_size, 1 if self.use_sep else 0)),
                thought_toks
            ), axis=1)
            ys = jnp.where(ys == 1, BinaryTreeTiTask.yes_idx, BinaryTreeTiTask.no_idx)

        if self.shuffle:
            idx = np.random.choice(xs.shape[0], size=xs.shape[0], replace=False)
            xs = xs[idx]
            ys = ys[idx]

        return xs.astype(int), ys.astype(int)


    def __iter__(self):
        return self


@functools.partial(jax.jit, static_argnums=(1,2,3,4,5))
def _add_chain(xs, depth, batch_size, add_sep, repeat_first, trace_to_start=True):
    nodes = xs - BinaryTreeTiTask.offset

    start_fac = 0 if repeat_first else 1
    facs = 2**np.arange(start_fac, depth + 2)

    first, last = nodes.T

    chain = (last[:,None] // facs).astype(int)

    target_idx = jnp.sum(first[:,None] < chain, axis=1)
    target_val = chain[jnp.arange(batch_size),target_idx]

    resp = jnp.where(first == target_val, BinaryTreeTiTask.yes_idx, BinaryTreeTiTask.no_idx)

    if trace_to_start:
        keep_mask = chain > 0
        chain = chain + BinaryTreeTiTask.offset
        chain = chain * keep_mask
        keep_idx = jnp.sum(keep_mask, axis=1)
        chain = chain.at[jnp.arange(batch_size), keep_idx].set(resp)
    else:
        stop_mask = first[:,None] > chain
        stop_mask = stop_mask.at[jnp.arange(batch_size), target_idx].set(False)
        chain = chain + BinaryTreeTiTask.offset
        chain = chain * (~stop_mask)
        chain = chain.at[jnp.arange(batch_size), target_idx + 1].set(resp)

    xs = jnp.concatenate((
        xs, 
        BinaryTreeTiTask.sep_idx * jnp.ones((batch_size, 1 if add_sep else 0)),
        chain
    ), axis=-1)

    return xs


@functools.partial(jax.jit, static_argnums=(1, 2))
def _unwrap(xs, depth, add_sep):
    seq_len = depth + 3 + 1 * add_sep

    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    res_mask = jnp.tril(jnp.ones((seq_len, seq_len)), k=1) - mask

    # keep prompt
    mask = mask[2:]
    res_mask = res_mask[2:]

    out = xs[:,None,:] * mask
    res = (xs[:,None,:] * res_mask).sum(axis=-1)

    final_mask = res.astype(bool)

    out = jnp.concatenate(out * final_mask[...,None])
    res = jnp.concatenate(res)

    return out, res


class Chain:
    def __init__(self, *tasks, sub_samp=False, weights=None) -> None:
        self.tasks = tasks
        self.sub_samp = sub_samp
        self.batch_size = self.tasks[0].batch_size
        self.weights = weights

        if self.weights is None:
            self.weights = np.ones(len(self.tasks))

        self.probs = self.weights / np.sum(self.weights)

        for task, p in zip(self.tasks, self.probs):
            task.batch_size = int(task.batch_size * p)
    

    def __next__(self):
        exs = [next(task) for task in self.tasks]
        xs, ys = zip(*exs)

        xs = jnp.concatenate(xs)
        ys = jnp.concatenate(ys)

        if self.sub_samp:
            idx = np.random.choice(len(xs), size=len(xs), replace=False)
            xs = xs[idx]
            ys = ys[idx]

        return xs, ys

    
    def __iter__(self):
        return self


def bt_rl_loss(params, state, traj, rew):
    logits = state.apply_fn({'params': params}, traj)

    traj = traj[:,3:]
    logits = logits[:,2:-1]

    sel = jnp.take_along_axis(logits, traj[...,None], axis=-1).squeeze()
    norm = jax.scipy.special.logsumexp(logits, axis=-1)
    log_p = sel - norm

    J = rew[:,None] * log_p
    exp_J = jnp.mean(J)
    return -exp_J


def bt_rl_adv_loss(params, state, traj, rew):
    logits = state.apply_fn({'params': params}, traj)

    traj = traj[:,3:]
    logits = logits[:,2:-1]

    sel = jnp.take_along_axis(logits, traj[...,None], axis=-1).squeeze()
    norm = jax.scipy.special.logsumexp(logits, axis=-1)
    log_p = sel - norm

    rew = (rew - jnp.mean(rew)) / jnp.std(rew)

    J = rew[:,None] * log_p
    exp_J = jnp.mean(J)
    return -exp_J


def bt_rew_fn(traj, ys):
    pred = extract_pred(traj)
    reward = (pred == ys).astype(float)
    return reward


def bt_rew_fn_with_punish(traj, ys):
    pred = extract_pred(traj)
    reward = (pred == ys).astype(float)
    reward = jnp.where(reward == 0, -0.01, reward)
    return reward


class StarfishTask:
    pad_idx = 0
    no_idx = 1
    yes_idx = 2
    sep_idx = 3
    offset = 3

    def __init__(self, depth, n_arms=2, samp_dist=1, cot=False, rl_prompt=False, trace_to_start=False, nouveau=False, batch_size=128) -> None:
        self.depth = depth
        self.n_arms = n_arms
        self.samp_dist = samp_dist
        self.cot = cot
        self.rl_prompt = rl_prompt
        self.trace_to_start = trace_to_start
        self.nouveau = nouveau
        self.batch_size = batch_size

        self.seed = new_seed()
        self.source = jax.random.key(self.seed)
    
    def __next__(self):
        k1, k2, self.source = jax.random.split(self.source, num=3)
        n_on = self.batch_size // 2
        n_off = self.batch_size - n_on

        xs_on = _star_samp_on(k1, self.depth, self.n_arms, self.samp_dist, n_on)
        xs_off = _star_samp_off(k2, self.depth, self.n_arms, self.samp_dist, n_off)
        xs = jnp.concatenate((xs_on, xs_off), axis=0)
        ys = jnp.concat((jnp.ones(n_on), jnp.zeros(n_off)))

        if self.cot:
            xs = _star_add_chain(xs, self.depth, self.n_arms, self.batch_size, trace_to_start=self.trace_to_start, nouveau=self.nouveau)
            ys = xs[:,1:]
            ys = ys.at[:,:2].set(0)   # mask prompt
            xs = xs[:,:-1]
        elif self.rl_prompt:
            xs = xs + StarfishTask.offset
            
            thought_toks = np.zeros((self.batch_size, 2 * self.depth))
            xs = jnp.concatenate((
                xs,
                StarfishTask.sep_idx * np.ones((self.batch_size, 1)),
                thought_toks
            ), axis=1)
            ys = ys + 1

        return xs.astype(int), ys.astype(int)

    def __iter__(self):
        return self

@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _star_samp_on(key, depth, n_arms, samp_dist, batch_size):
    key, source = jax.random.split(key)
    samp_dist = _split_samp_dist(key, samp_dist, batch_size)
    upr = n_arms * (depth - samp_dist)

    key, source = jax.random.split(source)
    parents = jax.random.randint(key, minval=1, maxval=upr, shape=batch_size)
    children = parents + samp_dist * n_arms

    key, source = jax.random.split(source)
    origin_idx = parents == 1
    accept_origin = jax.random.bernoulli(key, shape=batch_size)
    children = children - (origin_idx & accept_origin).astype(int)

    return jnp.stack((parents, children), axis=1)


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _star_samp_off(key, depth, n_arms, samp_dist, batch_size):
    key, source = jax.random.split(key)
    samp_dist = _split_samp_dist(key, samp_dist, batch_size)
    upr = n_arms * (depth - samp_dist)

    key, source = jax.random.split(source)
    parents = jax.random.randint(key, minval=2, maxval=upr, shape=batch_size)
    children = parents + samp_dist * n_arms

    key, source = jax.random.split(source)
    pert = jax.random.randint(key, minval=1, maxval=n_arms-1, shape=batch_size)
    children += pert

    wrap_idx = (children % n_arms) == 0
    children -= n_arms * wrap_idx

    return jnp.stack((parents, children), axis=1)


@functools.partial(jax.jit, static_argnums=(1,2,3,4,5))
def _star_add_chain(xs, depth, n_arms, batch_size, trace_to_start=True, nouveau=False):
    parents, children = xs.T

    diffs = n_arms * np.arange(depth + 2)
    chain = children[:,None] - diffs
    chain = chain + (chain == 0)           # connect to root

    target_idx = jnp.sum(parents[:,None] < chain, axis=1)
    target_val = chain[jnp.arange(batch_size),target_idx]

    resp = jnp.where(parents == target_val, BinaryTreeTiTask.yes_idx, BinaryTreeTiTask.no_idx)

    if trace_to_start:
        keep_mask = chain > 0
        chain = chain + StarfishTask.offset
        chain = chain * keep_mask
        keep_idx = jnp.sum(chain > 0, axis=1)
        chain = chain.at[jnp.arange(batch_size), keep_idx].set(resp)
    else:
        stop_mask = parents[:,None] > chain
        stop_mask = stop_mask.at[jnp.arange(batch_size), target_idx].set(False)
        chain = chain + BinaryTreeTiTask.offset
        chain = chain * (~stop_mask)
        chain = chain.at[jnp.arange(batch_size), target_idx + 1].set(resp)

    if not nouveau:
        xs = jnp.concatenate((
            xs + StarfishTask.offset, 
            StarfishTask.sep_idx * jnp.ones((batch_size, 1)),
            chain
        ), axis=-1)
    else:
        xs = jnp.concatenate((
            # StarfishTask.sep_idx * jnp.ones((batch_size, 1)),
            xs + StarfishTask.offset, 
            chain[:,1:]
        ), axis=-1)

    return xs


class CircleTask:
    pad_idx = 0
    no_idx = 1
    yes_idx = 2
    sep_idx = 3
    offset = 3

    def __init__(self, depth, samp_dist=1, cot=False, rl_prompt=False, trace_to_start=False, batch_size=128) -> None:
        self.depth = depth
        self.samp_dist = samp_dist
        self.cot = cot
        self.rl_prompt = rl_prompt
        self.trace_to_start = trace_to_start
        self.batch_size = batch_size

        self.seed = new_seed()
        self.source = jax.random.key(self.seed)
    
    def __next__(self):
        k1, k2, self.source = jax.random.split(self.source, num=3)
        n_on = n_off = self.batch_size // 2

        xs_on = _circle_samp_on(k1, self.depth, self.samp_dist, n_on)
        xs_off = _circle_samp_off(k2, self.depth, self.samp_dist, n_off)
        xs = jnp.concatenate((xs_on, xs_off), axis=0)
        ys = jnp.concat((jnp.ones(n_on), jnp.zeros(n_off)))

        if self.cot:
            xs = _circle_add_chain(xs, self.depth, self.batch_size, trace_to_start=self.trace_to_start)
            ys = xs[:,1:]
            ys = ys.at[:,:2].set(0)   # mask prompt
            xs = xs[:,:-1]
        elif self.rl_prompt:
            xs = xs + CircleTask.offset
            
            thought_toks = np.zeros((self.batch_size, 2 * self.depth))
            xs = jnp.concatenate((
                xs,
                StarfishTask.sep_idx * np.ones((self.batch_size, 1)),
                thought_toks
            ), axis=1)
            ys = ys + 1

        return xs.astype(int), ys.astype(int)

    def __iter__(self):
        return self

@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _circle_samp_on(key, depth, samp_dist, batch_size):
    key, source = jax.random.split(key)
    samp_dist = _split_samp_dist(key, samp_dist, batch_size)

    key, source = jax.random.split(source)
    parents = jax.random.randint(key, minval=0, maxval=depth, shape=batch_size)
    children = (parents + samp_dist) % depth + 1
    parents += 1  # adjust for 1-indexing

    key, source = jax.random.split(source)
    ring = jax.random.bernoulli(key, shape=batch_size)
    parents += ring * depth
    children += ring * depth

    return jnp.stack((parents, children), axis=1)


# TODO: consider varying samp_dist for off-chain examples
@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _circle_samp_off(key, depth, samp_dist, batch_size):
    key, source = jax.random.split(key)
    samp_dist = _split_samp_dist(key, samp_dist, batch_size)

    key, source = jax.random.split(source)
    parents = jax.random.randint(key, minval=0, maxval=depth, shape=batch_size)
    children = (parents + samp_dist) % depth + 1
    parents += 1  # adjust for 1-indexing

    key, source = jax.random.split(source)
    ring = jax.random.bernoulli(key, shape=batch_size)
    parents += ring * depth
    children += (1 - ring) * depth

    return jnp.stack((parents, children), axis=1)




@functools.partial(jax.jit, static_argnums=(1,2,3))
def _circle_add_chain(xs, depth, batch_size, trace_to_start=True):
    parents, children = xs.T
    ring = (children - 1) // depth
    inv_ring = 1 - ring

    diffs = np.arange(depth)
    chain = (children[:,None] - diffs - 1) % depth + 1
    on_chain = chain + (ring * depth)[:,None]
    off_chain = chain + (inv_ring * depth)[:,None]

    resp_mask = (parents[:,None] == on_chain) | (parents[:,None] == off_chain)
    keep_mask = (1 - jnp.cumsum(resp_mask, axis=1)).astype(bool) | resp_mask

    target_idx = jnp.sum(keep_mask, axis=1) - 1
    target_val = on_chain[jnp.arange(batch_size), target_idx]
    resp = jnp.where(parents == target_val, CircleTask.yes_idx, CircleTask.no_idx)

    on_chain += CircleTask.offset

    if trace_to_start:
        chain = jnp.concatenate((on_chain, resp[:,None]), axis=1)
    else:
        keep_idx = jnp.sum(keep_mask, axis=1)
        chain = on_chain * keep_mask
        chain = chain.at[jnp.arange(batch_size), keep_idx].set(resp)

    xs = jnp.concatenate((
        xs + CircleTask.offset, 
        CircleTask.sep_idx * jnp.ones((batch_size, 1)),
        chain,
        jnp.zeros((batch_size, 1))  # extra space for finish
    ), axis=-1)

    return xs

    
# task = StarfishTask(depth=10, samp_dist=(1,8), batch_size=10, cot=True, trace_to_start=False)
# xs, ys = next(task)

# print(xs)
# print(ys)

