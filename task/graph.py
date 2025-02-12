"""Graph connectedness algorithms"""

# <codecell>
import jax
import numpy as np

def fast_binary_pow(a, n):
    for _ in range(n):
        a = _bin_path(a)
    return a

@jax.jit
def _bin_path(a):
    c = a @ a + a
    c = (c > 0).astype(int)
    return c


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

        # do_cont = False
        # while not do_cont:
        #     self.adj = self.rng.binomial(n=1, p=self.p_connect, size=self.n_nodes**2).reshape(self.n_nodes, self.n_nodes)
        #     self.cnx = fast_binary_pow(self.adj, n_nodes)

        #     p = np.mean(self.cnx)
        #     if prob_thresh[0] < p < prob_thresh[1]:
        #         do_cont = True
        #     else:
        #         print(f'warn: rejecting graph with p={p:.2f}')

        ## Enforced binary tree
        self.adj = np.zeros((self.n_nodes, self.n_nodes))
        idx = 2
        while idx < self.n_nodes:
            row = idx // 2 - 1
            self.adj[row,idx-1] += 1
            self.adj[row,idx] += 1
            idx += 2

        self.cnx = fast_binary_pow(self.adj, n_nodes)
        
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


# task = GraphTiTask(31, n_dims=None, batch_size=6, samp_adj=False)
# print(np.mean(task.cnx > 0))
# task.adj
# xs, ys = next(task)
# xs

# %%
