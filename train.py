"""
Training utilities
"""

# <codecell>
from dataclasses import dataclass, field
from functools import partial
from typing import Iterable

from flax import traverse_util
from flax.serialization import to_state_dict
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
import wandb

from common import new_seed, merge_dicts, gen1, gen2


def create_train_state(rng=None, 
                       model=None, 
                       dummy_input=None, 
                       params=None, 
                       gamma=None, 
                       lr=1e-4, 
                       optim=optax.adamw, 
                       with_multistep_k=None,
                       clip=None,
                       **opt_kwargs):
    if params is None:
        params = model.init(rng, dummy_input)['params']

    tx = optim(learning_rate=lr, **opt_kwargs)

    tx_with_freeze = optax.multi_transform(
        {'learn': tx,
         'freeze': optax.set_to_zero()},
         traverse_util.path_aware_map(lambda path, _: 'freeze' if np.any([s.endswith('freeze') for s in path]) else 'learn', params)
    )

    if clip is not None:
        tx_with_freeze = optax.chain(
            optax.clip_by_global_norm(clip),
            tx_with_freeze
        )

    if with_multistep_k is not None:
        tx_with_freeze = optax.MultiSteps(tx_with_freeze, every_k_schedule=with_multistep_k)

    def apply_fn(variables, *args, **kwargs):
        logits = model.apply(variables, *args, **kwargs)

        if gamma is not None:
            logits_init = model.apply({'params': params}, *args, **kwargs)

            if type(logits_init) is tuple:
                val = (1 / gamma) * (logits[0] - logits_init[0])
                return val, logits[1]

            logits = (1 / gamma) * (logits - logits_init)
            return logits

        return logits

    return train_state.TrainState.create(
        apply_fn=apply_fn,
        params=params,
        tx=tx_with_freeze,
    )


def ce_mask(logits, labels):
    assert logits.shape[:2] == labels.shape, f'logit shape {logits.shape} not compatible with label shape {labels.shape}'

    out = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    mask = (labels != 0).astype(int)

    res = jnp.sum(out * mask)
    total = jnp.sum(mask)
    return res / total


def mse_mask(logits, labels):
    assert logits.shape[:2] == labels.shape, f'logit shape {logits.shape} not compatible with label shape {labels.shape}'

    targets = 2 * jax.nn.one_hot(labels, logits.shape[-1]) - 1
    out = ((targets - logits)**2).mean(axis=-1)
    mask = (labels != 0).astype(int)

    res = jnp.sum(out * mask)
    total = jnp.sum(mask)
    return res / total


def parse_loss_name(loss):
    if callable(loss):
        return loss

    loss_func = None
    if loss == 'bce':
        loss_func = optax.sigmoid_binary_cross_entropy
    elif loss == 'ce':
        loss_func = optax.softmax_cross_entropy_with_integer_labels
    elif loss == 'ce_mask':
        loss_func = ce_mask
    elif loss == 'mse':
        loss_func = optax.squared_error
    elif loss == 'mse_mask':
        loss_func = mse_mask
    else:
        raise ValueError(f'unrecognized loss name: {loss}')
    return loss_func


@partial(jax.jit, static_argnames=('loss',))
def train_step(state, batch, loss='bce'):
    x, labels = batch
    loss_func = parse_loss_name(loss)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)

        train_loss = loss_func(logits, labels)

        if loss == 'bce' and len(labels.shape) > 1:
            assert logits.shape == train_loss.shape
            train_loss = train_loss.mean(axis=-1)

        return train_loss.mean()
    
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@partial(jax.jit, static_argnames=('loss',))
def loss_and_acc(state, batch, loss='bce'):
    x, labels = batch
    logits = state.apply_fn({'params': state.params}, x)
    loss_func = parse_loss_name(loss)
    loss = loss_func(logits, labels).mean()

    if len(logits.shape) == 1:
        preds = logits > 0
    else:
        preds = logits.argmax(axis=-1)
    
    if len(labels.shape) == 2:
        # autoregressive branch
        mask = labels != 0
        res = jnp.sum((preds == labels) & mask)
        total = jnp.sum(mask)
        acc = res / total
    elif len(preds.shape) == 2:
        acc = jnp.mean(preds[:,-1] == labels)
    else:
        acc = jnp.mean(preds == labels)

    return {'loss': loss, 'acc': acc}


@partial(jax.jit, static_argnames=('loss'))
def decomp_flat_acc(state, batch, loss=None):
    x, labels = batch
    logits = state.apply_fn({'params': state.params}, x)

    if len(logits.shape) == 1:
        preds = logits > 0
    else:
        preds = logits.argmax(axis=-1)
    
    true_pos = labels * preds
    true_neg = (1 - labels) * (1 - preds)
    false_pos = (1 - labels) * preds
    false_neg = labels * (1 - preds)

    return {
        'true_pos': jnp.mean(true_pos),
        'true_neg': jnp.mean(true_neg),
        'false_pos': jnp.mean(false_pos),
        'false_neg': jnp.mean(false_neg),
    }


@partial(jax.jit, static_argnames=('loss'))
def gen_acc_cot2(state, batch, loss=None):
    xs, ys = batch
    ys = ys[:,2:]
    ans_idx = jnp.sum(ys != 0, axis=-1) - 1
    ans = ys[jnp.arange(len(ys)), ans_idx]

    traj = gen2(state, xs)
    preds = extract_pred(traj)

    return {'gen_acc': jnp.mean(preds == ans)}


@partial(jax.jit, static_argnames=('loss'))
def gen_acc_cot1(state, batch, loss=None):
    xs, ys = batch
    ys = ys[:,2:]
    ans_idx = jnp.sum(ys != 0, axis=-1) - 1
    ans = ys[jnp.arange(len(ys)), ans_idx]

    traj = gen1(state, xs)
    preds = extract_pred(traj)

    return {'gen_acc': jnp.mean(preds == ans)}


@partial(jax.jit, static_argnames=('loss'))
def gen_acc_rl(state, batch, loss=None):
    xs, ys = batch

    traj = gen2(state, xs)
    preds = extract_pred(traj)

    return {'gen_acc': jnp.mean(preds == ys)}


def extract_pred(traj):
    # assumes no/yes classification offset by 1 for padding
    no_occ = jnp.argmax(traj == 1, axis=1)
    no_occ = jnp.where(no_occ == 0, jnp.inf, no_occ)
    yes_occ = jnp.argmax(traj == 2, axis=1)
    yes_occ = jnp.where(yes_occ == 0, jnp.inf, yes_occ)

    preds = jnp.argmin(jnp.stack((no_occ, yes_occ), axis=1), axis=1) + 1
    preds = jnp.where(no_occ != yes_occ, preds, jnp.inf)
    return preds


def print_gen(step, hist):
    print(f'ITER {step}:  train_loss={hist["train"][-1]["loss"]:.4f}   train_acc={hist["train"][-1]["gen_acc"]:.4f}   test_loss={hist["test"][-1]["loss"]:.4f}   test_acc={hist["test"][-1]["gen_acc"]:.4f}')


def train(config, train_iter, 
          test_iter=None, 
          loss='ce', gamma=None,
          eval_fns: Iterable=None, print_fn=None,
          summary_fn=None,
          train_iters=10_000, test_iters=1, test_every=1_000, save_params=False,
          early_stop_n=None, early_stop_key='loss', early_stop_decision='min' ,
          optim=optax.adamw, k=1,
          seed=None, use_tqdm=False, wdb=None,
          **opt_kwargs):

    if seed is None:
        seed = new_seed()

    if test_iter is None:
        test_iter = train_iter
    
    if eval_fns is None:
        eval_fns = [loss_and_acc]
    
    if print_fn is None:
        print_fn = _print_status
    
    if isinstance(config, train_state.TrainState):
        state = config
    else:
        init_rng = jax.random.key(seed)
        model = config.to_model()

        samp_x, _ = next(train_iter)
        state = create_train_state(init_rng, model, samp_x, gamma=gamma, optim=optim, with_multistep_k=k, **opt_kwargs)

    hist = {
        'train': [],
        'test': [],
        'params': [state.params] if save_params else [],
        'summary': []
    }

    train_iters = k * train_iters
    it = zip(range(train_iters), train_iter)
    if use_tqdm:
        it = tqdm(it, total=train_iters)

    for step, batch in it:
        state = train_step(state, batch, loss=loss)

        if ((step + 1) % (test_every * k) == 0) or ((step + 1) == train_iters):
            all_train = []
            all_test = []

            for _, train_batch, test_batch in zip(range(test_iters), train_iter, test_iter):
                all_train.append(merge_dicts([fn(state, train_batch, loss=loss) for fn in eval_fns]))
                all_test.append(merge_dicts([fn(state, test_batch, loss=loss) for fn in eval_fns]))
            
            all_train = jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)).item(), *all_train)
            all_test = jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)).item(), *all_test)

            hist['train'].append(all_train)
            hist['test'].append(all_test)

            if summary_fn is not None:
                summ = summary_fn(state)
                hist['summary'].append(summ)

            print_fn((step + 1) // k, hist)

            if wdb is not None:
                train_obj = hist['train'][-1]
                test_obj = hist['test'][-1]
                
                train_log = {f'train_{k}': v for k, v in train_obj.items()}
                test_log = {f'test_{k}': v for k, v in test_obj.items()}
                log = train_log | test_log

                if summary_fn is not None:
                    summ_obj = hist['summary'][-1]
                    summ_log = {f'summary_{k}': v for k, v in summ_obj.items()}
                    log = log | summ_log
                
                wdb.log(log)

            if save_params:
                hist['params'].append(state.params)
        
            if early_stop_n is not None and len(hist['train']) > early_stop_n:
                last_n_metrics = np.array([getattr(m, early_stop_key) for m in hist['train'][-early_stop_n - 1:]])
                if early_stop_decision == 'min' and np.all(last_n_metrics[0] < last_n_metrics[1:]) \
                or early_stop_decision == 'max' and np.all(last_n_metrics[0] > last_n_metrics[1:]):
                    print(f'info: stopping early with {early_stop_key} =', last_n_metrics[-1])
                    break
    
    return state, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  train_loss={hist["train"][-1]["loss"]:.4f}   train_acc={hist["train"][-1]["acc"]:.4f}   test_loss={hist["test"][-1]["loss"]:.4f}   test_acc={hist["test"][-1]["acc"]:.4f}')


@dataclass
class Case:
    name: str
    config: dataclass
    train_task: Iterable | None = None
    test_task: Iterable | None = None
    train_args: dict = field(default_factory=dict)
    state: list = None
    hist: list = None
    info: dict = field(default_factory=dict)
    wdb_proj: str = None

    def run(self):
        if self.wdb_proj is not None:
            wdb = wandb.init(project=self.wdb_proj, config=to_state_dict(self.config))
        else:
            wdb = None

        self.state, self.hist = train(self.config, train_iter=self.train_task, test_iter=self.test_task, wdb=wdb, **self.train_args)
    
    def eval(self, task, eval_fns, n_iters=1, prefix=None):
        all_res = []
        loss = self.train_args.get('loss', None)

        for _ in range(n_iters):
            batch = next(task)
            all_res.append(merge_dicts([fn(self.state, batch, loss=loss) for fn in eval_fns]))

        all_res = jax.tree.map(lambda *xs: jnp.mean(jnp.array(xs)).item(), *all_res)
        if prefix is not None:
            all_res = {prefix: all_res}

        self.info = self.info | all_res

