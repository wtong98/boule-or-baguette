"""
Training utilities
"""

from dataclasses import dataclass, field
from functools import partial
import itertools
from typing import Any, Iterable

from flax import struct, traverse_util
from flax.training import train_state
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from common import new_seed


@struct.dataclass
class Metrics:
    accuracy: float
    loss: float
    count: int = 0

    @staticmethod
    def empty():
        return Metrics(accuracy=-1, loss=-1)
    
    def merge(self, other):
        total = self.count + 1
        acc = (self.count / total) * self.accuracy + (1 / total) * other.accuracy
        loss = (self.count / total) * self.loss + (1 / total) * other.loss
        return Metrics(acc, loss, count=total)


class TrainState(train_state.TrainState):
    metrics: Metrics
    init_params: Any = None


def create_train_state(rng=None, model=None, dummy_input=None, params=None, gamma=None, lr=1e-4, optim=optax.adamw, **opt_kwargs):
    if params is None:
        params = model.init(rng, dummy_input)['params']

    tx = optim(learning_rate=lr, **opt_kwargs)

    tx_with_freeze = optax.multi_transform(
        {'learn': tx,
         'freeze': optax.set_to_zero()},
         traverse_util.path_aware_map(lambda path, _: 'freeze' if np.any([s.endswith('freeze') for s in path]) else 'learn', params)
    )

    def apply_fn(variables, *args, **kwargs):
        logits = model.apply(variables, *args, **kwargs)

        if gamma is not None:
            logits_init = model.apply({'params': params}, *args, **kwargs)
            logits = (1 / gamma) * (logits - logits_init)

        return logits

    return TrainState.create(
        apply_fn=apply_fn,
        params=params,
        tx=tx_with_freeze,
        metrics=Metrics.empty()
        # init_params=params
    )


def ce_mask(logits, labels):
    assert logits.shape[:2] == labels.shape, f'logit shape {logits.shape} not compatible with label shape {labels.shape}'

    out = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    mask = (labels != 0).astype(int)

    res = jnp.sum(out * mask)
    total = jnp.sum(mask)
    return res / total


def parse_loss_name(loss):
    loss_func = None
    if loss == 'bce':
        loss_func = optax.sigmoid_binary_cross_entropy
    elif loss == 'ce':
        loss_func = optax.softmax_cross_entropy_with_integer_labels
    elif loss == 'ce_mask':
        loss_func = ce_mask
    elif loss == 'mse':
        loss_func = optax.squared_error
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
def compute_metrics(state, batch, loss='bce'):
    x, labels = batch
    logits = state.apply_fn({'params': state.params}, x)
    loss_func = parse_loss_name(loss)
    loss = loss_func(logits, labels).mean()

    if len(logits.shape) == 1:
        preds = logits > 0
    else:
        preds = logits.argmax(axis=-1)
    
    # TODO: test
    if len(labels.shape) == 2:
        # autoregressive branch
        mask = labels != 0
        res = jnp.sum((preds == labels) & mask)
        total = jnp.sum(mask)
        acc = res / total
    else:
        acc = jnp.mean(preds == labels)

    metrics = Metrics(accuracy=acc, loss=loss)
    metrics = state.metrics.merge(metrics)
    state = state.replace(metrics=metrics)
    return state


def train(config, data_iter, 
          test_iter=None, 
          loss='ce', gamma=None,
          train_iters=10_000, test_iters=100, test_every=1_000, save_params=False,
          early_stop_n=None, early_stop_key='loss', early_stop_decision='min' ,
          optim=optax.adamw,
          seed=None, use_tqdm=False,
          **opt_kwargs):

    if seed is None:
        seed = new_seed()
    
    if test_iter is None:
        test_iter = data_iter
    
    init_rng = jax.random.key(seed)
    model = config.to_model()

    samp_x, _ = next(data_iter)
    state = create_train_state(init_rng, model, samp_x, gamma=gamma, optim=optim, **opt_kwargs)

    hist = {
        'train': [],
        'test': [],
        'params': []
    }

    it = zip(range(train_iters), data_iter)
    if use_tqdm:
        it = tqdm(it, total=train_iters)

    for step, batch in it:
        state = train_step(state, batch, loss=loss)
        state = compute_metrics(state, batch, loss=loss)

        if ((step + 1) % test_every == 0) or ((step + 1) == train_iters):
            hist['train'].append(state.metrics)

            state = state.replace(metrics=Metrics.empty())
            test_state = state
            for _, test_batch in zip(range(test_iters), test_iter):
                test_state = compute_metrics(test_state, test_batch, loss=loss)
            
            hist['test'].append(test_state.metrics)

            _print_status(step+1, hist)

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
    print(f'ITER {step}:  train_loss={hist["train"][-1].loss:.4f}   train_acc={hist["train"][-1].accuracy:.4f}   test_acc={hist["test"][-1].accuracy:.4f}')


def reinforce(state, data_iter, 
              action_fn, reward_fn, rl_loss,
              train_iters=10_000, 
              test_iter=None, test_iters=10, test_every=1000, eval_loss='ce_mask',
              save_params=False,
              use_tqdm=False):

    if test_iter is None:
        test_iter = data_iter
    
    action_fn = jax.tree_util.Partial(action_fn)
    reward_fn = jax.tree_util.Partial(reward_fn)
    rl_loss = jax.tree_util.Partial(rl_loss)

    it = zip(range(train_iters), data_iter)
    if use_tqdm:
        it = tqdm(it, total=train_iters)

    hist = {
        'rew': [],
        'test': [],
        'params': []
    }
    
    for step, batch in it:
        state = rl_step(state, batch, action_fn, reward_fn, rl_loss)

        if ((step + 1) % test_every == 0) or ((step + 1) == train_iters):
            state = state.replace(metrics=Metrics.empty())
            test_state = state
            avg_rew = 0

            for _, test_batch in zip(range(test_iters), test_iter):
                test_state = compute_metrics(test_state, test_batch, loss=eval_loss)
                xs, ys = test_batch
                traj = action_fn(state, xs)
                rew = reward_fn(traj, ys)
                avg_rew += np.mean(rew) / test_iters
            
            hist['rew'].append(avg_rew)
            hist['test'].append(test_state.metrics)

            _print_rl_status(step+1, hist)

            if save_params:
                hist['params'].append(state.params)
    
    return state, hist


def _print_rl_status(step, hist):
    print(f'ITER {step}:  test_rew={hist["rew"][-1]:.4f} test_acc={hist["test"][-1].accuracy:.4f}')


def rl_step(state, batch, act_fn, rew_fn, rl_loss):
    xs, ys = batch
    traj = act_fn(state, xs)
    rew = rew_fn(traj, ys)
    loss_fn = lambda params: rl_loss(params, state ,traj, rew)
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


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

    def run(self):
        self.state, self.hist = train(self.config, data_iter=self.train_task, test_iter=self.test_task, **self.train_args)
    
    def get_flops(self):
        train_args = self.train_args
        loss = train_args.get('loss', None)
        return get_flops(train_step, self.state, next(self.train_task), loss=loss)
    
    def eval(self, task, key_name='eval_acc'):
        xs, labels = next(task)
        logits = self.state.apply_fn({'params': self.state.params}, xs)

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
        else:
            acc = jnp.mean(preds == labels)

        self.info[key_name] = acc
    
    def eval_mse(self, task, key_name='eval_mse'):
        xs, ys = next(task)
        ys_pred = self.state.apply_fn({'params': self.state.params}, xs)
        mse = np.mean((ys - ys_pred)**2)

        self.info[key_name] = mse


def eval_cases(all_cases, eval_task, key_name='eval_acc', use_mse=False, ignore_err=False):
    try:
        len(eval_task)
    except TypeError:
        eval_task = itertools.repeat(eval_task)

    for c, task in tqdm(zip(all_cases, eval_task), total=len(all_cases)):
        try:
            if use_mse:
                c.eval_mse(task, key_name)
            else:
                c.eval(task, key_name)
        except Exception as e:
            if ignore_err:
                continue
            else:
                raise e


# TODO: fix cost_analysis for FLOPs
def get_flops(fn, *args, **kwargs):
    """Borrowed from flax.nn.tabulate"""
    e = fn.lower(*args, **kwargs).compile()
    cost = e.cost_analysis()
    if cost is None:
        print('warn: unable to estimate flops')
        return 0
    flops = int(cost['flops']) if 'flops' in cost else -1
    return flops
