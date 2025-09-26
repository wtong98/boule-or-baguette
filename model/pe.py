"""Adapted from: https://github.com/crowsonkb/rope-flax/tree/main"""

from functools import wraps
from typing import Optional, Tuple

from einshape import jax_einshape as einshape
import flax.linen as nn
import jax
import jax.numpy as jnp


def rotate_half(x: jax.Array) -> jax.Array:
    x = einshape("...(dr)->...dr", x, r=2)
    x1, x2 = x[..., 0], x[..., 1]
    x = jnp.stack((-x2, x1), axis=-1)
    return einshape("...dr->...(dr)", x)


def apply_rotary_emb(
    freqs: jax.Array, t: jax.Array, start_index: int = 0, scale: float = 1.0
) -> jax.Array:
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    if end_index > t.shape[-1]:
        raise ValueError(
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * jnp.cos(freqs) * scale) + (rotate_half(t) * jnp.sin(freqs) * scale)
    return jnp.concatenate((t_left, t, t_right), axis=-1)


def freqs_lang(theta: float = 10000.0) -> callable:
    @wraps(freqs_lang)
    def init(key, shape, dtype=jnp.float32):
        dim = shape[-1] * 2
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
        return jnp.broadcast_to(jnp.log(freqs), shape)

    return init


class RoPE(nn.Module):
    dim: int
    num_heads: int = 1
    start_index: int = 0
    dtype: jnp.dtype = jnp.float32
    freqs_init: callable = freqs_lang()

    def setup(self):
        shape = self.num_heads, self.dim // 2
        self.freqs = self.param("freqs", self.freqs_init, shape)

    def get_freqs(self, pos: jax.Array) -> jax.Array:
        freqs = jnp.repeat(jnp.exp(self.freqs), 2, axis=-1)
        return pos[..., None, None] * freqs.astype(self.dtype)

    def __call__(self, x: jax.Array, pos: jax.Array) -> jax.Array:
        freqs = self.get_freqs(pos)
        return apply_rotary_emb(freqs, x, start_index=self.start_index)

