"""
Simple MLP model
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct

def parse_act_fn(fn: str):
    if fn == 'relu':
        return jax.nn.relu
    elif fn == 'linear':
        return lambda x: x
    elif fn == 'gelu':
        return jax.nn.gelu
    elif fn =='quadratic':
        return lambda x: x**2
    else:
        raise ValueError(f'function not recognized: {fn}')


@struct.dataclass
class MlpConfig:
    """Global hyperparamters"""
    n_vocab: int | None = None
    n_layers: int = 2
    n_emb: int = 64
    n_hidden: int = 128
    n_out: int = 1
    act_fn: str = 'relu'
    layer_norm: bool = False
    mup_scale: bool = False
    as_rf_model: bool = False
    use_bias: bool = True
    freeze_emb: bool = False

    def to_model(self):
        return MLP(self)


class MLP(nn.Module):

    config: MlpConfig

    @nn.compact
    def __call__(self, x):
        act_fn = parse_act_fn(self.config.act_fn)

        if self.config.n_vocab is not None:
            name = 'Embed_freeze' if (self.config.freeze_emb or self.config.as_rf_model) else None
            
            x = nn.Embed(
                num_embeddings=self.config.n_vocab,
                features=self.config.n_hidden,
                name=name)(x)
        
        x = x.reshape(x.shape[0], -1)

        for i in range(self.config.n_layers):
            name = None
            if self.config.as_rf_model:
                name = f'Dense_{i}_freeze'
            
            if self.config.mup_scale and i == 0:
                root_prec = np.sqrt(self.config.n_hidden)
                mup_init = jax.nn.initializers.truncated_normal(1 / root_prec)
                x = nn.Dense(self.config.n_hidden,
                               use_bias=self.config.use_bias,
                               kernel_init=mup_init,
                               name=name)(x)
                x = np.sqrt(self.config.n_hidden / x.shape[-1]) * x

            else:
                x = nn.Dense(self.config.n_hidden, 
                            use_bias=self.config.use_bias,
                            name=name)(x)

            if self.config.layer_norm:
                x = nn.LayerNorm()(x)

            x = act_fn(x)

        out = nn.Dense(self.config.n_out, use_bias=self.config.use_bias)(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out


@struct.dataclass
class MixerConfig:
    """Global hyperparamters"""
    n_vocab: int | None = None
    n_layers: int = 2
    n_hidden: int = 128
    n_channels: int = 16
    n_out: int = 1
    act_fn: str = 'relu'
    last_token_only: bool = True
    layer_norm: bool = False

    def to_model(self):
        return Mixer(self)


class Mixer(nn.Module):

    config: MixerConfig

    @nn.compact
    def __call__(self, x):
        act_fn = parse_act_fn(self.config.act_fn)

        if self.config.n_vocab is not None:
            name = 'Embed_freeze' if self.config.freeze_emb else None
            x = nn.Embed(
                num_embeddings=self.config.n_vocab,
                features=self.config.n_hidden,
                name=name)(x)
        
        assert len(x.shape) == 3

        for _ in range(self.config.n_layers):
            x = nn.Dense(self.config.n_hidden)(x)
            x = jnp.transpose(x, (0, 2, 1))

            x = nn.Dense(self.config.n_channels)(x)
            x = jnp.transpose(x, (0, 2, 1))

            if self.config.layer_norm:
                x = nn.LayerNorm()(x)

            x = act_fn(x)
    
        if self.config.last_token_only:
            x = x[:,-1,:]
        else:
            x = x.reshape(x.shape[0], -1)

        out = nn.Dense(self.config.n_out)(x)

        if self.config.n_out == 1:
            out = out.flatten()

        return out

