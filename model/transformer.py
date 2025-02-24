"""
Adapted from: https://github.com/google/flax/blob/main/examples/lm1b

License notice:
Copyright 2023 The Flax Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# <codecell>
from flax import linen as nn, struct

import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.append('../')
from common import new_seed


@struct.dataclass
class TransformerConfig:
    n_vocab: int | None = None
    n_layers: int = 2
    n_hidden: int = 128
    n_heads: int = 1
    n_out: int = 1
    max_len: int = 1024
    pos_emb: bool = True
    use_last_index_output: bool = False
    softmax_att: bool = True
    layer_norm: bool = True
    residual_connections: bool = True
    n_mlp_layers: int = 2
    return_final_logits_only: bool = True
    pure_linear_self_att: bool = False
    as_rf_model: bool = False
    use_simple_att: bool = False
    freeze_emb: bool = False
    use_bias: bool = True

    def to_model(self):
        return Transformer(self)


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0,
                    squeeze=False):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
            max_len: maximum possible length for the input.
            min_scale: float: minimum frequency-scale in sine grating.
            max_scale: float: maximum frequency-scale in sine grating.

    Returns:
            output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, :d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)

        if not squeeze:
            pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]

        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Args:
        config: TransformerConfig dataclass containing hyperparameters.
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Args:
            inputs: input data.

        Returns:
            output: `(bs, timesteps, in_dim)`
        """
        config = self.config
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                 ' but it is: %d' % inputs.ndim)
        length = inputs.shape[1]
        pos_emb_shape = (1, config.max_len, inputs.shape[-1])
        pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
                                                                pos_emb_shape,
                                                                None)
        
        pe = pos_embedding[:, :length, :]
        return inputs + pe


class SimpleSelfAttention(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        self.sow('intermediates', 'inputs', inputs)
        
        n_feats = inputs.shape[-1]
        n_heads = self.config.n_heads
        assert n_feats % n_heads == 0

        head_dim = n_feats // n_heads
        
        query = nn.DenseGeneral(features=(n_heads, head_dim), name='query', use_bias=False)(inputs)
        key = nn.DenseGeneral(features=(n_heads, head_dim), name='key', use_bias=False)(inputs)

        query = query / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
        # attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        self.sow('intermediates', 'attention_weights', attn_weights)

        out = jnp.einsum('...hqk,...kd->...qhd', attn_weights, inputs)
        # out = nn.DenseGeneral(features=n_feats, axis=(-2, -1), use_bias=False)(out)
        out = nn.DenseGeneral(features=1, axis=(-2, -1), use_bias=False)(out)
        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self,
                inputs,
                decoder_mask=None):

        assert inputs.ndim == 3

        if self.config.use_simple_att:
            x = SimpleSelfAttention(config=self.config)(inputs)
        else:
            x = nn.MultiHeadDotProductAttention(num_heads=self.config.n_heads, 
                                                qkv_features=self.config.n_hidden,
                                                use_bias=self.config.use_bias)(inputs_q=inputs, inputs_kv=inputs, mask=decoder_mask)

        if self.config.residual_connections:
            x = x + inputs

        if self.config.layer_norm:
            x = nn.LayerNorm()(x)
        
        if self.config.n_mlp_layers > 0:
            pre_mlp_x = x
            for i in range(self.config.n_mlp_layers):
                if i == 0:
                    x = nn.Dense(features=self.config.n_hidden, use_bias=self.config.use_bias)(pre_mlp_x)
                else:
                    x = nn.gelu(x)
                    x = nn.Dense(features=self.config.n_hidden, use_bias=self.config.use_bias)(x)
            
            if self.config.residual_connections:
                x = x + pre_mlp_x

            if self.config.layer_norm:
                x = nn.LayerNorm()(x)

        return x


class Transformer(nn.Module):

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):

        config = self.config
        y = inputs

        # Target Embedding
        if config.n_vocab is not None:
            assert inputs.ndim == 2  # (batch, len)

            name = 'Embed_freeze' if (self.config.freeze_emb or self.config.as_rf_model) else None
            y = nn.Embed(
                num_embeddings=self.config.n_vocab,
                features=self.config.n_hidden,
                name=name)(y)

        if config.pos_emb:
            y = AddPositionEmbs(config=config)(y)
        
        decoder_mask = nn.make_causal_mask(jnp.zeros(inputs.shape[:2]))
        
        for i in range(config.n_layers):
            name = f'transformer_block_{i}_freeze' if self.config.as_rf_model else None
            y = TransformerBlock(config=config, name=name)(y, decoder_mask=decoder_mask)
        
        if config.use_last_index_output:
            return y[:,-1,-1]

        logits = nn.Dense(config.n_out, use_bias=self.config.use_bias)(y)
        if config.return_final_logits_only:
            logits = logits[:,-1,:]

            if config.n_out == 1:
                logits = logits.flatten()

        return logits


def generate(state, xs, idx=2, beta=1, seed=None):
    if seed is None:
        seed = new_seed()

    source = jax.random.key(seed)
    while idx < xs.shape[1] - 1:
        key, source = jax.random.split(source)
        xs = _gen_pass(key, state, xs, idx, beta)
        idx += 1
    
    return xs


@jax.jit
def _gen_pass(key, state, xs, idx, beta):
    logits = state.apply_fn({'params': state.params}, xs)
    preds = jax.random.categorical(key, beta * logits)
    xs = xs.at[:,idx+1].set(preds[:,idx])
    return xs

