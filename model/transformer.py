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
from typing import Any

from flax import linen as nn, struct

import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.append('../')
from common import new_seed, t


@struct.dataclass
class TransformerConfig:
    n_vocab: int | None = None
    n_layers: int = 2
    n_hidden: int = 128
    n_heads: int = 1
    n_out: int = 1
    max_len: int = 1024
    pos_emb: bool = True
    layer_norm: bool = True
    residual_connections: bool = True
    n_mlp_layers: int = 2
    return_format: str = 'final_logit'
    as_rf_model: bool = False
    use_simple_att: bool = False
    freeze_emb: bool = False
    use_bias: bool = True
    mup_scale: bool = False
    linear_att: bool = False
    remove_att: bool = False
    unif_att: bool = False
    flash_att: bool = False
    dtype: Any = None

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
        pe[:, d_feature // 2:] = np.cos(position * div_term)

        if not squeeze:
            pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]

        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
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
    # NOTE: muP scale implemented only for single head case

    @nn.compact
    def __call__(self, inputs, mask=None, is_first=False):
        self.sow('intermediates', 'inputs', inputs)
        
        n_feats = inputs.shape[-1]
        n_heads = self.config.n_heads
        assert n_feats % n_heads == 0

        head_dim = n_feats // n_heads

        if self.config.mup_scale:
            kernel_init = nn.initializers.normal(np.sqrt(head_dim))
            prefac = 1 / head_dim
        else:
            kernel_init = nn.initializers.truncated_normal(1 / np.sqrt(head_dim))
            prefac = 1
        
        value = prefac * nn.DenseGeneral(features=(n_heads, head_dim), 
                                         name='value', 
                                         use_bias=False, 
                                         kernel_init=kernel_init,
                                         dtype=jnp.bfloat16 if self.config.flash_att else self.config.dtype)(inputs)

        query = prefac * nn.DenseGeneral(features=(n_heads, head_dim), 
                                         name='query', 
                                         use_bias=False, 
                                         kernel_init=kernel_init,
                                         dtype=jnp.bfloat16 if self.config.flash_att else self.config.dtype)(inputs)
        key = prefac * nn.DenseGeneral(features=(n_heads, head_dim), 
                                       name='key', 
                                       use_bias=False, 
                                       kernel_init=kernel_init,
                                       dtype=jnp.bfloat16 if self.config.flash_att else self.config.dtype)(inputs)
        fac = head_dim if self.config.mup_scale else np.sqrt(head_dim)

        if self.config.flash_att:
            out = jax.nn.dot_product_attention(query, key, value, 
                                                    bias=None, 
                                                    scale=(1/fac), 
                                                    is_causal=True,
                                                    implementation='cudnn')
        else:
            if self.config.unif_att:
                attn_weights = jnp.ones((1, inputs.shape[1], inputs.shape[1]))
            else:
                attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key) / fac

            if mask is not None:
                if self.config.linear_att:
                    attn_weights = jnp.where(mask, attn_weights, 0)
                else:
                    attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
                    attn_weights = jax.nn.softmax(attn_weights, axis=-1)


            self.sow('intermediates', 'attention_weights', attn_weights)
            out = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

        if self.config.mup_scale:
            kernel_init = nn.initializers.normal(np.sqrt(self.config.n_hidden))
            prefac = 1 / self.config.n_hidden
        else:
            kernel_init = nn.initializers.truncated_normal(1 / np.sqrt(self.config.n_hidden))
            prefac = 1

        out = prefac * nn.DenseGeneral(features=n_feats, 
                                       axis=(-2, -1), 
                                       use_bias=False, 
                                       name='out', 
                                       kernel_init=kernel_init,
                                       dtype=self.config.dtype)(out)

        self.sow('intermediates', 'attention_out', out)
        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self,
                inputs,
                decoder_mask=None,
                is_first=False):

        assert inputs.ndim == 3

        x = SimpleSelfAttention(config=self.config)(inputs, mask=decoder_mask, is_first=is_first)

        if self.config.residual_connections:
            if self.config.remove_att:
                x = inputs
            else:
                x = x + inputs

        if self.config.layer_norm:
            x = nn.LayerNorm()(x)
        
        if self.config.mup_scale:
            kernel_init = nn.initializers.normal(np.sqrt(self.config.n_hidden))
            prefac = 1 / self.config.n_hidden
        else:
            kernel_init = nn.initializers.truncated_normal(1 / np.sqrt(self.config.n_hidden))
            prefac = 1

        
        if self.config.n_mlp_layers > 0:
            pre_mlp_x = x
            for i in range(self.config.n_mlp_layers):
                if i == 0:
                    x = prefac * nn.Dense(features=self.config.n_hidden, 
                                          use_bias=self.config.use_bias, 
                                          kernel_init=kernel_init,
                                          dtype=self.config.dtype)(pre_mlp_x)
                else:
                    x = nn.relu(x)
                    x = prefac * nn.Dense(features=self.config.n_hidden, 
                                          use_bias=self.config.use_bias, 
                                          kernel_init=kernel_init,
                                          dtype=self.config.dtype)(x)
                
                self.sow('intermediates', f'layer_{i}', x)
            
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
                embedding_init=nn.initializers.normal(1),
                features=self.config.n_hidden,
                name=name,
                dtype=self.config.dtype)(y)

        if config.pos_emb:
            y = AddPositionEmbs(config=config)(y)
        
        decoder_mask = nn.make_causal_mask(jnp.zeros(inputs.shape[:2]))

        
        for i in range(config.n_layers):
            name = f'transformer_block_{i}_freeze' if self.config.as_rf_model else None
            y = TransformerBlock(config=config, name=name)(y, decoder_mask=decoder_mask, is_first=(i == 0))
        
        if self.config.mup_scale:
            kernel_init = nn.initializers.normal(1)
            prefac = 1 / self.config.n_hidden
            logits = prefac * nn.Dense(config.n_out, 
                                       use_bias=self.config.use_bias, 
                                       kernel_init=kernel_init,
                                       dtype=self.config.dtype)(y)
        else:
            logits = nn.Dense(config.n_out, 
                              use_bias=self.config.use_bias,
                              dtype=self.config.dtype)(y)

        if config.return_format is None:
            pass
        elif config.return_format == 'final_logit':
            logits = logits[:,-1]
        elif config.return_format == 'final_logit_up_to_pad':
            pred_idx = jnp.sum(inputs != 0, axis=1) - 1
            logits = logits[jnp.arange(logits.shape[0]), pred_idx]
        else:
            raise ValueError(f'unrecognized return format: {config.return_format}')

        if self.config.n_out == 1:
            logits = logits.squeeze()

        return logits


def transformer_phi(X, flatten=True):
    X_curr = X.reshape(*X.shape, 1, 1)

    X = jnp.repeat(jnp.expand_dims(X, axis=1), X.shape[1], axis=1)
    X = jnp.permute_dims(X, (0, 3, 1, 2))  # B x H x L x L
    X = jnp.tril(X)
    X = jnp.permute_dims(X, (0, 2, 3, 1))  # B x L x L x H
    X = t(X) @ X

    X = jnp.expand_dims(X, axis=2)
    X = X_curr * X                            # B x L x j x k x m

    X = jnp.permute_dims(X, (0, 1, 4, 2, 3))  # B x L x m x j x k

    if flatten:
        X = X.reshape(X.shape[0], X.shape[1], -1)
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    return X


@struct.dataclass
class TrLogRegConfig:
    n_vocab: float
    n_out: float = 1
    n_hidden: float = 32
    flatten: bool = False
    return_final_logits_only: bool = False
    pos_emb: bool = False
    max_len: int = 2

    def to_model(self):
        return TrLogReg(self)


class TrLogReg(nn.Module):
    config: TrLogRegConfig

    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(self.config.n_vocab, features=self.config.n_hidden, name='Embed_freeze')(inputs)

        if self.config.pos_emb:
            pos_emb_shape = (1, self.config.max_len, x.shape[-1])
            pos_embedding = sinusoidal_init(max_len=self.config.max_len)(None,
                                                                    pos_emb_shape,
                                                                    None)
        
            pe = pos_embedding[:, :x.shape[1], :]


            x = x + pe

        x = transformer_phi(x, flatten=self.config.flatten)
        
        if not self.config.flatten:
            x = nn.Dense(1, use_bias=False)(x).squeeze()

        x = nn.Dense(self.config.n_out, use_bias=False)(x)

        if self.config.return_final_logits_only:
            x = x[:,-1]

            if self.config.n_out == 1:
                x = x.flatten()

        return x


@struct.dataclass
class TrConfig:
    n_vocab: int
    n_out: int = 1
    n_hidden: int = 32
    return_final_logits_only: bool = False
    pos_emb: bool = False
    rand_pos_emb: bool = False
    big_pe: bool = True
    max_len: int = 2

    def to_model(self):
        return Tr(self)


class Tr(nn.Module):
    config: TrConfig

    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(self.config.n_vocab, features=self.config.n_hidden, name='Embed_freeze')(inputs) # B x L x H

        if self.config.pos_emb:
            if not self.config.rand_pos_emb:
                pos_emb_shape = (1, self.config.max_len, x.shape[-1])
                pos_embedding = sinusoidal_init(max_len=self.config.max_len)(None,
                                                                        pos_emb_shape,
                                                                        None)
            
                pe = pos_embedding[:, :x.shape[1], :]
            else:
                ps = jnp.arange(x.shape[1])[None]
                pe = nn.Embed(self.config.max_len, features=self.config.n_hidden, name='PE_freeze')(ps)
                if self.config.big_pe:
                    pe = pe * np.sqrt(self.config.n_hidden)





            x = x + pe

        Ax = nn.Dense(self.config.n_hidden, use_bias=False)(x)
        att = jnp.tril(x @ t(Ax))  # B x L x L
        x = att @ x

        x = nn.Dense(self.config.n_out, use_bias=False)(x)
        if self.config.return_final_logits_only:
            x = x[:,-1]

            if self.config.n_out == 1:
                x = x.flatten()

        return x


### COORDINATE CHECKING










                                



#         # logits_init = state.apply_fn({'params': state.params}, xs_init)
#         # att_init = intm['intermediates']['TransformerBlock_1']['SimpleSelfAttention_0']['attention_logits'][0]
        


#         # att = intm['intermediates']['TransformerBlock_1']['SimpleSelfAttention_0']['attention_logits'][0]

#         # norm_att = np.std(att - att_init).item()

#         # curr_norms_att.append(norm_att)
#         del state
    


#     # plt.plot(norms[:,1], 'o--')
#     # plt.yscale('log')

