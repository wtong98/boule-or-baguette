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
    as_rf_model: bool = False
    use_simple_att: bool = False
    freeze_emb: bool = False
    use_bias: bool = True
    mup_scale: bool = False
    linear_att: bool = False
    remove_att: bool = False

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
    # NOTE: muP scale implemented only for single head case

    @nn.compact
    def __call__(self, inputs, mask=None):
        self.sow('intermediates', 'inputs', inputs)
        
        n_feats = inputs.shape[-1]
        n_heads = self.config.n_heads
        assert n_feats % n_heads == 0

        head_dim = n_feats // n_heads
        
        query = nn.DenseGeneral(features=(n_heads, head_dim), name='query', use_bias=False)(inputs)
        key = nn.DenseGeneral(features=(n_heads, head_dim), name='key', use_bias=False)(inputs)
        value = nn.DenseGeneral(features=(n_heads, head_dim), name='value', use_bias=False)(inputs)

        attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)

        if mask is not None:
            if self.config.linear_att:
                attn_weights = jnp.where(mask, attn_weights, 0)
            else:
                attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
                attn_weights = jax.nn.softmax(attn_weights / head_dim, axis=-1)


        self.sow('intermediates', 'attention_weights', attn_weights)

        out = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        out = nn.DenseGeneral(features=n_feats, axis=(-2, -1), use_bias=False, name='out')(out)
        # out = nn.DenseGeneral(features=1, axis=(-2, -1), use_bias=False)(out)
        return out


class TransformerBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self,
                inputs,
                decoder_mask=None):

        assert inputs.ndim == 3

        if self.config.use_simple_att or self.config.mup_scale or self.linear_att:
            x = SimpleSelfAttention(config=self.config)(inputs, mask=decoder_mask)
        else:
            x = nn.MultiHeadDotProductAttention(num_heads=self.config.n_heads, 
                                                qkv_features=self.config.n_hidden,
                                                use_bias=self.config.use_bias)(inputs_q=inputs, inputs_kv=inputs, mask=decoder_mask, sow_weights=True)

        if self.config.residual_connections:
            if self.config.remove_att:
                x = inputs
            else:
                x = x + inputs

        if self.config.layer_norm:
            x = nn.LayerNorm()(x)
        
        if self.config.n_mlp_layers > 0:
            # NOTE: muP scale not implemented for MLP layers
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


## COORDINATE CHECKING
# import matplotlib.pyplot as plt

# import sys
# sys.path.append('../')
# from task.graph import *
# from common import *
# from train import *


# base_lr = 0.01
# depth = 5
# n_vocab = 2**depth + 4

# train_task = Chain(
#     BinaryTreeTiTask(depth=depth, samp_dist=(1), on_branch=True, cot=True),
#     BinaryTreeTiTask(depth=depth, samp_dist=(1), on_branch=False, fill_gaps=False, cot=True))

# xs_init, _ = next(train_task)

# all_norms = []
# for n_steps in tqdm([1]):
#     curr_norms = []
#     for n_hidden in [128, 256, 512, 1024, 2048]:
#         gamma0 = 1
#         gamma = gamma0 * np.sqrt(n_hidden)
#         lr = gamma0**2 * base_lr


#         config = TransformerConfig(n_vocab=n_vocab,
#                                 n_layers=2,
#                                 n_hidden=n_hidden,
#                                 n_heads=1,
#                                 n_out=n_vocab,
#                                 pos_emb=False,
#                                 layer_norm=False,
#                                 residual_connections=False,
#                                 n_mlp_layers=0,
#                                 return_final_logits_only=False,
#                                 use_bias=False,
#                                 freeze_emb=True,
#                                 mup_scale=False)


#         state = create_train_state(jax.random.key(new_seed()),
#                                 model=config.to_model(),
#                                 dummy_input=xs_init,
#                                 optim=optax.sgd,
#                                 # gamma=gamma,
#                                 lr=lr)

#         logits_init = state.apply_fn({'params': state.params}, xs_init)
        

#         state, hist = train(state,
#                             train_iter=iter(train_task), 
#                             loss='ce_mask',
#                             # gamma=gamma,
#                             test_every=1000,
#                             train_iters=n_steps, 
#                             optim=optax.sgd,
#                             lr=lr,
#                             seed=None)

#         logits = state.apply_fn({'params': state.params}, xs_init)
#         # norm = np.mean(np.linalg.norm(logits, axis=(-1, -2)))
#         norm = np.std(logits - logits_init).item()
#         curr_norms.append(norm)
#         del state
    
#     all_norms.append(curr_norms)

# for norms in all_norms:
#     plt.plot(norms, 'o--')

