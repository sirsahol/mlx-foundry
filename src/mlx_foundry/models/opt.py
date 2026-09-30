# Copyright © 2024 Apple Inc. & MLX Foundry contributors
# OPT architecture implementation for Apple Silicon MLX

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "opt"
    vocab_size: int = 50272
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    ffn_dim: int = 3072
    max_position_embeddings: int = 2048
    do_layer_norm_before: bool = True
    word_embed_proj_dim: int | None = 768


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.k_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.v_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.self_attn_layer_norm = nn.LayerNorm(args.hidden_size)
        self.fc1 = nn.Linear(args.hidden_size, args.ffn_dim, bias=True)
        self.fc2 = nn.Linear(args.ffn_dim, args.hidden_size, bias=True)
        self.final_layer_norm = nn.LayerNorm(args.hidden_size)
        self.do_layer_norm_before = args.do_layer_norm_before

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        if self.do_layer_norm_before:
            residual = x
            x = self.self_attn_layer_norm(x)
            x = residual + self.self_attn(x, mask=mask, cache=cache)
            residual = x
            x = self.final_layer_norm(x)
            x = residual + self.fc2(nn.relu(self.fc1(x)))
            return x
        else:
            residual = x
            x = self.self_attn(x, mask=mask, cache=cache)
            x = self.self_attn_layer_norm(residual + x)
            residual = x
            x = self.fc2(nn.relu(self.fc1(x)))
            x = self.final_layer_norm(residual + x)
            return x


class OPTDecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_positions = nn.Embedding(args.max_position_embeddings + 2, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.final_layer_norm = nn.LayerNorm(args.hidden_size)
        self.offset_bias = 2

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape
        hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        offset = 0
        if cache[0] is not None:
            offset = cache[0].offset

        position_ids = mx.arange(L) + (offset + self.offset_bias)
        hidden_states += self.embed_positions(position_ids)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.final_layer_norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = nn.Module()
        self.model.decoder = OPTDecoder(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        hidden_states = self.model.decoder(inputs, cache)
        return self.lm_head(hidden_states)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.decoder.layers
