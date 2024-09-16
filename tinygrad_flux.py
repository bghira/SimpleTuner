# This file incorporates code from the following:
# Github Name                    | License | Link
# black-forest-labs/flux         | Apache  | https://github.com/black-forest-labs/flux/tree/main/model_licenses

from tinygrad import Tensor, nn, dtypes, TinyJit
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad.helpers import fetch, tqdm, colored
import numpy as np

import math, time, argparse, tempfile
from typing import Callable, Union, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tinygrad T5 model."""

import math

from dataclasses import dataclass
from typing import List, Union

from tinygrad import nn, Tensor, dtypes

from sentencepiece import SentencePieceProcessor


# default config is t5-xxl
@dataclass
class T5Config:
    d_ff: int = 10240
    d_kv: int = 64
    d_model: int = 4096
    layer_norm_epsilon: float = 1e-6
    num_decoder_layers: int = 24
    num_heads: int = 64
    num_layers: int = 24
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    vocab_size: int = 32128


class T5Tokenizer:
    def __init__(self, spiece_path):
        self.spp = SentencePieceProcessor(str(spiece_path))

    def encode(self, text: str, max_length: int) -> list[int]:
        encoded = self.spp.Encode(text)
        if len(encoded) > max_length - 1:
            encoded = encoded[: max_length - 1]
        return encoded + [1] + [0] * (max_length - len(encoded) - 1)


class T5LayerNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        self.weight = Tensor.ones(hidden_size)
        self.variance_epsilon = eps

    def __call__(self, hidden_states: Tensor) -> Tensor:
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.cast(dtypes.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * Tensor.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [dtypes.float16, dtypes.bfloat16]:
            hidden_states = hidden_states.cast(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseGatedActDense:
    def __init__(self, config: T5Config):
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        hidden_gelu = self.wi_0(hidden_states).gelu()
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF:
    def __init__(self, config: T5Config):
        self.DenseReluDense = T5DenseGatedActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states


class T5Attention:
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor, num_buckets: int = 32, max_distance: int = 128
    ) -> Tensor:
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = Tensor.zeros_like(relative_position)
        num_buckets //= 2
        relative_buckets += (relative_position > 0).cast(dtypes.long) * num_buckets
        relative_position = Tensor.abs(relative_position)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            Tensor.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).cast(dtypes.long)

        relative_position_if_large = Tensor.min(
            Tensor.stack(
                relative_position_if_large,
                Tensor.full_like(relative_position_if_large, num_buckets - 1),
            ),
            axis=0,
        )
        relative_buckets += Tensor.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device=None) -> Tensor:
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = Tensor.arange(
            query_length, dtype=dtypes.long, device=device
        )[:, None]
        memory_position = Tensor.arange(key_length, dtype=dtypes.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def __call__(
        self, hidden_states: Tensor, position_bias: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, key_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer):
            """projects hidden states correctly to key/query states"""
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))

            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(hidden_states, self.k)
        value_states = project(hidden_states, self.v)

        # compute scores
        scores = Tensor.matmul(query_states, key_states.transpose(3, 2))
        # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            position_bias = self.compute_bias(
                key_length, key_length, device=scores.device
            )

        scores += position_bias
        attn_weights = Tensor.softmax(scores.float(), axis=-1).cast(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(
            Tensor.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        return attn_output, position_bias


class T5LayerSelfAttention:
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def __call__(
        self, hidden_states: Tensor, position_bias: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.SelfAttention(
            normed_hidden_states, position_bias=position_bias
        )
        hidden_states = hidden_states + attention_output
        return hidden_states, position_bias


class T5Block:
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        self.layer = []
        self.layer.append(
            T5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        self.layer.append(T5LayerFF(config))

    def __call__(
        self, hidden_states: Tensor, position_bias: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        self_attention_outputs, position_bias = self.layer[0](
            hidden_states, position_bias=position_bias
        )
        hidden_states = self_attention_outputs

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        return hidden_states, position_bias


class T5Stack:
    def __init__(self, config: T5Config, embed_tokens: nn.Embedding | None = None):
        self.config = config
        self.embed_tokens = embed_tokens
        self.block = [
            T5Block(config, has_relative_attention_bias=bool(i == 0))
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )

    def __call__(self, input_ids: Tensor) -> Tensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        inputs_embeds = self.embed_tokens(input_ids)

        position_bias = None

        hidden_states = inputs_embeds

        for layer_module in self.block:
            hidden_states, position_bias = layer_module(
                hidden_states, position_bias=position_bias
            )

        return self.final_layer_norm(hidden_states)


class T5EncoderModel:
    def __init__(self, config: T5Config):
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)

    def __call__(self, input_ids: Tensor) -> Tensor:
        return self.encoder(input_ids)


class T5Embedder:
    def __init__(self, max_length: int, spiece_path: str):
        self.tokenizer = T5Tokenizer(spiece_path)
        self.max_length = max_length
        config = T5Config()
        self.encoder = T5EncoderModel(config)

    def __call__(self, texts: Union[str, List[str]]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        toks = Tensor.cat(
            *[Tensor(self.tokenizer.encode(text, self.max_length)) for text in texts],
            dim=0,
        )
        return self.encoder(toks)


from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch
from tinygrad.nn import Linear, LayerNorm, Embedding, Conv2d

from typing import List, Optional, Union, Tuple, Dict
from abc import ABC, abstractmethod
from functools import lru_cache
from PIL import Image
import numpy as np
import re, gzip


@lru_cache()
def default_bpe():
    # Clip tokenizer, taken from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py (MIT license)
    return fetch(
        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz",
        "bpe_simple_vocab_16e6.txt.gz",
    )


class Tokenizer:
    """
    Namespace for CLIP Text Tokenizer components.
    """

    @staticmethod
    def get_pairs(word):
        """
        Return set of symbol pairs in a word.
        Word is represented as tuple of symbols (symbols being variable-length strings).
        """
        return set(zip(word, word[1:]))

    @staticmethod
    def whitespace_clean(text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    @staticmethod
    def bytes_to_unicode():
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a significant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    class ClipTokenizer:
        def __init__(self):
            self.byte_encoder = Tokenizer.bytes_to_unicode()
            merges = gzip.open(default_bpe()).read().decode("utf-8").split("\n")
            merges = merges[1 : 49152 - 256 - 2 + 1]
            merges = [tuple(merge.split()) for merge in merges]
            vocab = list(Tokenizer.bytes_to_unicode().values())
            vocab = vocab + [v + "</w>" for v in vocab]
            for merge in merges:
                vocab.append("".join(merge))
            vocab.extend(["<|startoftext|>", "<|endoftext|>"])
            self.encoder = dict(zip(vocab, range(len(vocab))))
            self.bpe_ranks = dict(zip(merges, range(len(merges))))
            self.cache = {
                "<|startoftext|>": "<|startoftext|>",
                "<|endoftext|>": "<|endoftext|>",
            }
            self.pat = re.compile(
                r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+""",
                re.IGNORECASE,
            )

        def bpe(self, token):
            if token in self.cache:
                return self.cache[token]
            word = tuple(token[:-1]) + (token[-1] + "</w>",)
            pairs = Tokenizer.get_pairs(word)

            if not pairs:
                return token + "</w>"

            while True:
                bigram = min(
                    pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
                )
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i < len(word):
                    try:
                        j = word.index(first, i)
                        new_word.extend(word[i:j])
                        i = j
                    except Exception:
                        new_word.extend(word[i:])
                        break

                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                word = new_word
                if len(word) == 1:
                    break
                pairs = Tokenizer.get_pairs(word)
            word = " ".join(word)
            self.cache[token] = word
            return word

        def encode(self, text: str, pad_with_zeros: bool = False) -> List[int]:
            bpe_tokens: List[int] = []
            text = Tokenizer.whitespace_clean(text.strip()).lower()
            for token in re.findall(self.pat, text):
                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                bpe_tokens.extend(
                    self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
                )
            # Truncation, keeping two slots for start and end tokens.
            if len(bpe_tokens) > 75:
                bpe_tokens = bpe_tokens[:75]
            return (
                [49406]
                + bpe_tokens
                + [49407]
                + ([0] if pad_with_zeros else [49407]) * (77 - len(bpe_tokens) - 2)
            )


class Embedder(ABC):
    input_key: str

    @abstractmethod
    def __call__(
        self, x: Union[str, List[str], Tensor]
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        pass


class Closed:
    """
    Namespace for OpenAI CLIP model components.
    """

    class ClipMlp:
        def __init__(self):
            self.fc1 = Linear(768, 3072)
            self.fc2 = Linear(3072, 768)

        def __call__(self, h: Tensor) -> Tensor:
            h = self.fc1(h)
            h = h.quick_gelu()
            h = self.fc2(h)
            return h

    class ClipAttention:
        def __init__(self):
            self.embed_dim = 768
            self.num_heads = 12
            self.head_dim = self.embed_dim // self.num_heads
            self.k_proj = Linear(self.embed_dim, self.embed_dim)
            self.v_proj = Linear(self.embed_dim, self.embed_dim)
            self.q_proj = Linear(self.embed_dim, self.embed_dim)
            self.out_proj = Linear(self.embed_dim, self.embed_dim)

        def __call__(
            self, hidden_states: Tensor, causal_attention_mask: Tensor
        ) -> Tensor:
            bsz, tgt_len, embed_dim = hidden_states.shape
            q, k, v = (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            )
            q, k, v = [
                x.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
                for x in (q, k, v)
            ]
            attn_output = Tensor.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_attention_mask
            )
            return self.out_proj(
                attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
            )

    class ClipEncoderLayer:
        def __init__(self):
            self.self_attn = Closed.ClipAttention()
            self.layer_norm1 = LayerNorm(768)
            self.mlp = Closed.ClipMlp()
            self.layer_norm2 = LayerNorm(768)

        def __call__(
            self, hidden_states: Tensor, causal_attention_mask: Tensor
        ) -> Tensor:
            residual = hidden_states
            hidden_states = self.layer_norm1(hidden_states)
            hidden_states = self.self_attn(hidden_states, causal_attention_mask)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states

    class ClipTextEmbeddings:
        def __init__(self):
            self.token_embedding = Embedding(49408, 768)
            self.position_embedding = Embedding(77, 768)

        def __call__(self, input_ids: Tensor, position_ids: Tensor) -> Tensor:
            return self.token_embedding(input_ids) + self.position_embedding(
                position_ids
            )

    class ClipEncoder:
        def __init__(self, layer_count: int = 12):
            self.layers = [Closed.ClipEncoderLayer() for _ in range(layer_count)]

        def __call__(
            self,
            x: Tensor,
            causal_attention_mask: Tensor,
            ret_layer_idx: Optional[int] = None,
        ) -> Tensor:
            # the indexing of layers is NOT off by 1, the original code considers the "input" as the first hidden state
            layers = (
                self.layers if ret_layer_idx is None else self.layers[:ret_layer_idx]
            )
            for l in layers:
                x = l(x, causal_attention_mask)
            return x

    class ClipTextTransformer:
        def __init__(self, ret_layer_idx: Optional[int] = None):
            self.embeddings = Closed.ClipTextEmbeddings()
            self.encoder = Closed.ClipEncoder()
            self.final_layer_norm = LayerNorm(768)
            self.ret_layer_idx = ret_layer_idx

        def __call__(self, input_ids: Tensor) -> Tensor:
            x = self.embeddings(
                input_ids, Tensor.arange(input_ids.shape[1]).reshape(1, -1)
            )
            x = self.encoder(
                x,
                Tensor.full((1, 1, 77, 77), float("-inf")).triu(1),
                self.ret_layer_idx,
            )
            return self.final_layer_norm(x) if (self.ret_layer_idx is None) else x

    class ClipTextModel:
        def __init__(self, ret_layer_idx: Optional[int]):
            self.text_model = Closed.ClipTextTransformer(ret_layer_idx=ret_layer_idx)


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L331
class FrozenClosedClipEmbedder(Embedder):
    def __init__(self, ret_layer_idx: Optional[int] = None):
        self.tokenizer = Tokenizer.ClipTokenizer()
        self.transformer = Closed.ClipTextModel(ret_layer_idx)
        self.input_key = "txt"

    def __call__(
        self, texts: Union[str, List[str], Tensor]
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(
            texts, (list, tuple)
        ), f"expected list of strings, got {type(texts).__name__}"
        tokens = Tensor.cat(
            *[Tensor(self.tokenizer.encode(text)) for text in texts], dim=0
        )
        return self.transformer.text_model(tokens.reshape(len(texts), -1))


class Open:
    """
    Namespace for OpenCLIP model components.
    """

    class MultiheadAttention:
        def __init__(self, dims: int, n_heads: int):
            self.dims = dims
            self.n_heads = n_heads
            self.d_head = self.dims // self.n_heads

            self.in_proj_bias = Tensor.empty(3 * dims)
            self.in_proj_weight = Tensor.empty(3 * dims, dims)
            self.out_proj = Linear(dims, dims)

        def __call__(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
            T, B, C = x.shape

            proj = x.linear(self.in_proj_weight.T, self.in_proj_bias)
            proj = proj.unflatten(-1, (3, C)).unsqueeze(0).transpose(0, -2)

            q, k, v = [
                y.reshape(T, B * self.n_heads, self.d_head)
                .transpose(0, 1)
                .reshape(B, self.n_heads, T, self.d_head)
                for y in proj.chunk(3)
            ]

            attn_output = Tensor.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            )
            attn_output = attn_output.permute(2, 0, 1, 3).reshape(T * B, C)

            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.reshape(T, B, C)

            return attn_output

    class Mlp:
        def __init__(self, dims, hidden_dims):
            self.c_fc = Linear(dims, hidden_dims)
            self.c_proj = Linear(hidden_dims, dims)

        def __call__(self, x: Tensor) -> Tensor:
            return x.sequential([self.c_fc, Tensor.gelu, self.c_proj])

    # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L210
    class ResidualAttentionBlock:
        def __init__(self, dims: int, n_heads: int, mlp_ratio: float):
            self.ln_1 = LayerNorm(dims)
            self.attn = Open.MultiheadAttention(dims, n_heads)

            self.ln_2 = LayerNorm(dims)
            self.mlp = Open.Mlp(dims, int(dims * mlp_ratio))

        def __call__(
            self, x: Tensor, attn_mask: Optional[Tensor] = None, transpose: bool = False
        ) -> Tensor:
            q_x = self.ln_1(x)
            attn_out = self.attn(
                q_x.transpose(0, 1) if transpose else q_x, attn_mask=attn_mask
            )
            attn_out = attn_out.transpose(0, 1) if transpose else attn_out
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x

    # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L317
    class ClipTransformer:
        def __init__(
            self, dims: int, layers: int, n_heads: int, mlp_ratio: float = 4.0
        ):
            self.resblocks = [
                Open.ResidualAttentionBlock(dims, n_heads, mlp_ratio)
                for _ in range(layers)
            ]

        def __call__(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
            for r in self.resblocks:
                x = r(x, attn_mask=attn_mask, transpose=True)
            return x

    # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/model.py#L220
    # https://github.com/mlfoundations/open_clip/blob/58e4e39aaabc6040839b0d2a7e8bf20979e4558a/src/open_clip/transformer.py#L661
    class ClipTextTransformer:
        def __init__(
            self,
            width: int,
            n_heads: int,
            layers: int,
            vocab_size: int = 49408,
            ctx_length: int = 77,
        ):
            self.token_embedding = Embedding(vocab_size, width)
            self.positional_embedding = Tensor.empty(ctx_length, width)
            self.transformer = Open.ClipTransformer(width, layers, n_heads)
            self.ln_final = LayerNorm(width)
            self.text_projection = Tensor.empty(width, width)
            self.attn_mask = Tensor.full((77, 77), float("-inf")).triu(1).realize()

        def __call__(self, text: Tensor) -> Tensor:
            seq_len = text.shape[1]

            x = self.token_embedding(text)
            x = x + self.positional_embedding[:seq_len]
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = self.ln_final(x)

            pooled = x[:, text.argmax(dim=-1)] @ self.text_projection
            return pooled

    class ClipVisionTransformer:
        def __init__(
            self, width: int, layers: int, d_head: int, image_size: int, patch_size: int
        ):
            grid_size = image_size // patch_size
            n_heads = width // d_head
            assert n_heads * d_head == width

            self.conv1 = Conv2d(
                3, width, kernel_size=patch_size, stride=patch_size, bias=False
            )

            self.class_embedding = Tensor.empty(width)
            self.positional_embedding = Tensor.empty(grid_size * grid_size + 1, width)
            self.transformer = Open.ClipTransformer(width, layers, n_heads)
            self.ln_pre = LayerNorm(width)
            self.ln_post = LayerNorm(width)
            self.proj = Tensor.empty(width, 1024)

        def __call__(self, x: Tensor) -> Tensor:
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            x = (
                self.class_embedding.reshape(1, 1, -1)
                .expand(x.shape[0], 1, -1)
                .cat(x, dim=1)
            )
            x = x + self.positional_embedding

            x = self.ln_pre(x)
            x = self.transformer(x)
            x = self.ln_post(x)

            pooled = x[:, 0] @ self.proj
            return pooled


# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L396
# https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/modules/encoders/modules.py#L498
class FrozenOpenClipEmbedder(Embedder):
    def __init__(
        self,
        dims: int,
        n_heads: int,
        layers: int,
        return_pooled: bool,
        ln_penultimate: bool = False,
    ):
        self.tokenizer = Tokenizer.ClipTokenizer()
        self.model = Open.ClipTextTransformer(dims, n_heads, layers)
        self.return_pooled = return_pooled
        self.input_key = "txt"
        self.ln_penultimate = ln_penultimate

    def tokenize(self, text: str, device: Optional[str] = None) -> Tensor:
        return Tensor(
            self.tokenizer.encode(text, pad_with_zeros=True),
            dtype=dtypes.int64,
            device=device,
        ).reshape(1, -1)

    def text_transformer_forward(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        for r in self.model.transformer.resblocks:
            x, penultimate = r(x, attn_mask=attn_mask), x
        return x.permute(1, 0, 2), penultimate.permute(1, 0, 2)

    def embed_tokens(self, tokens: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        x = (
            self.model.token_embedding(tokens)
            .add(self.model.positional_embedding)
            .permute(1, 0, 2)
        )
        x, penultimate = self.text_transformer_forward(
            x, attn_mask=self.model.attn_mask
        )

        if self.ln_penultimate:
            penultimate = self.model.ln_final(penultimate)

        if self.return_pooled:
            x = self.model.ln_final(x)
            index = (
                tokens.argmax(axis=-1)
                .reshape(-1, 1, 1)
                .expand(x.shape[0], 1, x.shape[-1])
            )
            pooled = x.gather(1, index).squeeze(1) @ self.model.text_projection
            return penultimate, pooled
        else:
            return penultimate

    def __call__(
        self, texts: Union[str, List[str], Tensor]
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(
            texts, (list, tuple)
        ), f"expected list of strings, got {type(texts).__name__}"
        tokens = Tensor.cat(*[self.tokenize(text) for text in texts], dim=0)
        return self.embed_tokens(tokens)


clip_configs: Dict = {
    "ViT-H-14": {
        "dims": 1024,
        "vision_cfg": {
            "width": 1280,
            "layers": 32,
            "d_head": 80,
            "image_size": 224,
            "patch_size": 14,
        },
        "text_cfg": {
            "width": 1024,
            "n_heads": 16,
            "layers": 24,
            "ctx_length": 77,
            "vocab_size": 49408,
        },
        "return_pooled": False,
        "ln_penultimate": True,
    }
}


class OpenClipEncoder:
    def __init__(self, dims: int, text_cfg: Dict, vision_cfg: Dict, **_):
        self.visual = Open.ClipVisionTransformer(**vision_cfg)

        text = Open.ClipTextTransformer(**text_cfg)
        self.transformer = text.transformer
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection

        self.attn_mask = Tensor.full((77, 77), float("-inf")).triu(1).realize()
        self.mean = Tensor([0.48145466, 0.45782750, 0.40821073]).reshape(-1, 1, 1)
        self.std = Tensor([0.26862954, 0.26130258, 0.27577711]).reshape(-1, 1, 1)

    # TODO:
    # Should be doable in pure tinygrad, would just require some work and verification.
    # This is very desirable since it would allow for full generation->evaluation in a single JIT call.
    def prepare_image(self, image: Image.Image) -> Tensor:
        SIZE = 224
        w, h = image.size
        scale = min(SIZE / h, SIZE / w)
        image = image.resize(
            (max(int(w * scale), SIZE), max(int(h * scale), SIZE)),
            Image.Resampling.BICUBIC,
        )
        w, h = image.size
        if w > SIZE:
            left = (w - SIZE) // 2
            image = image.crop((left, left + SIZE, 0, SIZE))
        elif h > SIZE:
            top = (h - SIZE) // 2
            image = image.crop((0, SIZE, top, top + SIZE))

        x = Tensor(np.array(image.convert("RGB")))
        x = x.permute(2, 0, 1).cast(dtypes.float32) / 255.0
        return (x - self.mean) / self.std

    def encode_tokens(self, tokens: Tensor) -> Tensor:
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = x[:, tokens.argmax(axis=-1)]
        x = x @ self.text_projection
        return x

    def get_clip_score(self, tokens: Tensor, image: Tensor) -> Tensor:
        image_features: Tensor = self.visual(image)
        image_features /= (
            image_features.square().sum(-1, keepdim=True).sqrt()
        )  # Frobenius Norm

        text_features = self.encode_tokens(tokens)
        text_features /= (
            text_features.square().sum(-1, keepdim=True).sqrt()
        )  # Frobenius Norm

        return (image_features * text_features).sum(axis=-1)


configs: dict = {
    "flux": {
        "in_channels": 64,
        "vec_in_dim": 768,
        "context_in_dim": 4096,
        "hidden_size": 3072,
        "mlp_ratio": 4.0,
        "num_heads": 24,
        "depth": 19,
        "depth_single_blocks": 38,
        "axes_dim": [16, 56, 56],
        "theta": 10_000,
        "qkv_bias": True,
    },
    "ae": {
        "scale_factor": 0.3611,
        "shift_factor": 0.1159,
        "resolution": 256,
        "in_channels": 3,
        "ch": 128,
        "out_ch": 3,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "z_channels": 16,
    },
    "urls": {
        "flux-schnell": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
        "flux-dev": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.sft",
        "ae": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors",
        "T5_1_of_2": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00001-of-00002.safetensors",
        "T5_2_of_2": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder_2/model-00002-of-00002.safetensors",
        "T5_tokenizer": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/tokenizer_2/spiece.model",
        "clip": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_encoder/model.safetensors",
    },
}


def tensor_identity(x: Tensor) -> Tensor:
    return x


# https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = Tensor.scaled_dot_product_attention(q, k, v)
    x = x.rearrange("B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = (
        Tensor.arange(0, dim, 2, dtype=dtypes.float32, device=pos.device) / dim
    )  # NOTE: this is torch.float64 in reference implementation
    omega = 1.0 / (theta**scale)
    out = pos.unsqueeze(-1) * omega.unsqueeze(
        0
    )  # equivalent to Tensor.einsum("...n,d->...nd", pos, omega)

    out = Tensor.stack(
        Tensor.cos(out), -Tensor.sin(out), Tensor.sin(out), Tensor.cos(out), dim=-1
    )
    out = out.rearrange("b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).cast(xq.dtype), xk_out.reshape(*xk.shape).cast(
        xk.dtype
    )


# Conditioner
class ClipEmbedder(FrozenClosedClipEmbedder):
    def __call__(
        self, texts: Union[str, List[str], Tensor]
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(
            texts, (list, tuple)
        ), f"expected list of strings, got {type(texts).__name__}"
        tokens = Tensor.cat(
            *[Tensor(self.tokenizer.encode(text)) for text in texts], dim=0
        )
        return self.transformer.text_model(tokens.reshape(len(texts), -1))[
            :, tokens.argmax(-1)
        ]


# https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py
class AttnBlock:
    def __init__(self, in_channels: int):
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.rearrange("b c h w -> b 1 (h w) c")
        k = k.rearrange("b c h w -> b 1 (h w) c")
        v = v.rearrange("b c h w -> b 1 (h w) c")
        h_ = Tensor.scaled_dot_product_attention(q, k, v)

        return h_.rearrange("b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def __call__(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock:
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def __call__(self, x):
        h = x
        h = self.norm1(h).swish()
        h = self.conv1(h)

        h = self.norm2(h).swish()
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Upsample:
    def __init__(self, in_channels: int):
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def __call__(self, x: Tensor):
        x = Tensor.interpolate(
            x, size=(x.shape[-2] * 2, x.shape[-1] * 2), mode="nearest"
        )
        x = self.conv(x)
        return x


class Decoder:
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = {}
        self.mid["block_1"] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid["attn_1"] = AttnBlock(block_in)
        self.mid["block_2"] = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = {}
            up["block"] = block
            if i_level != 0:
                up["upsample"] = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level]["block"][i_block](h)
            if i_level != 0:
                h = self.up[i_level]["upsample"](h)

        # end
        h = self.norm_out(h).swish()
        h = self.conv_out(h)
        return h


class AutoEncoder:
    def __init__(self, scale_factor: float, shift_factor: float, **decoder_params):
        self.decoder = Decoder(**decoder_params)

        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)


# https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
class EmbedND:
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = Tensor.cat(
            *[rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def timestep_embedding(
    t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0
):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = Tensor.exp(
        -math.log(max_period) * Tensor.arange(0, stop=half, dtype=dtypes.float32) / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = Tensor.cat(Tensor.cos(args), Tensor.sin(args), dim=-1)
    if dim % 2:
        embedding = Tensor.cat(
            *[embedding, Tensor.zeros_like(embedding[:, :1])], dim=-1
        )
    if Tensor.is_floating_point(t):
        embedding = embedding.cast(t.dtype)
    return embedding


class MLPEmbedder:
    def __init__(self, in_dim: int, hidden_dim: int):
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: Tensor) -> Tensor:
        return self.out_layer(self.in_layer(x).silu())


class QKNorm:
    def __init__(self, dim: int):
        self.query_norm = nn.RMSNorm(dim)
        self.key_norm = nn.RMSNorm(dim)

    def __call__(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class SelfAttention:
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation:
    def __init__(self, dim: int, double: bool):
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(vec.silu())[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock:
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False
    ):
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = [
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            Tensor.gelu,
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        ]

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = [
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            Tensor.gelu,
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        ]

    def __call__(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.rearrange(
            "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.rearrange(
            "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)

        # run actual attention
        q = Tensor.cat(txt_q, img_q, dim=2)
        k = Tensor.cat(txt_k, img_k, dim=2)
        v = Tensor.cat(txt_v, img_v, dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * (
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        ).sequential(self.img_mlp)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * (
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        ).sequential(self.txt_mlp)
        return img, txt


class SingleStreamBlock:
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = Tensor.gelu
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = Tensor.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(Tensor.cat(attn, self.mlp_act(mlp), dim=2))
        return x + mod.gate * output


class LastLayer:
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = [
            Tensor.silu,
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        ]

    def __call__(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = vec.sequential(self.adaLN_modulation).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


# https://github.com/black-forest-labs/flux/blob/main/src/flux/model.py
class Flux:
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(
        self,
        guidance_embed: bool,
        in_channels: int = 64,
        vec_in_dim: int = 768,
        context_in_dim: int = 4096,
        hidden_size: int = 3072,
        mlp_ratio: float = 4.0,
        num_heads: int = 24,
        depth: int = 19,
        depth_single_blocks: int = 38,
        axes_dim: list[int] = [16, 56, 56],
        theta: int = 10_000,
        qkv_bias: bool = True,
    ):

        self.guidance_embed = guidance_embed
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if guidance_embed
            else tensor_identity
        )
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )
            for _ in range(depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth_single_blocks)
        ]

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def __call__(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = Tensor.cat(txt_ids, img_ids, dim=1)
        pe = self.pe_embedder(ids)
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = Tensor.cat(txt, img, dim=1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


# https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py
class Util:
    def load_flow_model(name: str):
        # Loading Flux
        print("Init model")
        model = Flux(guidance_embed=(name != "flux-schnell"), **configs["flux"])
        state_dict = {
            k.replace("scale", "weight"): v
            for k, v in safe_load(fetch(configs["urls"][name])).items()
        }
        load_state_dict(model, state_dict)
        return model

    def load_T5(name: str, max_length: int = 512):
        # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
        print("Init T5")
        T5 = T5Embedder(max_length, fetch(configs["urls"]["T5_tokenizer"]))
        pt_1 = fetch(configs["urls"]["T5_1_of_2"])
        pt_2 = fetch(configs["urls"]["T5_2_of_2"])
        load_state_dict(T5.encoder, safe_load(pt_1) | safe_load(pt_2), strict=False)
        return T5

    def load_clip(name: str):
        print("Init Clip")
        clip = ClipEmbedder()
        load_state_dict(clip.transformer, safe_load(fetch(configs["urls"]["clip"])))
        return clip

    def load_ae(name: str) -> AutoEncoder:
        # Loading the autoencoder
        print("Init AE")
        ae = AutoEncoder(**configs["ae"])
        load_state_dict(ae, safe_load(fetch(configs["urls"]["ae"])))
        return ae


# https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py
def get_noise(
    num_samples: int, height: int, width: int, dtype: str, seed: int
) -> Tensor:
    Tensor.manual_seed(seed)
    return Tensor.randn(
        num_samples,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
    )


def prepare(
    T5: T5Embedder, clip: ClipEmbedder, img: Tensor, prompt: str | list[str]
) -> dict[str, Tensor]:
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = img.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = img.expand((bs, *img.shape[1:]))

    img_ids = Tensor.zeros(h // 2, w // 2, 3).contiguous()
    img_ids[..., 1] = img_ids[..., 1] + Tensor.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + Tensor.arange(w // 2)[None, :]
    img_ids = img_ids.rearrange("h w c -> 1 (h w) c")
    img_ids = img_ids.expand((bs, *img_ids.shape[1:]))

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = T5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = txt.expand((bs, *txt.shape[1:]))
    txt_ids = Tensor.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = vec.expand((bs, *vec.shape[1:]))

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    step_size = -1.0 / num_steps
    timesteps = Tensor.arange(1, 0 + step_size, step_size)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


@TinyJit
def run(model, *args):
    return model(*args).realize()


def denoise(
    model,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
) -> Tensor:
    # this is ignored for schnell
    guidance_vec = Tensor((guidance,), device=img.device, dtype=img.dtype).expand(
        (img.shape[0],)
    )
    for t_curr, t_prev in tqdm(list(zip(timesteps[:-1], timesteps[1:])), "Denoising"):
        t_vec = Tensor((t_curr,), device=img.device, dtype=img.dtype).expand(
            (img.shape[0],)
        )
        pred = run(model, img, img_ids, txt, txt_ids, t_vec, vec, guidance_vec)
        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return x.rearrange(
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


# https://github.com/black-forest-labs/flux/blob/main/src/flux/cli.py
if __name__ == "__main__":
    default_prompt = "bananas and a can of coke"
    parser = argparse.ArgumentParser(
        description="Run Flux.1", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--name", type=str, default="flux-schnell", help="Name of the model to load"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="width of the sample in pixels (should be a multiple of 16)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="height of the sample in pixels (should be a multiple of 16)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Set a seed for sampling"
    )
    parser.add_argument(
        "--prompt", type=str, default=default_prompt, help="Prompt used for sampling"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=Path(tempfile.gettempdir()) / "rendered.png",
        help="Output filename",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="number of sampling steps (default 4 for schnell, 50 for guidance distilled)",
    )  # noqa:E501
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="guidance value used for guidance distillation",
    )
    parser.add_argument("--offload", type=bool, default=False, help="offload to cpu")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="output directory"
    )
    args = parser.parse_args()

    if args.name not in ["flux-schnell", "flux-dev"]:
        raise ValueError(
            f"Got unknown model name: {args.name}, chose from flux-schnell and flux-dev"
        )

    if args.num_steps is None:
        args.num_steps = 4 if args.name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (args.height // 16)
    width = 16 * (args.width // 16)

    if args.seed is None:
        args.seed = Tensor._seed
    print(f"Generating with seed {args.seed}:\n{args.prompt}")
    t0 = time.perf_counter()

    # prepare input
    x = get_noise(1, height, width, dtype="bfloat16", seed=args.seed)

    # load text embedders
    T5 = Util.load_T5(args.name, max_length=256 if args.name == "flux-schnell" else 512)
    clip = Util.load_clip(args.name)

    # embed text to get inputs for model
    inp = prepare(T5, clip, x, prompt=args.prompt)
    for v in inp.values():
        v.realize()
    timesteps = get_schedule(
        args.num_steps, inp["img"].shape[1], shift=(args.name != "flux-schnell")
    )

    # done with text embedders
    del T5, clip

    # load model
    model = Util.load_flow_model(args.name)

    # denoise initial noise
    x = denoise(model, **inp, timesteps=timesteps, guidance=args.guidance)

    # done with model
    del model, run

    # load autoencoder
    ae = Util.load_ae(args.name)

    # decode latents to pixel space
    x = unpack(x.float(), height, width).realize()
    x = ae.decode(x).realize()

    t1 = time.perf_counter()

    # fn = output_name.format(idx=idx)
    print(f"Done in {t1 - t0:.1f}s. Saving {args.out}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = x[0].rearrange("c h w -> h w c")
    x = (127.5 * (x + 1.0)).cast("uint8")

    img = Image.fromarray(x.numpy())

    img.save(args.out)

    # validation!
    if (
        args.prompt == default_prompt
        and args.name == "flux-schnell"
        and args.seed == 0
        and args.width == args.height == 512
    ):
        ref_image = Tensor(np.array(Image.open("examples/flux1_seed0.png")))
        distance = (
            (
                (
                    (x.cast(dtypes.float) - ref_image.cast(dtypes.float))
                    / ref_image.max()
                )
                ** 2
            )
            .mean()
            .item()
        )
        assert distance < 4e-3, colored(f"validation failed with {distance=}", "red")
        print(colored(f"output validated with {distance=}", "green"))
