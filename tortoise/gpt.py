# copied from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

# TODO adapt some stuff from nanoGPT
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, softmax

if TYPE_CHECKING:
    from tortoise.autoregressive import TortoiseConfig


@dataclass
class GPTConfig:
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 30
    block_size: int = 1024
    vocab_size: int = 256
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1

    @classmethod
    def from_tortoise_config(cls, config: TortoiseConfig) -> "GPTConfig":
        return cls(
            config.n_embd,
            config.n_head,
            config.n_layer,
            block_size=config.max_speech_tokens + config.max_text_tokens + config.max_conditioning_inputs + 4,
            vocab_size=config.max_speech_tokens,
        )


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# TODO: use torch builtin attention
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
        att = softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            OrderedDict(
                {
                    "c_fc": nn.Linear(config.n_embd, 4 * config.n_embd),
                    "act": NewGELU(),
                    "c_proj": nn.Linear(4 * config.n_embd, config.n_embd),
                    "dropout": nn.Dropout(config.resid_pdrop),
                }
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model without input embedding"""

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        # n_params = sum(p.numel() for p in self.transformer.parameters())
        # print(f"number of parameters: {n_params / 1e6:.2f}M")

    def forward(
        self, tok_emb: Float[Tensor, "b t d"]
    ) -> Tuple[Float[Tensor, "b t d"], Float[Tensor, "b t vocab_size"], Optional[Float[Tensor, "1"]]]:
        b, t, d = tok_emb.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # forward the GPT model itself
        x = self.drop(tok_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


if __name__ == "__main__":
    gpt_config = GPTConfig()
    gpt = GPT(gpt_config)
    out = gpt(torch.randn((1, 23, gpt_config.n_embd)))
    print(out.shape)
