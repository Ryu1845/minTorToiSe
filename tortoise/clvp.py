from dataclasses import dataclass

import torch
import torch.nn as nn

from .gpt import CausalSelfAttention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class Block(nn.Module):
    """an Transformer block with scale and geglu"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.scale_attn = nn.Parameter(torch.zeros(1, 1, config.n_embd).fill_(0.1))
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=GEGELU(),
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward
        self.scale_mlp = nn.Parameter(torch.zeros(1, 1, config.n_embd).fill_(0.1))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) * self.scale_attn
        x = x + self.mlpf(self.ln_2(x)) * self.scale_mlp
        return x


@dataclass
class CLVPConfig:
    ...


class CLVP(nn.Module):
    """
    CLIP model retrofitted for performing contrastive evaluation between tokenized audio data and the corresponding
    transcribed text.
    """

    def __init__(self, config: CLVPConfig):
        super().__init__()
        self.text_emb = nn.Embedding(config.num_text_tokens, config.n_embd)
        self.to_text_latent = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.speech_emb = nn.Embedding(config.num_speech_tokens, config.n_embd)
        self.to_speech_latent = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.text_enc = nn.Sequential(*(Block(config) for _ in range(config.text_enc_depth)))
        self.speech_enc = nn.Sequential(*(Block(config) for _ in range(config.speech_enc_depth)))

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.text_mask_percentage = config.text_mask_percentage
        self.voice_mask_percentage = voice_mask_percentage
        self.wav_token_compression = wav_token_compression

        self.text_pos_emb = nn.Embedding(config.text_seq_len, config.n_embd)
        self.speech_pos_emb = nn.Embedding(config.num_speech_tokens, config.n_embd)
