from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import einsum, nn

from tortoise.gpt import CausalSelfAttention


def masked_mean(t, mask):
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


@dataclass
class CLVPConfig:
    n_embd: int = 768
    n_head: int = 12
    num_text_tokens: int = 256
    num_speech_tokens: int = 8192
    text_enc_depth: int = 20
    speech_enc_depth: int = 20
    text_mask_percentage: float = 0.2
    voice_mask_percentage: float = 0.2
    wav_token_compression: int = 1024
    text_seq_len: int = 350
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    block_size: int = 1024


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class Block(nn.Module):
    """a Transformer block with scale and geglu"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.scale_attn = nn.Parameter(torch.zeros(1, 1, config.n_embd).fill_(0.1))
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(config.n_embd, 4 * config.n_embd * 2)),
                    ("act", GEGLU()),
                    ("dropout", nn.Dropout(config.resid_pdrop)),
                    ("c_proj", nn.Linear(config.n_embd * 4, config.n_embd)),
                ]
            )
        )
        self.scale_mlp = nn.Parameter(torch.zeros(1, 1, config.n_embd).fill_(0.1))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x)) * self.scale_attn
        x = x + self.mlp(self.ln_2(x)) * self.scale_mlp
        return x


class Encoder(nn.Module):
    """Just a bunch of Blocks"""

    def __init__(self, config, n_layer):
        super().__init__()
        self.layers = nn.ModuleList([Block(config) for _ in range(n_layer)])

    def forward(self, inputs, mask=None):
        x = inputs
        for layer in self.layers:
            x = layer(x, mask)
        return x


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

        self.text_enc = Encoder(config, config.text_enc_depth)
        self.speech_enc = Encoder(config, config.speech_enc_depth)

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.text_mask_percentage = config.text_mask_percentage
        self.voice_mask_percentage = config.voice_mask_percentage
        self.wav_token_compression = config.wav_token_compression

        self.text_pos_emb = nn.Embedding(config.text_seq_len, config.n_embd)
        self.speech_pos_emb = nn.Embedding(config.num_speech_tokens, config.n_embd)

    def forward(self, text, speech_tokens, *, return_loss):
        batch_size, device = text.shape[0], text.device
        if self.training:
            text_mask = torch.rand_like(text.float()) > self.text_mask_percentage
            voice_mask = torch.rand_like(speech_tokens.float()) > self.voice_mask_percentage
        else:
            text_mask = torch.ones_like(text.float()).bool()
            voice_mask = torch.ones_like(speech_tokens.float()).bool()

        text_emb = self.text_emb(text)
        speech_emb = self.speech_emb(speech_tokens)

        text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))
        speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=device))

        enc_text = self.text_enc(text_emb, mask=text_mask)
        enc_speech = self.speech_enc(speech_emb, mask=voice_mask)

        text_latents = masked_mean(enc_text, text_mask)
        speech_latents = masked_mean(enc_speech, voice_mask)

        text_latents = self.to_text_latent(text_latents)
        speech_latents = self.to_speech_latent(speech_latents)

        text_latents = F.normalize(text_latents, p=2, dim=-1)
        speech_latents = F.normalize(speech_latents, p=2, dim=-1)

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum("n d, n d -> n", text_latents, speech_latents) * temp
            return sim

        sim = einsum("i d, j d -> i j", text_latents, speech_latents) * temp
        labels = torch.arange(batch_size, device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


if __name__ == "__main__":
    _config = CLVPConfig(text_mask_percentage=0.2, voice_mask_percentage=0.2)
    clip = CLVP(_config)
    loss = clip(
        torch.randint(0, 256, (2, 120)),
        torch.randint(0, 8192, (2, 250)),
        return_loss=True,
    )
    print(loss)
    nonloss = clip(
        torch.randint(0, 256, (2, 120)),
        torch.randint(0, 8192, (2, 250)),
        return_loss=False,
    )
    print(nonloss)
