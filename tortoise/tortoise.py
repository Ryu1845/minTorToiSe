"""
The TorToiSe autoregressive model
"""
from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn, Tensor
from torch.nn.functional import pad, cross_entropy

from .gpt import GPT, GPTConfig


@dataclass
class TortoiseConfig:
    """
    Args:
        n_layer: Number of layers in transformer stack.
        n_embd: Operating dimensions of the transformer
        n_head: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
        max_text_tokens: Maximum number of text tokens that will be encountered by model.
        max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
        max_conditioning_inputs: Maximum number of conditioning inputs provided to the model.
            If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
        mel_length_compression: The factor between <number_input_samples> and <mel_tokens>.
            Used to compute MEL token padding given wav input length.
        n_text_token:
        start_text_token:
        stop_text_token:
        n_mel_token:
        start_mel_token:
        stop_mel_token:
        train_solo_embeddings:
        use_mel_tokens_as_input:
        checkpointing:
    """

    n_layer: int = 30
    n_embd: int = 1024
    n_head: int = 16
    max_text_tokens: int = 402
    max_mel_tokens: int = 604
    max_conditioning_inputs: int = 2
    mel_length_compression: int = 1024
    n_text_token: int = 255
    start_text_token: int = 255
    stop_text_token: int = 0
    n_mel_token: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    train_solo_embeddings: bool = False
    use_mel_tokens_as_input: bool = True
    checkpointing: bool = False
    n_type: int = 1


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, seq_len: int, n_embd: int, init: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(seq_len, n_embd)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x: Tensor):
        seq_len = x.shape[1]
        return self.emb(torch.arange(0, seq_len, device=x.device))


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the 1d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, config):
        super().__init__()
        self.norm = nn.GroupNorm(32, config.n_embd)
        self.qkv = nn.Conv1d(config.n_embd, config.n_embd * 3, 1)
        self.attention = nn.MultiheadAttention(config.n_embd, config.n_head)
        self.proj_out = nn.Conv1d(config.n_embd, config.n_embd, kernel_size=1)

    def forward(self, x: Float[Tensor, "n c l"], mask: Optional[Tensor] = None):
        qkv = self.qkv(self.norm(x))  # [n c l] -> [n c*3 l]
        q, k, v = rearrange(qkv, "n c l -> n l c").chunk(3, dim=-1)  # [n c*3 l] -> 3 of [n l c]
        h, _ = self.attention(q, k, v, attn_mask=mask)
        h = rearrange(h, "n l c -> n c l")
        h = self.proj_out(h)  # [n c l] -> [n c l]
        return x + h


class ConditioningEncoder(nn.Module):
    def __init__(self, config, spec_dim: int = 80):
        super().__init__()
        self.init = nn.Conv1d(spec_dim, config.n_embd, kernel_size=1)
        self.attn = nn.Sequential(*(AttentionBlock(config) for _ in range(6)))

    def forward(self, speech: Float[Tensor, "batch spec_d length"]):
        out = self.init(speech)  # [n spec_d l] -> [n c l]
        out = self.attn(out)
        return out[:, :, 0]  # [n c l] -> [n c] (the first element)


class Tortoise(nn.Module):
    def __init__(self, config: TortoiseConfig):
        super().__init__()
        self.start_text_token = config.start_text_token
        self.stop_text_token = config.stop_text_token
        self.start_mel_token = config.start_mel_token
        self.stop_mel_token = config.stop_mel_token

        self.embed_text = nn.Embedding(config.n_text_token * config.n_type + 1, config.n_embd)
        self.embed_mel = nn.Embedding(config.n_mel_token * config.n_type + 1, config.n_embd)
        self.embed_pos_text = LearnedPositionEmbedding(config.max_text_tokens + 2, config.n_embd)
        self.embed_pos_mel = LearnedPositionEmbedding(
            config.max_mel_tokens + config.max_conditioning_inputs + 2, config.n_embd
        )

        self.transformer = GPT(GPTConfig.from_tortoise_config(config))

        self.final_norm = nn.LayerNorm(config.n_embd)

        self.text_head = nn.Linear(config.n_embd, config.n_text_token * config.n_type + 1)
        self.mel_head = nn.Linear(config.n_embd, config.n_mel_token * config.n_type + 1)

        self.embed_text.weight.data.normal_(mean=0.0, std=0.02)

    # TODO jaxtype
    def forward(self, text_inputs: Tensor, mel_inputs: Tensor, speech_conditioning_latent: Tensor):
        text_inputs = pad(text_inputs, (0, 1), value=self.stop_text_token)
        mel_inputs = pad(mel_inputs, (0, 1), value=self.stop_mel_token)

        condition = speech_conditioning_latent.unsqueeze(1)

        # build aligned inputs and targets
        text_targets = pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = pad(text_inputs, (1, 0), value=self.start_text_token)
        mel_targets = pad(mel_inputs, (0, 1), value=self.stop_mel_token)
        mel_inputs = pad(mel_inputs, (1, 0), value=self.start_mel_token)

        text_emb = self.embed_text(text_inputs) + self.embed_pos_text(text_inputs)
        # TODO investigate raw mels
        mel_emb = self.embed_mel(mel_inputs) + self.embed_pos_mel(mel_inputs)
        emb = torch.cat([condition, text_emb, mel_emb], dim=1)
        enc = self.final_norm(self.transformer(emb)[0])
        text_logits = self.text_head(enc[:, : text_emb.shape[1]]).permute(0, 2, 1)
        mel_logits = self.mel_head(enc[:, -mel_emb.shape[1] :]).permute(0, 2, 1)
        # TODO check return latent

        loss_text = cross_entropy(text_logits, text_targets.long())
        loss_mel = cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits


if __name__ == '__main__':
    torch.set_default_device("cuda")
    t_config = TortoiseConfig()
    conditioning_encoder = ConditioningEncoder(t_config, spec_dim=80)
    conditioning_latent = conditioning_encoder(speech=torch.randn(1, 80, 3272))
    print(conditioning_latent.shape)

    tortoise = Tortoise(t_config).eval()
    l_text, l_mel, mel_log = tortoise(
        text_inputs=torch.randint(high=120, size=(1, 250)),
        mel_inputs=torch.randint(high=8192, size=(1, 250)),
        speech_conditioning_latent=conditioning_latent,
    )
    print(l_text)
