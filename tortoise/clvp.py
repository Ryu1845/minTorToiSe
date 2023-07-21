import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import einsum, nn
from torch.nn.functional import cross_entropy, gelu, normalize, softmax


def masked_mean(t, mask):
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


@dataclass
class CLVPConfig:
    n_embd: int = 768
    n_head: int = 24  # TODO: figure out why this is 2x more than it should
    n_text_token: int = 256
    n_speech_token: int = 8192
    text_enc_depth: int = 20
    speech_enc_depth: int = 20
    wav_token_compression: int = 1024
    text_seq_len: int = 350
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    block_size: int = 1024


class GLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels * 2)

    def forward(self, x):
        x, gates = self.proj(x).chunk(2, dim=-1)
        return x * gelu(gates)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=1e-8) * self.weight


def apply_rope(x: torch.Tensor, freqs: torch.Tensor):
    seq_len = x.shape[2]
    freqs = freqs[:, :, -seq_len:]

    # rotate half
    x_rotated = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x_rotated.unbind(dim=-2)
    x_rotated = torch.cat((-x2, x1), dim=-1)

    return (x * freqs.cos()) + (x_rotated * freqs.sin())


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor, mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_head)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        q, k, v = (apply_rope(mat, rope_freqs) for mat in (q, k, v))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            mask = rearrange(mask, "b j -> b () () j")
            att = att.masked_fill(mask, -1e10)
        att = softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """a Transformer block with GLU"""

    def __init__(self, config):
        super().__init__()
        self.norm_attn = RMSNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.norm_mlp = RMSNorm(config.n_embd)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("act", GLU(config.n_embd, config.n_embd * 2)),
                    ("dropout", nn.Dropout(config.resid_pdrop)),
                    ("c_proj", nn.Linear(config.n_embd * 2, config.n_embd)),
                ]
            )
        )

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor, mask=None):
        x = x + self.attn(self.norm_attn(x), mask=mask, rope_freqs=rope_freqs)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Encoder(nn.Module):
    """Just a bunch of Blocks"""

    def __init__(self, config, n_layer: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(config) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)

    def forward(self, inputs: torch.Tensor, rope_freqs: torch.Tensor, mask=None):
        x = inputs
        for layer in self.layers:
            x = layer(x, mask=mask, rope_freqs=rope_freqs)
        x = self.norm(x)
        return x


class CLVP(nn.Module):
    """
    CLIP model retrofitted for performing contrastive evaluation between tokenized audio data and the corresponding
    transcribed text.
    """

    def __init__(self, config: CLVPConfig):
        super().__init__()
        self.text_emb = nn.Embedding(config.n_text_token, config.n_embd)
        self.to_text_latent = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.speech_emb = nn.Embedding(config.n_speech_token, config.n_embd)
        self.to_speech_latent = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.register_buffer(
            "rotary_inv_freq",
            1.0
            / (
                10_000
                ** (
                    torch.arange(start=0, end=config.n_embd // config.n_head, step=2).float()
                    / config.n_embd
                    // config.n_head
                )
            ),
        )
        t = torch.arange(config.text_seq_len).type_as(self.rotary_inv_freq)
        freqs = torch.einsum("i, j -> i j", t, self.rotary_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.freqs_text = rearrange(emb, "n d -> () () n d")

        t = torch.arange(config.n_speech_token).type_as(self.rotary_inv_freq)
        freqs = torch.einsum("i, j -> i j", t, self.rotary_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.freqs_speech = rearrange(emb, "n d -> () () n d")

        self.text_enc = Encoder(config, config.text_enc_depth)
        self.speech_enc = Encoder(config, config.speech_enc_depth)

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.wav_token_compression = config.wav_token_compression

    def forward(self, text, speech_tokens, *, return_loss: bool):
        batch_size, device = text.shape[0], text.device
        text_mask = torch.ones_like(text.float()).bool()
        voice_mask = torch.ones_like(speech_tokens.float()).bool()

        text_emb = self.text_emb(text)
        speech_emb = self.speech_emb(speech_tokens)

        enc_text = self.text_enc(inputs=text_emb, rope_freqs=self.freqs_text, mask=text_mask)
        enc_speech = self.speech_enc(inputs=speech_emb, rope_freqs=self.freqs_speech, mask=voice_mask)

        text_latents = masked_mean(enc_text, text_mask)
        speech_latents = masked_mean(enc_speech, voice_mask)

        text_latents = self.to_text_latent(text_latents)
        speech_latents = self.to_speech_latent(speech_latents)

        text_latents = normalize(text_latents, p=2, dim=-1)
        speech_latents = normalize(speech_latents, p=2, dim=-1)

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum("n d, n d -> n", text_latents, speech_latents) * temp
            return sim

        sim = einsum("i d, j d -> i j", text_latents, speech_latents) * temp
        labels = torch.arange(batch_size, device=device)
        loss = (cross_entropy(sim, labels) + cross_entropy(sim.t(), labels)) / 2
        return loss

    @classmethod
    def convert_old(cls):
        model = cls(CLVPConfig())
        model_path = hf_hub_download(repo_id="Gatozu35/tortoise-tts", filename="clvp2.safetensors")
        old_state_dict = load_file(model_path, device="cuda")

        new_state_dict = {
            "speech_emb.weight": old_state_dict["speech_emb.weight"],
            "text_emb.weight": old_state_dict["text_emb.weight"],
            "to_speech_latent.weight": old_state_dict["to_speech_latent.weight"],
            "to_text_latent.weight": old_state_dict["to_text_latent.weight"],
            "temperature": old_state_dict["temperature"],
            "speech_enc.norm.weight": old_state_dict["speech_transformer.transformer.norm.weight"],
            "speech_enc.norm.bias": old_state_dict["speech_transformer.transformer.norm.bias"],
            "rotary_inv_freq": old_state_dict["speech_transformer.transformer.attn_layers.rotary_pos_emb.inv_freq"],
            "text_enc.norm.weight": old_state_dict["text_transformer.transformer.norm.weight"],
            "text_enc.norm.bias": old_state_dict["text_transformer.transformer.norm.bias"],
        }
        for i in range(20):
            for kind in ("speech", "text"):
                new_state_dict[f"{kind}_enc.layers.{i}.attn.c_attn.weight"] = torch.cat(
                    [
                        old_state_dict[f"{kind}_transformer.transformer.attn_layers.layers.{i*2}.1.wrap.to_q.weight"],
                        old_state_dict[f"{kind}_transformer.transformer.attn_layers.layers.{i*2}.1.wrap.to_k.weight"],
                        old_state_dict[f"{kind}_transformer.transformer.attn_layers.layers.{i*2}.1.wrap.to_v.weight"],
                    ],
                    dim=0,
                )
                new_state_dict[f"{kind}_enc.layers.{i}.attn.c_proj.weight"] = old_state_dict[
                    f"{kind}_transformer.transformer.attn_layers.layers.{i*2}.1.wrap.to_out.weight"
                ]
                new_state_dict[f"{kind}_enc.layers.{i}.attn.c_proj.bias"] = old_state_dict[
                    f"{kind}_transformer.transformer.attn_layers.layers.{i*2}.1.wrap.to_out.bias"
                ]
                new_state_dict[f"{kind}_enc.layers.{i}.norm_attn.weight"] = old_state_dict[
                    f"{kind}_transformer.transformer.attn_layers.layers.{i*2}.0.0.g"
                ]
                new_state_dict[f"{kind}_enc.layers.{i}.norm_mlp.weight"] = old_state_dict[
                    f"{kind}_transformer.transformer.attn_layers.layers.{i*2+1}.0.0.g"
                ]
                new_state_dict[f"{kind}_enc.layers.{i}.mlp.act.proj.weight"] = old_state_dict[
                    f"{kind}_transformer.transformer.attn_layers.layers.{i*2+1}.1.wrap.net.0.proj.weight"
                ]
                new_state_dict[f"{kind}_enc.layers.{i}.mlp.act.proj.bias"] = old_state_dict[
                    f"{kind}_transformer.transformer.attn_layers.layers.{i*2+1}.1.wrap.net.0.proj.bias"
                ]
                new_state_dict[f"{kind}_enc.layers.{i}.mlp.c_proj.weight"] = old_state_dict[
                    f"{kind}_transformer.transformer.attn_layers.layers.{i*2+1}.1.wrap.net.3.weight"
                ]
                new_state_dict[f"{kind}_enc.layers.{i}.mlp.c_proj.bias"] = old_state_dict[
                    f"{kind}_transformer.transformer.attn_layers.layers.{i*2+1}.1.wrap.net.3.bias"
                ]

        model.load_state_dict(new_state_dict)
        save_file(new_state_dict, "clvp.safetensors")
        return model

    @classmethod
    def from_pretrained(cls):
        model_path = hf_hub_download(repo_id="Gatozu35/minTorToiSe", filename="clvp.safetensors")
        model = cls(CLVPConfig())
        model.load_state_dict(load_file(model_path, device="cuda"))
        return model


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_default_device("cuda")
    clip = CLVP.from_pretrained().eval()
    clip_loss = clip(
        torch.randint(0, 256, (2, 120)),
        torch.randint(0, 8192, (2, 250)),
        return_loss=True,
    )
    print(clip_loss)
    nonloss = clip(
        torch.randint(0, 256, (2, 120)),
        torch.randint(0, 8192, (2, 250)),
        return_loss=False,
    )
    print(nonloss)
