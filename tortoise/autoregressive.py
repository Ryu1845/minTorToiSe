"""
The TorToiSe autoregressive model
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn.functional import pad, softmax
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram

from tortoise.gpt import GPT, GPTConfig


@dataclass
class TortoiseConfig:
    """
    Args:
        n_layer: Number of layers in transformer stack.
        n_embd: Operating dimensions of the transformer
        n_head: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
        max_text_tokens: Maximum number of text tokens that will be encountered by model.
        max_speech_tokens: Maximum number of MEL tokens that will be encountered by model.
        max_conditioning_inputs: Maximum number of conditioning inputs provided to the model.
            If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
        mel_length_compression: The factor between <number_input_samples> and <speech_tokens>.
            Used to compute MEL token padding given wav input length.
        n_text_token:
        start_text_token:
        stop_text_token:
        n_speech_token:
        start_speech_token:
        stop_speech_token:
        train_solo_embeddings:
        use_speech_tokens_as_input:
        checkpointing:
    """

    n_layer: int = 30
    n_embd: int = 1024
    n_head: int = 16
    max_text_tokens: int = 402
    max_speech_tokens: int = 604
    max_conditioning_inputs: int = 2
    mel_length_compression: int = 1024
    n_text_token: int = 256
    start_text_token: int = 255
    stop_text_token: int = 0
    n_speech_token: int = 8194
    start_speech_token: int = 8192
    stop_speech_token: int = 8193
    train_solo_embeddings: bool = False
    use_speech_tokens_as_input: bool = True
    checkpointing: bool = False


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, seq_len: int, n_embd: int, init: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(seq_len, n_embd)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        return self.emb(torch.arange(0, seq_len, device=x.device))


class RelativePositionBias(nn.Module):
    def __init__(self, scale, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> () h i j")
        return qk_dots + (bias * self.scale)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the 1d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, n_embd: int, n_head: int, *, use_rel_pos: bool):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(32, n_embd)
        self.qkv = nn.Conv1d(n_embd, n_embd * 3, 1)
        self.proj_out = nn.Conv1d(n_embd, n_embd, kernel_size=1)
        if use_rel_pos:
            self.relative_pos_embeddings = RelativePositionBias(
                scale=(n_embd // n_head) ** 0.5, heads=n_head, num_buckets=32, max_distance=64
            )

    def forward(self, x: Float[Tensor, "n c l"], mask: Optional[Tensor] = None) -> Tensor:
        b, c, l = x.size()  # batch size, embedding dimensionality (n_embd), sequence length

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = (
            self.qkv(self.norm(x)).reshape(b * self.n_head, c * 3 // self.n_head, l).split(c // self.n_head, dim=1)
        )
        # q, k, v = self.qkv(x).split(c, dim=1)
        # q = q.view(b * self.n_head, c // self.n_head, l)  # (B*nh, hs, T)
        # k = k.view(b * self.n_head, c // self.n_head, l)  # (B*nh, hs, T)
        # v = v.view(b * self.n_head, c // self.n_head, l)  # (B*nh, hs, T)

        scale = 1 / math.sqrt(math.sqrt(c / self.n_head))
        att = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        if hasattr(self, "relative_pos_embeddings"):
            att = self.relative_pos_embeddings(att.reshape(b, self.n_head, att.shape[-2], att.shape[-1])).reshape(
                b * self.n_head, att.shape[-2], att.shape[-1]
            )
        if mask is not None:
            mask = rearrange(mask, "b j -> b () () j")
            att = att.masked_fill(mask, -1e10)
        att = softmax(att, dim=-1)
        att = torch.einsum("bts,bcs->bct", att, v)
        att = att.reshape(b, -1, l)
        return (x + self.proj_out(att)).reshape(b, c, l)


class ConditioningEncoder(nn.Module):
    def __init__(self, config: TortoiseConfig, spec_dim: int = 80):
        super().__init__()
        self.init = nn.Conv1d(spec_dim, config.n_embd, kernel_size=1)
        self.attn = nn.Sequential(*(AttentionBlock(config.n_embd, config.n_head, use_rel_pos=False) for _ in range(6)))

    def forward(self, speech: Float[Tensor, "batch spec_d length"]) -> Tensor:
        out = self.init(speech)  # [n spec_d l] -> [n c l]
        out = self.attn(out)
        # return torch.load("test_cond_enc.pth")
        return out[:, :, 0]  # [n c l] -> [n c] (the first element)

    @torch.inference_mode()
    def get_conditioning(self, speech_samples: List[Path]):
        cond_len = 132300
        speech_conditionings = []
        for speech_wav in speech_samples:
            wav, sr = torchaudio.load(str(speech_wav.resolve()))
            if sr != 22050:
                wav = resample(wav, orig_freq=sr, new_freq=22_050)
            gap = wav.shape[-1] - cond_len
            if gap < 0:
                wav = pad(wav, pad=(0, abs(gap)))
            elif gap > 0:
                wav = wav[:, :cond_len]

            mel_norm_file = str((Path(__file__).parent.parent / "mel_norms.pth").resolve())
            mel_norms = torch.load(mel_norm_file).cuda()
            get_mel = MelSpectrogram(
                n_fft=1024,
                hop_length=256,
                win_length=1024,
                power=2,
                n_mels=80,
                f_min=0,
                f_max=8000,
                sample_rate=22_050,
                normalized=False,
                norm="slaney",
            )
            mel = get_mel(wav.cuda().unsqueeze(0))

            # Dynamic range compression
            mel = mel.clamp(min=1e-5).log()
            mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
            speech_conditionings.append(self.forward(mel.squeeze(0)))

        return torch.stack(speech_conditionings, dim=1).mean(dim=1)

    @classmethod
    def convert_old(cls):
        model = cls(TortoiseConfig())
        model_path = hf_hub_download(repo_id="Gatozu35/tortoise-tts", filename="autoregressive.safetensors")
        old_state_dict = load_file(model_path, device="cuda")
        new_state_dict = {}
        for k, v in old_state_dict.items():
            if "conditioning_encoder" not in k:
                continue
            k = k.replace("conditioning_encoder.", "")  # noqa: PLW2901
            if ("attn.c" in k or "mlp.c" in k) and "weight" in k:  # transpose conv1d weight into linear
                v = v.permute(1, 0).contiguous()  # noqa: PLW2901
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        save_file(new_state_dict, "conditioning_encoder.safetensors")
        return model

    @classmethod
    def from_pretrained(cls):
        model_path = hf_hub_download(repo_id="Gatozu35/minTorToiSe", filename="conditioning_encoder.safetensors")
        model = cls(TortoiseConfig())
        model.load_state_dict(load_file(model_path, device="cuda"))
        return model


class Tortoise(nn.Module):
    def __init__(self, config: TortoiseConfig):
        super().__init__()
        self.start_text_token = config.start_text_token
        self.stop_text_token = config.stop_text_token
        self.start_speech_token = config.start_speech_token
        self.stop_speech_token = config.stop_speech_token

        self.embed_text = nn.Embedding(config.n_text_token, config.n_embd)
        self.embed_speech = nn.Embedding(config.n_speech_token, config.n_embd)
        self.embed_pos_text = LearnedPositionEmbedding(config.max_text_tokens + 2, config.n_embd)
        self.embed_pos_speech = LearnedPositionEmbedding(
            config.max_speech_tokens + config.max_conditioning_inputs + 2, config.n_embd
        )

        self.transformer = GPT(GPTConfig.from_tortoise_config(config))

        self.final_norm = nn.LayerNorm(config.n_embd)

        self.text_head = nn.Linear(config.n_embd, config.n_text_token)
        self.speech_head = nn.Linear(config.n_embd, config.n_speech_token)

        self.embed_text.weight.data.normal_(mean=0.0, std=0.02)

    # TODO jaxtype
    def forward(
        self, text_inputs: Tensor, speech_inputs: Tensor, speech_conditioning_latent: Tensor
    ) -> Tuple[Tensor, ...]:
        text_inputs = pad(text_inputs, (0, 1), value=self.stop_text_token)
        speech_inputs = pad(speech_inputs, (0, 1), value=self.stop_speech_token)

        condition = speech_conditioning_latent.unsqueeze(1)

        # build aligned inputs and targets
        text_targets = pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = pad(text_inputs, (1, 0), value=self.start_text_token)
        speech_targets = pad(speech_inputs, (0, 1), value=self.stop_speech_token)
        speech_inputs = pad(speech_inputs, (1, 0), value=self.start_speech_token)

        text_emb = self.embed_text(text_inputs) + self.embed_pos_text(text_inputs)
        # TODO investigate raw mels
        speech_emb = self.embed_speech(speech_inputs) + self.embed_pos_speech(speech_inputs)
        emb = torch.cat([condition, text_emb, speech_emb], dim=1)
        enc = self.final_norm(self.transformer(emb))
        latent = enc[:, -speech_emb.shape[1] : -2]  # -2 -> remove tokens added in this forward pass
        return latent

    @torch.inference_mode()
    def generate_speech_tokens(
        self,
        input_ids: Tensor,
        *,
        speech_conditioning_latent: Tensor,
        repetition_penalty: float = 2.0,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 50,
        max_length: int = 500,
    ) -> Tensor:
        eos_token_id_tensor = torch.tensor([self.stop_speech_token]).to(input_ids.device)
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        condition = speech_conditioning_latent.unsqueeze(1)

        input_ids = pad(input_ids, (0, 1), value=self.stop_text_token)
        input_ids = pad(input_ids, (1, 0), value=self.start_text_token)
        text_emb = self.embed_text(input_ids) + self.embed_pos_text(input_ids)

        emb = torch.cat([condition, text_emb], dim=1)

        speech_inputs = torch.tensor([[self.start_speech_token]])
        while True:
            # forward pass to get next token
            # print(speech_inputs)
            print(f"Generated {speech_inputs.shape[1]} tokens", end="\r")
            speech_emb = self.embed_speech(speech_inputs) + self.embed_pos_speech(speech_inputs)
            # print(speech_emb.shape)
            gpt_emb = torch.cat([emb, speech_emb], dim=1)
            hidden_state = self.transformer(gpt_emb)
            logits = self.speech_head(self.final_norm(hidden_state))
            scores = logits[:, -1, :]
            # print(scores)

            # repetition penalty
            score = torch.gather(scores, 1, speech_inputs)
            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            scores.scatter_(1, speech_inputs, score)

            # temperature
            scores = scores / temperature

            # top k
            indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
            scores = scores.masked_fill(indices_to_remove, -float("Inf"))

            # top p
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, -float("Inf"))

            # sample
            probs = nn.functional.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + self.stop_speech_token * (1 - unfinished_sequences)
            # update generated ids, model inputs, and length for next step
            speech_inputs = torch.cat([speech_inputs, next_tokens[:, None]], dim=-1)

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    break
            if speech_inputs.shape[-1] >= max_length:
                break
        print()
        speech_inputs = pad(speech_inputs, (0, max_length - speech_inputs.shape[1]), value=self.stop_speech_token)
        return self.fix_outputs(speech_inputs)

    def fix_outputs(self, speech_tokens: Tensor):
        speech_tokens = torch.where(speech_tokens == self.stop_speech_token, 83, speech_tokens)
        batch_size = speech_tokens.shape[0]
        for i in range(batch_size):
            speech_tokens[i, -3] = 45
            speech_tokens[i, -2] = 45
            speech_tokens[i, -1] = 248
        # speech_tokens[stop_token_indices.min().item():] = 83
        return speech_tokens[:, 1:]

    @classmethod
    def convert_old(cls):
        model = cls(TortoiseConfig())
        model_path = hf_hub_download(repo_id="Gatozu35/tortoise-tts", filename="autoregressive.safetensors")
        old_state_dict = load_file(model_path, device="cuda")
        deleted_parts = (
            "conditioning_encoder",  # conditioning_encoder is not really deleted but it's handled separately
            "attn.masked_bias",
        )
        name_changes = {
            "gpt": "transformer",
            "mel_head": "speech_head",
            "mel_pos_embedding": "embed_pos_speech",
            "text_pos_embedding": "embed_pos_text",
            "mel_embedding": "embed_speech",
            "text_embedding": "embed_text",
        }
        new_state_dict = {}
        for k, v in old_state_dict.items():
            if any(pattern in k for pattern in deleted_parts):
                continue
            for pattern, replacement in name_changes.items():
                if pattern in k:
                    k = k.replace(pattern, replacement)  # noqa: PLW2901
            if ("attn.c" in k or "mlp.c" in k) and "weight" in k:  # transpose conv1d weight into linear
                v = v.permute(1, 0).contiguous()  # noqa: PLW2901
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        save_file(new_state_dict, "autoregressive.safetensors")
        return model

    @classmethod
    def from_pretrained(cls):
        model_path = hf_hub_download(repo_id="Gatozu35/minTorToiSe", filename="autoregressive.safetensors")
        model = cls(TortoiseConfig())
        model.load_state_dict(load_file(model_path, device="cuda"))
        return model


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_default_device("cuda")
    # Tortoise.convert_old()
    # ConditioningEncoder.convert_old()
    # exit()
    with torch.inference_mode():
        t_config = TortoiseConfig()
        conditioning_encoder = ConditioningEncoder.from_pretrained().eval()
        conditioning_latent = conditioning_encoder(speech=torch.randn(1, 80, 395))
        print(conditioning_latent.shape)

        # tortoise = Tortoise(t_config).eval()
        tortoise = Tortoise.from_pretrained().eval()
        latent = tortoise(
            text_inputs=torch.randint(high=120, size=(1, 250)),
            speech_inputs=torch.randint(high=8192, size=(1, 250)),
            speech_conditioning_latent=conditioning_latent,
        )
        print(latent.shape)
