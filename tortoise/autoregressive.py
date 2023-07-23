"""
The TorToiSe autoregressive model
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, pad, softmax

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
    n_text_token: int = 255
    start_text_token: int = 255
    stop_text_token: int = 0
    n_speech_token: int = 8194
    start_speech_token: int = 8192
    stop_speech_token: int = 8193
    train_solo_embeddings: bool = False
    use_speech_tokens_as_input: bool = True
    checkpointing: bool = False
    n_type: int = 1


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

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.norm = nn.GroupNorm(32, n_embd)
        self.qkv = nn.Conv1d(n_embd, n_embd * 3, 1)
        self.proj_out = nn.Conv1d(n_embd, n_embd, kernel_size=1)
        self.relative_pos_embeddings = RelativePositionBias(
            scale=(n_embd // n_head) ** 0.5, heads=n_head, num_buckets=32, max_distance=64
        )

    def forward(self, x: Float[Tensor, "n c l"], mask: Optional[Tensor] = None) -> Tensor:
        b, c, l = x.size()  # batch size, embedding dimensionality (n_embd), sequence length

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.qkv(x).split(c, dim=1)
        q = q.view(b * self.n_head, c // self.n_head, l)  # (B*nh, hs, T)
        k = k.view(b * self.n_head, c // self.n_head, l)  # (B*nh, hs, T)
        v = v.view(b * self.n_head, c // self.n_head, l)  # (B*nh, hs, T)

        scale = 1 / math.sqrt(math.sqrt(c / self.n_head))
        att = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
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
        self.attn = nn.Sequential(*(AttentionBlock(config.n_embd, config.n_head) for _ in range(6)))

    def forward(self, speech: Float[Tensor, "batch spec_d length"]) -> Tensor:
        out = self.init(speech)  # [n spec_d l] -> [n c l]
        out = self.attn(out)
        return out[:, :, 0]  # [n c l] -> [n c] (the first element)


class Tortoise(nn.Module):
    def __init__(self, config: TortoiseConfig):
        super().__init__()
        self.start_text_token = config.start_text_token
        self.stop_text_token = config.stop_text_token
        self.start_speech_token = config.start_speech_token
        self.stop_speech_token = config.stop_speech_token

        self.embed_text = nn.Embedding(config.n_text_token * config.n_type + 1, config.n_embd)
        self.embed_speech = nn.Embedding(config.n_speech_token * config.n_type + 1, config.n_embd)
        self.embed_pos_text = LearnedPositionEmbedding(config.max_text_tokens + 2, config.n_embd)
        self.embed_pos_speech = LearnedPositionEmbedding(
            config.max_speech_tokens + config.max_conditioning_inputs + 2, config.n_embd
        )

        self.transformer = GPT(GPTConfig.from_tortoise_config(config))

        self.final_norm = nn.LayerNorm(config.n_embd)

        self.text_head = nn.Linear(config.n_embd, config.n_text_token * config.n_type + 1)
        self.speech_head = nn.Linear(config.n_embd, config.n_speech_token * config.n_type + 1)

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
        enc = self.final_norm(self.transformer(emb)[0])
        text_logits = self.text_head(enc[:, : text_emb.shape[1]]).permute(0, 2, 1)
        speech_logits = self.speech_head(enc[:, -speech_emb.shape[1] :]).permute(0, 2, 1)
        latent = enc[:, -speech_emb.shape[1] : -2]  # -2 -> remove tokens added in this forward pass

        loss_text = cross_entropy(text_logits, text_targets.long())
        loss_speech = cross_entropy(speech_logits, speech_targets.long())
        return loss_text.mean(), loss_speech.mean(), speech_logits, latent

    @torch.inference_mode()
    def generate_speech_tokens(
        self,
        input_ids: Tensor,
        *,
        speech_conditioning_latent: Tensor,
        repetition_penalty: float = 2.0,
        temperature: float = 0.2,
        top_p: float = 0.8,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        max_length: int = 250,
    ) -> Tensor:
        eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device) if eos_token_id is not None else None
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
            speech_inputs_idx = torch.where(
                speech_inputs >= self.start_speech_token,  # special tokens
                speech_inputs - self.start_speech_token,
                speech_inputs,
            )
            speech_emb = self.embed_speech(speech_inputs_idx) + self.embed_pos_speech(speech_inputs)
            # print(speech_emb.shape)
            gpt_emb = torch.cat([emb, speech_emb], dim=1)
            _hidden_state, logits, _loss = self.transformer(gpt_emb)
            scores = logits[:, -1, :]

            # top p
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, -float("Inf"))

            # temperature
            scores = scores / temperature

            # repetition penalty
            score = torch.gather(scores, 1, speech_inputs_idx)
            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            scores.scatter_(1, speech_inputs_idx, score)

            # sample
            probs = nn.functional.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    msg = "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    raise ValueError(msg)
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
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
        return speech_inputs


if __name__ == "__main__":
    torch.set_default_device("cuda")
    t_config = TortoiseConfig()
    conditioning_encoder = ConditioningEncoder(t_config, spec_dim=80)
    conditioning_latent = conditioning_encoder(speech=torch.randn(1, 80, 395))
    print(conditioning_latent.shape)

    tortoise = Tortoise(t_config).eval()
    l_text, l_speech, speech_log, latent = tortoise(
        text_inputs=torch.randint(high=120, size=(1, 250)),
        speech_inputs=torch.randint(high=8192, size=(1, 250)),
        speech_conditioning_latent=conditioning_latent,
    )
    print(l_text)
