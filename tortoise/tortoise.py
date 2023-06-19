from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn.functional import pad, cross_entropy

from .gpt import GPT


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

    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    max_text_tokens: int = 120
    max_mel_tokens: int = 250
    max_conditioning_inputs: int = 1
    mel_length_compression: int = 1024
    n_text_token: int = 256
    start_text_token: int = 256
    stop_text_token: int = 0
    n_mel_token: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    train_solo_embeddings: bool = False
    use_mel_tokens_as_input: bool = True
    checkpointing: bool = True
    n_type: int = 1


class Tortoise(nn.Module):
    def __init__(self, config: TortoiseConfig):
        super().__init__()
        self.start_text_token = config.start_text_token
        self.stop_text_token = config.stop_text_token
        self.start_mel_token = config.start_mel_token
        self.stop_mel_token = config.stop_mel_token

        self.embed_text = nn.Embedding(config.n_text_token * config.n_type + 1, config.n_embd)
        self.embed_mel = nn.Embedding(config.n_mel_token * config.n_type + 1, config.n_embd)
        self.embed_pos_text = LearnedPositionEmbeddings()
        self.embed_pos_mel = LearnedPositionEmbeddings()

        self.transformer = GPT(config)

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
        enc = self.final_norm(self.transformer(emb))
        text_logits = self.text_head(enc[:, : text_emb.shape[1]]).permute(0, 2, 1)
        mel_logits = self.mel_head(enc[:, -mel_emb.shape[1] :]).permute(0, 2, 1)
        # TODO check return latent

        loss_text = cross_entropy(text_logits, text_targets.long())
        loss_mel = cross_entropy(mel_logits, mel_targets.long())
        return loss_text.mean(), loss_mel.mean(), mel_logits
