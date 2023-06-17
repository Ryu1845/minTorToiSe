from dataclasses import dataclass

import torch.nn as nn

@dataclass
class TortoiseConfig:
    """
    Args:
        n_layer: Number of layers in transformer stack.
        n_embd: Operating dimensions of the transformer
        n_head: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
        max_text_tokens: Maximum number of text tokens that will be encountered by model.
        max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
        max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
        mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
        n_text_token:
        start_text_token:
        stop_text_token:
        n_mel_code:
        start_mel_token:
        stop_mel_token:
        train_solo_embeddings:
        use_mel_codes_as_input:
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
    start_text_tokens: int = 256
    n_mel_code: int = 8194
    start_mel_token: int = 8192
    stop_mel_token: int = 8193
    train_solo_embeddings: bool = False
    use_mel_codes_as_input: bool = True
    checkpointing: bool = True
    n_type: int = 1

class Tortoise(nn.Module):
    def __init__(self, config: TortoiseConfig):
        super().__init__()

    def forward(self, inputs):
        ...

