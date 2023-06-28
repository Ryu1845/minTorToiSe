from typing import List

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tortoise.tokenizer import Tokenizer
from tortoise.tortoise import ConditioningEncoder, Tortoise, TortoiseConfig


class Inference:
    """Inference class for tortoise"""

    def __init__(self):
        config = TortoiseConfig()
        self.conditioning_encoder = ConditioningEncoder(config, spec_dim=80).eval()
        self.autoregressive = Tortoise(config).eval()
        self.tokenizer = Tokenizer()

    @torch.inference_mode()
    def __call__(self, text: str, conditioning_speech: Float[Tensor, "n spec_d l"]):
        input_ids: List[int] = self.tokenizer.encode(text)
        text_inputs: Int[Tensor, "n l"] = torch.tensor(input_ids).unsqueeze(0)
        conditioning_latent = self.conditioning_encoder(speech=conditioning_speech)
        mel_inputs = self.autoregressive.generate_mel_tokens(
            text_inputs, speech_conditioning_latent=conditioning_latent
        )

        l_text, l_mel, mel_logits = self.autoregressive(
            text_inputs=text_inputs,
            mel_inputs=torch.randint(high=8192, size=(1, 250)),
            speech_conditioning_latent=conditioning_latent,
        )
        print(l_text, l_mel, mel_logits.shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    print("starting")
    inference = Inference()
    inference(
        "TorToiSe is a text-to-speech program that is capable of synthesizing speech "
        "in multiple voices with realistic prosody and intonation.",
        conditioning_speech=torch.randn(1, 80, 3272),
    )
