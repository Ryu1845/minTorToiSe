from typing import List

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tortoise.tokenizer import Tokenizer
from tortoise.tortoise import ConditioningEncoder, Tortoise, TortoiseConfig
from tortoise.clvp import CLVP, CLVPConfig


class Inference:
    """Inference class for tortoise"""

    def __init__(self):
        config = TortoiseConfig()
        self.conditioning_encoder = ConditioningEncoder(config, spec_dim=80).eval()
        self.autoregressive = Tortoise(config).eval()
        self.tokenizer = Tokenizer()
        self.clvp = CLVP(CLVPConfig())

    @torch.inference_mode()
    def __call__(
        self,
        text: str,
        conditioning_speech: Float[Tensor, "n spec_d l"],
        samples_to_generate: int = 1,
    ):
        input_ids: List[int] = self.tokenizer.encode(text)
        text_inputs: Int[Tensor, "1 l"] = torch.tensor(input_ids).unsqueeze(0)
        conditioning_latent = self.conditioning_encoder(speech=conditioning_speech)

        samples = []
        for _ in range(samples_to_generate):
            sample_speech_tokens: Int[Tensor, "1 length_speech_output"] = self.autoregressive.generate_speech_tokens(
                text_inputs, speech_conditioning_latent=conditioning_latent  # , max_length=80 # for testing
            )
            samples.append(sample_speech_tokens)

        clvp_results = []
        for sample in samples:
            clvp_result = self.clvp(text_inputs, sample, return_loss=False)
            # print(clvp_result)
            clvp_results.append(clvp_result)
        clvp_results = torch.cat(clvp_results, dim=0)
        best_result = samples[torch.argmax(clvp_results)]
        # print(best_result)

        _loss_text, _loss_speech, _speech_logits, latent = self.autoregressive(
            text_inputs=text_inputs,
            speech_inputs=best_result,
            speech_conditioning_latent=conditioning_latent,
        )


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_default_device("cpu")
    inference = Inference()
    inference(
        "TorToiSe is a text-to-speech program that is capable of synthesizing speech "
        "in multiple voices with realistic prosody and intonation.",
        conditioning_speech=torch.randn(1, 80, 3272),
    )
