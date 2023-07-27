from typing import List

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tortoise.autoregressive import ConditioningEncoder, Tortoise, TortoiseConfig
from tortoise.clvp import CLVP, CLVPConfig
from tortoise.diffuser import SpacedDiffuser, SpaceDiffuserConfig
from tortoise.tokenizer import Tokenizer
from tortoise.vocoder import UnivNetGenerator


class Inference:
    """Inference class for tortoise"""

    def __init__(self):
        print("Loading models...")
        self.conditioning_encoder = ConditioningEncoder.from_pretrained().eval()
        self.autoregressive = Tortoise.from_pretrained().eval()
        self.tokenizer = Tokenizer()
        self.clvp = CLVP.from_pretrained().eval()
        self.diffuser = SpacedDiffuser.from_pretrained().eval()
        self.vocoder = UnivNetGenerator.from_pretrained().eval()
        self.calm_token = 83  # token for coding silence # TODO: don't hardcode

    @torch.inference_mode()
    def __call__(
        self,
        text: str,
        conditioning_speech: Float[Tensor, "n spec_d l"],
        samples_to_generate: int = 1,
    ) -> Tensor:
        input_ids: List[int] = self.tokenizer.encode(text)
        text_inputs: Int[Tensor, "1 l"] = torch.tensor(input_ids).unsqueeze(0)
        conditioning_latent = self.conditioning_encoder(speech=conditioning_speech)

        print(f"Generating {samples_to_generate} samples...")
        samples = []
        for i in range(samples_to_generate):
            print(f"Generating sample nr {i} of {samples_to_generate}")
            sample_speech_tokens: Int[Tensor, "1 length_speech_output"] = self.autoregressive.generate_speech_tokens(
                text_inputs, speech_conditioning_latent=conditioning_latent  # , max_length=80  # for testing
            )
            samples.append(sample_speech_tokens)

        print("Ranking samples...")
        clvp_results = []
        for sample in samples:
            clvp_result = self.clvp(text_inputs, sample, return_loss=False)
            # print(clvp_result)
            clvp_results.append(clvp_result)
        clvp_results = torch.cat(clvp_results, dim=0)
        best_result = samples[torch.argmax(clvp_results)]
        # print(best_result)

        print("Generating latent...")
        _loss_text, _loss_speech, _speech_logits, latent = self.autoregressive(
            text_inputs=text_inputs,
            speech_inputs=best_result,
            speech_conditioning_latent=conditioning_latent,
        )

        # find 8 consecutive calm token and trim to that
        c_token_cnt = 0
        for idx, token in enumerate(best_result[0]):
            if c_token_cnt == 8:
                latent = latent[:, :idx]
                break
            if token == self.calm_token:
                c_token_cnt += 1
            else:
                c_token_cnt = 0

        print("Generating mel spectrogram...")
        # diffusion
        output_len = (
            latent.shape[1] * 4 * 24_000 // 22_050
        )  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        diffusion_temperature = 1  # TODO: don't hardcode
        noise = torch.randn(latent.shape[0], 100, output_len) * diffusion_temperature
        timestep_independent = self.diffuser.independent_timestep(latent, output_len)
        mel = self.diffuser.sample(noise=noise, embeddings=timestep_independent)

        MEL_MAX = 2.3143386840820312
        MEL_MIN = -11.512925148010254
        mel = ((mel + 1) / 2) * (MEL_MAX - MEL_MIN) + MEL_MIN  # denormalize
        mel = mel[:, :, :output_len]

        print("Generating waveform...")
        wav = self.vocoder.inference(mel)
        return wav


if __name__ == "__main__":
    import torchaudio

    torch.manual_seed(1)
    torch.set_default_device("cuda")  # type: ignore[no-untyped-call]
    inference = Inference()
    wav_tensor = inference(
        "TorToiSe is a text-to-speech program that is capable of synthesizing speech "
        "in multiple voices with realistic prosody and intonation.",
        conditioning_speech=torch.randn(1, 80, 3272),
    ).cpu()
    print(wav_tensor.shape)
    print("Saving output to test.wav...")
    torchaudio.save("test.wav", wav_tensor.squeeze(0), 24_000)
