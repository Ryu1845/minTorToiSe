from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torchaudio
from jaxtyping import Int
from torch import Tensor

from tortoise.autoregressive import ConditioningEncoder, Tortoise
from tortoise.clvp import CLVP
from tortoise.diffuser import DiffusionModel, SpacedDiffuser
from tortoise.tokenizer import Tokenizer
from tortoise.vocoder import UnivNetGenerator

torch.set_default_device("cuda")  # type: ignore[no-untyped-call]


@dataclass
class ARConfig:
    repetition_penalty: float = 2.0
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 50
    max_length: int = 500


@dataclass
class DiffuserConfig:
    conditioning_free_k: int = 2
    n_timestep: int = 200


@dataclass
class InferenceConfig:
    text: str
    conditioning_speech: Path
    output_path: Path = field(default_factory=lambda: Path("out.wav"))
    samples_to_generate: int = 1
    seed: int = 1
    ar_config: ARConfig = field(default_factory=lambda: ARConfig())
    diffuser_config: DiffuserConfig = field(default_factory=lambda: DiffuserConfig())


class Inference:
    """Inference class for tortoise"""

    def __init__(self):
        print("Loading models...")
        self.conditioning_encoder = ConditioningEncoder.from_pretrained().eval()
        self.autoregressive = Tortoise.from_pretrained().eval()
        self.tokenizer = Tokenizer()
        self.clvp = CLVP.from_pretrained().eval()
        self.diffuser = SpacedDiffuser(DiffusionModel.from_pretrained().eval())
        self.vocoder = UnivNetGenerator.from_pretrained().eval(inference=True)
        self.calm_token = 83  # token for coding silence # TODO: don't hardcode

    @torch.inference_mode()
    def __call__(
        self,
        text: str,
        conditioning_speech: Path,
        output_path: Path,
        samples_to_generate: int = 1,
        seed: int = 1,
        ar_config: Optional[Dict[str, Any]] = None,
        diffuser_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (self.diffuser.conditioning_free_k, self.diffuser.n_timestep) != (
            diffuser_config["conditioning_free_k"],
            diffuser_config["n_timestep"],
        ):
            self.diffuser = SpacedDiffuser(
                self.diffuser.model,
                n_timestep=diffuser_config["n_timestep"],
                conditioning_free_k=diffuser_config["conditioning_free_k"],
            )
        input_ids: List[int] = self.tokenizer.encode(text)
        text_inputs: Int[Tensor, "1 l"] = torch.tensor(input_ids).unsqueeze(0)

        torch.manual_seed(seed)

        conditioning_latent = self.conditioning_encoder.get_conditioning(speech_wav=str(conditioning_speech))
        # print("auto cond latent:", conditioning_latent)
        # print("diffusion cond latent:", torch.load("cond_latent.pth"))

        print(f"Generating {samples_to_generate} samples...")
        samples = []
        for i in range(samples_to_generate):
            print(f"Generating sample nr {i+1} of {samples_to_generate}")
            sample_speech_tokens: Int[Tensor, "1 length_speech_output"] = self.autoregressive.generate_speech_tokens(
                text_inputs,
                speech_conditioning_latent=conditioning_latent,
                **ar_config,  # , max_length=80  # for testing
            )
            # sample_speech_tokens = torch.load("codes.pth")
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
        latent = self.autoregressive(
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

        # print("latent:", latent)
        print("Generating mel spectrogram...")
        # diffusion
        output_len = (
            latent.shape[1] * 4 * 24_000 // 22_050
        )  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        diffusion_temperature = 1.0  # TODO: don't hardcode
        noise = torch.randn(latent.shape[0], 100, output_len) * diffusion_temperature
        # print("orig noise:", noise)
        timestep_independent = self.diffuser.model.independent_timestep(latent, output_len)
        # print("indep t:", timestep_independent)
        mel = self.diffuser.sample(noise=noise, embeddings=timestep_independent)

        MEL_MAX = 2.3143386840820312
        MEL_MIN = -11.512925148010254
        mel = ((mel + 1) / 2) * (MEL_MAX - MEL_MIN) + MEL_MIN  # denormalize
        mel = mel[:, :, :output_len]
        # print("mel:", mel)
        print("Generating waveform...")
        wav = self.vocoder.inference(mel)
        # print("wav:",wav)
        print(f"Saving output to {output_path}...")
        torchaudio.save(output_path, wav.cpu().squeeze(0), 24_000)


if __name__ == "__main__":
    import simple_parsing

    config = simple_parsing.parse(InferenceConfig)
    inference = Inference()
    # config = InferenceConfig(
    #    text="Tortoise is a text-to-speech program that is capable of synthesizing speech "
    #    "in multiple voices with realistic prosody and intonation.",
    #    conditioning_speech=Path("emma.wav"),
    # )
    inference(**asdict(config))
