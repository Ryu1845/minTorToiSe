import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import Tensor, inference_mode, nn, tensor
from torch.nn.functional import interpolate

from tortoise.autoregressive import AttentionBlock

LINEAR_BETA_SCHEDULE: np.ndarray = np.linspace(
    0.25 * 0.0001,  # beta_start = (1000/trained_diffusion_steps)*0.0001
    0.25 * 0.02,  # beta_end
    4000,  # trained_diffusion_steps
    dtype=np.float64,
)


def space_timesteps(n_timestep: int, section_counts: List[int]) -> Set[int]:
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    Args:
        n_timestep: the number of diffusion steps in the original process to divide up.
        section_counts:

    Returns:
        a set of diffusion steps from the original process to use.
    """
    size_per = n_timestep // len(section_counts)
    extra = n_timestep % len(section_counts)
    start_idx = 0
    all_steps = []
    for idx, section_count in enumerate(section_counts):
        size = size_per + (1 if idx < extra else 0)
        if size < section_count:
            msg = f"Cannot divide section of {size} steps into {section_count}"
            raise ValueError(msg)
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)

        current_idx = 0.0
        steps_taken = []
        for _ in range(section_count):
            steps_taken.append(start_idx + round(current_idx))
            current_idx += frac_stride
        all_steps += steps_taken
        start_idx += size
    return set(all_steps)


def into_tensor(array: np.ndarray, from_timesteps: Tensor, broadcast_shape: torch.Size) -> Tensor:
    result = torch.from_numpy(array).to(from_timesteps.device)[from_timesteps].float()
    return result.expand(broadcast_shape)


def sin_timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """
    Create sinusoidal timestep embeddings
    Args:
        timesteps: a 1D tensor of n indices, one per batch element. (may be fractional)
        dim: the dimension of the output
        max_period: controls the minimum frequency of the positional embeddings

    Returns:
        an [n dim] tensor of positional embeddings
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class TimestepEmbedSequential(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: Tensor, emb: Tensor) -> Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, emb)
        return outputs


class ResBlock(nn.Module):
    def __init__(self, channels: int, p_dropout: float):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0),
        )
        self.embedding_layers = nn.Sequential(nn.SiLU(), nn.Linear(channels, 2 * channels))
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(p=p_dropout),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: Tensor, embeddings: Tensor) -> Tensor:
        outputs = self.in_layers(inputs)
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        embeddings_out = self.embedding_layers(embeddings).unsqueeze(2)
        scale, shift = torch.chunk(embeddings_out, chunks=2, dim=1)
        outputs = out_norm(outputs) * (1 + scale) + shift
        outputs = out_rest(outputs)
        return inputs + outputs


class DiffusionLayer(nn.Module):
    def __init__(self, channels: int, p_dropout: float, n_head: int):
        super().__init__()
        self.resblock = ResBlock(channels, p_dropout)
        self.attention = AttentionBlock(channels, n_head, use_rel_pos=True)

    def forward(self, inputs: Tensor, time_embeddings: Tensor) -> Tensor:
        return self.attention(self.resblock(inputs, time_embeddings))


@dataclass
class DiffusionConfig:
    n_embd: int = 1024
    n_layer: int = 10
    n_head: int = 16
    p_dropout: float = 0.0
    in_channels: int = 100
    out_channels: int = 200
    latent_channels: int = 1024
    conditioning_free_k: int = 2


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.unconditioned_embedding = nn.Parameter(torch.randn(1, config.n_embd, 1))
        self.unconditioned_latent = nn.Parameter(torch.randn(1, 2 * config.n_embd))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            DiffusionLayer(config.n_embd, config.p_dropout, config.n_head),
            DiffusionLayer(config.n_embd, config.p_dropout, config.n_head),
            DiffusionLayer(config.n_embd, config.p_dropout, config.n_head),
        )
        self.input_block = nn.Conv1d(
            in_channels=config.in_channels, out_channels=config.n_embd, kernel_size=3, stride=1, padding=1
        )
        self.embed_time = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd), nn.SiLU(), nn.Linear(config.n_embd, config.n_embd)
        )
        self.norm = nn.GroupNorm(32, config.n_embd)
        self.integrating_conv = nn.Conv1d(in_channels=config.n_embd * 2, out_channels=config.n_embd, kernel_size=1)
        self.layers = nn.ModuleList(
            [DiffusionLayer(config.n_embd, config.p_dropout, config.n_head) for _ in range(config.n_layer)]
            + [ResBlock(config.n_embd, config.p_dropout) for _ in range(3)]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(32, config.n_embd),
            nn.SiLU(),
            nn.Conv1d(config.n_embd, config.out_channels, kernel_size=3, padding=1),
        )
        self.latent_conditioner = nn.Sequential(
            nn.Conv1d(in_channels=config.latent_channels, out_channels=config.n_embd, kernel_size=3, padding=1),
            AttentionBlock(config.n_embd, config.n_head, use_rel_pos=True),
            AttentionBlock(config.n_embd, config.n_head, use_rel_pos=True),
            AttentionBlock(config.n_embd, config.n_head, use_rel_pos=True),
            AttentionBlock(config.n_embd, config.n_head, use_rel_pos=True),
        )

    def forward(
        self,
        inputs: Tensor,
        timesteps: Tensor,
        precomputed_embeddings: Optional[Tensor] = None,
        *,
        conditioning_free: bool,
    ) -> Tensor:
        batch_size, *_ = inputs.shape
        if conditioning_free:
            code_embedding = self.unconditioned_embedding.repeat(batch_size, 1, inputs.shape[-1])
        else:
            assert precomputed_embeddings is not None, "precomputed_embeddings should be set when not conditioning free"
            code_embedding = precomputed_embeddings
        time_embedding = self.embed_time(sin_timestep_embedding(timesteps, self.n_embd))
        code_embedding = self.conditioning_timestep_integrator(code_embedding, time_embedding)
        inputs = self.input_block(inputs)
        inputs = torch.cat([inputs, code_embedding], dim=1)
        inputs = self.integrating_conv(inputs)
        for layer in self.layers:
            inputs = layer(inputs, time_embedding)
        return self.out(inputs)

    def independent_timestep(self, latent: Tensor, seq_len: int) -> Tensor:
        latent = rearrange(latent, "b s c -> b c s")
        embedding = self.latent_conditioner(latent)
        unconditioned_latent = torch.load("cond_latent.pth")
        cond_scale, cond_shift = torch.chunk(unconditioned_latent, 2, dim=1)
        embedding = self.norm(embedding) * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1)
        # maybe add the self.training branch?
        expanded_embedding = interpolate(embedding, size=seq_len, mode="nearest")
        return expanded_embedding

    @classmethod
    def convert_old(cls) -> "DiffusionModel":
        model_path = hf_hub_download(repo_id="Gatozu35/tortoise-tts", filename="diffusion_decoder.safetensors")
        old_state_dict = load_file(model_path, device="cuda")
        name_changes = {
            "attn": "attention",
            "resblk": "resblock",
            "time_embed": "embed_time",
            "inp_block": "input_block",
            "code_norm": "norm",
            "emb_layers": "embedding_layers",
            "conditioning_timestep_integrator": "conditioning_timestep_integrator.layers",
        }
        deleted_parts = ("code_converter", "mel_head", "contextual_embedder", "code_embedding")
        new_state_dict = {}
        for k, v in old_state_dict.items():
            if any(pattern in k for pattern in deleted_parts):
                continue
            for pattern, replacement in name_changes.items():
                if pattern in k:
                    k = k.replace(pattern, replacement)  # noqa: PLW2901
            new_state_dict[k] = v
        # setting a diffusion conditioning latent from the original repo
        # afaict it's always the same
        new_state_dict["unconditioned_latent"] = load_file(
            hf_hub_download(repo_id="Gatozu35/minTorToiSe", filename="diffuser_cond_latent.safetensors"), device="cuda"
        )["conditioning_latent"]
        model = cls(DiffusionConfig())
        save_file(new_state_dict, "diffuser.safetensors")
        model.load_state_dict(new_state_dict)
        return model

    @classmethod
    def from_pretrained(cls) -> "DiffusionModel":
        model_path = hf_hub_download(repo_id="Gatozu35/minTorToiSe", filename="diffuser.safetensors")
        model = cls(DiffusionConfig())
        model.load_state_dict(load_file(model_path, device="cuda"))
        return model


class SpacedDiffuser:
    def __init__(self, model: DiffusionModel, n_timestep: int = 200, conditioning_free_k: int = 2):
        self.model = model
        self.conditioning_free_k = conditioning_free_k
        timesteps = space_timesteps(4000, [n_timestep])

        betas: np.ndarray = np.array(LINEAR_BETA_SCHEDULE, dtype=np.float64)  # for more precision
        alphas_cumprod = np.cumprod(1.0 - betas, axis=0)

        betas = []
        timestep_map = []
        last_alpha_cumprod = 1.0
        for idx, alpha_cumprod in enumerate(alphas_cumprod):
            if idx in timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(idx)
        self.n_timestep = len(betas)
        self.timestep_map = torch.tensor(timestep_map)

        betas: np.ndarray = np.array(betas, dtype=np.float64)
        self.betas = betas
        alphas_cumprod = np.cumprod(1.0 - betas, axis=0)

        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1.0)

        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(1.0 - betas) / (1.0 - alphas_cumprod)

    # TODO: add comments describing p mean variance etc
    @inference_mode()
    def sample(self, noise: Tensor, embeddings: Tensor) -> Tensor:
        # TODO: find another variable name for noise
        for i in reversed(range(self.n_timestep)):
            print(f"Timestep {self.n_timestep-i} of {self.n_timestep}", end="\r")
            # print(noise[0,0,:5])
            timesteps = tensor([i])  # assumes batch size is one
            # print(timesteps)
            # p mean variance
            _batch_size, n_channel = noise.shape[:2]
            output = self.model.forward(
                noise, self.timestep_map[timesteps], precomputed_embeddings=embeddings, conditioning_free=False
            )
            # print(output[0,0,:5])
            output_no_cond = self.model.forward(noise, self.timestep_map[timesteps], conditioning_free=True)
            # print(output_no_cond[0,0,:5])
            output, variable_values = torch.split(output, n_channel, dim=1)
            output_no_cond, _ = torch.split(output_no_cond, n_channel, dim=1)
            min_log = into_tensor(self.posterior_log_variance_clipped, timesteps, noise.shape)
            max_log = into_tensor(np.log(self.betas), timesteps, noise.shape)
            log_variance = ((variable_values + 1) / 2) * max_log + (1 - (variable_values + 1) / 2) * min_log
            # variance = torch.exp(log_variance)
            conditioning_free_k = self.conditioning_free_k * (
                1 - timesteps[0].item() / self.n_timestep
            )  # ramp conditioning free
            output = (1 + conditioning_free_k) * output - conditioning_free_k * output_no_cond
            # print(output[0,0,:5])
            epsilon = output
            predicted_xstart = (
                into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, noise.shape) * noise
                - into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, noise.shape) * epsilon
            )  # predict from eps
            # print(predicted_xstart[0,0,:5])

            # q post mean variance
            posterior_mean = (
                into_tensor(self.posterior_mean_coef1, timesteps, noise.shape) * predicted_xstart.clamp(-1, 1)
                + into_tensor(self.posterior_mean_coef2, timesteps, noise.shape) * noise
            )
            # print(posterior_mean[0,0,:5])
            _posterior_variance = into_tensor(self.posterior_variance, timesteps, noise.shape)
            _posterior_log_variance_clipped = into_tensor(self.posterior_log_variance_clipped, timesteps, noise.shape)
            mean = posterior_mean
            assert mean.shape == log_variance.shape == predicted_xstart.shape == noise.shape

            # new_noise = torch.rand_like(noise)
            new_noise = torch.randn(*noise.shape)
            # new_noise = torch.load("new_noise.pth")
            non_zero_mask = (
                (timesteps != 0).float().view(-1, 1)
            )  # no noise when t == 0 # TODO: nb of ones should be batch size
            # TODO: check cond_fn
            sample = mean + non_zero_mask * torch.exp(0.5 * log_variance) * new_noise
            noise = sample
        print()
        return noise


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_default_device("cuda:0")
    # SpacedDiffuser.convert_old()
    # exit()
    with torch.inference_mode():
        diffuser = SpacedDiffuser(DiffusionModel.from_pretrained().eval(), n_timestep=80)
        timestep_independent = diffuser.model.independent_timestep(torch.randn(1, 100, 1024), 348)
        test_input = torch.randn(1, 100, 348)
        mel = diffuser.sample(test_input, timestep_independent)
        print(mel.shape)
