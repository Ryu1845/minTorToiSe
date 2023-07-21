import math
from dataclasses import dataclass, field
from typing import List, Set, Optional

import numpy as np
import torch
from einops import rearrange
from torch import nn, Tensor, inference_mode, tensor
from torch.nn.functional import interpolate

from tortoise.tortoise import AttentionBlock

LINEAR_BETA_SCHEDULE = np.linspace(
    0.25 * 0.0001,  # beta_start = (1000/trained_diffusion_steps)*0.0001
    0.25 * 0.02,  # beta_end
    4000,  # trained_diffusion_steps
    dtype=np.float64,
)


def space_timesteps(n_timestep: int, section_counts: List[int]):
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
            raise ValueError(f"Cannot divide section of {size} steps into {section_count}")
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


def into_tensor(array: np.ndarray, from_timesteps: Tensor, broadcast_shape: torch.Size):
    result = torch.from_numpy(array).to(from_timesteps.device)[from_timesteps].float()
    return result.expand(broadcast_shape)


def sin_timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000):
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
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half)) / half
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class TimestepEmbedSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, inputs: Tensor, emb: Tensor):
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
        self.out_norm = nn.GroupNorm(32, channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=p_dropout),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: Tensor, embeddings: Tensor):
        outputs = self.in_layers(inputs)
        outputs = self.out_norm(outputs)
        outputs = rearrange(outputs, "n c l -> n l c")
        embeddings_out = self.embedding_layers(embeddings)
        scale, shift = torch.chunk(embeddings_out, chunks=2, dim=1)
        outputs = outputs * (1 - scale) + shift
        outputs = rearrange(outputs, "n l c -> n c l")
        outputs = self.out_layers(outputs)
        return inputs + outputs


class DiffusionLayer(nn.Module):
    def __init__(self, channels: int, p_dropout: float, n_head: int):
        super().__init__()
        self.resblock = ResBlock(channels, p_dropout)
        self.attention = AttentionBlock(channels, n_head)

    def forward(self, inputs: Tensor, time_embeddings: Tensor):
        return self.attention(self.resblock(inputs, time_embeddings))


@dataclass
class SpaceDiffuserConfig:
    n_embd: int = 1024
    n_layer: int = 10
    n_head: int = 16
    p_dropout: float = 0.0
    in_channels: int = 100
    out_channels: int = 200
    latent_channels: int = 1024
    conditioning_free_k: int = 2
    betas: np.ndarray = field(default_factory=lambda: LINEAR_BETA_SCHEDULE)
    timesteps: Set[int] = field(default_factory=lambda: space_timesteps(4000, [200]))


class SpacedDiffuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.conditioning_free_k = config.conditioning_free_k

        betas = np.array(config.betas, dtype=np.float64)  # for more precision
        alphas_cumprod = np.cumprod(1.0 - betas, axis=0)

        betas = []
        last_alpha_cumprod = 1.0
        for idx, alpha_cumprod in enumerate(alphas_cumprod):
            if idx in config.timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        self.n_timestep = len(betas)

        self.betas = betas = np.array(betas, dtype=np.float64)
        alphas_cumprod = np.cumprod(1.0 - betas, axis=0)

        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1.0)

        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(1.0 - betas) / (1.0 - alphas_cumprod)

        self.unconditioned_embedding = nn.Parameter(torch.randn(1, config.n_embd, 1))
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
            AttentionBlock(config.n_embd, config.n_head),
            AttentionBlock(config.n_embd, config.n_head),
            AttentionBlock(config.n_embd, config.n_head),
            AttentionBlock(config.n_embd, config.n_head),
        )

    def forward(
        self,
        inputs: Tensor,
        timesteps: Tensor,
        precomputed_embeddings: Optional[Tensor] = None,
        *,
        conditioning_free: bool,
    ):
        batch_size, *_ = inputs.shape
        if conditioning_free:
            code_embedding = self.unconditioned_embedding.repeat(batch_size, 1, inputs.shape[-1])
        else:
            code_embedding = precomputed_embeddings
        time_embedding = self.embed_time(sin_timestep_embedding(timesteps, self.n_embd))
        code_embedding = self.conditioning_timestep_integrator(code_embedding, time_embedding)
        inputs = self.input_block(inputs)
        inputs = torch.cat([inputs, code_embedding], dim=1)
        inputs = self.integrating_conv(inputs)
        for layer in self.layers:
            inputs = layer(inputs, time_embedding)
        return self.out(inputs)

    def independent_timestep(self, latent: Tensor, seq_len: int):
        latent = rearrange(latent, "b s c -> b c s")
        embedding = self.latent_conditioner(latent)
        embedding = self.norm(embedding)
        # maybe add the self.training branch?
        expanded_embedding = interpolate(embedding, size=seq_len, mode="nearest")
        return expanded_embedding

    # TODO: add comments describing p mean variance etc
    @inference_mode()
    def sample(self, noise: Tensor, embeddings: Tensor):
        # TODO: find another variable name for noise
        for i in reversed(range(self.n_timestep)):
            timesteps = tensor([i])  # assumes batch size is one
            # p mean variance
            _batch_size, n_channel = noise.shape[:2]
            output = self(noise, timesteps, precomputed_embeddings=embeddings, conditioning_free=False)
            output_no_cond = self(noise, timesteps, conditioning_free=True)
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
            epsilon = output
            predicted_xstart = (
                into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, noise.shape) * noise
                - into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, noise.shape) * epsilon
            )  # predict from eps
            # q post mean variance
            posterior_mean = into_tensor(
                self.posterior_mean_coef1, timesteps, noise.shape
            ) * predicted_xstart + into_tensor(self.posterior_mean_coef2, timesteps, noise.shape)
            _posterior_variance = into_tensor(self.posterior_variance, timesteps, noise.shape)
            _posterior_log_variance_clipped = into_tensor(self.posterior_log_variance_clipped, timesteps, noise.shape)
            mean = posterior_mean
            assert mean.shape == log_variance.shape == predicted_xstart.shape == noise.shape

            new_noise = torch.rand_like(noise)
            non_zero_mask = (timesteps != 0).view(-1, 1)  # no noise when t == 0 # TODO: nb of ones should be batch size
            # TODO: check cond_fn
            sample = mean + non_zero_mask * torch.exp(0.5 * log_variance) * new_noise
            noise = sample
        return noise


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.set_default_device("cuda:0")
    test_config = SpaceDiffuserConfig()
    diffuser = SpacedDiffuser(test_config)
    timestep_independent = diffuser.independent_timestep(torch.randn(1, 100, 1024), 348)
    test_input = torch.randn(1, 100, 348)
    mel = diffuser.sample(test_input, timestep_independent)
    print(mel.shape)
