"""
model.py
--------
Name      : Numan Hussan
Roll No.  : MSDS25067
Assignment: DL Spring 2025 - Assignment 5 (Bonus) - Diffusion Models

This file contains:
    1. Diffusion        -> the forward (noising) and reverse (denoising/sampling)
                            diffusion process, implemented using the proper
                            closed-form variance-schedule equations (NOT a
                            naive "add noise directly to image" operation).
    2. SinusoidalTimeEmb -> time-step embedding used to tell the UNet "how
                            noisy" the current image is.
    3. UNet              -> the denoising neural network (predicts the noise
                            that was added at step t).
    4. custom_mse_loss   -> a manually written loss function (per assignment
                            rule: "Loss must be a customized function").

This file is imported by both:
    - MSDS25067_05.py        (training script)
    - test_single_sample.ipynb (single-sample inference/testing notebook)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Forward / Reverse Diffusion process
# ---------------------------------------------------------------------------
class Diffusion:
    """
    Implements the DDPM forward and reverse process.

    Forward process (training time):
        We do NOT add raw noise directly to the pixel values. Instead we use
        the closed-form equation derived from repeatedly applying the Markov
        chain q(x_t | x_t-1) for t steps:

            x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * eps

        where alpha_hat_t = product of (1 - beta_i) for i = 1..t.
        This equation IS the mathematically correct result of injecting
        Gaussian noise step-by-step T times - it is just the efficient,
        closed-form way of computing the result of that loop, so the image
        only looks like noise once t is large, exactly as the assignment
        describes (Figure 1, left->right).

    Reverse process (sampling/testing time):
        Starting from pure Gaussian noise x_T, we iteratively remove the
        noise predicted by the trained UNet, one step at a time, until we
        get back x_0 (a generated image). This matches Figure 1 right->left.
    """

    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02,
                 img_size=64, device="cpu"):
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device

        # linear noise (variance) schedule, beta_t
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_timesteps(self, n):
        """Randomly sample n timesteps in [1, T) for a training batch."""
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=self.device)

    def noise_images(self, x, t):
        """
        Forward process: produce x_t from x_0 using the closed-form formula
        (this is what step T_5 of the assignment - 'forward process function'
        - refers to).

        Returns:
            x_t        : the noised image at step t
            noise (eps): the actual noise that was added (this is the
                         training target for the UNet)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return x_t, eps

    @torch.no_grad()
    def sample(self, model, n_samples, channels=3, starting_noise=None):
        """
        Reverse process (this IS the "test function which will accept noise
        and create an image from it" required by the assignment): start
        from noise and iteratively denoise using the trained model,
        producing n_samples generated images.

        Args:
            model         : trained UNet
            n_samples     : how many images to generate (ignored if
                             starting_noise is given - n_samples is then
                             inferred from its batch dimension)
            channels      : number of image channels (default 3, RGB)
            starting_noise: optional tensor of shape
                             (n_samples, channels, img_size, img_size) to
                             use as x_T instead of freshly sampled noise.
                             Pass this in to explicitly hand the function
                             "noise" and get back an image, e.g.:

                                 noise = torch.randn(1, 3, 64, 64)
                                 img = diffusion.sample(model, n_samples=1,
                                                         starting_noise=noise)

                             If left as None (default), random Gaussian
                             noise is generated internally - same behavior
                             as before.
        """
        model.eval()
        if starting_noise is not None:
            x = starting_noise.to(self.device)
            n_samples = x.shape[0]
        else:
            x = torch.randn((n_samples, channels, self.img_size, self.img_size), device=self.device)

        for i in reversed(range(1, self.timesteps)):
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t)

            alpha = self.alpha[i]
            alpha_hat = self.alpha_hat[i]
            beta = self.beta[i]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        model.train()
        x = x.clamp(-1, 1)
        x = (x + 1) / 2  # back to [0, 1] for saving/displaying
        return x


# ---------------------------------------------------------------------------
# 2. Time-step embedding
# ---------------------------------------------------------------------------
class SinusoidalTimeEmb(nn.Module):
    """Standard transformer-style sinusoidal embedding for timestep t."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# 3. Building blocks + UNet
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Conv block with GroupNorm + SiLU + a time-embedding injection."""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_bias = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_bias
        h = self.block2(h)
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, x_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(x_ch, x_ch, kernel_size=4, stride=2, padding=1)
        self.res = ResidualBlock(x_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        return x


class UNet(nn.Module):
    """
    Small UNet used to predict the noise eps added at timestep t.
    Architecture chosen to be light enough to train quickly in Colab on a
    small (20 images x 5 classes) dataset, while still following the
    standard diffusion-UNet design (down path, bottleneck, up path with
    skip connections, time embedding injected at every residual block).
    """

    def __init__(self, in_channels=3, base_ch=32, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        self.down1 = Down(base_ch, base_ch * 2, time_emb_dim)
        self.down2 = Down(base_ch * 2, base_ch * 4, time_emb_dim)

        self.bottleneck = ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # up1: x comes from bottleneck (base_ch*4 channels), skip2 has base_ch*4 channels
        self.up1 = Up(x_ch=base_ch * 4, skip_ch=base_ch * 4, out_ch=base_ch * 2, time_emb_dim=time_emb_dim)
        # up2: x comes from up1 output (base_ch*2 channels), skip1 has base_ch*2 channels
        self.up2 = Up(x_ch=base_ch * 2, skip_ch=base_ch * 2, out_ch=base_ch, time_emb_dim=time_emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.in_conv(x)
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)

        x = self.bottleneck(x, t_emb)

        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)

        return self.out_conv(x)


# ---------------------------------------------------------------------------
# 4. Custom loss function (written manually, not nn.MSELoss())
# ---------------------------------------------------------------------------
def custom_mse_loss(predicted_noise, true_noise):
    """
    Manually implemented mean-squared-error between the noise the UNet
    predicted and the actual noise that was injected during the forward
    process. Written by hand (instead of calling nn.MSELoss()) to satisfy
    the assignment requirement that the loss be a "customized function".
    """
    diff = predicted_noise - true_noise
    squared_error = diff ** 2
    return torch.mean(squared_error)
