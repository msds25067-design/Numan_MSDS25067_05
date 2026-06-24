"""
MSDS25067_05_allCode.py
------------------------
Name      : Numan Hussan
Roll No.  : MSDS25067
Assignment: DL Spring 2025 - Assignment 5 (Bonus) - Diffusion Models

REQUIRED BY ASSIGNMENT: a single .py file containing the code of ALL
code files in the project, for the grader's convenience.
This file simply concatenates: model.py + MSDS25067_05.py + the code
cells of test_single_sample.ipynb (in that order).
Do NOT run this file directly - use MSDS25067_05.py for training and
test_single_sample.ipynb for inference. This file is for review only.
"""

# ============================== model.py ==============================

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

# =========================== MSDS25067_05.py ============================

"""
MSDS25067_05.py
----------------
Name      : Numan Hussan
Roll No.  : MSDS25067
Assignment: DL Spring 2025 - Assignment 5 (Bonus) - Image Generation Using
            Diffusion Models

WHAT THIS SCRIPT DOES
----------------------
1. Loads a small custom dataset (5 animal classes x ~20 images each, as
   required by the assignment) using a custom Dataset/DataLoader class.
2. Builds the forward diffusion process and a UNet denoising model
   (both defined in model.py).
3. Trains the UNet to predict the noise added at each timestep, using a
   manually written (custom) loss function.
4. Saves:
      - the trained model checkpoint   -> Saved_Models/diffusion_model.pt
      - a loss curve                   -> outputs/loss_curve.png
      - a forward-noising visualization (Figure-1 style)
                                        -> outputs/forward_process.png
      - generated sample images        -> outputs/generated_samples.png

HOW TO RUN (command line, accepts dataset path as required by Readme.txt)
---------------------------------------------------------------------------
    python MSDS25067_05.py --data_dir /path/to/dataset --epochs 100

See Readme.txt / the Colab guide for the full list of arguments.
"""

import os
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import Diffusion, UNet, custom_mse_loss


# ---------------------------------------------------------------------------
# 1. Custom Dataset / DataLoader
# ---------------------------------------------------------------------------
class AnimalDataset(Dataset):
    """
    Expects a directory structure like:

        data_dir/
            class_a/
                img001.jpg
                img002.jpg
                ...
            class_b/
                ...
            ...

    Per assignment instructions: pick 5 classes out of the 15 available,
    and only ~20 images per class (you do not need to use the whole
    dataset). Both choices are controlled with --classes and
    --images_per_class.
    """

    def __init__(self, root_dir, classes=None, images_per_class=20, img_size=64, seed=42):
        self.root_dir = root_dir
        self.img_size = img_size

        all_classes = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        if not all_classes:
            raise RuntimeError(f"No class sub-folders found inside {root_dir}")

        rng = random.Random(seed)
        if classes is None or len(classes) == 0:
            # randomly choose 5 classes if the user didn't specify which ones
            self.classes = rng.sample(all_classes, k=min(5, len(all_classes)))
        else:
            self.classes = classes

        self.image_paths = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            rng.shuffle(files)
            chosen = files[:images_per_class]
            self.image_paths.extend(os.path.join(cls_dir, f) for f in chosen)

        if len(self.image_paths) == 0:
            raise RuntimeError("No images found - check --data_dir / --classes.")

        print(f"[Dataset] Using classes: {self.classes}")
        print(f"[Dataset] Total images loaded: {len(self.image_paths)}")

        # Necessary transformations before feeding data into the diffusion process:
        # resize -> tensor -> normalize to [-1, 1] (standard for diffusion models)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# 2. Visualization helpers (Figure-1 style forward-noising plot)
# ---------------------------------------------------------------------------
def save_forward_process_plot(diffusion, sample_img, out_path, n_steps_to_show=8):
    """Shows the same image getting noisier as t increases (Figure 1)."""
    sample_img = sample_img.unsqueeze(0)
    steps = torch.linspace(0, diffusion.timesteps - 1, n_steps_to_show).long()

    fig, axes = plt.subplots(1, n_steps_to_show, figsize=(2 * n_steps_to_show, 2.2))
    for ax, t_val in zip(axes, steps):
        t = torch.tensor([t_val], device=sample_img.device)
        x_t, _ = diffusion.noise_images(sample_img, t)
        img = (x_t[0].clamp(-1, 1) + 1) / 2
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"t={t_val.item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] Forward process visualization -> {out_path}")


def save_loss_curve(losses, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Custom MSE loss")
    plt.title("Training loss curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] Loss curve -> {out_path}")


def save_generated_samples(samples, out_path):
    n = samples.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.2))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, samples):
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] Generated samples -> {out_path}")


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    classes = args.classes.split(",") if args.classes else None
    dataset = AnimalDataset(
        root_dir=args.data_dir,
        classes=classes,
        images_per_class=args.images_per_class,
        img_size=args.img_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    diffusion = Diffusion(timesteps=args.timesteps, img_size=args.img_size, device=device)
    model = UNet(in_channels=3, base_ch=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Save a forward-process visualization once, before training starts.
    sample_img = dataset[0].to(device)
    save_forward_process_plot(
        diffusion, sample_img, os.path.join(args.output_dir, "forward_process.png")
    )

    losses = []
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in loader:
            batch = batch.to(device)
            t = diffusion.sample_timesteps(batch.shape[0])
            x_t, true_noise = diffusion.noise_images(batch, t)

            predicted_noise = model(x_t, t)
            loss = custom_mse_loss(predicted_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())

        avg = sum(epoch_losses) / len(epoch_losses)
        print(f"[Epoch {epoch + 1}/{args.epochs}] avg loss: {avg:.5f}")

        if (epoch + 1) % max(1, args.sample_every) == 0:
            samples = diffusion.sample(model, n_samples=4)
            save_generated_samples(
                samples, os.path.join(args.output_dir, f"samples_epoch_{epoch + 1}.png")
            )

    save_loss_curve(losses, os.path.join(args.output_dir, "loss_curve.png"))

    checkpoint_path = os.path.join(args.model_dir, "diffusion_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "img_size": args.img_size,
        "timesteps": args.timesteps,
        "base_channels": args.base_channels,
    }, checkpoint_path)
    print(f"[Saved] Model checkpoint -> {checkpoint_path}")

    # Final test: generate and save a batch of images from pure noise.
    final_samples = diffusion.sample(model, n_samples=args.n_test_samples)
    save_generated_samples(
        final_samples, os.path.join(args.output_dir, "final_generated_samples.png")
    )


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Diffusion model - DL Assignment 5 (Bonus)")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to dataset root folder (one sub-folder per animal class)")
    parser.add_argument("--classes", type=str, default="",
                         help="Comma-separated list of exactly 5 class names to use "
                              "(leave empty to pick 5 random classes)")
    parser.add_argument("--images_per_class", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=300,
                         help="Number of diffusion steps T (assignment default is 1000; "
                              "300 is used here for faster training on Colab - adjust freely)")
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--n_test_samples", type=int, default=4)
    parser.add_argument("--sample_every", type=int, default=20,
                         help="Save a sample image grid every N epochs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_dir", type=str, default="Saved_Models")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)

# ===================== test_single_sample.ipynb (code cells) =====================

# If running in Google Colab, make sure model.py and the Saved_Models folder
# are in the same directory as this notebook (see Readme.txt / Colab guide).
import torch
import matplotlib.pyplot as plt

from model import Diffusion, UNet

# ---- Settings ----
CHECKPOINT_PATH = "Saved_Models/diffusion_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---- Load checkpoint ----
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

img_size = checkpoint["img_size"]
timesteps = checkpoint["timesteps"]
base_channels = checkpoint["base_channels"]

model = UNet(in_channels=3, base_ch=base_channels).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

diffusion = Diffusion(timesteps=timesteps, img_size=img_size, device=DEVICE)
print("Model loaded. img_size =", img_size, " timesteps =", timesteps)

# ---- Generate a single sample from pure noise ----
with torch.no_grad():
    sample = diffusion.sample(model, n_samples=1)

img = sample[0].permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.axis("off")
plt.title("Generated sample (noise -> image)")
plt.show()

# ---- Alternative: explicitly create noise yourself, then pass it in ----
# (This matches the assignment wording exactly: "write a test function
# which will accept noise and create an image from it".)

noise = torch.randn(1, 3, img_size, img_size)   # create noise explicitly

with torch.no_grad():
    sample2 = diffusion.sample(model, n_samples=1, starting_noise=noise)

img2 = sample2[0].permute(1, 2, 0).cpu().numpy()
plt.figure(figsize=(4, 4))
plt.imshow(img2)
plt.axis("off")
plt.title("Generated from explicitly-created noise")
plt.show()

