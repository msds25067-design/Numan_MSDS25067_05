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
