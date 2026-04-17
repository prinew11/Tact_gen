"""
Conditional DDPM training: diffuse RGB -> heightmap.

The U-Net receives a 4-channel tensor [noisy_height, diffuse_rgb] and predicts
the noise that was added to the heightmap (epsilon-prediction). At inference
the diffuse image is kept fixed across all denoising steps and DDIM sampling
produces the heightmap.

Usage (from src/training):
    python train.py --data_root "D:/homework/lund/CS_project/dataset_resize" \
                    --output_dir "D:/homework/lund/CS_project/Tact_gen/checkpoints" \
                    --epochs 200 --batch_size 4 --image_size 256

Resume:
    python train.py ... --resume "D:/.../checkpoints/latest"

The final EMA checkpoint is saved to <output_dir>/final/ and a copy is also
kept at <output_dir>/latest/ for easy loading.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import PairedTextureDataset  # noqa: E402


def build_model(image_size: int):
    from diffusers import UNet2DModel

    return UNet2DModel(
        sample_size=image_size,
        in_channels=4,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def build_scheduler(num_train_timesteps: int = 1000):
    from diffusers import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False,
    )


def save_checkpoint(
    out_dir: Path,
    model,
    scheduler,
    ema_state_dict: dict,
    extra: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    scheduler.save_pretrained(out_dir)
    torch.save(ema_state_dict, out_dir / "ema.pt")
    with open(out_dir / "train_meta.json", "w") as f:
        json.dump(extra, f, indent=2)


class SimpleEMA:
    """Plain parameter-space EMA (avoids version churn in diffusers.EMAModel)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.state_dict().items():
            if not torch.is_floating_point(p):
                self.shadow[n].copy_(p)
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()})


@torch.no_grad()
def sample_preview(model, scheduler, diffuse_t: torch.Tensor, num_steps: int = 50, device: str = "cuda"):
    from diffusers import DDIMScheduler

    ddim = DDIMScheduler.from_config(scheduler.config)
    ddim.set_timesteps(num_steps)
    b, _, h, w = diffuse_t.shape
    height_t = torch.randn((b, 1, h, w), device=device)
    model.eval()
    for t in ddim.timesteps:
        inp = torch.cat([height_t, diffuse_t], dim=1)
        pred = model(inp, t).sample
        height_t = ddim.step(pred, t, height_t).prev_sample
    model.train()
    return height_t


def save_preview_grid(out_path: Path, diffuse_t: torch.Tensor, height_pred: torch.Tensor, height_gt: torch.Tensor) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = (diffuse_t.clamp(-1, 1).cpu().numpy() + 1) / 2
    hp = (height_pred.clamp(-1, 1).cpu().numpy() + 1) / 2
    hg = (height_gt.clamp(-1, 1).cpu().numpy() + 1) / 2
    n = d.shape[0]
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        ax = ax[None, :]
    for i in range(n):
        ax[i, 0].imshow(np.transpose(d[i], (1, 2, 0)))
        ax[i, 0].set_title("diffuse"); ax[i, 0].axis("off")
        ax[i, 1].imshow(hp[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[i, 1].set_title("pred height"); ax[i, 1].axis("off")
        ax[i, 2].imshow(hg[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[i, 2].set_title("gt height"); ax[i, 2].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="D:/homework/lund/CS_project/dataset_resize")
    p.add_argument("--output_dir", default="D:/homework/lund/CS_project/Tact_gen/checkpoints")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--preview_every", type=int, default=10)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_timesteps", type=int, default=1000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_dataset = PairedTextureDataset(args.data_root, image_size=args.image_size, train=True)
    n_total = len(full_dataset)
    n_val = max(4, int(n_total * args.val_fraction))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=gen)

    val_ds.dataset = PairedTextureDataset(args.data_root, image_size=args.image_size, train=False)

    print(f"Dataset: total={n_total}  train={n_train}  val={n_val}")
    print(f"Device: {device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=min(4, n_val), shuffle=False, num_workers=0)

    model = build_model(args.image_size).to(device)
    scheduler = build_scheduler(args.num_timesteps)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    ema = SimpleEMA(model, decay=args.ema_decay)

    start_epoch = 0
    if args.resume:
        from diffusers import UNet2DModel, DDPMScheduler  # noqa: F401

        resume_dir = Path(args.resume)
        model = build_model(args.image_size)
        model.load_state_dict(UNet2DModel.from_pretrained(resume_dir).state_dict())
        model.to(device)
        ema = SimpleEMA(model, decay=args.ema_decay)
        ema_path = resume_dir / "ema.pt"
        if ema_path.exists():
            ema.load_state_dict(torch.load(ema_path, map_location=device))
        meta_path = resume_dir / "train_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            start_epoch = int(meta.get("epoch", 0))
        print(f"Resumed from {resume_dir} at epoch {start_epoch}")

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    preview_batch = next(iter(val_loader))
    preview_diffuse = preview_batch[0].to(device)
    preview_gt = preview_batch[1].to(device)

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t_start = time.time()
        for step, (diffuse, height) in enumerate(train_loader):
            diffuse = diffuse.to(device, non_blocking=True)
            height = height.to(device, non_blocking=True)
            bsz = height.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(height)
            noisy_height = scheduler.add_noise(height, noise, timesteps)
            model_in = torch.cat([noisy_height, diffuse], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(model_in, timesteps).sample
                loss = F.mse_loss(pred, noise)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ema.update(model)
            epoch_loss += loss.item()
            global_step += 1
            if step % 20 == 0:
                print(f"  ep {epoch+1:03d} step {step:04d}/{len(train_loader):04d}  loss={loss.item():.4f}")

        dt = time.time() - t_start
        mean_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1:03d}/{args.epochs}  mean_loss={mean_loss:.4f}  elapsed={dt:.1f}s")

        if (epoch + 1) % args.preview_every == 0 or (epoch + 1) == args.epochs:
            ema_model = build_model(args.image_size).to(device)
            ema.copy_to(ema_model)
            preds = sample_preview(ema_model, scheduler, preview_diffuse, num_steps=25, device=device)
            save_preview_grid(
                out / "previews" / f"epoch_{epoch+1:03d}.png",
                preview_diffuse, preds, preview_gt,
            )
            del ema_model
            if device == "cuda":
                torch.cuda.empty_cache()

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ema_model = build_model(args.image_size).to(device)
            ema.copy_to(ema_model)
            meta = {
                "epoch": epoch + 1,
                "mean_loss": mean_loss,
                "image_size": args.image_size,
                "num_timesteps": args.num_timesteps,
                "ema_decay": args.ema_decay,
            }
            # save_checkpoint(out / f"ckpt_epoch{epoch+1:03d}", ema_model, scheduler, ema.state_dict(), meta)
            save_checkpoint(out / "latest", ema_model, scheduler, ema.state_dict(), meta)
            del ema_model
            if device == "cuda":
                torch.cuda.empty_cache()

    ema_model = build_model(args.image_size).to(device)
    ema.copy_to(ema_model)
    save_checkpoint(
        out / "final", ema_model, scheduler, ema.state_dict(),
        {"epoch": args.epochs, "image_size": args.image_size, "num_timesteps": args.num_timesteps},
    )
    save_checkpoint(
        out / "latest", ema_model, scheduler, ema.state_dict(),
        {"epoch": args.epochs, "image_size": args.image_size, "num_timesteps": args.num_timesteps},
    )
    print(f"Done. Final checkpoint: {out/'final'}")


if __name__ == "__main__":
    main()
