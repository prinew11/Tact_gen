"""
Conditional DDPM training: diffuse RGB -> heightmap.

Splits the dataset into train / val / test. Only the final EMA
checkpoint is saved. Per-epoch train & val losses are tracked and a
loss curve is saved to <output_dir>/loss_curve.png at the end.

Usage (from src/training):
    python train.py --data_root "D:/homework/lund/CS_project/dataset_resize" \
                    --output_dir "D:/homework/lund/CS_project/Tact_gen/models" \
                    --epochs 200 --batch_size 4 --image_size 256
"""
from __future__ import annotations

import argparse
import json
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
from losses import fabrication_aware_loss  # noqa: E402
from cnc_params import MAX_SLOPE_PX, MIN_FEAT_PX  # noqa: E402


def build_model(image_size: int):
    from diffusers import UNet2DModel

    return UNet2DModel(
        sample_size=image_size,
        in_channels=4,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
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


def save_model(
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

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()})


@torch.no_grad()
def compute_eval_loss(model, scheduler, loader, device: str) -> float:
    """Average epsilon-MSE over a dataloader at random timesteps."""
    model.eval()
    total = 0.0
    n = 0
    for diffuse, height in loader:
        diffuse = diffuse.to(device, non_blocking=True)
        height = height.to(device, non_blocking=True)
        bsz = height.shape[0]
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        noise = torch.randn_like(height)
        noisy_height = scheduler.add_noise(height, noise, timesteps)
        model_in = torch.cat([noisy_height, diffuse], dim=1)
        pred = model(model_in, timesteps).sample
        loss = F.mse_loss(pred, noise)
        total += loss.item() * bsz
        n += bsz
    model.train()
    return total / max(1, n)


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    test_loss: float | None,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="train", linewidth=2)
    ax.plot(epochs, val_losses, label="val", linewidth=2)
    if test_loss is not None:
        ax.axhline(test_loss, color="red", linestyle="--",
                   label=f"test (final): {test_loss:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (epsilon-prediction)")
    ax.set_title("Conditional DDPM training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="D:/homework/lund/CS_project/dataset_processed")
    p.add_argument("--output_dir", default="D:/homework/lund/CS_project/Tact_gen/models")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--val_fraction", type=float, default=0.10)
    p.add_argument("--test_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--num_timesteps", type=int, default=1000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Split into train / val / test
    full_dataset = PairedTextureDataset(args.data_root, image_size=args.image_size, train=True)
    eval_dataset = PairedTextureDataset(args.data_root, image_size=args.image_size, train=False)
    n_total = len(full_dataset)
    n_val = max(4, int(n_total * args.val_fraction))
    n_test = max(4, int(n_total * args.test_fraction))
    n_train = n_total - n_val - n_test
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test], generator=gen)
    # Val / test use the eval-mode dataset (no augmentation) via shared indices
    val_ds.dataset = eval_dataset
    test_ds.dataset = eval_dataset

    print(f"Dataset: total={n_total}  train={n_train}  val={n_val}  test={n_test}")
    print(f"Device: {device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.image_size).to(device)

    scheduler = build_scheduler(args.num_timesteps)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    ema = SimpleEMA(model, decay=args.ema_decay)

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_losses: list[float] = []
    val_losses: list[float] = []
    slope_losses: list[float] = []
    feat_losses: list[float] = []
    alphas_cumprod = scheduler.alphas_cumprod
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_slope = 0.0
        epoch_feat = 0.0
        n_batches = 0
        t_start = time.time()
        for step, (diffuse, height) in enumerate(train_loader):
            diffuse = diffuse.to(device, non_blocking=True)
            height = height.to(device, non_blocking=True)
            bsz = height.shape[0]
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
            noise = torch.randn_like(height)
            noisy_height = scheduler.add_noise(height, noise, timesteps)
            model_in = torch.cat([noisy_height, diffuse], dim=1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(model_in, timesteps).sample
                loss, parts = fabrication_aware_loss(
                    pred_noise=pred,
                    target_noise=noise,
                    noisy_h=noisy_height,
                    alphas_cumprod=alphas_cumprod,
                    timesteps=timesteps,
                    max_slope_px=MAX_SLOPE_PX,
                    min_feat_px=MIN_FEAT_PX,
                    current_step=global_step,
                )

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
            epoch_loss += parts["total"]
            epoch_mse += parts["mse"]
            epoch_slope += parts["slope"]
            epoch_feat += parts["feat"]
            n_batches += 1
            if step % 20 == 0:
                print(
                    f"  ep {epoch+1:03d} step {step:04d}/{len(train_loader):04d}  "
                    f"total={parts['total']:.4f}  mse={parts['mse']:.4f}  "
                    f"slope={parts['slope']:.4f}  feat={parts['feat']:.4f}"
                )
            global_step += 1

        mean_train_loss = epoch_loss / max(1, n_batches)
        mean_slope_loss = epoch_slope / max(1, n_batches)
        mean_feat_loss = epoch_feat / max(1, n_batches)

        # Evaluate on val set (with EMA weights)
        ema_model = build_model(args.image_size).to(device)
        ema.copy_to(ema_model)
        mean_val_loss = compute_eval_loss(ema_model, scheduler, val_loader, device)
        del ema_model
        if device == "cuda":
            torch.cuda.empty_cache()

        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)
        slope_losses.append(mean_slope_loss)
        feat_losses.append(mean_feat_loss)

        dt = time.time() - t_start
        print(f"Epoch {epoch+1:03d}/{args.epochs}  "
              f"train_loss={mean_train_loss:.4f}  val_loss={mean_val_loss:.4f}  "
              f"slope={mean_slope_loss:.4f}  feat={mean_feat_loss:.4f}  "
              f"elapsed={dt:.1f}s")

    # Final test evaluation
    ema_model = build_model(args.image_size).to(device)
    ema.copy_to(ema_model)
    test_loss = compute_eval_loss(ema_model, scheduler, test_loader, device)
    print(f"\nFinal test loss: {test_loss:.4f}")

    # Save final checkpoint only
    save_model(
        out / "improved", ema_model, scheduler, ema.state_dict(),
        {
            "epoch": args.epochs,
            "image_size": args.image_size,
            "num_timesteps": args.num_timesteps,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "test_loss": test_loss,
        },
    )
    del ema_model

    # Save loss log + plot
    with open(out / "loss_log.json", "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "slope_losses": slope_losses,
                "feat_losses": feat_losses,
                "test_loss": test_loss,
            },
            f,
            indent=2,
        )
    plot_loss_curves(train_losses, val_losses, test_loss, out / "loss_curve.png")

    print(f"Done. Final checkpoint: {out/'improved'}")
    print(f"Loss curve: {out/'loss_curve.png'}")


if __name__ == "__main__":
    main()
