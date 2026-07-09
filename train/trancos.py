import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import DenseSwin
from utils import MODEL_CONFIG, TRANCOS, TRANCOS_MASTER, EarlyStopper, GameMetrics

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("-b", "--batch", type=int, default=4, help="data batch size")
parser.add_argument(
    "-lr_b",
    "--learning_rate_backbone",
    type=float,
    default=1e-5,
    help="learning rate of backbone",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=1e-4,
    help="learning rate of the density head",
)
parser.add_argument("-d", "--decay", type=float, default=0.05, help="weight decay")
parser.add_argument(
    "-cw",
    "--count_weight",
    type=float,
    default=0.01,
    help="weight of the auxiliary count loss (0 disables it)",
)
parser.add_argument(
    "-ds",
    "--density_scale",
    type=float,
    default=1000.0,
    help="the network regresses density * this factor; see the note in the loss",
)
parser.add_argument("-f", "--frames", type=int, default=8, help="temporal clip length")
parser.add_argument("-w", "--workers", type=int, default=4, help="dataloader workers")
parser.add_argument("--clip", type=float, default=1.0, help="gradient norm clip")
parser.add_argument("--patience", type=int, default=15, help="early stopping patience")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no_amp", action="store_true", help="disable bfloat16 autocast")
parser.add_argument("--no_augment", action="store_true", help="disable augmentation")
parser.add_argument(
    "--train_split",
    default="training",
    choices=["training", "trainval"],
    help="'trainval' is the standard protocol when reporting on the test split",
)
parser.add_argument("--resume", default=None, help="checkpoint to resume from")
args = parser.parse_args()

TARGET_SIZE = (224, 384)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# A full-resolution density map averages ~4e-4 per pixel. Asked to regress that
# directly, the head is pushed to zero, the output ReLU saturates and training
# dies. Regressing `density * SCALE` keeps the target in the same range as the
# activations. Counts and GAME are always reported in true vehicle units.
SCALE = args.density_scale


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(args.seed)
torch.backends.cudnn.benchmark = True

# bfloat16 has the dynamic range to hold a sum-reduced loss; fp16 does not
use_amp = (
    not args.no_amp and device.startswith("cuda") and torch.cuda.is_bf16_supported()
)
if not args.no_amp and not use_amp:
    print("bfloat16 unsupported on this device, running in fp32")


resume_ckpt = None
if args.resume:
    resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
    saved = resume_ckpt["args"]
    # OneCycleLR bakes total_steps into its state_dict, so a run resumed under a
    # different schedule length overruns it partway through the next epoch
    mismatch = {
        k: (saved.get(k), getattr(args, k))
        for k in ("epochs", "batch", "train_split")
        if saved.get(k) != getattr(args, k)
    }
    if mismatch:
        raise SystemExit(
            "--resume needs the same LR schedule as the original run; differs in: "
            + ", ".join(f"{k}: {was} -> {now}" for k, (was, now) in mismatch.items())
        )


def build_set(split, augment):
    return TRANCOS(
        TRANCOS_MASTER.images,
        TRANCOS_MASTER.split(split),
        target_size=TARGET_SIZE,
        num_frames=args.frames,
        augment=augment,
        cache_dir=TRANCOS_MASTER.cache,
    )


def build_loader(dataset, shuffle):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )


training_loader = build_loader(build_set(args.train_split, not args.no_augment), True)
validation_loader = build_loader(build_set("validation", False), False)
test_loader = build_loader(build_set("test", False), False)

model = DenseSwin(size=(args.frames, *TARGET_SIZE)).to(device)

# Only the backbone and the density head lie on the gradient path of D.
# The neck/head are skipped entirely via density_only=True, and train/ucsd.py
# only ever restores backbone.* and density_head.* from these checkpoints.
params = [
    {"params": model.backbone.parameters(), "lr": args.learning_rate_backbone},
    {"params": model.density_head.parameters(), "lr": args.learning_rate},
]
optimizer = torch.optim.AdamW(params, weight_decay=args.decay)

scheduler = OneCycleLR(
    optimizer,
    max_lr=[args.learning_rate_backbone, args.learning_rate],
    epochs=args.epochs,
    steps_per_epoch=len(training_loader),
    pct_start=0.1,  # 10% of training warming up
    anneal_strategy="cos",
)

density_loss = torch.nn.MSELoss()
count_loss = torch.nn.SmoothL1Loss()

train_metrics = GameMetrics()
val_metrics = GameMetrics()
stopper = EarlyStopper(patience=args.patience) if args.patience > 0 else None

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
writer = SummaryWriter(
    os.path.join(MODEL_CONFIG.logs, f"DenseSwinTrainer_TRANCOS_{timestamp}")
)
writer.add_text("Hyperparameters", "\n".join(f"**{k}**: {v}" for k, v in vars(args).items()))
writer.flush()

os.makedirs(MODEL_CONFIG.checkpoints, exist_ok=True)
best_path = os.path.join(MODEL_CONFIG.checkpoints, f"DenseSwin_TRANCOS_{timestamp}_best.pth")
last_path = os.path.join(MODEL_CONFIG.checkpoints, f"DenseSwin_TRANCOS_{timestamp}_last.pth")

best_game0 = float("inf")
start_epoch = 0

if resume_ckpt:
    model.load_state_dict(resume_ckpt["model"])
    optimizer.load_state_dict(resume_ckpt["optimizer"])
    scheduler.load_state_dict(resume_ckpt["scheduler"])
    best_game0 = resume_ckpt["best_game0"]
    start_epoch = resume_ckpt["epoch"]
    print(f"resumed from {args.resume} at epoch {start_epoch}")


def predict(frame, roi):
    """Predicted density map in SCALE units, ROI-masked. (B, 1, H, W)"""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        _, D = model(frame, density_only=True)
    # every frame of the clip is the same still image, so collapse the time axis
    return D.float().mean(dim=2) * roi


def run_epoch(loader, metrics, train, desc):
    model.train(train)
    total = 0.0
    grad_norm = 0.0

    with torch.set_grad_enabled(train), tqdm(loader, desc=desc) as bar:
        for frame, density, roi, count in bar:
            frame, density, roi, count = (
                t.to(device, non_blocking=True) for t in (frame, density, roi, count)
            )

            if train:
                optimizer.zero_grad(set_to_none=True)

            D = predict(frame, roi)

            loss = density_loss(D, density * SCALE)
            if args.count_weight:
                # in true vehicle units, so this term reads as a count error
                loss = loss + args.count_weight * count_loss(
                    D.flatten(1).sum(dim=1) / SCALE, count
                )

            if train:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                grad_norm += norm.item()
                optimizer.step()
                scheduler.step()

            total += loss.item()
            metrics.update(D.detach() / SCALE, density)
            bar.set_postfix(loss=loss.item())

    return total / len(loader), grad_norm / len(loader)


for epoch in range(start_epoch, args.epochs):

    avg_loss, grad_norm = run_epoch(
        training_loader, train_metrics, True, f"Training Epoch {epoch+1}"
    )
    avg_vloss, _ = run_epoch(
        validation_loader, val_metrics, False, f"Validation Epoch {epoch+1}"
    )

    train_game = train_metrics.compute()
    val_game = val_metrics.compute()

    writer.add_scalars("Loss", {"Training": avg_loss, "Validation": avg_vloss}, epoch + 1)
    writer.add_scalars("Game/Train", train_game, epoch + 1)
    writer.add_scalars("Game/Validation", val_game, epoch + 1)
    writer.add_scalar("GradNorm", grad_norm, epoch + 1)
    for i, group in enumerate(optimizer.param_groups):
        writer.add_scalar(f"LR/group_{i}", group["lr"], epoch + 1)
    writer.flush()

    train_metrics.reset()
    val_metrics.reset()

    game0 = val_game["0"]
    print(
        f"epoch {epoch+1}: loss={avg_loss:.4f} vloss={avg_vloss:.4f} "
        f"train GAME-0={train_game['0']:.3f} val GAME-0={game0:.3f} |g|={grad_norm:.2f}"
    )

    meta = {"epoch": epoch + 1, "best_game0": min(best_game0, game0), "args": vars(args)}

    # only `last` carries the optimiser moments needed to resume; `best` is the
    # artifact you transfer or deploy, and stays half the size without them
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            **meta,
        },
        last_path,
    )

    # GAME-0 is the reported metric, so select on it rather than on the loss
    if game0 < best_game0:
        best_game0 = game0
        torch.save({"model": model.state_dict(), **meta}, best_path)

    # stop on the metric we select on, not on the loss
    if stopper and stopper.early_stop(game0):
        print(f"early stopping at epoch {epoch+1}")
        break

"""
TEST -- run once, on the best checkpoint, against the held-out split
"""
if os.path.exists(best_path):
    best = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
else:
    # resumed at or past the final epoch, so this run never wrote a best
    print("no best checkpoint from this run; testing the weights as loaded")

test_metrics = GameMetrics()
run_epoch(test_loader, test_metrics, False, "Test")
test_game = test_metrics.compute()


writer.add_text(
    "Test", "\n".join(f"**GAME-{lvl}**: {err:.3f}" for lvl, err in test_game.items())
)
writer.flush()
writer.close()

print(f"best val GAME-0: {best_game0:.3f}")
print("test: " + "  ".join(f"GAME-{lvl}={err:.3f}" for lvl, err in test_game.items()))
