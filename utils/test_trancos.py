import argparse
import os
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models import DenseSwin
from utils import MODEL_CONFIG, TRANCOS, TRANCOS_MASTER, GameMetrics

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    help="load model checkpoint",
)
parser.add_argument(
    "-ds",
    "--density_scale",
    type=float,
    default=1000.0,
    help="the network regresses density * this factor; see the note in the loss",
)
args = parser.parse_args()

TARGET_SIZE = (224, 384)
SCALE = args.density_scale


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(args.seed)
torch.backends.cudnn.benchmark = True

use_amp = (
    not args.no_amp and device.startswith("cuda") and torch.cuda.is_bf16_supported()
)
if not args.no_amp and not use_amp:
    print("bfloat16 unsupported on this device, running in fp32")


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


test_loader = build_loader(build_set("test", False), False)

model = DenseSwin(size=(args.frames, *TARGET_SIZE)).to(device)

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
writer = SummaryWriter(
    os.path.join(MODEL_CONFIG.logs, f"DenseSwinTrainer_TRANCOS_{timestamp}")
)


def predict(frame, roi):
    """Predicted density map in SCALE units, ROI-masked. (B, 1, H, W)"""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        _, D = model(frame, density_only=True)
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

            D = predict(frame, roi)

            metrics.update(D.detach() / SCALE, density)

    return total / len(loader), grad_norm / len(loader)


checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)

test_metrics = GameMetrics()
run_epoch(test_loader, test_metrics, False, "Test")
test_game = test_metrics.compute()
writer.add_text(
    "Test", "\n".join(f"**GAME-{lvl}**: {err:.3f}" for lvl, err in test_game.items())
)
writer.flush()
writer.close()
