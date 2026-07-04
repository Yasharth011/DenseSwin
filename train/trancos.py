from models import DenseSwin
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from datetime import datetime
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import math
from utils import TRANCOS_MASTER, TRANCOS, MODEL_CONFIG, GameMetrics

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epoch", default=1)
parser.add_argument("-b", "--batch", help="data batch size", default=1)
parser.add_argument(
    "-lr_b", "--learning_rate_backbone", help="learning rate of backbone", default=1e-5
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    help="learning rate of neck, density and regression head ",
    default=1e-4,
)
parser.add_argument("-d", "--decay", help="weight decay", default=0.05)
args = parser.parse_args()

EPOCHS = int(args.epochs)
BATCH = int(args.batch)
LR_B = float(args.learning_rate_backbone)
LR = float(args.learning_rate)
DECAY = float(args.decay)

transform = v2.Compose(
    [
        v2.Resize((224, 384)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

training_set = TRANCOS(
    TRANCOS_MASTER.images,
    TRANCOS_MASTER.results,
    TRANCOS_MASTER.training,
    transform=transform,
)
validation_set = TRANCOS(
    TRANCOS_MASTER.images,
    TRANCOS_MASTER.results,
    TRANCOS_MASTER.validation,
    transform=transform,
)

training_loader = torch.utils.data.DataLoader(
    training_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
)

model = DenseSwin(num_class=3)
model.to(device=device)

# freeze decoder and linear layers
for name, layer in model.named_children():
    if name in ["neck", "head"]:
        for param in layer.parameters():
            param.requires_grad = False

D_loss = torch.nn.MSELoss()

train_metrics = GameMetrics()
val_metrics = GameMetrics()

params = [
    {"params": model.backbone.parameters(), "lr": LR_B},
    {"params": model.density_head.parameters(), "lr": LR},
]
optimizer = torch.optim.AdamW(params, weight_decay=DECAY)

scheduler = OneCycleLR(
    optimizer,
    max_lr=[LR_B, LR],
    epochs=EPOCHS,
    steps_per_epoch=len(training_loader),
    pct_start=0.1,  # 10% of training warming up
    anneal_strategy="cos",
)

timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
writer = SummaryWriter(
    os.path.join(MODEL_CONFIG.logs, f"DenseSwinTrainer_CBT2015_{timestamp}")
)

best_vloss = float("inf")
epoch = 0
early_stop = True
for epoch in range(EPOCHS):

    """
    TRAINING
    """
    # set model to training mode
    model.train(True)

    with tqdm(training_loader) as tepoch:

        tepoch.set_description(f"Training Epoch {epoch+1}")
        running_loss = 0
        total_loss = 0

        for i, data in enumerate(tepoch):

            frame, map, roi = [x.to(device) for x in data]

            optimizer.zero_grad()

            _, D = model(frame)

            D = D*roi
            map = map*roi

            loss = D_loss(D, map)
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            total_loss += loss.item()

            train_metrics.update(D, map)

            tepoch.set_postfix(loss=loss.item())

            # log loss per batch
            epoch_batch = 10 ** (math.floor(math.log10(len(training_loader))))
            if i % epoch_batch == epoch_batch - 1:
                batch_avg_loss = running_loss / 100  # loss per batch
                tb_x = epoch * len(training_loader) + i + 1
                writer.add_scalar("Loss/Train_Batch_Avg", batch_avg_loss, tb_x)
                running_loss = 0.0

        avg_loss = total_loss / len(training_loader)

    """
    VALIDATION
    """
    # set the model to evaluation mode
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():

        with tqdm(validation_loader) as tepoch:

            tepoch.set_description(f"Validation Epoch {epoch+1}")
            running_vloss = 0.0

            for i, data in enumerate(tepoch):

                frame, map, roi = [x.to(device) for x in data]

                _, D = model(frame)

                D = D * roi
                map = map * roi

                vloss = D_loss(D, map)

                running_vloss += vloss.item()

                val_metrics.update(D, map)

                tepoch.set_postfix(loss=vloss.item())

            avg_vloss = running_vloss / len(validation_loader)

    train_game = train_metrics.compute()
    val_game = val_metrics.compute()

    writer.add_scalars(
        "Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch + 1,
    )

    writer.add_scalars(
        "Game/Train",
        train_game,
        epoch + 1,
    )

    writer.add_scalars(
        "Game/Validation",
        val_game,
        epoch + 1,
    )

    writer.flush()

    train_metrics.reset()
    val_metrics.reset()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(
            model.state_dict(),
            os.path.join(
                MODEL_CONFIG.checkpoints,
                f"DenseSwin_CBT2015_{timestamp}_epoch{epoch+1}.pth",
            ),
        )

# log hyperparams
text = f"""
**Epochs**: {EPOCHS}
**Learning Rate**: Head = {LR} | Backbone = {LR_B}
**Decay**: {DECAY}
"""
writer.add_text("Hyperparameters", text)
writer.flush()

writer.close()
