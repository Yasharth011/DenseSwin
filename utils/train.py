from models import DenseSwin
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from utils import (
    TrafficDensityDataset,
    TEST_DATASET,
    TRAIN_DATASET,
    MODEL_CONFIG,
    EarlyStopper,
    early_stopper,
    RegressionMetrics,
)
import torch
import os
from datetime import datetime
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epoch", default=1)
parser.add_argument("-b", "--batch", help="data batch size", default=1)
parser.add_argument(
    "-wds", "--weight_dense_swin", help="weight of dense swin", default=1
)
parser.add_argument(
    "-wd", "--weight_density_head", help="weight of density head", default=0.25
)
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
W_DS = float(args.weight_dense_swin)
W_D = float(args.weight_density_head)
LR_B = float(args.learning_rate_backbone)
LR = float(args.learning_rate)
D = float(args.decay)

transform = v2.Compose(
    [
        v2.Resize((224, 384)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

training_set = TrafficDensityDataset(
    TRAIN_DATASET.videos,
    TRAIN_DATASET.csv,
    transform=transform,
)
validation_set = TrafficDensityDataset(
    TEST_DATASET.videos,
    TEST_DATASET.csv,
    transform=transform,
)

training_loader = torch.utils.data.DataLoader(
    training_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
)

model = DenseSwin().to(device)

D_loss = torch.nn.MSELoss()
DS_loss = torch.nn.SmoothL1Loss()

train_metrics = RegressionMetrics()
val_metrics = RegressionMetrics()

params = [
    {"params": model.backbone.parameters(), "lr": LR_B},
    {"params": model.density_head.parameters(), "lr": LR},
    {"params": model.neck.parameters(), "lr": LR},
    {"params": model.head.parameters(), "lr": LR},
]
optimizer = torch.optim.AdamW(params, weight_decay=D)

scheduler = OneCycleLR(
    optimizer,
    max_lr=[LR_B, LR, LR, LR],
    epochs=EPOCHS,
    steps_per_epoch=len(training_loader),
    pct_start=0.1,  # 10% of training warming up
    anneal_strategy="cos",
)

early_stopper = EarlyStopper(patience=5, min_delta=0.001)

timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
writer = SummaryWriter(os.path.join(MODEL_CONFIG.logs, f"DenseSwinTrainer_{timestamp}"))


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

            frame, conD, map = [x.to(device) for x in data]

            optimizer.zero_grad()

            F_ds, D = model(frame)

            loss = (W_DS * DS_loss(F_ds, conD)) + (W_D * D_loss(D, map))
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            total_loss +=loss.item()

            train_metrics.update(F_ds, conD)

            tepoch.set_postfix(loss=loss.item())

            # log loss per batch
            if i % 100 == 99:
                batch_avg_loss = running_loss / 100  # loss per batch
                tb_x = epoch * len(training_loader) + i + 1
                writer.add_scalar("Loss/train", batch_avg_loss, tb_x)
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

                frame, conD, map = [x.to(device) for x in data]

                F_ds, D = model(frame)

                vloss = (W_DS * DS_loss(F_ds, conD)) + (W_D * D_loss(D, map))

                running_vloss += vloss.item()

                val_metrics.update(F_ds, conD)

                tepoch.set_postfix(loss=vloss.item())

            avg_vloss = running_vloss / len(validation_loader)

    train_mae, train_mse, train_r2 = train_metrics.compute()
    val_mae, val_mse, val_r2 = val_metrics.compute()

    print(
        f"Train: loss= {avg_loss} | mae= {train_mae} | mse= {train_mse} | r2= {train_r2}"
    )
    print(
        f"Validation: loss= {avg_vloss} | mae= {val_mae} | mse= {val_mse} | r2={val_r2}"
    )

    writer.add_scalars(
        "Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch + 1,
    )

    writer.add_scalars(
        "Metrics/Train",
        {"MAE": train_mae, "MSE": train_mse, "R2": train_r2},
        epoch + 1,
    )

    writer.add_scalars(
        "Metrics/Validation",
        {"MAE": val_mae, "MSE": val_mse, "R2": val_r2},
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
                MODEL_CONFIG.checkpoints, f"DenseSwin_{timestamp}_epoch{epoch+1}.pth"
            ),
        )

    # Check for early stop
    early_stop = early_stopper.early_stop(avg_vloss)
    if early_stop:
        print("Early Stopping")
        break

# log hyperparams
text = f"""
**Epochs**: {EPOCHS} | Early Stop = {(epoch+1) if early_stop else 'No'}
**Weight**: Dense Swin = {W_DS} | Density Head = {W_D} 
**Learning Rate**: Head = {LR} | Backbone = {LR_B}
**Decay**: {D}
**Best Loss**: {best_vloss}
"""
writer.add_text("Hyperparameters", text)
writer.flush()

writer.close()
