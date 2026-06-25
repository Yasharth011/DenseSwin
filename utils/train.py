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
)
import torch
import os
from datetime import datetime
import argparse
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epoch", default=1)
parser.add_argument(
    "-wds", "--weight_dense_swin", help="weight of dense swin", default=1
)
parser.add_argument(
    "-wd", "--weight_density_head", help="weight of density head", default=0.25
)
args = parser.parse_args()

EPOCHS = int(args.epochs)
W_DS = float(args.weight_dense_swin)
W_D = float(args.weight_density_head)

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
    training_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

model = DenseSwin().to(device)

D_loss = torch.nn.MSELoss()
DS_loss = torch.nn.SmoothL1Loss()

params = [
    {"params": model.backbone.parameters(), "lr": 1e-5},
    {"params": model.density_head.parameters(), "lr": 1e-4},
    {"params": model.neck.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(), "lr": 1e-4},
]
optimizer = torch.optim.AdamW(params, weight_decay=0.05)

early_stopper = EarlyStopper(patience=5, min_delta=0.001)

timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
writer = SummaryWriter(
    os.path.join(MODEL_CONFIG.logs, f"DenseSwinTrainer_{timestamp}")
)

best_vloss = float("inf")

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

            running_loss += loss.item()

            tepoch.set_postfix(loss=loss.item())

            # log loss per batch
            if i % 100 == 99:
                batch_avg_loss = running_loss / 100  # loss per batch
                tb_x = epoch * len(training_loader) + i + 1
                writer.add_scalar("Loss/train", batch_avg_loss, tb_x)
                total_loss += running_loss
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

                tepoch.set_postfix(loss=vloss.item())

            avg_vloss = running_vloss / len(validation_loader)

    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    writer.add_scalars(
        "Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch + 1,
    )
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "model_{}_{}".format(timestamp, epoch + 1)
        torch.save(
            model.state_dict(),
            os.path.join(MODEL_CONFIG.checkpoints, f"DenseSwin_{W_DS}_{W_D}_{timestamp}_epoch{epoch+1}.pth"),
        )

    # Check for early stop
    if early_stopper.early_stop(avg_vloss):
        print("Early Stopping")
        break

writer.close()
