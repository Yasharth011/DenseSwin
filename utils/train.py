from models import swin3d, DensityConv
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from utils import TrafficDensityDataset, TEST_DATASET, TRAIN_DATASET, MODEL_CONFIG
import torch
import os
from datetime import datetime
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epoch", default=1)
args = parser.parse_args()

EPOCHS = args.epochs
TARGET_SIZE = (224, 384)

transform = v2.Compose(
    [
        v2.Resize(TARGET_SIZE),
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

training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=4, shuffle=False
)

model = DensityConv()

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(
    [{"params": model.parameters(), "lr": 1e-5}], weight_decay=1e-4
)


def train_one_epoch(epoch_index, tb_writer):

    running_loss = 0
    last_loss = 0

    for i, data in enumerate(training_loader):
        breakpoint()
        frame, map = data

        frame = frame.to(device)
        map = map.to(device)

        optimizer.zero_grad()

        swin3d_features = swin3d(frame)

        _, density_head = model(swin3d_features, TARGET_SIZE)

        loss = loss_fn(density_head, map)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(
    os.path.join(MODEL_CONFIG.logs, f"density_module_trainer{timestamp}")
)

epoch_number = 0

best_vloss = 1_000_000.0

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0

    # set the model to evaluation mode
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, data in enumerate(validation_loader):

            frame, map = data

            frame = frame.to(device)
            map = map.to(device)

            optimizer.zero_grad()

            swin3d_features = swin3d(frame)

            _, density_head = model(swin3d_features, TARGET_SIZE)

            vloss = loss_fn(density_head, map)
            running_vloss += vloss

    avg_vloss = running_vloss / (len(validation_loader) + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch_number + 1,
    )
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "model_{}_{}".format(timestamp, epoch_number)
        torch.save(
            model.state_dict(),
            os.path.join(
                MODEL_CONFIG.checkpoints, f"density_module_{epoch_number}.pth"
            ),
        )

    epoch_number += 1
