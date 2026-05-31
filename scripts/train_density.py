from timm.data.config import resolve_data_config
from models import swin, DensityConv
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import TrafficDensityDataset, Rescale, ToTensor, DATASET_CONFIG
import torch
from datetime import datetime

transform = transforms.Compose(
    [
        Rescale((224, 384)),
        ToTensor(**resolve_data_config(swin.pretrained_cfg, model=swin)),
    ]
)

training_set = TrafficDensityDataset(
    DATASET_CONFIG.frames_csv,
    DATASET_CONFIG.frames,
    transform=transform,
)
validation_set = TrafficDensityDataset(
    DATASET_CONFIG.frames_csv,
    DATASET_CONFIG.frames,
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
        images, density_maps = data

        optimizer.zero_grad()

        c5_output = swin(images)

        _, density_head = model(c5_output)
        target_size = density_maps.shape[-2, :]

        loss = loss_fn(density_head, density_maps, target_size)
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
writer = SummaryWriter("runs/density_module_trainer{}".format(timestamp))
epoch_number = 0

EPOCHS = 5

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
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            
            vc5_output = swin(vinputs)
            vtarget_size = vlabels.shape[-2:]
            
            _, voutputs = model(vc5_output, target_size=vtarget_size)
            
            vloss = loss_fn(voutputs, vlabels)
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
            model.state_dict(), f"../checkpoints/density_module_{epoch_number}.pth"
        )

    epoch_number += 1
