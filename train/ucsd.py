from models import DenseSwin
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
import os
from datetime import datetime
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import math
from utils import (
    UCSD,
    UCSD_MASTER,
    MODEL_CONFIG,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", help="number of epoch", default=1)
parser.add_argument("-b", "--batch", help="data batch size", default=8)
parser.add_argument("-c", "--checkpoint", help="model checkpoint")
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
CHECKPOINT = str(args.checkpoint)
LR = float(args.learning_rate)
DECAY = float(args.decay)

checkpoint = torch.load(
    os.path.join(MODEL_CONFIG.checkpoints, CHECKPOINT), weights_only=True
)
checkpoint = {
        key: value for key, value in checkpoint.items()
        if key.startswith("backbone.") or key.startswith("density_head.")
    }

transform = v2.Compose(
    [
        v2.Resize((224, 384)),
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

master_train_set = UCSD(UCSD_MASTER.videos, UCSD_MASTER.csv, 8, transform, augment=True)
master_val_set = UCSD(UCSD_MASTER.videos, UCSD_MASTER.csv, 8, transform, augment=False)

# inverse-frequency class weights: light/medium/heavy are 65%/18%/17% of the data
label_map = master_train_set.label_map
class_counts = master_train_set.csv["class"].str.strip().str.lower().value_counts()
class_weights = torch.tensor(
    [len(master_train_set) / (len(label_map) * class_counts[cls]) for cls in label_map],
    dtype=torch.float32,
).to(device)

CEloss = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

train_metrics = MetricCollection(
    {
        "accuracy": Accuracy(num_classes=3, average="weighted", task="multiclass"),
        "precision": Precision(num_classes=3, average="weighted", task="multiclass"),
        "recall": Recall(num_classes=3, average="weighted", task="multiclass"),
        "f1score": F1Score(num_classes=3, average="weighted", task="multiclass"),
    }
).to(device)
val_metrics = MetricCollection(
    {
        "accuracy": Accuracy(num_classes=3, average="weighted", task="multiclass"),
        "precision": Precision(num_classes=3, average="weighted", task="multiclass"),
        "recall": Recall(num_classes=3, average="weighted", task="multiclass"),
        "f1score": F1Score(num_classes=3, average="weighted", task="multiclass"),
    }
).to(device)

timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
writer = SummaryWriter(
    os.path.join(MODEL_CONFIG.logs, f"DenseSwinTrainer_UCSD_{timestamp}")
)

fold_best_metrics = []

# 4 Fold Cross Validation
for fold in range(4):

    print(f"FOLD {fold} : ")

    model = DenseSwin(num_class=3)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device=device)

    # freeze back and density head
    for name, layer in model.named_children():
        if name in ["backbone", "density_head"]:
            for param in layer.parameters():
                param.requires_grad = False

    checkpoint_path = os.path.join(
                        MODEL_CONFIG.checkpoints,
                        f"DenseSwin_UCSD_{timestamp}_fold{fold}.pth",
                    )

    training_set = torch.utils.data.Subset(
        master_train_set, master_train_set.get_subset(UCSD_MASTER.train_csv, fold)
    )
    validation_set = torch.utils.data.Subset(
        master_val_set, master_val_set.get_subset(UCSD_MASTER.test_csv, fold)
    )

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
    )

    params = [
        {"params": model.neck.parameters(), "lr": LR},
        {"params": model.head.parameters(), "lr": LR},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=DECAY)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LR, LR],
        epochs=EPOCHS,
        steps_per_epoch=len(training_loader),
        pct_start=0.1,  # 10% of training warming up
        anneal_strategy="cos",
    )

    best_vloss = float("inf")
    best_epoch_metrics = None
    epoch = 0
    for epoch in range(EPOCHS):

        """
        TRAINING
        """
        model.train(True)
        model.backbone.eval()
        model.density_head.eval()

        with tqdm(training_loader) as tepoch:

            tepoch.set_description(f"Training Epoch {epoch+1}")
            running_loss = 0
            total_loss = 0

            for i, data in enumerate(tepoch):

                frame, label = [x.to(device) for x in data]

                optimizer.zero_grad()

                F_ds, _ = model(frame)

                loss = CEloss(F_ds, label)
                loss.backward()

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                total_loss += loss.item()

                F_ds = torch.argmax(F_ds, dim=1)
                train_metrics.update(F_ds, label)

                tepoch.set_postfix(loss=loss.item())

                # log loss per batch
                epoch_batch = 10 ** (math.floor(math.log10(len(training_loader))))
                if i % epoch_batch == epoch_batch - 1:
                    batch_avg_loss = running_loss / 100  # loss per batch
                    tb_x = epoch * len(training_loader) + i + 1
                    writer.add_scalars(
                        f"Loss/Train_Batch_Avg", {f"{fold}": batch_avg_loss}, tb_x
                    )
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

                    frame, label = [x.to(device) for x in data]

                    F_ds, _ = model(frame)

                    vloss = CEloss(F_ds, label)

                    running_vloss += vloss.item()

                    F_ds = torch.argmax(F_ds, dim=1)
                    val_metrics.update(F_ds, label)

                    tepoch.set_postfix(loss=vloss.item())

                avg_vloss = running_vloss / len(validation_loader)

        train_accuracy, train_precision, train_recall, train_f1score = (
            train_metrics.compute().values()
        )
        val_accuracy, val_precision, val_recall, val_f1score = (
            val_metrics.compute().values()
        )

        writer.add_scalars(
            "Loss",
            {f"Training_Fold{fold}": avg_loss, f"Validation_Fold{fold}": avg_vloss},
            epoch + 1,
        )

        writer.add_scalars(
            "Metrics/Train",
            {
                f"Accuracy{fold}": train_accuracy,
                f"Precision{fold}": train_precision,
                f"Recall{fold}": train_recall,
                f"F1Score{fold}": train_f1score,
            },
            epoch + 1,
        )

        writer.add_scalars(
            "Metrics/Validation",
            {
                f"Accuracy{fold}": val_accuracy,
                f"Precision{fold}": val_precision,
                f"Recall{fold}": val_recall,
                f"F1Score{fold}": val_f1score,
            },
            epoch + 1,
        )

        writer.flush()

        train_metrics.reset()
        val_metrics.reset()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch_metrics = {
                "accuracy": val_accuracy.item(),
                "precision": val_precision.item(),
                "recall": val_recall.item(),
                "f1score": val_f1score.item(),
            }
            torch.save(model.state_dict(), checkpoint_path)

    fold_best_metrics.append(best_epoch_metrics)

    # reset model for next fold
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

mean_metrics = {
    key: sum(fold[key] for fold in fold_best_metrics) / len(fold_best_metrics)
    for key in fold_best_metrics[0]
}

# log hyperparams
text = f"""
**Epochs**: {EPOCHS}
**Learning Rate**: Head = {LR}
**Decay**: {DECAY}

**Mean Validation Metrics (across folds, at each fold's best-val-loss epoch)**:
- Accuracy: {mean_metrics["accuracy"]:.4f}
- Precision: {mean_metrics["precision"]:.4f}
- Recall: {mean_metrics["recall"]:.4f}
- F1 Score: {mean_metrics["f1score"]:.4f}
"""
writer.add_text("Hyperparameters", text)
writer.flush()

writer.close()
