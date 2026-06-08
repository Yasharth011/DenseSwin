import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from ultralytics.models import YOLO
from utils import TEST_DATASET, TRAIN_DATASET
from decord import VideoReader
from decord import cpu
import json

model = YOLO("yolo11x.pt")

device = "0" if torch.cuda.is_available() else "cpu"

# COCO indices for traffic elements (2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASSES = [2, 3, 5, 7]
# detecting person to detect overlapped bikes by detecting the person


def annotate_frame(frames, results, file_path):

    for result in results:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_xywh = result.boxes.xywh.cpu().numpy()

        for frame in frames:
            for xyxy, xywh in zip(boxes_xyxy, boxes_xywh):
                x1, y1, x2, y2 = map(int, xyxy)
                x_center, y_center = int(xywh[0]), int(xywh[1])
                draw = ImageDraw.Draw(frame)
                draw.rectangle((x1, y1, x2, y2), width=2)
                draw.point((x_center, y_center))

            frame.save(file_path)


def create_dataset(dataset, BATCH_SIZE=10, num_frames=8, annotate=False, CONF=0.15):
    video_path = dataset.videos
    video_len = len(os.listdir(video_path))
    csv_path = dataset.csv
    i = 1
    dataset_rows = []

    processed_videos = set()
    try:
        df = pd.read_csv(csv_path, header=None)
        processed_videos = set(df[0].to_list())

        if len(processed_videos) == video_len:
            print("Processed all videos already")
            exit()

        print(
            f"Found existing processed videos. Resuming pipeline. {len(processed_videos)} image(s) already annotated."
        )
        video_len = video_len - len(processed_videos)

    except Exception as e:
        print(f"Could not read existing CSV, starting fresh. Error: {e}")

    for file in os.scandir(dataset.videos):

        if i > BATCH_SIZE:
            print("Batch Complete")
            break

        if file.is_file() and (file.name not in processed_videos):

            print(f"{i}/{video_len}  {file.name}")

            vr = VideoReader(file.path, ctx=cpu(0))

            frames_batch = np.linspace(0, len(vr) - 1, num=num_frames, dtype=int)

            frames = [
                Image.fromarray(frame) for frame in vr.get_batch(frames_batch).asnumpy()
            ]

            results = model.predict(
                source=frames,
                conf=CONF,
                classes=VEHICLE_CLASSES,
                verbose=False,
                device=device,
            )

            bboxes = [
                result.boxes.xywh.cpu().numpy().flatten().tolist() for result in results
            ]

            if annotate:
                file_path = os.path.join(
                    dataset.frames, f"{os.path.splitroot(file.name)}.png"
                )
                annotate_frame(frames, results, file_path)

            dataset_rows.append(
                [file.name, json.dumps(frames_batch.tolist()), json.dumps(bboxes)]
            )

            i = i + 1

    if len(dataset_rows):
        df = pd.DataFrame(dataset_rows)
        df.to_csv(csv_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="create train dataset")
    parser.add_argument("-b", "--batch", help="size of batch to process")
    parser.add_argument("-n", "--num_frames", help="number of frames to extract")
    parser.add_argument("-a", "--annotate", help="annotate image with bounding boxes")
    parser.add_argument("-c", "--conf", help="confidence of yolo model")
    args = parser.parse_args()

    if args.dataset == "train":
        create_dataset(
            TRAIN_DATASET,
            int(args.batch),
            int(args.num_frames),
            args.annotate,
            args.conf,
        )
    elif args.dataset == "test":
        create_dataset(
            TEST_DATASET,
            int(args.batch),
            int(args.num_frames),
            args.annotate,
            args.conf,
        )
