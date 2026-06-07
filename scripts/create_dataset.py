import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from ultralytics.models import YOLO
from utils import TEST_DATASET, TRAIN_DATASET
from decord import VideoReader
from decord import cpu

model = YOLO("yolo11x.pt")

device = "0" if torch.cuda.is_available() else "cpu"

# COCO indices for traffic elements (2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASSES = [2, 3, 5, 7]
# detecting person to detect overlapped bikes by detecting the person


def detect_vehicles(dataset, image, annotated_name, conf=0.10):

    results = model.predict(
        source=image, conf=conf, classes=VEHICLE_CLASSES, verbose=False, device=device
    )

    all_boxes = []

    for result in results:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_xywh = (
            result.boxes.xywh.cpu().numpy()
        )  # [x_center, y_center, width, height]

        if len(boxes_xywh) > 0:
            all_boxes.append(boxes_xywh)

        for xyxy, xywh in zip(boxes_xyxy, boxes_xywh):
            x1, y1, x2, y2 = map(int, xyxy)
            x_center, y_center = int(xywh[0]), int(xywh[1])

            draw = ImageDraw.Draw(image)
            draw.rectangle((x1, y1, x2, y2), width=2)
            draw.point((x_center, y_center))

    image.save(os.path.join(dataset.annotated_frames, annotated_name))

    if len(all_boxes) > 0:
        vehicle_bboxes = np.vstack(all_boxes)
    else:
        vehicle_bboxes = np.empty((0, 4))

    return vehicle_bboxes


def create_dataset(dataset, BATCH_SIZE):
    video_path = dataset.videos
    video_len = len(os.listdir(video_path))
    main_csv_path = dataset.main_csv
    i = 1
    dataset_rows = []

    processed_videos = set()
    try:
        df = pd.read_csv(main_csv_path, header=None)
        processed_videos = set(df[0].to_list())

        if len(processed_videos) == video_len:
            print("Processed all videos already")

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
            bbox_rows = []

            for j in range(len(vr)):

                image = Image.fromarray(vr[j].asnumpy())

                file_name = f"{(file.name).split('.')[0]}_{j}.png"
                annotated_name = f"annot_{(file.name).split('.')[0]}_{j}.png"

                bbox_rows.append(
                    detect_vehicles(dataset, image, annotated_name).flatten().tolist()
                )

                image.save(os.path.join(dataset.frames, file_name))

            csv_name = f"{(file.name).split('.')[0]}.csv"
            csv_path = os.path.join(dataset.csv_dir, csv_name)
            csv_df = pd.DataFrame(bbox_rows)
            csv_df.to_csv(csv_path, mode="w", header=False, index=False)

            dataset_rows.append([file.name, csv_name])

            i = i + 1

    if len(dataset_rows):
        df = pd.DataFrame(dataset_rows)
        df.to_csv(main_csv_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="create train dataset")
    parser.add_argument("-b", "--batch", help="size of batch to process")
    args = parser.parse_args()

    if args.dataset == "train":
        create_dataset(TRAIN_DATASET, int(args.batch))
    elif args.dataset == "test":
        create_dataset(TEST_DATASET, int(args.batch))
    else:
        print("Invalid Dataset")
