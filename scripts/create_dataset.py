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


def get_conD(bboxes, roi_area=224 * 384, max_expected_cars=30):
    """Implementation inspired from SSANET for Traffic Congestion Estimation"""

    # values from paper
    a = 20.0
    b = 10.0

    omega_o = 1.0
    omega_k = 1.0 / max_expected_cars

    conS_list = []
    velocity_list = []

    num_frames = len(bboxes)

    for bbox_list in bboxes:

        bbox_list = np.array(bbox_list).reshape(-1, 4)

        k = len(bbox_list)  # Density: number of vehicles

        # Calculate Occupancy (o) using area of bounding boxes
        total_box_area = 0

        for box in bbox_list:
            w, h = box[2], box[3]
            total_box_area += w * h

        o = min(1.0, total_box_area / roi_area)

        conS = (omega_o * o) + (omega_k * k)
        conS_list.append(conS)

    for t in range(num_frames - 1):
        boxes_t = np.array(bboxes[t]).reshape(-1, 4)
        boxes_t1 = np.array(bboxes[t + 1]).reshape(-1, 4)

        if len(boxes_t) == 0 or len(boxes_t1) == 0:
            velocity_list.append(10.0)
            continue

        displacements = []
        for box_a in boxes_t:
            cx_a, cy_a = box_a[0], box_a[1]

            min_dist = float("inf")
            for box_b in boxes_t1:
                cx_b, cy_b = box_b[0], box_b[1]
                dist = np.sqrt((cx_b - cx_a) ** 2 + (cy_b - cy_a) ** 2)
                if dist < min_dist:
                    min_dist = dist
            displacements.append(min_dist)

        v_star = np.mean(displacements)
        velocity_list.append(v_star)

    sequence_conD_sum = 0.0
    for t in range(num_frames - 1):
        sequence_conD_sum += (a * conS_list[t]) / (b + velocity_list[t])

    conD = sequence_conD_sum / (num_frames - 1)
    return float(conD)


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
    headers = False
    i = 1
    dataset_rows = []

    processed_videos = set()
    try:
        df = pd.read_csv(csv_path, header=0)
        processed_videos = set(df['file_name'].to_list())

        if len(processed_videos) == video_len:
            print("Processed all videos")
            exit()

        print(
            f"Found existing processed videos. Resuming pipeline. {len(processed_videos)} image(s) already annotated."
        )
        video_len = video_len - len(processed_videos)

    except Exception as e:
        headers = ["file_name", "frames_batch", "conD", "bounding_boxes"]
        print(f"Pandas: {e}")

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

            conD = get_conD(bboxes)

            if annotate:
                file_path = os.path.join(
                    dataset.frames, f"{os.path.splitroot(file.name)}.png"
                )
                annotate_frame(frames, results, file_path)

            dataset_rows.append(
                [
                    file.name,
                    json.dumps(frames_batch.tolist()),
                    json.dumps(conD),
                    json.dumps(bboxes),
                ]
            )

            i = i + 1

    if len(dataset_rows):
        df = pd.DataFrame(dataset_rows)
        df.to_csv(csv_path, mode="a", header=headers, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="create train or test dataset")
    parser.add_argument("-b", "--batch", help="size of batch to process", default=10)
    parser.add_argument(
        "-n", "--num_frames", help="number of frames to extract", default=8
    )
    parser.add_argument(
        "-a", "--annotate", help="annotate image with bounding boxes", default=False
    )
    parser.add_argument("-c", "--conf", help="confidence of yolo model", default=0.15)
    args = parser.parse_args()

    if args.dataset == "train":
        create_dataset(
            dataset=TRAIN_DATASET,
            BATCH_SIZE=int(args.batch),
            num_frames=int(args.num_frames),
            annotate=args.annotate,
            CONF=args.conf,
        )
    elif args.dataset == "test":
        create_dataset(
            dataset=TEST_DATASET,
            BATCH_SIZE=int(args.batch),
            num_frames=int(args.num_frames),
            annotate=args.annotate,
            CONF=args.conf,
        )
