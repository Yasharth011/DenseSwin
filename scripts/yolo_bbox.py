import os
import numpy as np
import cv2
import pandas as pd
import torch
from ultralytics import YOLO
from utils import DATASET_CONFIG

model = YOLO("yolo11x.pt")

device = "0" if torch.cuda.is_available() else "cpu"

# COCO indices for traffic elements (0: person, 2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASSES = [0, 2, 3, 5, 7]
# detecting person to detect overlapped bikes by detecting the person

output_dir = str(DATASET_CONFIG.annotated_frames)


def detect_vehicles(image_path, conf=0.10):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return np.empty((0, 4))

    results = model.predict(
        source=image_path,
        conf=conf,
        classes=VEHICLE_CLASSES,
        verbose=False,
        device=device
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

            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 0), 2)
            cv2.circle(
                img, (x_center, y_center), radius=4, color=(0, 0, 255), thickness=-1
            )

    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"annotated_{base_name}")
    cv2.imwrite(save_path, img)

    if len(all_boxes) > 0:
        vehicle_bboxes = np.vstack(all_boxes)
    else:
        vehicle_bboxes = np.empty((0, 4))

    return vehicle_bboxes


frames_path = DATASET_CONFIG.frames
csv_path = DATASET_CONFIG.frames_csv
frames_len = len(os.listdir(frames_path))

dataset_rows = []
i = 0

# create set of images already processed 
processed_images = set()
try: 
    df = pd.read_csv(csv_path)
    processed_images = set(df[0].to_list())
    print(f"Found existing tracking file. Resuming pipeline. {len(processed_images)} images already annotated.")
except Exception as e:
    print(f"Could not read existing CSV, starting fresh. Error: {e}")

frames_len = frames_len - len(processed_images)

BATCH_SIZE = 9000

for frame in os.scandir(frames_path):

    if os.path.basename(frame) in processed_images:
        continue

    bboxes = detect_vehicles(frame.path)
    flat_boxes = bboxes.flatten().tolist()
    row_data = [os.path.basename(frame.path)] + bboxes.flatten().tolist()
    dataset_rows.append(row_data)
    i += 1
    print(f"{i}/{frames_len}")

    if i == BATCH_SIZE:
        df = pd.DataFrame(dataset_rows)
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print("Finished Batch")
