from PIL import Image
import pandas as pd
import torch
import json
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes
import os
from scipy.ndimage import gaussian_filter
import numpy as np
from decord import VideoReader, cpu


class TrafficDensityDataset(Dataset):
    """Traffic Dataset for Density Estimation"""

    def __init__(self, csv_path, root_dir, transform=None):
        self.csv = pd.read_csv(csv_path, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name, frames_batch, bboxes = self.csv.iloc[idx]
        frames_batch, bboxes = json.loads(frames_batch), json.loads(bboxes)

        vr = VideoReader(os.path.join(self.root_dir, file_name), ctx=cpu(0))
        frames = vr.get_batch(frames_batch)
        frame_list = []

        for frame, bbox in frames, bboxes:

            frame = Image.fromarray(frame)

            if self.transform:
                bbox = np.array(bbox).reshape(-1, 4)
                bbox = torch.from_numpy(bbox).to(dtype=torch.float32)
                bbox = BoundingBoxes(
                    bbox, format="XYWH", canvas_size=(frame.height, frame.width)
                )
                frame = self.transform(frame, bbox)

            frame_list.append(frame)

        return frame_list


class ToDensityMap(torch.nn.Module):
    """Convert images to density maps"""

    def forward(self, image, bbox):
        W, H = image.size
        density_map = np.zeros((W, H), dtype=np.float32)

        for x, y, w, h in bbox:
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                single_point_map = np.zeros((H, W), dtype=np.float32)
                single_point_map[y, x] = 1.0

                sigma = max(2, int((w + h) / 8))

                blurred_point = gaussian_filter(single_point_map, sigma=sigma)

                density_map += blurred_point

        if density_map.sum() > 0 and len(bbox) > 0:
            density_map = (density_map / density_map.sum()) * len(bbox)

        return density_map
