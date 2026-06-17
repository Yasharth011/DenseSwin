from PIL import Image
import pandas as pd
import torch
import json
from torch.utils.data import Dataset
import os
from scipy.ndimage import gaussian_filter
import numpy as np
from decord import VideoReader, cpu


class TrafficDensityDataset(Dataset):
    """Traffic Dataset for Density Estimation"""

    def __init__(self, root_dir, csv_path, transform=None):
        self.csv = pd.read_csv(csv_path, header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def _getmap(self, image, bbox):
        W, H = image.size
        density_map = np.zeros((H, W), dtype=np.float32)

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

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name, frames_batch, conD, bboxes = self.csv.iloc[idx]
        frames_batch, bboxes = (
            json.loads(frames_batch),
            json.loads(bboxes),
        )

        vr = VideoReader(os.path.join(self.root_dir, file_name), ctx=cpu(0))

        frames = vr.get_batch(frames_batch).asnumpy()
        frame_list = []
        map_list = []

        for frame, bbox in zip(frames, bboxes):

            frame = Image.fromarray(frame)

            bbox = np.array(bbox).reshape(-1, 4)
            bbox = torch.from_numpy(bbox).to(dtype=torch.float32)

            map = self._getmap(frame, bbox)
            map = torch.from_numpy(map).unsqueeze(0).float()

            if self.transform:
                frame = self.transform(frame)
                map = self.transform(map)

            frame_list.append(frame)
            map_list.append(map)

        frame_list = torch.stack(frame_list, dim=0).permute(1, 0, 2, 3)
        map_list = torch.stack(map_list, dim=0).permute(1, 0, 2, 3)

        return (frame_list, conD, map_list)
