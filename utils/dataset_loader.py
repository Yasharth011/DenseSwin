from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from scipy.ndimage import gaussian_filter
import numpy as np
from decord import VideoReader, cpu


class TrafficDensityDataset(Dataset):
    """Traffic Dataset for Density Estimation"""

    def __init__(
        self, csv_file, video_dir, csv_dir, transform=None, tensor_transform=None
    ):
        self.csv = pd.read_csv(csv_file, header=None)
        self.video_dir = video_dir
        self.csv_dir = csv_dir
        self.transform = transform
        self.tensor_transform = tensor_transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_name, csv_name = self.csv.iloc[idx]
        video_path = os.path.join(self.video_dir, video_name)
        csv_path = os.path.join(self.csv_dir, csv_name)
        df = pd.read_csv(csv_path, header=None)

        vr = VideoReader(video_path, ctx=cpu(0))

        frame_list = []

        for i, data in df.iterrows():

            frame = Image.fromarray(vr[i].asnumpy())

            if self.transform:
                bbox = data.dropna().values.astype("float").reshape(-1, 4)
                frame = self.transform({"frame": frame, "bbox": bbox})

            frame_list.append(frame)

        return frame_list


class ToDensityMap(object):
    """Convert images to density maps"""

    def __call__(self, sample):
        image, bbox = sample.values()
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
