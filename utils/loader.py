from PIL import Image
import pandas as pd
import torch
import json
from torch.utils.data import Dataset
import os
from scipy.ndimage import gaussian_filter
import numpy as np
from decord import VideoReader, cpu
from torchvision import tv_tensors


class CBT2015(Dataset):
    """Dataset Loader for CBT2015"""

    def __init__(self, root_dir, csv_path, transform=None):
        self.csv = pd.read_csv(csv_path, header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def _getmap(self, size, bbox):
        H, W = size
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

        conD = (conD - self.csv["conD"].min()) / (
            self.csv["conD"].max() - self.csv["conD"].min()
        )

        vr = VideoReader(os.path.join(self.root_dir, file_name), ctx=cpu(0))

        frames = vr.get_batch(frames_batch).asnumpy()
        frame_list = []
        map_list = []

        for frame, bbox in zip(frames, bboxes):

            frame = Image.fromarray(frame)
            H, W = frame.size

            bbox = np.array(bbox).reshape(-1, 4)
            tv_bbox = tv_tensors.BoundingBoxes(
                data=bbox,
                format=tv_tensors.BoundingBoxFormat.CXCYWH,
                canvas_size=(H, W),
            )

            if self.transform:
                frame, tv_bbox = self.transform(frame, tv_bbox)
                H, W = frame.shape[-2:]

            map = self._getmap((H, W), tv_bbox.cpu().numpy())
            map = torch.from_numpy(map).unsqueeze(0).float()

            frame_list.append(frame)
            map_list.append(map)

        frame_list = torch.stack(frame_list, dim=0).permute(1, 0, 2, 3)
        map_list = torch.stack(map_list, dim=0).permute(1, 0, 2, 3)

        return (frame_list, conD, map_list)


class UCSD(Dataset):
    """Dataset Loader for UCSD"""

    def __init__(self, root_dir, csv_path, num_frames, transform=None):
        self.csv = pd.read_csv(
            csv_path,
            names=["index", "filename", "class"],
            index_col=0,
            skiprows=1,
            sep="\t",
        )
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.label_map = {"light": 0, "medium": 1, "heavy": 2}

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name, label = self.csv.iloc[idx]

        label = self.label_map[label.strip().lower()]

        vr = VideoReader(os.path.join(self.root_dir, file_name), ctx=cpu(0))

        # skip the corrupted first frame in UCSD
        frames_batch = np.linspace(1, len(vr) - 1, num=self.num_frames, dtype=int)

        frames = [
            Image.fromarray(frame) for frame in vr.get_batch(frames_batch).asnumpy()
        ]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

        return (tensor, label)

    def get_subset(self, path, fold=0):

        with open(path, "r") as file:
            lines = file.readlines()
            indices = [int(idx) for idx in lines[fold].strip().split(",")]

        return indices
