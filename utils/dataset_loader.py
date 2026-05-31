from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from scipy.ndimage import gaussian_filter
import numpy as np

class TrafficDensityDataset(Dataset):
    """Traffic Dataset for Density Estimation"""

    def __init__(self, csv_file, root_dir, transform=None, transform_tensor=None):
        self.frames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.transform_tensor = transform_tensor

    def __len__(self):
        return len(self.frames)

    def _generate_map(self, image_shape, bboxes):
        H, W = image_shape
        density_map = np.zeros(image_shape, dtype=np.float32)
    
        for x, y, w, h in bboxes:
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                single_point_map = np.zeros(image_shape, dtype=np.float32)
                single_point_map[y, x] = 1.0
    
                sigma = max(2, int((w + h) / 8))
    
                blurred_point = gaussian_filter(single_point_map, sigma=sigma)
    
                density_map += blurred_point
    
        if density_map.sum() > 0 and len(bboxes) > 0:
            density_map = (density_map / density_map.sum()) * len(bboxes)
    
        return density_map

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.frames.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        points = self.frames.iloc[idx, 1:].values.astype("float").reshape(-1, 2)

        sample = {"image": image, "points": points}

        if self.transform:
            sample = self.transform(sample)

        density_array = self._generate_map(sample["image"].size, sample["points"])

        if self.transform_tensor:
            sample = self.transform_tensor(sample)

        return {"image": image, "density_map": density_array}


class Rescale(object):
    """Rescale the image in a given sample to a given size"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def _scale_points(self, points, orig_hw, new_hw):
        orig_h, orig_w = orig_hw
        new_h, new_w = new_hw

        scaled_points = []

        for x, y in points:
            new_x = x * (new_w / orig_w)
            new_y = y * (new_h / orig_h)
            scaled_points.append((new_x, new_y))

        return scaled_points

    def __call__(self, sample):
        image, points = sample["image"], sample["points"]
        w, h = image.size

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size

            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_h, new_w))
        pts = self._scale_points(points, (h, w), (new_h, new_w))

        return {"image": img, "points": pts}


class ToTensor(object):
    """Convert imgaes to tensor"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image, points = sample["image"], sample["points"]

        image = self.transform(image).unsqueeze(0)

        return {"image": image, "points": points}
