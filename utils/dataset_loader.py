from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from scipy.ndimage import gaussian_filter
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as transforms


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

    def _generate_map(self, image_shape, bbox):
        W, H = image_shape
        density_map = np.zeros((W, H), dtype=np.float32)

        for x, y, w, h in bbox:
            x, y = int(x), int(y)
            if 0 <= x < W and 0 <= y < H:
                single_point_map = np.zeros(image_shape, dtype=np.float32)
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

        video_name, csv_name = self.csv.iloc[idx]
        video_path = os.path.join(self.video_dir, video_name)
        csv_path = os.path.join(self.csv_dir, csv_name)
        df = pd.read_csv(csv_path, header=None)

        vr = VideoReader(video_path, ctx=cpu(0))
        frame_list = []
        map_list = []

        for i, data in df.iterrows():
            image = Image.fromarray(vr[i].asnumpy())
            bbox = data.dropna().values.astype("float").reshape(-1, 4)

            if self.transform:
                image, bbox = self.transform(image, bbox)
                map = self._generate_map(image.size, bbox)

            else:
                map = self._generate_map(image.size, bbox)

            frame_list.append(image)
            map_list.append(map)

        if self.tensor_transform:
            image = self.tensor_transform(frame_list)
            map = self.tensor_transform(map_list)
        else:
            image = frame_list
            map = map_list

        return {"image": image, "density_map": map}


class Rescale(object):
    """Rescale the image and its bounding boxes to a given size"""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def _scale_bboxes(self, bboxes, orig_hw, new_hw):
        orig_h, orig_w = orig_hw
        new_h, new_w = new_hw

        x_ratio = new_w / orig_w
        y_ratio = new_h / orig_h

        scaled_bboxes = []

        for x, y, w, h in bboxes:
            new_x = x * x_ratio
            new_y = y * y_ratio
            new_w = w * x_ratio
            new_h = h * y_ratio
            scaled_bboxes.append([new_x, new_y, new_w, new_h])

        return scaled_bboxes

    def __call__(self, image, bbox):
        w, h = image.size

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_w, new_h))

        scaled_bbox = self._scale_bboxes(bbox, (h, w), (new_h, new_w))

        return img, scaled_bbox


class ToTensor(object):
    """Convert imgaes to tensor"""

    def __call__(self, sample):
        transform = transforms.Compose([ToTensor()])

        tensor = [transform(s) for s in sample]  # C, H, W

        tensor = torch.stack(tensor, dim=0)  # T, C, H, W

        tensor = tensor.permute(1, 0, 2, 3)  # C, T, H, W

        tensor = tensor.unsqueeze(0)  # B, C, T, H, W

        return tensor
