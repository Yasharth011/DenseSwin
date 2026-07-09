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
from torchvision.transforms import v2
from scipy.signal import fftconvolve
from scipy.io import loadmat
import torch.nn.functional as F
import torchvision.transforms.functional as TF


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

        if self.transform:
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

        vr = VideoReader(os.path.join(self.root_dir, file_name + ".avi"), ctx=cpu(0))

        # skip the corrupted first frame in UCSD
        frames_batch = np.linspace(1, len(vr) - 1, num=self.num_frames, dtype=int)

        frames = [
            Image.fromarray(frame) for frame in vr.get_batch(frames_batch).asnumpy()
        ]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            frames = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

        return (frames, label)

    def get_subset(self, path, fold=0):

        with open(path, "r") as file:
            lines = file.readlines()
            indices = [int(idx) for idx in lines[fold].strip().split(",")]

        return indices


class TRANCOS(Dataset):
    """Dataset Loader for TRANCOS.

    Each item is ``(frame, density, roi, count)``:
        frame   -- (3, T, H, W) normalised clip, the still image repeated T times
        density -- (1, H, W) ROI-masked density map that sums to ``count``
        roi     -- (1, H, W) binary region-of-interest mask
        count   -- scalar, the number of vehicles inside the ROI

    The density map stays in true vehicle units, so GAME-0 computed against it
    is the absolute count error and is directly comparable to published TRANCOS
    numbers.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # bump whenever _targets changes, so stale .npz files are not reused
    CACHE_VERSION = 2

    def __init__(
        self,
        root_dir,
        csv_path,
        target_size=None,
        num_frames=8,
        augment=False,
        min_crop=0.75,
        cache_dir=None,
    ):
        self.root_dir = root_dir
        self.csv = pd.read_csv(csv_path, header=None)
        self.target_size = target_size
        self.num_frames = num_frames
        self.augment = augment
        self.min_crop = min_crop
        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        self.jitter = (
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            if augment
            else None
        )

    def __len__(self):
        return len(self.csv)

    def _getmap(self, path: str, kernel_size: int = 90, sigma: float = 15.0):

        dots = np.array(Image.open(path))
        red_channel = dots[:, :, 0].astype(np.float64) / 255.0

        radius = (kernel_size - 1) / 2.0
        y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
        total = kernel.sum()
        if total != 0:
            kernel /= total

        map = fftconvolve(red_channel, kernel, mode="same")

        return map

    def _targets(self, file_name):
        """Full-resolution ROI-masked density map and mask, cached on disk.

        The 90x90 FFT convolution and the .mat read cost more than the forward
        pass at small batch sizes, and neither depends on the epoch.
        """
        cache = (
            os.path.join(self.cache_dir, f"{file_name}.v{self.CACHE_VERSION}.npz")
            if self.cache_dir
            else None
        )
        if cache and os.path.exists(cache):
            with np.load(cache) as stored:
                return stored["density"], stored["roi"].astype(np.float32)

        dots_path = os.path.join(self.root_dir, file_name + "dots.png")
        density = self._getmap(dots_path)
        roi = loadmat(os.path.join(self.root_dir, file_name + "mask.mat"))["BW"]

        h, w = density.shape
        roi = roi[:h, :w].astype(np.float32)

        # the TRANCOS protocol counts vehicles inside the ROI only
        density = density * roi

        # Convolving with 'same' and then masking sheds 2-6% of the Gaussian
        # mass over the image border and the ROI edge. Left alone that biases
        # every target low by up to ~2 vehicles, which is the size of the metric
        # we are trying to report. Rescale so the map sums to the exact number
        # of annotated dots inside the ROI.
        dots = np.array(Image.open(dots_path))[:h, :w, 0] > 128
        n_dots = float((dots & (roi > 0)).sum())
        total = density.sum()
        if total > 0:
            density = density * (n_dots / total)

        density = density.astype(np.float32)

        if cache:
            # the map is mostly zeros and the mask is binary, so this is ~10x
            # smaller than a raw float32 dump of both
            tmp = f"{cache}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, density=density, roi=roi.astype(bool))
            os.replace(tmp, cache)

        return density, roi

    def _resize(self, x, mode):
        kwargs = {"align_corners": False} if mode != "nearest" else {}
        return F.interpolate(
            x.unsqueeze(0), size=self.target_size, mode=mode, **kwargs
        ).squeeze(0)

    def _augment(self, frame, density, roi):

        if torch.rand(()) < 0.5:
            frame, density, roi = (
                torch.flip(t, dims=[-1]) for t in (frame, density, roi)
            )

        h, w = density.shape[-2:]
        for _ in range(10):
            scale = self.min_crop + torch.rand(()).item() * (1.0 - self.min_crop)
            ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
            top = int(torch.randint(0, h - ch + 1, ()))
            left = int(torch.randint(0, w - cw + 1, ()))
            window = (..., slice(top, top + ch), slice(left, left + cw))
            # a crop that misses the ROI entirely carries no supervision
            if roi[window].sum() > 0:
                frame, density, roi = frame[window], density[window], roi[window]
                break

        return self.jitter(frame), density, roi

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.splitext(self.csv.iloc[idx, 0])[0]

        frame = Image.open(os.path.join(self.root_dir, file_name + ".jpg")).convert(
            "RGB"
        )
        density, roi = self._targets(file_name)

        frame = TF.to_tensor(frame)
        density = torch.from_numpy(density).unsqueeze(0)
        roi = torch.from_numpy(roi).unsqueeze(0)

        if self.augment:
            frame, density, roi = self._augment(frame, density, roi)

        # measured after cropping, so it reflects the vehicles actually visible
        count = density.sum()

        frame = TF.normalize(frame, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

        if self.target_size:
            frame = self._resize(frame, "bilinear")
            density = self._resize(density, "bilinear")
            roi = self._resize(roi, "nearest")

        # resampling does not conserve mass -- restore the true count
        density = density * roi
        total = density.sum()
        if total > 0:
            density = density * (count / total)

        frame = frame.unsqueeze(1).expand(-1, self.num_frames, -1, -1).contiguous()

        return frame, density, roi, count
