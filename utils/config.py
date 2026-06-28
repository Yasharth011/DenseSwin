from dataclasses import dataclass, field
import os


@dataclass
class CBT2015Config:
    parent_path: str

    videos: str = field(init=False)
    frames: str = field(init=False)
    csv: str = field(init=False)

    def __post_init__(self):
        self.videos = os.path.join(self.parent_path, "videos")
        self.frames = os.path.join(self.parent_path, "frames")
        self.csv = os.path.join(self.parent_path, "traffic_video.csv")


@dataclass
class ModelConfig:
    parent_path: str

    logs: str = field(init=False)
    checkpoints: str = field(init=False)

    def __post_init__(self):
        self.logs = os.path.join(self.parent_path, "logs")
        self.checkpoints = os.path.join(self.parent_path, "checkpoints")


CBT2015_TRAIN = CBT2015Config("dataset/CBT2015/train")
CBT2015_VAL = CBT2015Config("dataset/CBT2015/validation")
MODEL_CONFIG = ModelConfig("")
