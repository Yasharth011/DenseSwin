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
class UCSDConfig:
    parent_path: str

    videos: str = field(init=False)
    csv: str = field(init=False)
    train_csv: str = field(init=False)
    test_csv: str = field(init=False)

    def __post_init__(self):
        self.videos = os.path.join(self.parent_path, "video")
        self.csv = os.path.join(self.parent_path, "ImageMaster")
        self.train_csv = os.path.join(self.parent_path, "EvalSet_train")
        self.test_csv = os.path.join(self.parent_path, "EvalSet_test")


@dataclass
class TRANCOSConfig:
    parent_path: str

    images: str = field(init=False)
    training: str = field(init=False)
    validation: str = field(init=False)
    trainval: str = field(init=False)
    test: str = field(init=False)
    cache: str = field(init=False)

    def __post_init__(self):
        self.images = os.path.join(self.parent_path, "images")
        image_sets = os.path.join(self.parent_path, "image_sets")
        self.training = os.path.join(image_sets, "training.txt")
        self.validation = os.path.join(image_sets, "validation.txt")
        self.trainval = os.path.join(image_sets, "trainval.txt")
        self.test = os.path.join(image_sets, "test.txt")
        self.cache = os.path.join(self.parent_path, "cache")

    def split(self, name: str) -> str:
        return getattr(self, name)


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
TRANCOS_MASTER = TRANCOSConfig("dataset/TRANCOS_v3")
UCSD_MASTER = UCSDConfig("dataset/UCSD")

MODEL_CONFIG = ModelConfig("")
