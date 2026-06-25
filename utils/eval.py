from decord import VideoReader, cpu
from torchvision import transforms
from models import DenseSwin
import torch
from torchvision.transforms import ToTensor, Resize
import numpy as np
from PIL import Image
import argparse
from utils import MODEL_CONFIG
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", help="model weights checkpoint")
parser.add_argument(
    "-v", "--video", help="video for evaludation")
args = parser.parse_args()

checkpoint = torch.load(os.path.join(MODEL_CONFIG.checkpoints, args.checkpoint), weights_only=True)
model = DenseSwin()
model.load_state_dict(checkpoint)
model.eval()
model.to(device=device)

vr = VideoReader(args.video, ctx=cpu(0))

frames_batch = np.linspace(0, len(vr) - 1, num=8, dtype=int)

frames = [Image.fromarray(frame) for frame in vr.get_batch(frames_batch).asnumpy()]

transform = transforms.Compose([Resize((224, 384)), ToTensor()])

frame_list = []

for frame in frames: 
    frame = transform(frame)
    frame_list.append(frame)

tensor = torch.stack(frame_list, dim=0).permute(1, 0, 2, 3)
tensor = tensor.unsqueeze(0)
tensor = tensor.to(device=device)

conD, _ = model(tensor)
print('Congestion factor : ', conD)
