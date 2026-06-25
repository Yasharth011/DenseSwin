from decord import VideoReader, cpu
from torchvision import transforms
from models import DenseSwin
import torch
from torchvision.transforms import ToTensor, Resize
import numpy as np
from PIL import Image

checkpoint = torch.load('checkpoints/density_module_3.pth', weights_only=True)
model = DenseSwin().to(device='cuda:0')
model.load_state_dict(checkpoint)
model.eval()

vr = VideoReader('test3.mp4', ctx=cpu(0))

frames_batch = np.linspace(0, len(vr) - 1, num=8, dtype=int)

frames = [Image.fromarray(frame) for frame in vr.get_batch(frames_batch).asnumpy()]

transform = transforms.Compose([Resize((224, 384)), ToTensor()])

frame_list = []

for frame in frames: 
    frame = transform(frame)
    frame_list.append(frame)

tensor = torch.stack(frame_list, dim=0).permute(1, 0, 2, 3)
tensor = tensor.unsqueeze(0)
tensor = tensor.to(device='cuda:0')
conD, _ = model(tensor)
print('Congestion factor : ',conD)
