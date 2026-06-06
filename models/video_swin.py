import torch 
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)

swin3d = model.features

if torch.cuda.is_available:
    swin3d = swin3d.cuda().eval()
else: 
    swin3d = swin3d.eval()
