import torch.nn as nn
import torch.nn.functional as F 

def conv_bn_reul(in_ch, out_ch):
    return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

class DensityConv(nn.Module):
    def __init__(self, in_ch: int = 768):
        super().__init__()
        self.conv_blocks = nn.Sequential(
                conv_bn_reul(in_ch, 512),
                conv_bn_reul(512, 256),
                conv_bn_reul(256, 256)
                )
        self.density_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, swin_c5, target_size):
        F_dm =  self.conv_blocks(swin_c5)

        features = F.interpolate(
                F_dm, 
                size=target_size,
                mode='bilinear',
                align_corners=False
                )

        D = self.density_head(features)
        D = F.relu(D)

        return F_dm, D
