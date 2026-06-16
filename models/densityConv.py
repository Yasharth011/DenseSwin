import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_reul(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DensityConv(nn.Module):
    def __init__(self, in_ch: int = 768):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            conv_bn_reul(in_ch, 512), conv_bn_reul(512, 256), conv_bn_reul(256, 256)
        )
        self.conv_align = conv_bn_reul(256, 768)
        self.density_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, swin3d_features, target_size):
        h, w = target_size
        B, T, H, W, C = swin3d_features.shape
        swin3d_features = swin3d_features.permute(0, 2, 1, 3, 4)
        flattened_batch = swin3d_features.reshape(B * T, C, H, W)
        features = self.conv_blocks(flattened_batch)
        F_dm = self.conv_align(features)
        F_dm = torch.sigmoid(F_dm) # activation function

        interpolated_features = F.interpolate(
            features, size=target_size, mode="bilinear", align_corners=False
        )

        D = self.density_head(interpolated_features)
        D = F.relu(D)

        F_dm = F_dm.view(B, T, 256, H, W)
        D = D.view(B, T, 1, h, w)

        return F_dm, D
