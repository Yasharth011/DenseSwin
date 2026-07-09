import torch.nn as nn
import torch
from torch import Tensor
from models import VideoSwinE, VideoSwinD, DensityHead
import torch.nn.functional as F
from torchvision.models.video.swin_transformer import Swin3D_T_Weights
from typing import Optional


class DenseSwin(nn.Module):
    """
    Implements DenseSwin model for traffic density estimation
    Args:
        linear_ch (int): Input channel size for Regression Head
        backbone (nn.Module): Backbone Network, Default: VSwinTransformer
        density_head (nn.Module): Density Map Network, Default: densityConv
        neck (nn.Module): Neck Network, Default: VSwinTransformer(Decoder)
        num_class (nn.Module): number of classes for Classification, Default: 1 (Regression)
    """

    def __init__(
        self,
        linear_ch: int = 96,
        size: Optional[tuple[int, int, int]] = (8, 224, 384),
        backbone: Optional[nn.Module] = None,
        density_head: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        num_class: int = 1,
    ):
        super().__init__()

        if backbone is None:
            self.backbone = VideoSwinE(
                patch_size=[2, 4, 4],
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=[8, 7, 7],
                stochastic_depth_prob=0.2,
            )
            self.backbone.load_state_dict(
                Swin3D_T_Weights.DEFAULT.get_state_dict(progress=True), strict=False
            )
        else:
            self.backbone = backbone

        if density_head is None:
            self.density_head = DensityHead(size, [768, 512, 256])
        else:
            self.density_head = density_head

        if neck is None:
            self.neck = VideoSwinD(
                embed_dim=768,
                depths=[2, 6, 2, 2],
                num_heads=[24, 12, 6, 3],
                window_size=[8, 7, 7],
                stochastic_depth_prob=0.2,
            )
        else:
            self.neck = neck

        self.num_class = num_class
        self.head = nn.Linear(linear_ch, self.num_class)

    def forward(
        self, x: Tensor, density_only: bool = False
    ) -> tuple[Optional[Tensor], Tensor]:
        # x: B, C, T, H, W

        x = self.backbone(x)

        F_dm, D = self.density_head(x, density_only=density_only)

        # D branches off before the neck, so a density-only objective gives
        # neck/head/conv_align no gradient -- running them would be dead compute.
        if density_only:
            return None, D

        x = self.neck(x + F_dm)

        x = F.adaptive_avg_pool3d(x, 1)

        x = torch.flatten(x, 1)

        x = self.head(x)

        if self.num_class == 1:
            x = x.squeeze(-1)

        return x, D
