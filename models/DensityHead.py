import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class SpatialPoolPyramid(nn.Module):
    """
    Implements a Spatial Pool Pyramid from https://github.com/weizheliu/Context-Aware-Crowd-Counting
    Args :
        in_ch (int): input channel size
        out_ch (int): output channel size
        size (Tuple[int, ...]): tuple of the adaptive pooling grid sizes
    """

    def __init__(self, in_ch: int, out_ch: int, sizes: tuple[int, ...]):
        super().__init__()

        self.scales = nn.ModuleList([self._make_scale(in_ch, size) for size in sizes])

        self.bottleneck = nn.Conv3d(in_ch * 2, out_ch, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        self.weight_net = nn.Conv3d(in_ch, in_ch, kernel_size=1)

    def __make_weight(self, feature, scale_feature):
        weight_feature = feature - scale_feature
        return torch.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool3d(output_size=(1, size, size))
        conv = nn.Conv3d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x: Tensor) -> Tensor:
        # x : B, C, T, H, W

        T, H, W = x.size(2), x.size(3), x.size(4)

        multi_scales = [
            F.interpolate(input=stage(x), size=(T, H, W), mode="trilinear")
            for stage in self.scales
        ]

        weights = [
            self.__make_weight(x, scale_feature) for scale_feature in multi_scales
        ]

        overall_features = [
            (
                multi_scales[0] * weights[0]
                + multi_scales[1] * weights[1]
                + multi_scales[2] * weights[2]
                + multi_scales[3] * weights[3]
            )
            / (weights[0] + weights[1] + weights[2] + weights[3])
        ] + [x]

        bottle = self.bottleneck(torch.cat(overall_features, 1))

        return self.relu(bottle)


class DensityHead(nn.Module):
    """
    Implements a Dilated CNNs with Spatial Pyramid Pooling to create density feature map
    Args :
        size (Tuple[int,int]): target size (T, H, W) of density feature map
        ch_dims (List[int]): list of channel size for each layer
        layer (nn.Module, optional): Dilated Conv Block Default: None.
        pooling layer (nn.Module, optional): Spatial Pooling Layer Default: None.
    """

    def __init__(
        self,
        size: tuple[int, int, int],
        ch_dims: list[int],
        layer: Optional[Callable[..., nn.Module]] = None,
        pooling_layer: Optional[nn.Module] = None,
    ):
        super().__init__()

        if layer is None:
            layer = lambda in_ch, out_ch: nn.Sequential(
                nn.Conv3d(
                    in_ch, out_ch, kernel_size=3, padding=(1, 2, 2), dilation=(1, 2, 2)
                ),
                nn.ReLU(),
            )

        if pooling_layer is None:
            self.pooling_layer = SpatialPoolPyramid(
                ch_dims[-1], ch_dims[-1], (1, 2, 3, 6)
            )
        else:
            self.pooling_layer = pooling_layer

        layers: list[nn.Module] = []
        for i in range(len(ch_dims) - 1):
            layers.append(layer(ch_dims[i], ch_dims[i + 1]))
        self.conv_block = nn.Sequential(*layers)

        self.conv_align = layer(ch_dims[-1], ch_dims[0])

        self.density_head = nn.Sequential(nn.Conv3d(ch_dims[-1], 1, kernel_size=1))
        self.density_act = nn.ReLU()

        nn.init.normal_(self.density_head[0].weight, std=0.01)
        nn.init.zeros_(self.density_head[0].bias)

        self.size = size

    def forward(self, x: Tensor, density_only: bool = False) -> tuple[Tensor, Tensor]:
        # x : B, C, T, H, W

        F_dm: Tensor = self.conv_block(x)
        F_dm = self.pooling_layer(F_dm)

        D = self.density_head(F_dm)
        D = F.interpolate(D, size=self.size, mode="trilinear", align_corners=False)
        D = self.density_act(D)

        if density_only:
            return None, D

        return self.conv_align(F_dm), D
