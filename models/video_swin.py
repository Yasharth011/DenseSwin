""" Credits to pytorch Video Swin Transformer Implementation : https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py """

from torch.cuda import is_available
import torch.nn as nn
import torch
from torch import Tensor
from typing import Callable, Optional
from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock
from torchvision.models.video.swin_transformer import (
    ShiftedWindowAttention3d,
    PatchEmbed3d,
    Swin3D_T_Weights
)
from functools import partial


class VideoSwinT(nn.Module):
    """
    Implements 3D Swin Transformer from the `"Video Swin Transformer" <https://arxiv.org/abs/2106.13230>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 400.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
        patch_embed (nn.Module, optional): Patch Embedding layer. Default: None.
    """

    def __init__(
        self,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        num_heads: list[int],
        window_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 400,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        patch_embed: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        if block is None:
            block = partial(SwinTransformerBlock, attn_layer=ShiftedWindowAttention3d)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        if patch_embed is None:
            patch_embed = PatchEmbed3d

        # split image into non-overlapping patches
        self.patch_embed = patch_embed(
            patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer
        )
        self.pos_drop = nn.Dropout(p=dropout)

        layers: list[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: list[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=ShiftedWindowAttention3d,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: B C T H W
        x = self.patch_embed(x)  # B _T _H _W C
        x = self.pos_drop(x)
        x = self.features(x)  # B _T _H _W C
        x = x.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W
        return x

model = VideoSwinT(
        patch_size=[2, 4, 4],      
        embed_dim=96,               
        depths=[2, 2, 6, 2],      
        num_heads=[3, 6, 12, 24],   
        window_size=[8, 7, 7],   
        stochastic_depth_prob=0.2   
)
weights = Swin3D_T_Weights.DEFAULT
model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

swin3d = model.eval()
if torch.cuda.is_available():
    swin3d = swin3d.cuda()

