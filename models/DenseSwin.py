import torch.nn as nn
import torch
from torch import Tensor
from models import vSwinD, vSwinE, DensityConv


class DenseSwim(nn.Module) : 
    def __init__(self, backbone, density_module , neck ,output_size , in_channels =128, hidden_dim= 64):
        super().__init__()

        self.backbone = backbone
        self.density_module = density_module
        self.neck = neck 
        self.max_pool_3d = nn.AdaptiveAvgPool3d(output_size)
        self.linear_reg_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim ,1),
            nn.ReLU()
        )
        
    def forward(self , x):
        B, C, T, H, W = x.shape
        features = self.backbone (x)
        F_dm, D = self.density_module(features, (H, W) )
        output = self.neck(features + F_dm)
        pooled_output = self.max_pool_3d(output)
        regressed_output = self.linear_reg_head(pooled_output)
        return regressed_output
    

        
        

