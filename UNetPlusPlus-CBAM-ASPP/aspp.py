"""
ASPP (Atrous Spatial Pyramid Pooling) — DeepLabV3 style.
Bottleneck multi-scale context module with parallel dilated convs + global pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=6, dilation=6),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=12, dilation=12),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=18, dilation=18),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid_ch, 1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_ch * 5, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        feat1 = self.conv1(x)
        feat6 = self.conv6(x)
        feat12 = self.conv12(x)
        feat18 = self.conv18(x)
        global_feat = F.interpolate(self.global_branch(x), size=(h, w), mode="bilinear", align_corners=True)

        fused = torch.cat([feat1, feat6, feat12, feat18, global_feat], dim=1)
        return self.fusion(fused)
