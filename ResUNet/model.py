import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual conv block with skip connection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class ResUNet(nn.Module):
    """
    ResUNet: U-Net with residual blocks instead of standard double conv.
    Zhang et al., 2018.
    """

    def __init__(self, in_ch=3, out_ch=1, base_ch=64):
        super().__init__()

        self.enc1 = ResidualBlock(in_ch, base_ch)
        self.enc2 = ResidualBlock(base_ch, base_ch * 2)
        self.enc3 = ResidualBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ResidualBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(base_ch * 8, base_ch * 16)

        self.up4 = UpBlock(base_ch * 16, base_ch * 8)
        self.dec4 = ResidualBlock(base_ch * 16, base_ch * 8)

        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.dec3 = ResidualBlock(base_ch * 8, base_ch * 4)

        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.dec2 = ResidualBlock(base_ch * 4, base_ch * 2)

        self.up1 = UpBlock(base_ch * 2, base_ch)
        self.dec1 = ResidualBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


if __name__ == "__main__":
    m = ResUNet(3, 1, base_ch=64)
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
