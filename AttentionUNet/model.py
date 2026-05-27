import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections. Suppresses irrelevant regions."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from decoder (upsampled)
        # x: skip connection from encoder
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net (Oktay et al., MIDL 2018).
    U-Net with attention gates on skip connections.
    """

    def __init__(self, in_ch=3, out_ch=1, base_ch=64):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = UpBlock(base_ch * 16, base_ch * 8)
        self.ag4 = AttentionGate(F_g=base_ch * 8, F_l=base_ch * 8, F_int=base_ch * 4)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.ag3 = AttentionGate(F_g=base_ch * 4, F_l=base_ch * 4, F_int=base_ch * 2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.ag2 = AttentionGate(F_g=base_ch * 2, F_l=base_ch * 2, F_int=base_ch)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = UpBlock(base_ch * 2, base_ch)
        self.ag1 = AttentionGate(F_g=base_ch, F_l=base_ch, F_int=base_ch // 2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with attention gates
        d4 = self.up4(b)
        e4_gated = self.ag4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4_gated], dim=1))

        d3 = self.up3(d4)
        e3_gated = self.ag3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_gated], dim=1))

        d2 = self.up2(d3)
        e2_gated = self.ag2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_gated], dim=1))

        d1 = self.up1(d2)
        e1_gated = self.ag1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_gated], dim=1))

        return self.out_conv(d1)


if __name__ == "__main__":
    m = AttentionUNet(3, 1, base_ch=64)
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
