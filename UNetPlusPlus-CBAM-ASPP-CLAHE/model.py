import torch
import torch.nn as nn
from cbam import CBAM
from aspp import ASPPBlock


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


class NestedUNetV2(nn.Module):
    """U-Net++ + CBAM (all blocks) + ASPP (bottleneck)."""

    def __init__(self, in_ch=3, out_ch=1, filters=(32, 64, 128, 256, 512)):
        super().__init__()
        self.depth = len(filters)
        self.filters = filters

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder + CBAM
        self.enc_blocks = nn.ModuleList()
        self.enc_cbam = nn.ModuleList()
        prev = in_ch
        for f in filters:
            self.enc_blocks.append(DoubleConv(prev, f))
            self.enc_cbam.append(CBAM(f))
            prev = f

        # ASPP at bottleneck (after deepest encoder, before decoder)
        self.aspp = ASPPBlock(filters[-1], filters[-1])

        # Decoder + CBAM
        self.convs = nn.ModuleList()
        self.dec_cbam = nn.ModuleList()
        for i in range(self.depth - 1):
            conv_row = nn.ModuleList()
            cbam_row = nn.ModuleList()
            for j in range(self.depth - 1 - i):
                in_dim = filters[i + 1] + filters[i] * (j + 1)
                conv_row.append(DoubleConv(in_dim, filters[i]))
                cbam_row.append(CBAM(filters[i]))
            self.convs.append(conv_row)
            self.dec_cbam.append(cbam_row)

        self.out1 = nn.Conv2d(filters[0], out_ch, 1)
        self.out2 = nn.Conv2d(filters[0], out_ch, 1)
        self.out3 = nn.Conv2d(filters[0], out_ch, 1)
        self.out4 = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x):
        # Encoder
        enc = []
        for i, (blk, cbam) in enumerate(zip(self.enc_blocks, self.enc_cbam)):
            x = blk(x)
            x = cbam(x)
            enc.append(x)
            if i < self.depth - 1:
                x = self.pool(x)

        # ASPP at bottleneck
        enc[-1] = self.aspp(enc[-1])

        # Decoder
        X = [[None] * self.depth for _ in range(self.depth)]
        for i in range(self.depth):
            X[i][0] = enc[i]

        for j in range(1, self.depth):
            for i in range(self.depth - j):
                up = self.up(X[i + 1][j - 1])
                skip_list = [up]
                for k in range(j):
                    skip_list.append(X[i][k])
                cat = torch.cat(skip_list, dim=1)
                X[i][j] = self.convs[i][j - 1](cat)
                X[i][j] = self.dec_cbam[i][j - 1](X[i][j])

        out1 = self.out1(X[0][1])
        out2 = self.out2(X[0][2])
        out3 = self.out3(X[0][3])
        out4 = self.out4(X[0][4])
        return [out1, out2, out3, out4]


if __name__ == "__main__":
    m = NestedUNetV2(3, 1)
    x = torch.randn(2, 3, 256, 256)
    outs = m(x)
    for i, o in enumerate(outs):
        print(f"  L{i+1}: {o.shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
