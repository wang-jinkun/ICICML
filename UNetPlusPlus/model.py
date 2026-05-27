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


class NestedUNet(nn.Module):
    """
    U-Net++ (Zhou et al., DLMIA 2018).
    Nested dense skip pathways + deep supervision.

    Node grid (i=row from top, j=column from left):
      X_00 -> X_01 -> X_02 -> X_03 -> X_04  (outputs)
         |      |       |       |
          X_10 -> X_11 -> X_12 -> X_13
             |      |       |
              X_20 -> X_21 -> X_22
                 |      |
                  X_30 -> X_31
                     |
                      X_40 (bottleneck)

    Deep supervision on X_01 through X_04.
    """

    def __init__(self, in_ch=3, out_ch=1, filters=(32, 64, 128, 256, 512), deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_levels = len(filters)

        # Downsampling
        self.pool = nn.MaxPool2d(2)

        # Encoder conv blocks (column 0)
        self.encoders = nn.ModuleList()
        prev_ch = in_ch
        for f in filters:
            self.encoders.append(DoubleConv(prev_ch, f))
            prev_ch = f

        # Decoder nodes: convs[i][j] for X_{i, j+1}
        # convs[i][0] = X_{i,1}, convs[i][1] = X_{i,2}, etc.
        self.decoder_nodes = nn.ModuleList()
        for i in range(self.num_levels - 1):  # 0..3 (shallow to deep)
            row = nn.ModuleList()
            for j in range(self.num_levels - 1 - i):  # how many decoder nodes on this row
                # in_ch = filters[i+1] (upsampled from deeper) + filters[i] * (j+1) (dense skips)
                in_ch = filters[i + 1] + filters[i] * (j + 1)
                row.append(DoubleConv(in_ch, filters[i]))
            self.decoder_nodes.append(row)

        # Upsampling for each decoder node
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Deep supervision output heads (1×1 conv)
        if deep_supervision:
            self.out_heads = nn.ModuleList([
                nn.Conv2d(filters[0], out_ch, 1) for _ in range(self.num_levels - 1)
            ])
        else:
            self.out_conv = nn.Conv2d(filters[0], out_ch, 1)

    def forward(self, x):
        # Encoder: produce X_{0,0}, X_{1,0}, ..., X_{4,0}
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            skips.append(x)
            if i < self.num_levels - 1:
                x = self.pool(x)

        # skips[i] = X_{i,0} (encoder outputs at each level)

        # Decoder: compute nodes column by column
        # nodes[i] stores the last computed node at level i
        nodes = [skips[i] for i in range(self.num_levels)]  # X_{i,0}

        outputs = []

        for col in range(self.num_levels - 1):  # column 1..4
            new_nodes = [None] * self.num_levels
            for i in range(self.num_levels - 1 - col):  # rows that get a new node
                # Upsample from deeper level's previous column
                up = self.up(nodes[i + 1])  # upsample(X_{i+1, col})

                # Dense skip connections: all previous nodes in same row
                inputs = [up]
                for k in range(col + 1):  # X_{i,0}, X_{i,1}, ..., X_{i,col}
                    if k == 0:
                        inputs.append(skips[i])  # X_{i,0}
                    else:
                        # Previously computed node at row i, column k
                        # This needs to come from previous columns
                        pass

                # Actually, I need to track all previously computed nodes
                # Let me restructure...

                cat = torch.cat(inputs, dim=1)
                new_nodes[i] = self.decoder_nodes[i][col](cat)

            # Update nodes with newly computed ones
            for i in range(self.num_levels - 1 - col):
                nodes[i] = new_nodes[i]

            if self.deep_supervision and col > 0:
                outputs.append(self.out_heads[col](nodes[0]))

        # Final output from X_{0,4}
        if self.deep_supervision:
            outputs.append(self.out_heads[-1](nodes[0]))
            return [torch.sigmoid(o) for o in outputs]
        else:
            return torch.sigmoid(self.out_conv(nodes[0]))


class NestedUNetV2(nn.Module):
    """
    Cleaner U-Net++ implementation using explicit node tracking.

    Node grid: X[i][j] where i = depth level (0=top/shallow, 4=bottom/deep)
                           j = column index (0=encoder, 1..4=decoder nodes)

    X[i][j] receives:
      - upsample(X[i+1][j-1]) if j > 0
      - dense skips from X[i][0], X[i][1], ..., X[i][j-1]
    """

    def __init__(self, in_ch=3, out_ch=1, filters=(32, 64, 128, 256, 512)):
        super().__init__()
        self.depth = len(filters)
        self.filters = filters

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder blocks: X_{i,0}
        self.enc_blocks = nn.ModuleList()
        prev = in_ch
        for f in filters:
            self.enc_blocks.append(DoubleConv(prev, f))
            prev = f

        # Decoder blocks: convs[i][j] builds X_{i, j+1}
        # Row i has (depth - 1 - i) decoder nodes
        self.convs = nn.ModuleList()
        for i in range(self.depth - 1):
            row = nn.ModuleList()
            for j in range(self.depth - 1 - i):
                in_dim = filters[i + 1] + filters[i] * (j + 1)
                row.append(DoubleConv(in_dim, filters[i]))
            self.convs.append(row)

        # Deep supervision heads for X_{0,2}, X_{0,3}, X_{0,4}
        # (Paper uses all except X_{0,1}, but we include all for flexibility)
        self.out1 = nn.Conv2d(filters[0], out_ch, 1)  # X_0_1
        self.out2 = nn.Conv2d(filters[0], out_ch, 1)  # X_0_2
        self.out3 = nn.Conv2d(filters[0], out_ch, 1)  # X_0_3
        self.out4 = nn.Conv2d(filters[0], out_ch, 1)  # X_0_4

    def forward(self, x):
        # Encoder
        enc = []
        for i, blk in enumerate(self.enc_blocks):
            x = blk(x)
            enc.append(x)
            if i < self.depth - 1:
                x = self.pool(x)

        # X is a 2D list: X[level][column]
        # Initialize column 0 with encoder outputs
        X = [[None] * self.depth for _ in range(self.depth)]
        for i in range(self.depth):
            X[i][0] = enc[i]

        # Build columns left to right
        for j in range(1, self.depth):
            for i in range(self.depth - j):
                # Upsample from deeper level, previous column
                up = self.up(X[i + 1][j - 1])

                # Dense skip connections from same row, all previous columns
                skip_list = [up]
                for k in range(j):
                    skip_list.append(X[i][k])

                cat = torch.cat(skip_list, dim=1)
                X[i][j] = self.convs[i][j - 1](cat)

        # Deep supervision outputs
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
        print(f"  Output L{i+1}: {o.shape}")
    print(f"Params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
