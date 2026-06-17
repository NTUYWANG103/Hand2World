import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x


class SimpleAdapter(nn.Module):
    """PixelUnshuffle (×8) + Conv2d + N ResidualBlocks. Maps (B, C, F, H, W) → (B, out, F, H', W')."""
    def __init__(self, in_dim, out_dim, kernel_size, stride, downscale_factor=8, num_residual_blocks=1):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        self.conv = nn.Conv2d(in_dim * downscale_factor * downscale_factor, out_dim,
                              kernel_size=kernel_size, stride=stride, padding=0)
        self.residual_blocks = nn.Sequential(*[_ResidualBlock(out_dim) for _ in range(num_residual_blocks)])

    def forward(self, x):
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        x = self.residual_blocks(self.conv(self.pixel_unshuffle(x)))
        return x.view(bs, f, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
