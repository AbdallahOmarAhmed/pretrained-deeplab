from torch import nn


class Pooling(nn.Module):
    def __init__(self, in_ch, out_ch, img_size, bias):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.UpsamplingBilinear2d(img_size)

    def forward(self, x):
        return self.up(self.relu(self.bn(self.conv(self.pool(x)))))