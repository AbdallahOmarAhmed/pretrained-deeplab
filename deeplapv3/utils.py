import torch
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


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, img_size, bias=False):
        super(ASPP, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, dilation=6, padding='same', bias=bias)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, dilation=12, padding='same', bias=bias)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.conv4 = nn.Conv2d(in_ch, out_ch, 3, dilation=18, padding='same', bias=bias)
        self.bn4 = nn.BatchNorm2d(out_ch)

        self.pool = Pooling(in_ch, out_ch, img_size, bias)

        self.conv_end = nn.Conv2d(out_ch*5, out_ch, 1, bias=bias)
        self.bn_end = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        x5 = self.pool(x)
        return self.relu(self.bn_end(self.conv_end(torch.cat([x1, x2, x3, x4, x5], dim=1))))
