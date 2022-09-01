import torch
from torch import nn

from utils import *
from backbones import choose_backbone


class Encoder(nn.Module):
    def __init__(self, backbone, in_ch, out_ch, pretrained=True):
        super().__init__()
        self.backbone = choose_backbone(backbone, in_ch, pretrained)
        # in_ch ---> out_ch of the backbone
        in_ch = self.backbone.feature_info.channels()[-1]
        self.aspp = ASPP(in_ch, out_ch)

    def forward(self, x):
        res, x = self.backbone(x)
        x = self.aspp(x)
        return res, x


class Decoder(nn.Module):
    def __init__(self, num_classes, in_ch, reduction=4):
        super().__init__()
        out_ch = in_ch[1]//reduction
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.reduce = nn.Conv2d(in_ch[0], out_ch, 1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(out_ch)

        self.cat_conv = nn.Conv2d(in_ch[1]+out_ch, in_ch[1], 3, bias=False, padding=1)
        self.cat_bn = nn.BatchNorm2d(in_ch[1])

        self.end_conv = nn.Conv2d(in_ch[1], num_classes, 1)

    def forward(self, res, x):
        out = torch.cat((self.up(x), self.reduce_bn(self.reduce(res))), dim=1)
        out = self.up(self.cat_bn(self.cat_conv(out)))
        return self.end_conv(out)

