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
