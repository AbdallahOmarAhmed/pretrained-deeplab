from .utils import *
from .backbones import choose_backbone
import warnings
warnings.filterwarnings("ignore")


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

    def res_channels(self):
        return self.backbone.feature_info.channels()[0]


class Decoder(nn.Module):
    def __init__(self, num_classes, in_ch1, in_ch2):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.reduce = nn.Conv2d(in_ch1, in_ch1, 1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(in_ch1)

        self.cat_conv = nn.Conv2d(in_ch2+in_ch1, in_ch2, 3, bias=False, padding=1)
        self.cat_bn = nn.BatchNorm2d(in_ch2)

        self.conv = nn.Conv2d(in_ch2, in_ch2, 3, bias=False, padding=1)
        self.bn = nn.BatchNorm2d(in_ch2)

        self.end_conv = nn.Conv2d(in_ch2, num_classes, 1)

    def forward(self, res, x):
        out = torch.cat((self.up(x), self.relu(self.reduce_bn(self.reduce(res)))), dim=1)
        out = self.relu(self.cat_bn(self.cat_conv(out)))
        out = self.end_conv(self.relu(self.bn(self.conv(out))))
        return self.up(out)


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, input, num_classes, encoder_out=256, pretrained=None):
        super().__init__()

        # info
        self.backbone = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained == 'all'
        self.backbone_pretrained = pretrained == 'backbone'

        self.encoder = Encoder(backbone, input, encoder_out, self.backbone_pretrained)
        res_ch = self.encoder.res_channels()
        self.decoder = Decoder(num_classes, res_ch, encoder_out)

    def forward(self, x):
        res, x = self.encoder(x)
        out = self.decoder(res, x)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self):
        print('backbone :', self.backbone)
        print('parameters :', self.count_parameters())
        print('number of classes :', self.num_classes)
        if self.pretrained:
            print('is pretrained : yes')
        else:
            print('backbone pretrained :', self.backbone_pretrained)
            

if __name__ == '__main__':
    backbone = input()
    model = DeepLabV3Plus(backbone, 3, 10)
    print(model.count_parameters())
