import torch
from torch import nn
import timm
import warnings
warnings.filterwarnings("ignore")


class Resnet(nn.Module):
    def __init__(self, backbone, input, pretrained=True):
        super().__init__()
        assert backbone in timm.list_models('resnet*', pretrained=pretrained), backbone + ' is not supported!'
        self.model = timm.create_model(backbone,
                                       in_chans=input,
                                       pretrained=pretrained,
                                       features_only=True,
                                       output_stride=16,
                                       out_indices=(2, 4))
        for name, layer in self.model.layer4.named_modules():
            if isinstance(layer, nn.Conv2d):
                # name[0] is the block number
                d = 2 ** (int(name[0]) + 1)
                layer.dilation = (d, d)
                layer.padding = 'same'

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    backbone = 'resnet18'
    model = Resnet(backbone, 3, pretrained=False)
    print(f'backbone: {backbone}')
    print(f'output reduction: {model.model.feature_info.reduction()}')
    x = torch.randn(1, 3, 256, 256)
    print(f'input size: {x.shape}')
    out = model(x)
    print(f'output size: {out[0].shape, out[1].shape}')
    print(model)
