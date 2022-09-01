import timm
import torch
from torch import nn


def Resnet(backbone, input, pretrained=True):
    assert backbone in timm.list_models('resnet*', pretrained=pretrained), backbone + ' is not supported!'
    model = timm.create_model(backbone, in_chans=input, pretrained=pretrained,
                              features_only=True, output_stride=16, out_indices=(1, 4))
    for name, layer in model.layer4.named_modules():
        if isinstance(layer, nn.Conv2d):
            # name[0] is the block number
            d = 2 ** (int(name[0]) + 1)
            layer.dilation = (d, d)
            layer.padding = 'same'
    return model


def choose_backbone(backbone, input, pretrained=True):
    if backbone.startswith('resnet'):
        return Resnet(backbone, input, pretrained=pretrained)
    else:
        raise Exception(backbone, "is not supported!")


if __name__ == '__main__':
    backbone = input()
    model = choose_backbone(backbone, 3, pretrained=False)
    print(f'backbone: {backbone}')
    print(f'output reduction: {model.feature_info.reduction()}')
    x = torch.randn(1, 3, 256, 256)
    print(f'input size: {x.shape}')
    out = model(x)
    print(f'output size: {out[0].shape, out[1].shape}')
    print(model)
