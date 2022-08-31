import timm
import torch
from torch import nn


def Resnet(backbone, input, pretrained=True):
    assert backbone in timm.list_models('resnet*', pretrained=pretrained), backbone + ' is not supported!'
    model = timm.create_model(backbone, in_chans=input, pretrained=pretrained,
                              features_only=True, output_stride=16, out_indices=(2, 4))
    for name, layer in model.layer4.named_modules():
        if isinstance(layer, nn.Conv2d):
            # name[0] is the block number
            d = 2 ** (int(name[0]) + 1)
            layer.dilation = (d, d)
            layer.padding = 'same'
    return model


