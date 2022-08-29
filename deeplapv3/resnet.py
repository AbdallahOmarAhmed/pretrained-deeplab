from torch import nn
import timm


class Resnet(nn.Module):
    def __init__(self, backbone, input, pretrained=True):
        super().__init__()
        assert backbone in timm.list_models('resnet*', pretrained=pretrained), backbone + ' is not supported!'
        self.model = timm.create_model(backbone, in_chans=input, pretrained=pretrained, features_only=True)
        for name, layer in self.model.layer4.named_modules():
            if isinstance(layer, nn.Conv2d):
                # name[0] is the block number
                d = 2 ** (int(name[0]) + 1)
                layer.stride = (1, 1)
                layer.dilation = (d, d)

    def forward(self, x):
        return self.model(x)
